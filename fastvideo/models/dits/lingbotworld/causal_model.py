# SPDX-License-Identifier: Apache-2.0
"""
Causal LingBot-World transformer (Fast / DMD-distilled variant).

Adapted from the official LingBot-World implementation:
https://github.com/Robbyant/lingbot-world/blob/main/wan/modules/model_fast.py

Differences from ``LingBotWorldTransformer3DModel`` (base-cam):
  * Block-causal self-attention with a per-layer KV cache
    (CausVid-style; see https://arxiv.org/abs/2412.07772).
  * Image conditioning via channel concatenation: the transformer
    consumes 36 channels (16 noise + 4 mask + 16 VAE latents).
  * Designed for chunked autoregressive inference driven by
    ``LingBotCausalDMDDenoisingStage``.
"""

import math
import os
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits.lingbotworld import (
    CausalLingBotWorldVideoConfig)
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_all_to_all_4D,
    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (PatchEmbed,
                                               WanCamControlPatchEmbedding)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.lingbotworld.model import (
    LingBotWorldCamConditioner)
from fastvideo.models.dits.wanvideo import (WanT2VCrossAttention,
                                            WanTimeTextImageEmbedding)
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)


class CausalLingBotSelfAttention(nn.Module):
    """Block-causal self-attention with a rolling KV cache.

    Mirrors the official ``CausalWanSelfAttention`` in model_fast.py:
      * ``local_attn_size == -1``: the cache holds the entire video
        (global attention over all previously generated frames).
      * ``local_attn_size > 0``: the cache rolls, keeping the first
        ``sink_size`` frames pinned as an attention sink.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 eps: float = 1e-6) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.eps = eps

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: dict,
        current_start: int = 0,
        frame_seqlen: int = 1560,
        max_attention_size: int | None = None,
    ) -> torch.Tensor:
        # Ulysses sequence parallelism: incoming q/k/v are
        # [B, L_local, H, d] (sequence-sharded, all heads).  All-to-all
        # swaps to [B, L_full, H/P, d] so each rank attends over the
        # full sequence for its head slice; the KV cache is then
        # rank-local ([B, kv_size, H/P, d]) with no cross-rank
        # coordination (matching official image2video_fast.py).
        #
        # Comm fusion: q/k/v are concatenated along head_dim (the last
        # dim) so ONE all-to-all moves all three. Scattering happens on
        # the head dim (dim 2), so the per-head [q|k|v] feature concat
        # is preserved on every rank; a single split recovers the
        # tensors. 1 NCCL launch instead of 3 with a 3x larger message
        # utilizes the PIX link far better at small chunk sizes.
        sp_size = get_sp_world_size()
        if sp_size > 1:
            head_dim = q.shape[-1]
            qkv = torch.cat([q, k, v], dim=-1)
            qkv = sequence_model_parallel_all_to_all_4D(
                qkv, scatter_dim=2, gather_dim=1)
            q, k, v = qkv.split(head_dim, dim=-1)
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(q, cos, sin,
                                        is_neox_style=False).type_as(v)
        roped_key = _apply_rotary_emb(k, cos, sin,
                                      is_neox_style=False).type_as(v)

        # Optional fp8 KV cache: cache tensors are float8_e4m3fn with
        # PER-HEAD scales calibrated on the first write (K is RoPE'd -
        # rotation preserves norm - and V is a projection output, so
        # per-head ranges are stable across a session). Per-head scales
        # cut quantization noise vs per-tensor: attention errors
        # compound chunk-to-chunk through the t=0 KV refresh, so cache
        # fidelity directly bounds long-rollout drift.
        #
        # flashinfer's fa2 kernel only takes SCALAR k/v scales, so the
        # per-head factors are folded outside the kernel instead:
        #   QK^T: q_h' = q_h * k_scale_h  (per-head scale commutes into q)
        #   PV:   out_h' = out_h * v_scale_h (softmax weights sum to 1,
        #         so the V scale factors out of the weighted sum)
        # The kernel then runs with k_scale=v_scale=1.
        # LINGBOT_CUDA_GRAPH: manual CUDA-graph capture of the steady
        # rolled streaming state. During capture/replay no host sync is
        # allowed, so cache positions come from PYTHON ints kept in the
        # cache dict (py_global_end/py_local_end) instead of tensor
        # .item() reads, tensor position updates are skipped (baked
        # constants are stable in the steady rolled state), and fp8
        # scale recalibration is frozen (kv_cache["freeze_scales"]).
        graph_mode = (os.environ.get("LINGBOT_CUDA_GRAPH",
                                     "false").lower() == "true"
                      or os.environ.get("LINGBOT_STEP_GRAPH",
                                        "false").lower() == "true")
        fp8_kv = kv_cache["k"].dtype == torch.float8_e4m3fn
        if fp8_kv and kv_cache.get("freeze_scales"):
            k_scale = kv_cache["k_scale"].view(1, 1, -1, 1)
            v_scale = kv_cache["v_scale"].view(1, 1, -1, 1)
            k_write = (roped_key.float() / k_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn)
            v_write = (v.float() / v_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn)
        elif fp8_kv:
            # Growth-only running-max recalibration. The original
            # first-chunk absmax calibration clamped later chunks at
            # +-448 whenever activations exceeded the initial range
            # (user-visible quality loss on long rollouts). Now every
            # write updates the per-head scale; when a head's range
            # grows, the already-cached fp8 content for that head is
            # rescaled in place (dequant-requant by old/new ratio -
            # values only SHRINK, so no clamping, and the rescale
            # fires rarely once ranges stabilize).
            #
            # FP8_KV_PERCENTILE (e.g. "0.999"): calibrate the per-head
            # scale on that abs-value quantile instead of absmax. A
            # single outlier no longer stretches the head's whole e4m3
            # grid; values beyond the percentile clamp at +-448 (rare
            # by construction) while the bulk of the distribution gets
            # a finer grid. Standard KV-quant tradeoff; env-gated for
            # visual A/B.
            pct = float(os.environ.get("FP8_KV_PERCENTILE", "0") or 0)
            if 0.0 < pct < 1.0:
                # Subsample tokens 4x to bound the per-write quantile
                # (sort) cost; K/V head ranges are statistically
                # stationary across tokens so the subsample is
                # representative.
                k_abs = roped_key[:, ::4].float().abs()
                v_abs = v[:, ::4].float().abs()
                k_new = (k_abs.permute(2, 0, 1, 3).reshape(
                    k_abs.shape[2], -1).quantile(pct, dim=1) /
                         448.0).clamp(min=1e-6)
                v_new = (v_abs.permute(2, 0, 1, 3).reshape(
                    v_abs.shape[2], -1).quantile(pct, dim=1) /
                         448.0).clamp(min=1e-6)
            else:
                k_new = (roped_key.float().abs().amax(dim=(0, 1, 3)) /
                         448.0).clamp(min=1e-6)
                v_new = (v.float().abs().amax(dim=(0, 1, 3)) /
                         448.0).clamp(min=1e-6)
            valid = (kv_cache.get("py_local_end", 0) if graph_mode else
                     int(kv_cache["local_end_index"].item()))
            for name, new_scale in (("k_scale", k_new), ("v_scale",
                                                         v_new)):
                old_scale = kv_cache[name]
                grew = new_scale > old_scale
                if bool(grew.any().item()):
                    if valid > 0:
                        buf = kv_cache[name[0]]  # "k" or "v"
                        ratio = torch.where(
                            grew, old_scale /
                            new_scale.clamp(min=1e-6),
                            torch.ones_like(old_scale)).view(
                                1, 1, -1, 1).to(torch.bfloat16)
                        # Rescale in small token-slices with bf16
                        # arithmetic: upcasting a multi-GB fp8 cache
                        # in one shot would transiently allocate 2-4x
                        # its size and OOM at the 81f memory ceiling.
                        step = 8192
                        for s in range(0, valid, step):
                            e = min(s + step, valid)
                            buf[:, s:e] = (
                                buf[:, s:e].to(torch.bfloat16) *
                                ratio).to(torch.float8_e4m3fn)
                    kv_cache[name].copy_(
                        torch.maximum(old_scale, new_scale))
            k_scale = kv_cache["k_scale"].view(1, 1, -1, 1)
            v_scale = kv_cache["v_scale"].view(1, 1, -1, 1)
            k_write = (roped_key.float() / k_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn)
            v_write = (v.float() / v_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn)
        else:
            k_write = roped_key
            v_write = v

        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        num_new_tokens = q.shape[1]
        kv_cache_size = kv_cache["k"].shape[1]

        if max_attention_size is None:
            if self.local_attn_size == -1:
                max_attention_size = kv_cache_size
            else:
                max_attention_size = self.local_attn_size * frame_seqlen

        if graph_mode:
            global_end_index = kv_cache.get("py_global_end", 0)
            local_end_index_prev = kv_cache.get("py_local_end", 0)
        else:
            global_end_index = int(kv_cache["global_end_index"].item())
            local_end_index_prev = int(kv_cache["local_end_index"].item())

        if self.local_attn_size == -1:
            # Global attention: cache covers the full video.
            local_end_index = (local_end_index_prev + current_end -
                               global_end_index)
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k_write
            kv_cache["v"][:, local_start_index:local_end_index] = v_write
        elif (current_end > global_end_index) and (
                num_new_tokens + local_end_index_prev > kv_cache_size):
            # Roll the cache, keeping sink tokens pinned.
            num_evicted_tokens = (num_new_tokens + local_end_index_prev -
                                  kv_cache_size)
            num_rolled_tokens = (local_end_index_prev - num_evicted_tokens -
                                 sink_tokens)
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:
                              sink_tokens + num_evicted_tokens +
                              num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:
                              sink_tokens + num_evicted_tokens +
                              num_rolled_tokens].clone()
            local_end_index = (local_end_index_prev + current_end -
                               global_end_index - num_evicted_tokens)
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k_write
            kv_cache["v"][:, local_start_index:local_end_index] = v_write
        else:
            local_end_index = (local_end_index_prev + current_end -
                               global_end_index)
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k_write
            kv_cache["v"][:, local_start_index:local_end_index] = v_write

        key_window = kv_cache["k"][:,
                                   max(0, local_end_index -
                                       max_attention_size):local_end_index]
        value_window = kv_cache["v"][:,
                                     max(0, local_end_index -
                                         max_attention_size):local_end_index]

        if fp8_kv:
            x = self._fp8_attention(roped_query, key_window, value_window,
                                    kv_cache)
        else:
            x = self.attn(roped_query, key_window, value_window)

        if graph_mode:
            kv_cache["py_global_end"] = current_end
            kv_cache["py_local_end"] = local_end_index
        else:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        if sp_size > 1:
            # [B, L_full, H/P, d] -> [B, L_local, H, d]
            x = sequence_model_parallel_all_to_all_4D(
                x, scatter_dim=1, gather_dim=2)

        return x

    @torch.compiler.disable
    def _fp8_attention(self, q: torch.Tensor, k_fp8: torch.Tensor,
                       v_fp8: torch.Tensor,
                       kv_cache: dict) -> torch.Tensor:
        """Attention over the fp8 KV cache window.

        Default: custom Triton prefill kernel (vLLM c9e50123 pattern)
        that loads fp8 tiles from HBM and dequantizes IN REGISTERS
        with the per-head scales - no q/output scale-fold passes, no
        dequantized copy in global memory. Beats the flashinfer
        fold-outside path by ~15% at chunk-sized windows.

        Fallback (FP8_KV_TRITON=false): flashinfer fa2 with per-head
        factors folded outside the kernel (k_scale_h into q via QK^T
        commutation, v_scale_h onto the output).
        """
        assert q.shape[0] == 1, "fp8 KV attention path assumes batch=1"
        if os.environ.get("FP8_KV_TRITON", "true").lower() == "true":
            from fastvideo.models.dits.lingbotworld.triton_fp8_attention import (  # noqa: E501
                triton_fp8_prefill)
            out = triton_fp8_prefill(
                q.squeeze(0).contiguous(),
                k_fp8.squeeze(0),
                v_fp8.squeeze(0),
                kv_cache["k_scale"],
                kv_cache["v_scale"],
            )
            return out.unsqueeze(0)

        from flashinfer import single_prefill_with_kv_cache
        head_dim = q.shape[-1]
        sm_scale = head_dim**-0.5
        k_scale = kv_cache["k_scale"].view(1, -1, 1)  # [1, H, 1]
        v_scale = kv_cache["v_scale"].view(1, -1, 1)
        q_scaled = (q.squeeze(0).float() * k_scale).to(q.dtype)
        out = single_prefill_with_kv_cache(
            q_scaled,
            k_fp8.squeeze(0),
            v_fp8.squeeze(0),
            causal=False,
            kv_layout="NHD",
            o_dtype=q.dtype,
            sm_scale=sm_scale,
        )
        out = (out.float() * v_scale).to(q.dtype)
        return out.unsqueeze(0)


class CausalLingBotWorldTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = True,
                 eps: float = 1e-6,
                 quant_config=None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.to_q")
        self.to_k = ReplicatedLinear(dim, dim, bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.to_k")
        self.to_v = ReplicatedLinear(dim, dim, bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.to_v")
        self.to_out = ReplicatedLinear(dim, dim, bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.to_out")
        self.attn1 = CausalLingBotSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            eps=eps)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads

        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise NotImplementedError(
                f"QK Norm type '{qk_norm}' not supported")
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 2. Cross-attention (T2V style; image conditioning is via
        # channel concat, not cross attention)
        self.attn2 = WanT2VCrossAttention(dim,
                                          num_heads,
                                          qk_norm=qk_norm,
                                          eps=eps,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.attn2")
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh",
                       quant_config=quant_config,
                       prefix=f"{prefix}.ffn")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)
        self.cam_conditioner = LingBotWorldCamConditioner(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        frame_seqlen: int | None = None,
        max_attention_size: int | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden_states: [B, L, D]; temb: [B, temb_seq_len, 6, D]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        temb_seq_len = temb.shape[1]
        tokens_per_temb = hidden_states.shape[1] // temb_seq_len
        if frame_seqlen is None:
            frame_seqlen = tokens_per_temb
        else:
            frame_seqlen = int(frame_seqlen)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        e = self.scale_shift_table + temb.float()
        assert e.shape == (bs, temb_seq_len, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            e.chunk(6, dim=2)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()).unflatten(
                dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
            (1 + scale_msa) + shift_msa).flatten(1, 2).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            kv_cache,
            current_start=current_start,
            frame_seqlen=frame_seqlen,
            max_attention_size=max_attention_size,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0],
                                               device=hidden_states.device)
        # gate_msa: [B, temb_seq_len, 1, D] -> broadcast over tokens
        gate = gate_msa.squeeze(2)
        if temb_seq_len == 1:
            gate = gate  # [B, 1, D] broadcasts over L
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # Camera conditioning (after self-attention residual update)
        if c2ws_plucker_emb is not None:
            hidden_states = self.cam_conditioner(hidden_states,
                                                 c2ws_plucker_emb)
            norm_hidden_states = self.self_attn_residual_norm.norm(
                hidden_states).to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa.squeeze(2),
            c_scale_msa.squeeze(2))
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output,
                                          c_gate_msa.squeeze(2))
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class CausalLingBotWorldTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = CausalLingBotWorldVideoConfig(
    )._fsdp_shard_conditions
    _compile_conditions = CausalLingBotWorldVideoConfig()._compile_conditions
    _supported_attention_backends = CausalLingBotWorldVideoConfig(
    )._supported_attention_backends
    param_names_mapping = CausalLingBotWorldVideoConfig().param_names_mapping
    reverse_param_names_mapping = CausalLingBotWorldVideoConfig(
    ).reverse_param_names_mapping
    lora_param_names_mapping = CausalLingBotWorldVideoConfig(
    ).lora_param_names_mapping

    def __init__(self, config: CausalLingBotWorldVideoConfig,
                 hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.arch_config.local_attn_size
        self.sink_size = config.arch_config.sink_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)
        self.patch_embedding_wancamctrl = WanCamControlPatchEmbedding(
            in_chans=6 * 64,
            embed_dim=inner_dim,
            patch_size=config.patch_size)
        self.c2ws_mlp = MLP(inner_dim,
                            inner_dim,
                            inner_dim,
                            bias=True,
                            act_type="silu")

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        quant_config = getattr(config, "quant_config", None)
        self.blocks = nn.ModuleList([
            CausalLingBotWorldTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.arch_config.local_attn_size,
                config.arch_config.sink_size,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                quant_config=quant_config,
                prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32,
                                            compute_dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        self.independent_first_frame = False

        self.__post_init__()

    def _compute_rope(self, post_patch_num_frames: int,
                      post_patch_height: int, post_patch_width: int,
                      start_frame: int,
                      device: torch.device) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
        """fp64-accurate RoPE table, returned as fp32 on device."""
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=(torch.float32
                   if current_platform.is_mps() else torch.float64),
            rope_theta=10000,
            start_frame=start_frame,
        )
        return (freqs_cos.to(device=device, dtype=torch.float32),
                freqs_sin.to(device=device, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        start_frame: int = 0,
        y: torch.Tensor | None = None,
        max_attention_size: int | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Run one chunk with KV caching.

        Args:
            hidden_states: noise latents [B, 16, F_chunk, H, W]
            y: conditioning latents [B, 20, F_chunk, H, W]
               (4 mask channels + 16 VAE-encoded channels).
               Concatenated with hidden_states along channels
               before the patch embedding (total 36 = in_channels).
            c2ws_plucker_emb: raw camera Plucker embedding chunk
               [B, C, F_chunk, H, W] (optional).
        """
        if kv_cache is None:
            raise NotImplementedError(
                "CausalLingBotWorldTransformer3DModel only supports "
                "inference with a KV cache.")

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        if y is not None:
            hidden_states = torch.cat(
                [hidden_states, y.to(hidden_states.dtype)], dim=1)

        batch_size, num_channels, num_frames, height, width = \
            hidden_states.shape
        assert num_channels == self.in_channels, (
            f"Expected {self.in_channels} input channels "
            f"(noise + conditioning), got {num_channels}. "
            "Pass `y` conditioning latents.")
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_seqlen = post_patch_height * post_patch_width

        # Rotary embeddings with absolute start frame offset. The table
        # is generated in float64 for accuracy but CACHED and applied in
        # float32: leaving cos/sin in float64 silently promotes the
        # whole RoPE apply over q/k to double precision every forward
        # (~7% of GPU time in the nsys profile).
        #
        # CUDA-graph mode: replay skips ALL python, so per-chunk tables
        # must live at a FIXED address. When `_static_rope` buffers are
        # installed (by the streaming graph harness, which copies the
        # chunk's table in before each replay), use them directly.
        static_rope = getattr(self, "_static_rope", None)
        if static_rope is not None:
            freqs_cos, freqs_sin = static_rope
        else:
            rope_key = (post_patch_num_frames, post_patch_height,
                        post_patch_width, int(start_frame))
            rope_cache = getattr(self, "_rope_cache", None)
            if rope_cache is None:
                rope_cache = {}
                self._rope_cache = rope_cache
            cached = rope_cache.get(rope_key)
            if cached is None:
                freqs_cos, freqs_sin = self._compute_rope(
                    post_patch_num_frames, post_patch_height,
                    post_patch_width, start_frame,
                    hidden_states.device)
                # Bounded cache: chunked generation revisits few
                # distinct (shape, start_frame) keys, but guard against
                # unbounded growth in long interactive sessions.
                if len(rope_cache) > 64:
                    rope_cache.clear()
                rope_cache[rope_key] = (freqs_cos, freqs_sin)
            else:
                freqs_cos, freqs_sin = cached
        freqs_cis = (freqs_cos, freqs_sin)

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Camera conditioning embedding
        c2ws_hidden = None
        if c2ws_plucker_emb is not None:
            c2ws_hidden = self.patch_embedding_wancamctrl(
                c2ws_plucker_emb.to(device=hidden_states.device,
                                    dtype=orig_dtype))
            c2ws_hidden = c2ws_hidden + self.c2ws_mlp(c2ws_hidden)

        # Ulysses SP: shard the chunk tokens across ranks.  Attention
        # all-to-alls to head-sharding internally; everything else
        # (norms, FFN, cross-attn) is pointwise over tokens.
        sp_size = get_sp_world_size()
        original_seq_len = hidden_states.shape[1]
        if sp_size > 1:
            assert hidden_states.shape[1] % sp_size == 0, (
                f"Chunk token count {hidden_states.shape[1]} must be "
                f"divisible by sp_size {sp_size}")
            hidden_states, original_seq_len = sequence_model_parallel_shard(
                hidden_states, dim=1)
            if c2ws_hidden is not None:
                c2ws_hidden, _ = sequence_model_parallel_shard(
                    c2ws_hidden, dim=1)

        # Text embedding (pad to text_len)
        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                encoder_hidden_states.size(0),
                self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2))
        ],
                                          dim=1)

        temb, timestep_proj, encoder_hidden_states, _ = \
            self.condition_embedder(timestep.flatten(),
                                    encoder_hidden_states, None)
        timestep_proj = timestep_proj.unflatten(
            1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps() else encoder_hidden_states
        assert encoder_hidden_states.dtype == orig_dtype

        # Transformer blocks
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                kv_cache=kv_cache[block_index],
                crossattn_cache=(crossattn_cache[block_index]
                                 if crossattn_cache is not None else None),
                current_start=current_start,
                frame_seqlen=frame_seqlen,
                max_attention_size=max_attention_size,
                c2ws_plucker_emb=c2ws_hidden,
            )

        # Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(
            2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)

        # Ulysses SP: gather sequence shards back to the full chunk
        if sp_size > 1:
            hidden_states = sequence_model_parallel_all_gather_with_unpad(
                hidden_states, original_seq_len, dim=1)

        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def unpatchify(self, x, grid_sizes):
        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


# Entry point for model registry
EntryClass = CausalLingBotWorldTransformer3DModel
