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
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits.lingbotworld import (
    CausalLingBotWorldVideoConfig)
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
        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(q, cos, sin,
                                        is_neox_style=False).type_as(v)
        roped_key = _apply_rotary_emb(k, cos, sin,
                                      is_neox_style=False).type_as(v)

        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        num_new_tokens = q.shape[1]
        kv_cache_size = kv_cache["k"].shape[1]

        if max_attention_size is None:
            if self.local_attn_size == -1:
                max_attention_size = kv_cache_size
            else:
                max_attention_size = self.local_attn_size * frame_seqlen

        global_end_index = int(kv_cache["global_end_index"].item())
        local_end_index_prev = int(kv_cache["local_end_index"].item())

        if self.local_attn_size == -1:
            # Global attention: cache covers the full video.
            local_end_index = (local_end_index_prev + current_end -
                               global_end_index)
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
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
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = (local_end_index_prev + current_end -
                               global_end_index)
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        key_window = kv_cache["k"][:,
                                   max(0, local_end_index -
                                       max_attention_size):local_end_index]
        value_window = kv_cache["v"][:,
                                     max(0, local_end_index -
                                         max_attention_size):local_end_index]

        x = self.attn(roped_query, key_window, value_window)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        return x


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
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
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
                                          eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
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

        # Rotary embeddings with absolute start frame offset
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
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
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
