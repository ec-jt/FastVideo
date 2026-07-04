# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World causal DMD (Fast) image-to-video pipeline.

Ports the official chunked autoregressive inference loop from
https://github.com/Robbyant/lingbot-world/blob/main/wan/image2video_fast.py
onto FastVideo's modular pipeline architecture.

Key properties (from FastVideo/LingBot-World-Fast-Diffusers):
  * Single causal transformer (36 input channels =
    16 noise + 4 mask + 16 VAE conditioning), no transformer_2.
  * 4 DMD denoising timesteps on a shift-5.0 flow-match schedule.
  * Per-chunk KV cache (3 latent frames per chunk); after each chunk
    the clean latents are re-run at timestep 0 to refresh the cache.
  * Image conditioning via channel concatenation (`y`), not by
    replacing the first latent frame.
"""

import os
import sys

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage)
from fastvideo.pipelines.stages.causal_denoising import (
    CausalDMDDenosingStage)

logger = init_logger(__name__)


class LingBotCausalDMDDenoisingStage(CausalDMDDenosingStage):
    """Causal DMD denoising with LingBot-style `y` channel conditioning."""

    def _encode_conditioning(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the 20-channel conditioning tensor `y`.

        y = concat([mask (4ch), vae_latent (16ch)], dim=1)
        following the official image2video_fast.py.
        """
        device = get_local_torch_device()
        assert batch.pil_image is not None, (
            "LingBot-World Fast is an I2V model; provide image_path.")
        # batch.pil_image was converted by InputValidationStage
        # (is_causal=True path) into a normalized tensor [1, 3, 1, H, W].
        img = batch.pil_image
        assert isinstance(img, torch.Tensor) and img.dim() == 5

        num_pixel_frames = batch.num_frames
        height, width = batch.height, batch.width

        video_condition = torch.cat([
            img.to(device=device, dtype=torch.float32),
            img.new_zeros(img.shape[0], img.shape[1], num_pixel_frames - 1,
                          height, width).to(device=device,
                                            dtype=torch.float32)
        ],
                                    dim=2)

        self.vae = self.vae.to(device)
        latent_condition = self.vae.encode(video_condition).mean.float()

        # Normalize into the transformer's latent space (inverse of
        # DecodingStage._denormalize_latents).
        cfg = getattr(self.vae, "config", None)
        if (cfg is not None and hasattr(cfg, "latents_mean")
                and hasattr(cfg, "latents_std")
                and cfg.latents_mean is not None):
            latents_mean = torch.tensor(cfg.latents_mean,
                                        device=device,
                                        dtype=latent_condition.dtype).view(
                                            1, -1, 1, 1, 1)
            latents_std = torch.tensor(cfg.latents_std,
                                       device=device,
                                       dtype=latent_condition.dtype).view(
                                           1, -1, 1, 1, 1)
            latent_condition = (latent_condition -
                                latents_mean) / latents_std
        else:
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                latent_condition = latent_condition - self.vae.shift_factor
            if hasattr(self.vae, "scaling_factor"):
                latent_condition = (latent_condition *
                                    self.vae.scaling_factor)

        if fastvideo_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

        # Truncate the conditioning latents to the (possibly truncated)
        # latent frame count used for generation.
        latent_condition = latent_condition[:, :, :num_latent_frames]

        # Build the 4-channel first-frame mask (official msk logic)
        temporal_ratio = 4
        msk = torch.ones(1,
                         num_pixel_frames,
                         latent_height,
                         latent_width,
                         device=device)
        msk[:, 1:] = 0
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=temporal_ratio,
                                    dim=1), msk[:, 1:]
        ],
                        dim=1)
        msk = msk.view(1, msk.shape[1] // temporal_ratio, temporal_ratio,
                       latent_height, latent_width)
        msk = msk.transpose(1, 2)  # [1, 4, T_lat, H, W]
        msk = msk[:, :, :num_latent_frames]

        y = torch.cat([msk, latent_condition], dim=1)  # [1, 20, T, H, W]
        return y.to(target_dtype)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_size = self.transformer.config.arch_config.patch_size
        patch_ratio = patch_size[-1] * patch_size[-2]
        self.frame_seq_length = latent_seq_length // patch_ratio

        # DMD timesteps
        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(),
                 torch.tensor([0], dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents  # [B, 16, T, H, W]

        # Truncate latent frames to a multiple of the chunk size
        # (official: lat_f -= lat_f % chunk_size).
        t_full = latents.shape[2]
        t = t_full - (t_full % self.num_frames_per_block)
        if t != t_full:
            logger.info(
                "[LingBot-Fast] Truncating latent frames %d -> %d "
                "(multiple of %d)", t_full, t, self.num_frames_per_block)
            latents = latents[:, :, :t]
        b, c, t, h, w = latents.shape

        prompt_embeds = batch.prompt_embeds
        if not isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = prompt_embeds[0]
        assert torch.isnan(prompt_embeds).sum() == 0

        # Conditioning tensor y (mask + VAE latents)
        y = self._encode_conditioning(batch, fastvideo_args, t, h, w,
                                      target_dtype)

        # KV caches.
        # Official image2video_fast.py: with local_attn_size == -1 the
        # cache covers the FULL video (frame_seqlen * lat_f), not the
        # sliding window.  The base class sizes by
        # sliding_window_num_frames, which overflows for videos longer
        # than the window, so size it explicitly here.
        kv_cache = self._initialize_full_kv_cache(
            batch_size=b,
            num_latent_frames=t,
            dtype=target_dtype,
            device=latents.device)
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=b,
            max_text_len=fastvideo_args.pipeline_config.
            text_encoder_configs[0].arch_config.text_len,
            dtype=target_dtype,
            device=latents.device)

        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        max_attention_size = kv_cache[0]["k"].shape[1]

        with self.progress_bar(total=len(block_sizes) *
                               len(timesteps)) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[:, :, start_index:start_index +
                                          current_num_frames]
                current_y = y[:, :, start_index:start_index +
                              current_num_frames]

                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)
                    t_expand = t_cur.repeat(latent_model_input.shape[0])

                    with torch.autocast(device_type="cuda",
                                        dtype=target_dtype,
                                        enabled=autocast_enabled), \
                        set_forward_context(current_timestep=i,
                                            attn_metadata=None,
                                            forward_batch=batch):
                        t_expanded_noise = t_cur * torch.ones(
                            (latent_model_input.shape[0], 1),
                            device=latent_model_input.device,
                            dtype=torch.long)
                        pred_noise_btchw = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=start_index *
                            self.frame_seq_length,
                            start_frame=start_index,
                            y=current_y,
                            max_attention_size=max_attention_size,
                        ).permute(0, 2, 1, 3, 4)

                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler).unflatten(
                            0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1],
                            dtype=torch.long,
                            device=pred_video_btchw.device)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(batch.generator[0] if isinstance(
                                batch.generator, list) else
                                       batch.generator)).to(self.device)
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise.flatten(0, 1), next_timestep).unflatten(
                                0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
                            0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                # Write back the clean chunk
                latents[:, :, start_index:start_index +
                        current_num_frames] = current_latents

                # Refresh KV cache with the clean chunk at timestep 0
                context_noise = getattr(fastvideo_args.pipeline_config,
                                        "context_noise", 0)
                t_context = torch.ones(
                    [b, 1], device=latents.device,
                    dtype=torch.long) * int(context_noise)
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=None,
                                        forward_batch=batch):
                    self.transformer(
                        current_latents.to(target_dtype),
                        prompt_embeds,
                        t_context,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start_index * self.frame_seq_length,
                        start_frame=start_index,
                        y=current_y,
                        max_attention_size=max_attention_size,
                    )

                start_index += current_num_frames

        batch.latents = latents

        # The KV caches for a 21-latent-frame rollout are multi-GB
        # (13.4GB fp8 / 26.8GB bf16); free them eagerly so back-to-back
        # generations do not OOM while the previous run's caches are
        # still referenced by this frame's locals.
        for entry in kv_cache:
            entry.clear()
        for entry in crossattn_cache:
            entry.clear()
        del kv_cache, crossattn_cache, y
        torch.cuda.empty_cache()

        return batch


    def _initialize_full_kv_cache(self, batch_size, num_latent_frames,
                                  dtype, device) -> list[dict]:
        """Allocate a KV cache covering the full video.

        Used when local_attn_size == -1 (global attention over all
        previously generated frames), matching the official
        image2video_fast.py: kv_size = frame_seqlen * lat_f.

        Under Ulysses SP the cache is head-sharded: each rank stores
        num_heads // sp_size heads for the full sequence (official:
        self_kv_shape = [B, kv_size, local_num_heads, head_dim]).

        With FP8_KV_CACHE=true the cache is float8_e4m3fn (half the
        memory and read bandwidth) with per-layer scalar k/v scales
        calibrated on first write; attention reads it via flashinfer
        with scales folded into softmax.
        """
        sp_size = get_sp_world_size()
        num_attention_heads = (self.transformer.num_attention_heads //
                               sp_size)
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * num_latent_frames

        fp8_kv = os.environ.get("FP8_KV_CACHE",
                                "false").lower() == "true"
        kv_dtype = torch.float8_e4m3fn if fp8_kv else dtype
        if fp8_kv:
            logger.info("[LingBot-Fast] Using fp8 (e4m3) KV cache")

        cache = []
        for _ in range(self.num_transformer_blocks):
            # torch.empty, not zeros: the attention read window
            # [local_end - max_attention_size, local_end] only ever
            # covers previously written tokens, so zero-init is pure
            # overhead (nsys showed the FillFunctor memsets at ~4% of
            # GPU time, re-issued for every generation).
            entry = {
                "k":
                torch.empty([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=kv_dtype,
                            device=device),
                "v":
                torch.empty([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=kv_dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            }
            if fp8_kv:
                # Per-head scales, [num_local_heads]; 0.0 means "not
                # yet calibrated" - the attention layer fills these on
                # the first write.
                entry["k_scale"] = torch.zeros([num_attention_heads],
                                               dtype=torch.float32,
                                               device=device)
                entry["v_scale"] = torch.zeros([num_attention_heads],
                                               dtype=torch.float32,
                                               device=device)
            cache.append(entry)
        return cache


class TAEHVDecodingStage(DecodingStage):
    """Decoding stage with optional TAEHV tiny-VAE decode.

    When ``USE_TAEHV_DECODE=true``, decodes latents with madebyollin's
    TAEHV (taew2_1 weights - LingBot-World uses the Wan 2.1 VAE) which
    is ~5x faster and needs <0.5GB peak vs the full AutoencoderKLWan
    decoder. TAEHV operates on the NORMALIZED diffusion latent space
    directly (no denormalize step). Falls back to the full VAE when
    the flag is unset or loading fails.

    Env:
      USE_TAEHV_DECODE: "true" to enable (default false)
      TAEHV_DIR:        dir containing taehv.py (default /taehv)
      TAEHV_WEIGHTS:    checkpoint (default $TAEHV_DIR/taew2_1.pth)
    """

    def __init__(self, vae, pipeline=None) -> None:
        super().__init__(vae, pipeline)
        self._taehv = None
        self._taehv_failed = False

    def _get_taehv(self, device: torch.device):
        if self._taehv is not None or self._taehv_failed:
            return self._taehv
        try:
            taehv_dir = os.environ.get("TAEHV_DIR", "/taehv")
            weights = os.environ.get(
                "TAEHV_WEIGHTS", os.path.join(taehv_dir, "taew2_1.pth"))
            if taehv_dir not in sys.path:
                sys.path.insert(0, taehv_dir)
            from taehv import TAEHV  # type: ignore[import-not-found]
            model = TAEHV(checkpoint_path=weights)
            model = model.to(device=device, dtype=torch.float16).eval()
            self._taehv = model
            logger.info("TAEHV decoder loaded from %s", weights)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "TAEHV decode requested but failed to load (%s); "
                "falling back to full VAE decode.", e)
            self._taehv_failed = True
        return self._taehv

    @torch.no_grad()
    def decode(self, latents: torch.Tensor,
               fastvideo_args: FastVideoArgs) -> torch.Tensor:
        if os.environ.get("USE_TAEHV_DECODE", "false").lower() != "true":
            return super().decode(latents, fastvideo_args)
        taehv = self._get_taehv(latents.device)
        if taehv is None:
            return super().decode(latents, fastvideo_args)
        # TAEHV wants NTCHW latents in the normalized (~Gaussian)
        # diffusion space; batch.latents is [B, C, T, H, W] and already
        # normalized (DecodingStage would denormalize for the full VAE,
        # TAEHV was trained on the normalized space so we skip it).
        x = latents.to(dtype=torch.float16).permute(0, 2, 1, 3, 4)
        video = taehv.decode_video(x,
                                   parallel=True,
                                   show_progress_bar=False)
        # NTCHW [0,1] -> [B, C, T, H, W]
        return video.permute(0, 2, 1, 3, 4).clamp(0, 1)


class LingBotWorldCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    """LingBot-World Fast (causal DMD-distilled) image-to-video pipeline."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Match the official generate_fast.py schedule:
        # scheduler.set_timesteps(1000, shift=flow_shift)
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.modules["scheduler"].set_timesteps(num_inference_steps=1000,
                                                denoising_strength=1.0)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))

        self.add_stage(stage_name="denoising_stage",
                       stage=LingBotCausalDMDDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module(
                               "transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="decoding_stage",
                       stage=TAEHVDecodingStage(vae=self.get_module("vae")))


EntryClass = LingBotWorldCausalDMDPipeline
