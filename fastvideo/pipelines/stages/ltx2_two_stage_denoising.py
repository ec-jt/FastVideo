# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 two-stage denoising: CFG Stage 1 + distilled Stage 2.

Upstream equivalent: ``TI2VidTwoStagesPipeline`` from
``LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py``

Resolution convention (matches upstream):
  The user specifies the **full (Stage 2)** resolution
  (e.g. 1024×1536).  Stage 1 runs at **half** that
  (e.g. 512×768).

Stage 1: Half resolution with full CFG/STG/modality guidance
          (30 steps, Euler sampler).
Upsample: 2× spatial via ``LatentUpsampler``.
Stage 2: Full resolution refinement with distilled LoRA, simple
          denoising (3 steps, ``STAGE_2_DISTILLED_SIGMA_VALUES``,
          no CFG).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadataBuilder)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS,
    DEFAULT_LTX2_AUDIO_DOWNSAMPLE, DEFAULT_LTX2_AUDIO_HOP_LENGTH,
    DEFAULT_LTX2_AUDIO_MEL_BINS, DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
    VideoLatentShape)
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.ltx2_denoising import (
    _ltx2_sigmas)
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    STAGE_2_DISTILLED_SIGMA_VALUES)
from fastvideo.pipelines.stages.ltx2_stage_utils import (
    debug_save_latent, noise_latent_at_sigma,
    run_spatial_upsample)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)
from fastvideo.utils import is_vsa_available

logger = init_logger(__name__)

try:
    vsa_available = is_vsa_available()
except ImportError:
    vsa_available = False


class LTX2TwoStageDenoisingStage(PipelineStage):
    """Two-stage denoising: CFG Stage 1 + simple distilled Stage 2.

    Stage 1 reuses the multi-modal CFG/STG/modality guidance logic
    from ``LTX2DenoisingStage``.  Stage 2 reuses the simple Euler
    denoising loop from ``LTX2DistilledDenoisingStage``.

    Between stages, the pipeline applies a distilled LoRA via the
    ``LoRAPipeline`` infrastructure (managed externally by the
    pipeline class).
    """

    def __init__(
        self,
        transformer,
        spatial_upsampler: LatentUpsampler | None = None,
        per_channel_statistics=None,
        vae=None,
        merge_stage2_lora_fn: Callable[[], None] | None = None,
        unmerge_stage2_lora_fn: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.spatial_upsampler = spatial_upsampler
        self.per_channel_statistics = per_channel_statistics
        self._debug_vae = vae
        self._merge_stage2_lora = merge_stage2_lora_fn
        self._unmerge_stage2_lora = unmerge_stage2_lora_fn

    # ─────────────────────────────────────────────────────────
    # Audio latent initialisation
    # ─────────────────────────────────────────────────────────

    def _init_audio_latents(
        self,
        batch: ForwardBatch,
        latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
        noise_scale: float = 1.0,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Create initial audio latents and timestep template."""
        audio_prompt_embeds = batch.extra.get(
            "ltx2_audio_prompt_embeds")
        audio_context = (
            audio_prompt_embeds[0]
            if audio_prompt_embeds else None
        )
        if audio_context is None:
            return None, None, None

        fps_value = batch.fps
        if isinstance(fps_value, list):
            fps_value = fps_value[0] if fps_value else None
        if fps_value is None:
            fps_value = 1.0
        duration = float(batch.num_frames) / float(fps_value)
        audio_shape = AudioLatentShape.from_duration(
            batch=latents.shape[0],
            duration=duration,
            channels=DEFAULT_LTX2_AUDIO_CHANNELS,
            mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
            sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
            hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
            audio_latent_downsample_factor=(
                DEFAULT_LTX2_AUDIO_DOWNSAMPLE),
        )

        gen = self._make_generator(
            batch, latents, fastvideo_args)

        audio_patch_shape = (
            audio_shape.batch,
            audio_shape.frames,
            audio_shape.channels * audio_shape.mel_bins,
        )
        audio_latents_patch = torch.randn(
            audio_patch_shape,
            generator=gen,
            device=latents.device,
            dtype=latents.dtype,
        ) * noise_scale

        if hasattr(self.transformer, "audio_patchifier"):
            audio_latents = (
                self.transformer.audio_patchifier.unpatchify(
                    audio_latents_patch, audio_shape))
        else:
            audio_latents = audio_latents_patch.view(
                audio_shape.batch,
                audio_shape.frames,
                audio_shape.channels,
                audio_shape.mel_bins,
            ).permute(0, 2, 1, 3).contiguous()

        audio_timestep_template = torch.ones(
            (latents.shape[0], audio_shape.frames),
            device=latents.device,
            dtype=torch.float32,
        )

        return (audio_latents, audio_context,
                audio_timestep_template)

    def _make_generator(
        self,
        batch: ForwardBatch,
        latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
    ) -> torch.Generator | None:
        """Build a device-local generator for audio latents."""
        gen = None
        if (fastvideo_args.ltx2_initial_latent_path
                and batch.seed is not None):
            gen = torch.Generator(
                device=latents.device,
            ).manual_seed(batch.seed)
        elif batch.generator is not None:
            if isinstance(batch.generator, list):
                gen = batch.generator[0]
            else:
                gen = batch.generator
        if (gen is not None
                and gen.device.type != latents.device.type):
            if batch.seed is None:
                gen = torch.Generator(device=latents.device)
            else:
                gen = torch.Generator(
                    device=latents.device,
                ).manual_seed(batch.seed)
        return gen

    # ─────────────────────────────────────────────────────────
    # Simple Euler denoising loop (no guidance, for Stage 2)
    # ─────────────────────────────────────────────────────────

    def _simple_denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigmas: torch.Tensor,
        audio_latents: torch.Tensor | None,
        audio_context: torch.Tensor | None,
        audio_timestep_template: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
        batch: ForwardBatch,
        stage_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run simple Euler denoising (no CFG/STG/modality)."""
        if hasattr(self.transformer, "patchifier"):
            video_shape = VideoLatentShape.from_torch_shape(
                latents.shape)
            token_count = (
                self.transformer.patchifier.get_token_count(
                    video_shape))
        else:
            token_count = 1

        timestep_template = torch.ones(
            (latents.shape[0], token_count),
            device=latents.device,
            dtype=torch.float32,
        )

        for step_index in tqdm(
            range(len(sigmas) - 1),
            desc=f"Denoising {stage_label}",
        ):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_timestep = (
                audio_timestep_template * sigma
                if audio_timestep_template is not None
                else None
            )

            with torch.autocast(
                device_type="cuda",
                dtype=target_dtype,
                enabled=autocast_enabled,
            ), set_forward_context(
                current_timestep=sigma.item(),
                attn_metadata=None,
                forward_batch=batch,
            ):
                outputs = self.transformer(
                    hidden_states=latents.to(target_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    audio_hidden_states=audio_latents,
                    audio_encoder_hidden_states=audio_context,
                    audio_timestep=audio_timestep,
                )
                if isinstance(outputs, tuple):
                    denoised, audio_denoised = outputs
                else:
                    denoised = outputs
                    audio_denoised = None

            # Euler step
            sigma_f = sigma.to(torch.float32)
            dt = sigma_next - sigma
            velocity = (
                (latents.float() - denoised.float())
                / sigma_f
            ).to(latents.dtype)
            latents = (
                latents.float() + velocity.float() * dt
            ).to(latents.dtype)

            if (audio_denoised is not None
                    and audio_latents is not None):
                audio_velocity = (
                    (audio_latents.float()
                     - audio_denoised.float())
                    / sigma_f
                ).to(audio_latents.dtype)
                audio_latents = (
                    audio_latents.float()
                    + audio_velocity.float() * dt
                ).to(audio_latents.dtype)

        return latents, audio_latents

    # ─────────────────────────────────────────────────────────
    # Guided denoising loop (CFG/STG/modality, for Stage 1)
    # ─────────────────────────────────────────────────────────

    def _guided_denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        sigmas: torch.Tensor,
        audio_latents: torch.Tensor | None,
        audio_context_p: torch.Tensor | None,
        audio_context_n: torch.Tensor | None,
        audio_timestep_template: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
        batch: ForwardBatch,
        stage_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run guided Euler denoising with CFG/STG/modality."""
        if hasattr(self.transformer, "patchifier"):
            video_shape = VideoLatentShape.from_torch_shape(
                latents.shape)
            token_count = (
                self.transformer.patchifier.get_token_count(
                    video_shape))
        else:
            token_count = 1

        timestep_template = torch.ones(
            (latents.shape[0], token_count),
            device=latents.device,
            dtype=torch.float32,
        )

        # Multi-modal CFG parameters
        cfg_v = batch.ltx2_cfg_scale_video
        cfg_a = batch.ltx2_cfg_scale_audio
        mod_v = batch.ltx2_modality_scale_video
        mod_a = batch.ltx2_modality_scale_audio
        rescale = batch.ltx2_rescale_scale
        stg_v = batch.ltx2_stg_scale_video
        stg_a = batch.ltx2_stg_scale_audio
        stg_blk_v = batch.ltx2_stg_blocks_video
        stg_blk_a = batch.ltx2_stg_blocks_audio
        do_stg_v = stg_v != 0.0
        do_stg_a = stg_a != 0.0
        do_stg = do_stg_v or do_stg_a
        do_cfg = cfg_v != 1.0 or cfg_a != 1.0
        do_mod = mod_v != 1.0 or mod_a != 1.0
        do_guidance = do_cfg or do_mod or do_stg

        use_vsa = (
            vsa_available
            and envs.FASTVIDEO_ATTENTION_BACKEND
            == "VIDEO_SPARSE_ATTN"
        )
        vsa_builder = (
            VideoSparseAttentionMetadataBuilder()
            if use_vsa else None
        )

        for step_index in tqdm(
            range(len(sigmas) - 1),
            desc=f"Denoising {stage_label}",
        ):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_ts = (
                audio_timestep_template * sigma
                if audio_timestep_template is not None
                else None
            )
            attn_meta = None
            if vsa_builder is not None:
                attn_meta = vsa_builder.build(
                    current_timestep=step_index,
                    raw_latent_shape=latents.shape[2:5],
                    patch_size=(
                        fastvideo_args.pipeline_config
                        .dit_config.patch_size),
                    VSA_sparsity=fastvideo_args.VSA_sparsity,
                    device=latents.device,
                )

            with torch.autocast(
                device_type="cuda",
                dtype=target_dtype,
                enabled=autocast_enabled,
            ), set_forward_context(
                current_timestep=sigma.item(),
                attn_metadata=attn_meta,
                forward_batch=batch,
            ):
                # Pass 1: Full conditioning
                pos_out = self.transformer(
                    hidden_states=latents.to(target_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    audio_hidden_states=audio_latents,
                    audio_encoder_hidden_states=(
                        audio_context_p),
                    audio_timestep=audio_ts,
                )
                if isinstance(pos_out, tuple):
                    pos_vid, pos_aud = pos_out
                else:
                    pos_vid = pos_out
                    pos_aud = None

                if do_guidance:
                    neg_vid = pos_vid
                    neg_aud = pos_aud
                    mod_vid = pos_vid
                    mod_aud = pos_aud
                    ptb_vid = pos_vid
                    ptb_aud = pos_aud

                    # Pass 2: text CFG
                    if do_cfg and neg_prompt_embeds is not None:
                        neg_out = self.transformer(
                            hidden_states=latents.to(
                                target_dtype),
                            encoder_hidden_states=(
                                neg_prompt_embeds),
                            encoder_attention_mask=None,
                            timestep=timestep,
                            audio_hidden_states=audio_latents,
                            audio_encoder_hidden_states=(
                                audio_context_n),
                            audio_timestep=audio_ts,
                        )
                        if isinstance(neg_out, tuple):
                            neg_vid, neg_aud = neg_out
                        else:
                            neg_vid = neg_out

                    # Pass 3: Modality-isolated
                    if do_mod:
                        mod_out = self.transformer(
                            hidden_states=latents.to(
                                target_dtype),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=None,
                            timestep=timestep,
                            audio_hidden_states=audio_latents,
                            audio_encoder_hidden_states=(
                                audio_context_p),
                            audio_timestep=audio_ts,
                            skip_cross_modal_attn=True,
                        )
                        if isinstance(mod_out, tuple):
                            mod_vid, mod_aud = mod_out
                        else:
                            mod_vid = mod_out

                    # Pass 4: STG perturbed
                    if do_stg:
                        ptb_out = self.transformer(
                            hidden_states=latents.to(
                                target_dtype),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=None,
                            timestep=timestep,
                            audio_hidden_states=audio_latents,
                            audio_encoder_hidden_states=(
                                audio_context_p),
                            audio_timestep=audio_ts,
                            skip_video_self_attn_blocks=(
                                stg_blk_v
                                if do_stg_v else None),
                            skip_audio_self_attn_blocks=(
                                stg_blk_a
                                if do_stg_a else None),
                        )
                        if isinstance(ptb_out, tuple):
                            ptb_vid, ptb_aud = ptb_out
                        else:
                            ptb_vid = ptb_out

                    # Multi-modal guidance formula
                    vid = (
                        pos_vid
                        + (cfg_v - 1) * (pos_vid - neg_vid)
                        + (mod_v - 1) * (pos_vid - mod_vid)
                        + stg_v * (pos_vid - ptb_vid)
                    )
                    aud = None
                    if pos_aud is not None:
                        aud = (
                            pos_aud
                            + (cfg_a - 1)
                            * (pos_aud - neg_aud)
                            + (mod_a - 1)
                            * (pos_aud - mod_aud)
                            + stg_a
                            * (pos_aud - ptb_aud)
                        )

                    # Guidance rescaling
                    if rescale > 0:
                        f_v = pos_vid.std() / vid.std()
                        f_v = rescale * f_v + (1 - rescale)
                        vid = vid * f_v
                        if aud is not None:
                            f_a = pos_aud.std() / aud.std()
                            f_a = (
                                rescale * f_a
                                + (1 - rescale))
                            aud = aud * f_a

                    pos_vid = vid
                    pos_aud = aud

            # Euler step
            sigma_f = sigma.to(torch.float32)
            dt = sigma_next - sigma
            vel = (
                (latents.float() - pos_vid.float()) / sigma_f
            ).to(latents.dtype)
            latents = (
                latents.float() + vel.float() * dt
            ).to(latents.dtype)

            if pos_aud is not None and audio_latents is not None:
                a_vel = (
                    (audio_latents.float()
                     - pos_aud.float()) / sigma_f
                ).to(audio_latents.dtype)
                audio_latents = (
                    audio_latents.float()
                    + a_vel.float() * dt
                ).to(audio_latents.dtype)

        return latents, audio_latents

    # ─────────────────────────────────────────────────────────
    # Main forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError(
                "Latents must be provided before denoising.")

        prompt_embeds = batch.prompt_embeds[0]
        neg_prompt_embeds = None
        if batch.do_classifier_free_guidance:
            if not batch.negative_prompt_embeds:
                raise ValueError(
                    "CFG enabled but negative_prompt_embeds "
                    "is empty")
            neg_prompt_embeds = batch.negative_prompt_embeds[0]

        if prompt_embeds.device != batch.latents.device:
            prompt_embeds = prompt_embeds.to(
                batch.latents.device)
        if (neg_prompt_embeds is not None
                and neg_prompt_embeds.device
                != batch.latents.device):
            neg_prompt_embeds = neg_prompt_embeds.to(
                batch.latents.device)

        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not fastvideo_args.disable_autocast

        has_upsampler = (
            self.spatial_upsampler is not None
            and self.per_channel_statistics is not None
        )

        if not has_upsampler:
            logger.warning(
                "[LTX2-TwoStage] No spatial upsampler. "
                "Falling back to single-stage guided.")
            self._run_single_stage_guided(
                batch=batch,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
            )
        else:
            self._run_two_stage(
                batch=batch,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
            )

        logger.info("[LTX2-TwoStage] Denoising done.")
        return batch

    # ─────────────────────────────────────────────────────────
    # Two-stage implementation
    # ─────────────────────────────────────────────────────────

    def _run_two_stage(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Execute two-stage denoising.

        Resolution convention (matches upstream
        ``TI2VidTwoStagesPipeline``):

        * The user specifies the **full (Stage 2)** resolution
          via ``batch.height / batch.width``.  The latents
          prepared by the latent-preparation stage correspond
          to this full resolution.
        * Stage 1 runs at **half** the user-specified resolution.
          We create fresh half-res noise for Stage 1.
        * After Stage 1, the spatial upsampler doubles H and W
          back to the full resolution for Stage 2.

        This matches upstream lines 117-118::

            stage_1_output_shape = VideoPixelShape(
                ..., width=width // 2, height=height // 2, ...)
        """
        full_latents = batch.latents
        assert full_latents is not None
        device = full_latents.device

        # Full-resolution latent dimensions (from user spec).
        b, c, f, full_h, full_w = full_latents.shape
        # Stage 1 runs at half spatial resolution.
        half_h, half_w = full_h // 2, full_w // 2

        # ── Stage 1: half-resolution with CFG guidance ────
        sigmas = _ltx2_sigmas(
            steps=batch.num_inference_steps,
            latent=None,
            device=device,
        )

        logger.info(
            "[LTX2-TwoStage] Stage 1: %dx%d latent "
            "(half-res, %d steps, guided)",
            half_h, half_w, len(sigmas) - 1,
        )

        gen = None
        if batch.seed is not None:
            gen = torch.Generator(device=device).manual_seed(
                batch.seed)
        # Fresh noise at half resolution for Stage 1.
        half_noise = torch.randn(
            (b, c, f, half_h, half_w),
            generator=gen,
            device=device,
            dtype=full_latents.dtype,
        )

        # Temporarily update batch dimensions for Stage 1
        # (half resolution).
        orig_h = batch.height
        orig_w = batch.width
        sr = (fastvideo_args.pipeline_config.vae_config
              .arch_config.spatial_compression_ratio)
        batch.height = half_h * sr
        batch.width = half_w * sr

        # Init audio
        (audio_latents, audio_ctx_p,
         audio_ts_tpl) = self._init_audio_latents(
            batch, half_noise, fastvideo_args,
            noise_scale=1.0,
        )
        audio_neg = batch.extra.get(
            "ltx2_audio_negative_embeds")
        audio_ctx_n = audio_neg[0] if audio_neg else None

        half_denoised, audio_latents = (
            self._guided_denoise_loop(
                latents=half_noise,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                sigmas=sigmas,
                audio_latents=audio_latents,
                audio_context_p=audio_ctx_p,
                audio_context_n=audio_ctx_n,
                audio_timestep_template=audio_ts_tpl,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                stage_label="Stage 1 (half-res, guided)",
            ))

        # Restore original batch dimensions after Stage 1.
        batch.height = orig_h
        batch.width = orig_w

        debug_save_latent(
            half_denoised, "two_stage_s1", audio_latents)

        # ── Spatial upsample ─────────────────────────────
        logger.info(
            "[LTX2-TwoStage] Upsampling 2x "
            "(%dx%d -> %dx%d)",
            half_h, half_w, full_h, full_w,
        )
        assert self.spatial_upsampler is not None
        assert self.per_channel_statistics is not None

        upsampled = run_spatial_upsample(
            half_latents=half_denoised,
            spatial_upsampler=self.spatial_upsampler,
            per_channel_statistics=(
                self.per_channel_statistics),
            target_h=full_h,
            target_w=full_w,
            batch_size=b,
        )

        # ── Stage 2: full resolution with distilled LoRA ──
        # Merge distilled LoRA before Stage 2
        if self._merge_stage2_lora is not None:
            self._merge_stage2_lora()

        s2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            device=device,
            dtype=torch.float32,
        )

        logger.info(
            "[LTX2-TwoStage] Stage 2: full-res %dx%d "
            "(%d steps, simple, start_sigma=%.6f)",
            full_h, full_w,
            len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
            STAGE_2_DISTILLED_SIGMA_VALUES[0],
        )

        # Noise the upsampled latent to Stage 2 start sigma.
        # Matches upstream ``denoise_audio_video`` with
        # ``noise_scale=distilled_sigmas[0]``.
        start_sigma = s2_sigmas[0]
        s2_latents = noise_latent_at_sigma(
            upsampled, start_sigma)

        # Re-noise audio for Stage 2 (same as distilled).
        if audio_latents is not None:
            audio_latents = noise_latent_at_sigma(
                audio_latents, start_sigma)

        # Rebuild audio timestep template for full-res
        # (audio shape doesn't change with video resolution).
        if audio_latents is not None:
            audio_ts_tpl = torch.ones(
                (b, audio_latents.shape[2]
                 if audio_latents.dim() == 4
                 else audio_latents.shape[1]),
                device=device,
                dtype=torch.float32,
            )

        s2_latents, audio_latents = (
            self._simple_denoise_loop(
                latents=s2_latents,
                prompt_embeds=prompt_embeds,
                sigmas=s2_sigmas,
                audio_latents=audio_latents,
                audio_context=audio_ctx_p,
                audio_timestep_template=audio_ts_tpl,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                stage_label="Stage 2 (full-res, simple)",
            ))

        debug_save_latent(
            s2_latents, "two_stage_s2", audio_latents)

        # Unmerge distilled LoRA after Stage 2
        if self._unmerge_stage2_lora is not None:
            self._unmerge_stage2_lora()

        batch.latents = s2_latents
        batch.extra["ltx2_audio_latents"] = audio_latents

    # ─────────────────────────────────────────────────────────
    # Single-stage fallback
    # ─────────────────────────────────────────────────────────

    def _run_single_stage_guided(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        latents = batch.latents
        assert latents is not None

        sigmas = _ltx2_sigmas(
            steps=batch.num_inference_steps,
            latent=None,
            device=latents.device,
        )

        (audio_latents, audio_ctx_p,
         audio_ts_tpl) = self._init_audio_latents(
            batch, latents, fastvideo_args,
        )
        audio_neg = batch.extra.get(
            "ltx2_audio_negative_embeds")
        audio_ctx_n = audio_neg[0] if audio_neg else None

        latents, audio_latents = self._guided_denoise_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
            sigmas=sigmas,
            audio_latents=audio_latents,
            audio_context_p=audio_ctx_p,
            audio_context_n=audio_ctx_n,
            audio_timestep_template=audio_ts_tpl,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            fastvideo_args=fastvideo_args,
            batch=batch,
            stage_label="single-stage (guided)",
        )

        batch.latents = latents
        batch.extra["ltx2_audio_latents"] = audio_latents

    # ─────────────────────────────────────────────────────────
    # Verification
    # ─────────────────────────────────────────────────────────

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "latents", batch.latents,
            [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "prompt_embeds", batch.prompt_embeds,
            V.list_not_empty)
        result.add_check(
            "num_inference_steps",
            batch.num_inference_steps,
            V.positive_int)
        return result