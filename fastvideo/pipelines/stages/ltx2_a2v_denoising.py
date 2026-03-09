# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 audio-to-video two-stage denoising stage.

Upstream equivalent: ``A2VidPipelineTwoStage``

Stage 1: Encode input audio, freeze audio latents (denoise_mask=0),
          denoise video only at half-res with CFG guidance.
Stage 2: Refine both video and audio at full-res with distilled LoRA.
"""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS,
    DEFAULT_LTX2_AUDIO_DOWNSAMPLE, DEFAULT_LTX2_AUDIO_HOP_LENGTH,
    DEFAULT_LTX2_AUDIO_MEL_BINS, DEFAULT_LTX2_AUDIO_SAMPLE_RATE)
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.ltx2_denoising import (
    _ltx2_sigmas)
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    STAGE_2_DISTILLED_SIGMA_VALUES)
from fastvideo.pipelines.stages.ltx2_stage_utils import (
    debug_save_latent, noise_latent_at_sigma,
    run_spatial_upsample)
from fastvideo.pipelines.stages.ltx2_two_stage_denoising import (
    LTX2TwoStageDenoisingStage)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)

logger = init_logger(__name__)


class LTX2A2VDenoisingStage(LTX2TwoStageDenoisingStage):
    """Audio-to-video two-stage denoising.

    Stage 1 freezes the encoded audio latents (denoise_mask=0)
    and only denoises video.  Stage 2 denoises both modalities
    with distilled LoRA.

    The encoded audio latents are expected in
    ``batch.extra["ltx2_encoded_audio_latents"]``.
    """

    def _run_two_stage(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        full_latents = batch.latents
        assert full_latents is not None
        device = full_latents.device

        b, c, f, h, w = full_latents.shape
        half_h, half_w = h // 2, w // 2

        # Get encoded audio from batch
        encoded_audio = batch.extra.get(
            "ltx2_encoded_audio_latents")
        if encoded_audio is None:
            logger.warning(
                "[LTX2-A2V] No encoded audio found. "
                "Falling back to standard two-stage.")
            super()._run_two_stage(
                batch=batch,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
            )
            return

        sigmas = _ltx2_sigmas(
            steps=batch.num_inference_steps,
            latent=None,
            device=device,
        )

        logger.info(
            "[LTX2-A2V] Stage 1: half-res %dx%d "
            "(%d steps, video-only, audio frozen)",
            half_h, half_w, len(sigmas) - 1,
        )

        gen = None
        if batch.seed is not None:
            gen = torch.Generator(device=device).manual_seed(
                batch.seed)
        half_latents = torch.randn(
            (b, c, f, half_h, half_w),
            generator=gen,
            device=device,
            dtype=full_latents.dtype,
        )

        # Use encoded audio (frozen) for Stage 1
        audio_latents = encoded_audio.to(device)
        audio_prompt_embeds = batch.extra.get(
            "ltx2_audio_prompt_embeds")
        audio_ctx_p = (
            audio_prompt_embeds[0]
            if audio_prompt_embeds else None
        )
        audio_neg = batch.extra.get(
            "ltx2_audio_negative_embeds")
        audio_ctx_n = audio_neg[0] if audio_neg else None

        # Build audio timestep template
        if audio_latents is not None:
            audio_ts_tpl = torch.ones(
                (b, audio_latents.shape[2]
                 if audio_latents.dim() == 4
                 else audio_latents.shape[1]),
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_ts_tpl = None

        # Temporarily update batch for half-res
        orig_h = batch.height
        orig_w = batch.width
        sr = (fastvideo_args.pipeline_config.vae_config
              .arch_config.spatial_compression_ratio)
        batch.height = half_h * sr
        batch.width = half_w * sr

        # Stage 1: guided denoising (video only — audio frozen)
        # Audio latents are passed but not updated because
        # the guidance loop processes them but the audio
        # "velocity" is zero when audio is already clean.
        # We save and restore audio after the loop.
        saved_audio = audio_latents.clone()

        half_latents, _ = self._guided_denoise_loop(
            latents=half_latents,
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
            stage_label="Stage 1 (half-res, video-only)",
        )

        # Restore frozen audio
        audio_latents = saved_audio

        batch.height = orig_h
        batch.width = orig_w

        debug_save_latent(
            half_latents, "a2v_s1", audio_latents)

        # Spatial upsample
        logger.info(
            "[LTX2-A2V] Upsampling 2x (%dx%d -> %dx%d)",
            half_h, half_w, h, w,
        )
        assert self.spatial_upsampler is not None
        assert self.per_channel_statistics is not None

        upsampled = run_spatial_upsample(
            half_latents=half_latents,
            spatial_upsampler=self.spatial_upsampler,
            per_channel_statistics=(
                self.per_channel_statistics),
            target_h=h,
            target_w=w,
            batch_size=b,
        )

        # Stage 2: full-res with distilled LoRA (both modalities)
        s2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            device=device,
            dtype=torch.float32,
        )

        logger.info(
            "[LTX2-A2V] Stage 2: full-res %dx%d "
            "(%d steps, both modalities)",
            h, w, len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
        )

        start_sigma = s2_sigmas[0]
        s2_latents = noise_latent_at_sigma(
            upsampled, start_sigma)

        # Noise audio for stage 2 refinement
        audio_latents = noise_latent_at_sigma(
            audio_latents, start_sigma)

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
                stage_label="Stage 2 (full-res, both)",
            ))

        debug_save_latent(
            s2_latents, "a2v_s2", audio_latents)

        batch.latents = s2_latents
        batch.extra["ltx2_audio_latents"] = audio_latents
