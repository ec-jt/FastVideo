# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 retake denoising stage.

Upstream equivalent: ``RetakePipeline``

Single-stage pipeline that re-generates a temporal region of an
existing video.  Encodes the source video/audio to latents, applies
a temporal region mask, adds noise, and re-denoises.

Supports both distilled (8 steps) and non-distilled (30+ steps) modes.
"""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import VideoLatentShape
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.ltx2_denoising import (
    _ltx2_sigmas)
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    DISTILLED_SIGMA_VALUES)
from fastvideo.pipelines.stages.ltx2_stage_utils import (
    debug_save_latent, noise_latent_at_sigma)
from fastvideo.pipelines.stages.ltx2_two_stage_denoising import (
    LTX2TwoStageDenoisingStage)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)

logger = init_logger(__name__)


def _build_temporal_mask(
    num_frames_latent: int,
    start_frame: int,
    end_frame: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 1D temporal mask for the retake region.

    Returns a tensor of shape ``(num_frames_latent,)`` where
    values are 1.0 inside ``[start_frame, end_frame)`` and
    0.0 outside.
    """
    mask = torch.zeros(
        num_frames_latent, device=device, dtype=dtype)
    start = max(0, min(start_frame, num_frames_latent))
    end = max(0, min(end_frame, num_frames_latent))
    if start < end:
        mask[start:end] = 1.0
    return mask


class LTX2RetakeDenoisingStage(LTX2TwoStageDenoisingStage):
    """Retake denoising: re-generate a temporal region.

    Expects the following in ``batch.extra``:

    * ``ltx2_encoded_video_latents``: Encoded source video latents.
    * ``ltx2_encoded_audio_latents``: Encoded source audio latents
      (optional).
    * ``ltx2_retake_start_frame``: Start frame index in latent space.
    * ``ltx2_retake_end_frame``: End frame index in latent space.
    * ``ltx2_retake_distilled``: Whether to use distilled mode.
    * ``ltx2_retake_regenerate_video``: Whether to regenerate video.
    * ``ltx2_retake_regenerate_audio``: Whether to regenerate audio.
    """

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
            if batch.negative_prompt_embeds:
                neg_prompt_embeds = (
                    batch.negative_prompt_embeds[0])

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

        self._run_retake(
            batch=batch,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            fastvideo_args=fastvideo_args,
        )

        logger.info("[LTX2-Retake] Denoising done.")
        return batch

    def _run_retake(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Execute the retake pipeline (single-stage)."""
        device = batch.latents.device

        # Get encoded source latents
        encoded_video = batch.extra.get(
            "ltx2_encoded_video_latents")
        encoded_audio = batch.extra.get(
            "ltx2_encoded_audio_latents")

        # Get retake parameters
        start_frame = batch.extra.get(
            "ltx2_retake_start_frame", 0)
        end_frame = batch.extra.get(
            "ltx2_retake_end_frame", 0)
        distilled = batch.extra.get(
            "ltx2_retake_distilled", False)
        regen_video = batch.extra.get(
            "ltx2_retake_regenerate_video", True)
        regen_audio = batch.extra.get(
            "ltx2_retake_regenerate_audio", True)

        # Choose sigma schedule
        if distilled:
            sigmas = torch.tensor(
                DISTILLED_SIGMA_VALUES,
                device=device,
                dtype=torch.float32,
            )
            logger.info(
                "[LTX2-Retake] Distilled mode: %d steps",
                len(DISTILLED_SIGMA_VALUES) - 1)
        else:
            sigmas = _ltx2_sigmas(
                steps=batch.num_inference_steps,
                latent=None,
                device=device,
            )
            logger.info(
                "[LTX2-Retake] Full mode: %d steps",
                len(sigmas) - 1)

        # Use encoded video as initial latent
        if encoded_video is not None:
            latents = encoded_video.to(device)
        else:
            latents = batch.latents

        # Build temporal mask for video
        b, c, f, h, w = latents.shape
        if regen_video and start_frame < end_frame:
            video_mask = _build_temporal_mask(
                f, start_frame, end_frame,
                device=device,
            )
            # Expand to [B, 1, F, 1, 1]
            video_mask_5d = video_mask.view(
                1, 1, f, 1, 1).expand(b, 1, f, 1, 1)
        else:
            # Regenerate everything or nothing
            video_mask_5d = torch.ones(
                b, 1, f, 1, 1,
                device=device,
                dtype=torch.float32,
            ) if regen_video else torch.zeros(
                b, 1, f, 1, 1,
                device=device,
                dtype=torch.float32,
            )

        # Save clean latent for blending
        clean_video = latents.clone()

        # Add noise to the region being regenerated
        latents = noise_latent_at_sigma(
            latents, sigmas[0].item())

        # Blend: keep clean outside mask, noised inside
        latents = (
            latents * video_mask_5d
            + clean_video * (1.0 - video_mask_5d)
        ).to(latents.dtype)

        # Audio handling
        audio_latents = None
        audio_ctx_p = None
        audio_ts_tpl = None

        audio_prompt_embeds = batch.extra.get(
            "ltx2_audio_prompt_embeds")
        audio_ctx_p = (
            audio_prompt_embeds[0]
            if audio_prompt_embeds else None
        )
        audio_neg = batch.extra.get(
            "ltx2_audio_negative_embeds")
        audio_ctx_n = audio_neg[0] if audio_neg else None

        if encoded_audio is not None:
            audio_latents = encoded_audio.to(device)
            if regen_audio:
                audio_latents = noise_latent_at_sigma(
                    audio_latents, sigmas[0].item())
            audio_ts_tpl = torch.ones(
                (b, audio_latents.shape[2]
                 if audio_latents.dim() == 4
                 else audio_latents.shape[1]),
                device=device,
                dtype=torch.float32,
            )
        else:
            (audio_latents, audio_ctx_p,
             audio_ts_tpl) = self._init_audio_latents(
                batch, latents, fastvideo_args,
            )

        # Run denoising
        if distilled or neg_prompt_embeds is None:
            # Simple denoising (no guidance)
            latents, audio_latents = (
                self._simple_denoise_loop(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    sigmas=sigmas,
                    audio_latents=audio_latents,
                    audio_context=audio_ctx_p,
                    audio_timestep_template=audio_ts_tpl,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    fastvideo_args=fastvideo_args,
                    batch=batch,
                    stage_label="retake (simple)",
                ))
        else:
            # Guided denoising
            latents, audio_latents = (
                self._guided_denoise_loop(
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
                    stage_label="retake (guided)",
                ))

        # Blend result with clean latent outside mask
        latents = (
            latents * video_mask_5d
            + clean_video * (1.0 - video_mask_5d)
        ).to(latents.dtype)

        debug_save_latent(
            latents, "retake_final", audio_latents)

        batch.latents = latents
        batch.extra["ltx2_audio_latents"] = audio_latents

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
        return result
