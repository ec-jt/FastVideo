# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 keyframe interpolation denoising stage.

Upstream equivalent: ``KeyframeInterpolationPipeline``

Two-stage pipeline that interpolates between keyframe images at
arbitrary frame indices.  Uses ``VideoConditionByKeyframeIndex``
(guiding latent) for ALL frames — softer conditioning than the
standard I2V which replaces the latent at frame 0.
"""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.ltx2_two_stage_denoising import (
    LTX2TwoStageDenoisingStage)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)

logger = init_logger(__name__)


class LTX2KeyframeDenoisingStage(LTX2TwoStageDenoisingStage):
    """Keyframe interpolation two-stage denoising.

    Identical to ``LTX2TwoStageDenoisingStage`` but designed for
    keyframe interpolation where all image conditionings use the
    guiding latent approach (``VideoConditionByKeyframeIndex``)
    rather than latent replacement.

    The keyframe conditioning is handled at the pipeline level
    (image encoding stage) — this denoising stage is structurally
    the same as the standard two-stage stage.  It exists as a
    separate class for clarity and future customization.

    Keyframe latents are expected in
    ``batch.extra["ltx2_keyframe_latents"]`` as a list of
    ``(encoded_image, frame_idx, strength)`` tuples.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        keyframes = batch.extra.get("ltx2_keyframe_latents")
        if keyframes:
            logger.info(
                "[LTX2-Keyframe] %d keyframe(s) provided",
                len(keyframes),
            )
        else:
            logger.info(
                "[LTX2-Keyframe] No keyframes — running "
                "standard two-stage T2V")

        # Delegate to parent two-stage logic
        return super().forward(batch, fastvideo_args)

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
