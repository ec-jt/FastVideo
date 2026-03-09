# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 IC-LoRA (In-Context LoRA) denoising stage.

Upstream equivalent: ``ICLoraPipeline``

Two-stage distilled pipeline with IC-LoRA for control signals
(depth, pose, edges).  Stage 1 uses IC-LoRA + distilled model,
Stage 2 uses distilled model only (no IC-LoRA).

Both stages use simple denoising (no CFG) since the distilled
model is used.
"""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES,
    LTX2DistilledDenoisingStage)
from fastvideo.pipelines.stages.ltx2_stage_utils import (
    debug_save_latent, noise_latent_at_sigma,
    run_spatial_upsample)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)

logger = init_logger(__name__)


class LTX2ICLoraDenoisingStage(LTX2DistilledDenoisingStage):
    """IC-LoRA two-stage distilled denoising.

    Inherits from ``LTX2DistilledDenoisingStage`` since both stages
    use simple (non-guided) denoising with the distilled model.

    The IC-LoRA weights are managed by the pipeline class via
    ``LoRAPipeline.merge_lora_weights()`` /
    ``unmerge_lora_weights()``.

    Reference video conditioning is expected in
    ``batch.extra["ltx2_reference_latents"]`` as a list of
    ``(encoded_video, downscale_factor, strength)`` tuples.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        ref_latents = batch.extra.get(
            "ltx2_reference_latents")
        if ref_latents:
            logger.info(
                "[LTX2-ICLoRA] %d reference conditioning(s)",
                len(ref_latents),
            )
        else:
            logger.info(
                "[LTX2-ICLoRA] No reference latents — "
                "running standard distilled pipeline")

        # Delegate to parent distilled denoising
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
        return result
