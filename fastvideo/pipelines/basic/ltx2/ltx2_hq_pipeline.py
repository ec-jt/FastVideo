# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 HQ two-stage pipeline with Res2s sampler.

Upstream equivalent: ``TI2VidTwoStagesHQPipeline``
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.ltx2.ltx2_two_stage_pipeline import (
    LTX23TwoStagePipeline)
from fastvideo.pipelines.stages import (
    DecodingStage, InputValidationStage,
    LTX2AudioDecodingStage,
    LTX2LatentPreparationStage,
    LTX2TextEncodingStage)
from fastvideo.pipelines.stages.ltx2_hq_denoising import (
    LTX2HQDenoisingStage)

logger = init_logger(__name__)


class LTX23HQPipeline(LTX23TwoStagePipeline):
    """HQ two-stage pipeline using Res2s second-order sampler.

    Same structure as ``LTX23TwoStagePipeline`` but uses the Res2s
    sampler for both stages, allowing fewer steps (15 vs 30) for
    comparable quality.  Supports per-stage LoRA strength control.
    """

    def create_pipeline_stages(
        self, fastvideo_args: FastVideoArgs,
    ):
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=LTX2TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LTX2LatentPreparationStage(
                transformer=self.get_module("transformer"),
            ),
        )

        spatial_upsampler = self._load_spatial_upsampler(
            fastvideo_args)

        vae = self.get_module("vae")
        per_channel_stats = self._get_per_channel_stats(vae)

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2HQDenoisingStage(
                transformer=self.get_module("transformer"),
                spatial_upsampler=spatial_upsampler,
                per_channel_statistics=per_channel_stats,
                vae=vae,
            ),
        )

        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=LTX2AudioDecodingStage(
                audio_decoder=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae")),
        )


EntryClass = LTX23HQPipeline
