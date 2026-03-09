# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 keyframe interpolation pipeline.

Upstream equivalent: ``KeyframeInterpolationPipeline``
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
from fastvideo.pipelines.stages.ltx2_keyframe_denoising import (
    LTX2KeyframeDenoisingStage)

logger = init_logger(__name__)


class LTX23KeyframeInterpolationPipeline(LTX23TwoStagePipeline):
    """Keyframe interpolation two-stage pipeline.

    Interpolates between keyframe images at arbitrary frame indices
    to generate smooth video transitions.  All images use guiding
    latent conditioning (``VideoConditionByKeyframeIndex``) for
    softer control than standard I2V.
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
            stage=LTX2KeyframeDenoisingStage(
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


EntryClass = LTX23KeyframeInterpolationPipeline
