# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 audio-to-video two-stage pipeline.

Upstream equivalent: ``A2VidPipelineTwoStage``
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
from fastvideo.pipelines.stages.ltx2_a2v_denoising import (
    LTX2A2VDenoisingStage)

logger = init_logger(__name__)


class LTX23A2VPipeline(LTX23TwoStagePipeline):
    """Audio-to-video two-stage pipeline.

    Takes an input audio file and generates video synchronized to it.
    Stage 1 freezes audio latents and denoises video only.
    Stage 2 refines both modalities with distilled LoRA.

    Audio encoding is expected to be handled externally (e.g. via
    a preprocessing stage) and stored in
    ``batch.extra["ltx2_encoded_audio_latents"]``.
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
            stage=LTX2A2VDenoisingStage(
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


EntryClass = LTX23A2VPipeline
