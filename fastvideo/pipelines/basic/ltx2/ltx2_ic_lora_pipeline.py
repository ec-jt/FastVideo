# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 IC-LoRA (In-Context LoRA) pipeline.

Upstream equivalent: ``ICLoraPipeline``

Two-stage distilled pipeline with IC-LoRA for control signals.
Stage 1 uses IC-LoRA + distilled model, Stage 2 uses distilled
model only.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.ltx2.ltx2_distilled_pipeline import (
    LTX23DistilledPipeline)
from fastvideo.pipelines.stages import (
    DecodingStage, InputValidationStage,
    LTX2AudioDecodingStage,
    LTX2LatentPreparationStage,
    LTX2TextEncodingStage)
from fastvideo.pipelines.stages.ltx2_ic_lora_denoising import (
    LTX2ICLoraDenoisingStage)

logger = init_logger(__name__)


class LTX23ICLoraPipeline(LTX23DistilledPipeline):
    """IC-LoRA two-stage distilled pipeline.

    Extends ``LTX23DistilledPipeline`` to support IC-LoRA weights
    for control signal conditioning (depth, pose, edges).

    Stage 1: IC-LoRA + distilled model at half resolution.
    Stage 2: Distilled model only (no IC-LoRA) at full resolution.

    The IC-LoRA weights are loaded via the ``LoRAPipeline``
    infrastructure.  The pipeline manages LoRA state transitions
    between stages.
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
        per_channel_stats = None
        if vae is not None:
            if hasattr(vae, "per_channel_statistics"):
                per_channel_stats = (
                    vae.per_channel_statistics)
            elif hasattr(vae, "decoder") and hasattr(
                vae.decoder, "per_channel_statistics"
            ):
                per_channel_stats = (
                    vae.decoder.per_channel_statistics)
            elif hasattr(vae, "model") and hasattr(
                vae.model, "per_channel_statistics"
            ):
                per_channel_stats = (
                    vae.model.per_channel_statistics)

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2ICLoraDenoisingStage(
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


EntryClass = LTX23ICLoraPipeline
