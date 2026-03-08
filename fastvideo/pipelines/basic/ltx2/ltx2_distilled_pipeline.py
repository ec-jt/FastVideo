# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 distilled two-stage text-to-video pipeline.

Uses fixed sigma schedules and simple denoising (no CFG/STG) for
fast inference with the distilled model checkpoint.

Two-stage architecture:
  Stage 1 – Generate at half spatial resolution (8 steps).
  Upsample – 2× spatial upsampling via LatentUpsampler.
  Stage 2 – Refine at full resolution (3 steps).
"""

import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import (
    PipelineComponentLoader)
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.composed_pipeline_base import (
    ComposedPipelineBase)
from fastvideo.pipelines.stages import (
    DecodingStage, InputValidationStage,
    LTX2AudioDecodingStage,
    LTX2LatentPreparationStage,
    LTX2TextEncodingStage)
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    LTX2DistilledDenoisingStage)

logger = init_logger(__name__)

# Default search paths for the spatial upsampler weights.
_UPSAMPLER_SEARCH_PATHS = [
    # Inside the model directory
    "spatial_upsampler/model.safetensors",
    "spatial_upsampler.safetensors",
    # Common external locations
    "/mnt/nvme0/models/Lightricks/LTX-2.3/"
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
]


class LTX23DistilledPipeline(ComposedPipelineBase):
    """Distilled LTX-2.3 two-stage pipeline.

    Key differences from the standard ``LTX2Pipeline``:

    * Uses ``LTX2DistilledDenoisingStage`` which runs a two-stage
      denoising loop (half-res → upsample → full-res refine).
    * No classifier-free guidance, STG, or modality guidance —
      single forward pass per denoising step.
    * Designed for ``guidance_scale=1.0``.
    * Loads a ``LatentUpsampler`` for 2× spatial upsampling
      between stages.
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

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

        # Load spatial upsampler for two-stage pipeline
        spatial_upsampler = self._load_spatial_upsampler(
            fastvideo_args)

        # Get per_channel_statistics from the VAE decoder.
        # The VAE is LTX2CausalVideoAutoencoder which wraps
        # a VideoDecoder at vae.decoder.per_channel_statistics.
        vae = self.get_module("vae")
        per_channel_stats = None
        if vae is not None:
            # Try direct attribute first
            if hasattr(vae, "per_channel_statistics"):
                per_channel_stats = vae.per_channel_statistics
            # Then try vae.decoder (LTX2CausalVideoAutoencoder)
            elif hasattr(vae, "decoder") and hasattr(
                vae.decoder, "per_channel_statistics"
            ):
                per_channel_stats = (
                    vae.decoder.per_channel_statistics)
            # Then try vae.model (LTX2VideoDecoder wrapper)
            elif hasattr(vae, "model") and hasattr(
                vae.model, "per_channel_statistics"
            ):
                per_channel_stats = (
                    vae.model.per_channel_statistics)
        if per_channel_stats is not None:
            logger.info(
                "Using VAE per_channel_statistics "
                "for latent normalization")
        else:
            logger.warning(
                "Could not find per_channel_statistics "
                "on VAE module")

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2DistilledDenoisingStage(
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

    def _load_spatial_upsampler(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> LatentUpsampler | None:
        """Try to load the spatial upsampler from known paths.

        Returns ``None`` if no weights are found (falls back to
        single-stage pipeline).
        """
        import torch

        # Check environment variable override
        upsampler_path = os.environ.get(
            "LTX2_SPATIAL_UPSAMPLER_PATH")

        if upsampler_path and Path(upsampler_path).exists():
            logger.info(
                "Loading spatial upsampler from env: %s",
                upsampler_path)
            return LatentUpsampler.from_pretrained(
                upsampler_path,
                device="cpu",
                dtype=torch.bfloat16,
            )

        # Search known paths
        for rel_path in _UPSAMPLER_SEARCH_PATHS:
            # Try relative to model directory first
            candidate = os.path.join(
                self.model_path, rel_path)
            if os.path.exists(candidate):
                logger.info(
                    "Loading spatial upsampler from: %s",
                    candidate)
                return LatentUpsampler.from_pretrained(
                    candidate,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
            # Try absolute path
            if os.path.isabs(rel_path) and os.path.exists(
                rel_path
            ):
                logger.info(
                    "Loading spatial upsampler from: %s",
                    rel_path)
                return LatentUpsampler.from_pretrained(
                    rel_path,
                    device="cpu",
                    dtype=torch.bfloat16,
                )

        logger.warning(
            "Spatial upsampler weights not found. "
            "Searched: %s. "
            "Set LTX2_SPATIAL_UPSAMPLER_PATH env var or "
            "place weights in model directory. "
            "Falling back to single-stage pipeline.",
            _UPSAMPLER_SEARCH_PATHS,
        )
        return None

    def initialize_pipeline(
        self, fastvideo_args: FastVideoArgs,
    ):
        tokenizer = self.get_module("tokenizer")
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_index = self._load_config(self.model_path)
        logger.info(
            "Loading distilled pipeline modules from "
            "config: %s",
            model_index,
        )

        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        model_index.pop("workload_type", None)

        if len(model_index) <= 1:
            raise ValueError(
                "model_index.json must contain at least one "
                "pipeline module")

        required_modules = self.required_config_modules
        modules: dict[str, Any] = {}

        for module_name, module_spec in model_index.items():
            if (not isinstance(module_spec, list)
                    or len(module_spec) < 1):
                continue
            transformers_or_diffusers = module_spec[0]
            if transformers_or_diffusers is None:
                if module_name in self.required_config_modules:
                    self.required_config_modules.remove(
                        module_name)
                continue
            if module_name not in required_modules:
                continue
            if (loaded_modules is not None
                    and module_name in loaded_modules):
                modules[module_name] = loaded_modules[
                    module_name]
                continue

            component_model_path = os.path.join(
                self.model_path, module_name)
            if (module_name == "tokenizer"
                    and not os.path.isdir(
                        component_model_path)):
                gemma_path = os.path.join(
                    self.model_path, "text_encoder", "gemma")
                if os.path.isdir(gemma_path):
                    component_model_path = gemma_path
                else:
                    raise ValueError(
                        "Tokenizer directory missing and "
                        "Gemma weights were not found.")

            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=(
                    transformers_or_diffusers),
                fastvideo_args=fastvideo_args,
            )
            logger.info(
                "Loaded module %s from %s",
                module_name, component_model_path,
            )
            modules[module_name] = module

        if ("tokenizer" in required_modules
                and "tokenizer" not in modules):
            gemma_path = os.path.join(
                self.model_path, "text_encoder", "gemma")
            if os.path.isdir(gemma_path):
                modules["tokenizer"] = (
                    AutoTokenizer.from_pretrained(
                        gemma_path, local_files_only=True))

        for module_name in required_modules:
            if (module_name not in modules
                    or modules[module_name] is None):
                raise ValueError(
                    f"Required module {module_name} was not "
                    "loaded properly")

        return modules


EntryClass = LTX23DistilledPipeline
