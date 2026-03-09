# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 two-stage text-to-video pipeline.

Upstream equivalent: ``TI2VidTwoStagesPipeline`` from
``LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py``

Resolution convention (matches upstream):
  The user specifies the **full (Stage 2)** resolution
  (e.g. 1024×1536).  Stage 1 internally runs at half
  (512×768).

Stage 1: Generate at half resolution with full CFG guidance.
Stage 2: Refine at full resolution with distilled LoRA (no CFG).
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
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.stages import (
    DecodingStage, InputValidationStage,
    LTX2AudioDecodingStage,
    LTX2LatentPreparationStage,
    LTX2TextEncodingStage)
from fastvideo.pipelines.stages.ltx2_two_stage_denoising import (
    LTX2TwoStageDenoisingStage)

logger = init_logger(__name__)

_UPSAMPLER_SEARCH_PATHS = [
    "spatial_upsampler/model.safetensors",
    "spatial_upsampler.safetensors",
    "/mnt/nvme0/models/Lightricks/LTX-2.3/"
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
]

# Search paths for the distilled LoRA weights (Stage 2).
_DISTILLED_LORA_SEARCH_PATHS = [
    "distilled_lora/model.safetensors",
    "distilled_lora.safetensors",
    "/mnt/nvme0/models/Lightricks/LTX-2.3/"
    "ltx-2.3-22b-distilled-lora-384.safetensors",
]


class LTX23TwoStagePipeline(LoRAPipeline):
    """Two-stage LTX-2.3 pipeline: CFG Stage 1 + distilled Stage 2.

    Extends ``LoRAPipeline`` to support loading a distilled LoRA
    for Stage 2 refinement.  Stage 1 uses the base (non-distilled)
    model with full multi-modal guidance.

    Key differences from ``LTX23DistilledPipeline``:

    * Stage 1 uses CFG/STG/modality guidance (multiple forward
      passes per step).
    * Stage 2 applies a distilled LoRA on top of the base model.
    * Designed for ``guidance_scale > 1.0`` with negative prompts.
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

        spatial_upsampler = self._load_spatial_upsampler(
            fastvideo_args)

        vae = self.get_module("vae")
        per_channel_stats = self._get_per_channel_stats(vae)

        # Auto-discover distilled LoRA for Stage 2.
        # LoRAPipeline.__init__ only calls convert_to_lora_layers()
        # when fastvideo_args.lora_path is set at init time.
        # Since we auto-discover the path here (after __init__),
        # we must proactively initialise the LoRA layer
        # infrastructure so that set_lora_adapter() works in
        # the merge callback.
        # Ref: upstream TI2VidTwoStagesPipeline loads the
        # distilled LoRA via stage_2_model_ledger.
        distilled_lora_path = self._find_distilled_lora(
            fastvideo_args)
        if distilled_lora_path:
            logger.info(
                "Found distilled LoRA for Stage 2: %s",
                distilled_lora_path)
            # Ensure LoRA layers are initialised even though
            # fastvideo_args.lora_path was None at __init__
            # time.  convert_to_lora_layers() is idempotent
            # (guarded by self.lora_initialized).
            self.convert_to_lora_layers()
        else:
            logger.warning(
                "Distilled LoRA not found. Stage 2 will "
                "use base model without LoRA (lower quality)."
                " Set LTX2_DISTILLED_LORA_PATH env var.")

        # Build LoRA merge/unmerge callbacks for the
        # denoising stage to call between stages.
        lora_pipeline_ref = self

        def merge_distilled_lora():
            if distilled_lora_path:
                logger.info(
                    "[TwoStage] Loading distilled LoRA "
                    "from %s", distilled_lora_path)
                logger.info(
                    "[TwoStage] lora_initialized=%s, "
                    "lora_layers keys=%s",
                    lora_pipeline_ref.lora_initialized,
                    list(lora_pipeline_ref.lora_layers.keys()),
                )
                lora_pipeline_ref.set_lora_adapter(
                    "distilled_stage2",
                    distilled_lora_path,
                )
                lora_pipeline_ref.merge_lora_weights()
                logger.info(
                    "[TwoStage] Merged distilled LoRA "
                    "for Stage 2 (%d layers converted)",
                    sum(
                        sum(1 for _ in ll.all_lora_layers())
                        for ll
                        in lora_pipeline_ref.lora_layers
                        .values()
                    ),
                )

        def unmerge_distilled_lora():
            if distilled_lora_path:
                lora_pipeline_ref.unmerge_lora_weights()
                logger.info(
                    "[TwoStage] Unmerged distilled LoRA "
                    "after Stage 2")

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2TwoStageDenoisingStage(
                transformer=self.get_module("transformer"),
                spatial_upsampler=spatial_upsampler,
                per_channel_statistics=per_channel_stats,
                vae=vae,
                merge_stage2_lora_fn=merge_distilled_lora,
                unmerge_stage2_lora_fn=unmerge_distilled_lora,
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

    @staticmethod
    def _get_per_channel_stats(vae):
        """Extract per_channel_statistics from the VAE."""
        if vae is None:
            return None
        if hasattr(vae, "per_channel_statistics"):
            return vae.per_channel_statistics
        if hasattr(vae, "decoder") and hasattr(
            vae.decoder, "per_channel_statistics"
        ):
            return vae.decoder.per_channel_statistics
        if hasattr(vae, "model") and hasattr(
            vae.model, "per_channel_statistics"
        ):
            return vae.model.per_channel_statistics
        logger.warning(
            "Could not find per_channel_statistics on VAE")
        return None

    def _load_spatial_upsampler(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> LatentUpsampler | None:
        """Try to load the spatial upsampler."""
        import torch

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

        for rel_path in _UPSAMPLER_SEARCH_PATHS:
            candidate = os.path.join(
                self.model_path, rel_path)
            if os.path.exists(candidate):
                logger.info(
                    "Loading spatial upsampler: %s",
                    candidate)
                return LatentUpsampler.from_pretrained(
                    candidate,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
            if os.path.isabs(rel_path) and os.path.exists(
                rel_path
            ):
                logger.info(
                    "Loading spatial upsampler: %s",
                    rel_path)
                return LatentUpsampler.from_pretrained(
                    rel_path,
                    device="cpu",
                    dtype=torch.bfloat16,
                )

        logger.warning(
            "Spatial upsampler not found. "
            "Set LTX2_SPATIAL_UPSAMPLER_PATH or place "
            "weights in model directory.")
        return None

    def _find_distilled_lora(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> str | None:
        """Find the distilled LoRA weights for Stage 2.

        Searches environment variable, model directory, and
        known external paths.
        """
        # Check environment variable override
        lora_path = os.environ.get(
            "LTX2_DISTILLED_LORA_PATH")
        if lora_path and os.path.exists(lora_path):
            return lora_path

        # Check fastvideo_args.lora_path (if user specified)
        if (fastvideo_args.lora_path
                and os.path.exists(fastvideo_args.lora_path)):
            return fastvideo_args.lora_path

        # Search known paths
        for rel_path in _DISTILLED_LORA_SEARCH_PATHS:
            candidate = os.path.join(
                self.model_path, rel_path)
            if os.path.exists(candidate):
                return candidate
            if os.path.isabs(rel_path) and os.path.exists(
                rel_path
            ):
                return rel_path

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
            "Loading two-stage pipeline modules: %s",
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


EntryClass = LTX23TwoStagePipeline
