# SPDX-License-Identifier: Apache-2.0
"""
Convert LTX-2.3 nvfp4 quantized weights to FastVideo/Diffusers format.

This script handles the nvfp4 quantization format which stores:
- weight: uint8 tensor (packed 4-bit weights)
- weight_scale: float8_e4m3fn tensor (per-block scales)
- weight_scale_2: float32 scalar (global scale)
- bias: bfloat16 tensor

Example usage:
    python scripts/checkpoint_conversion/convert_ltx2_nvfp4_weights.py \\
        --source /path/to/LTX-2.3-nvfp4/ltx-2.3-22b-dev-nvfp4.safetensors \\
        --output /path/to/output/LTX2.3-nvfp4-Diffusers \\
        --gemma-path /path/to/google/gemma-3-12b-it
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "transformer": ("model.diffusion_model.",),
    "vae": ("vae.",),
    "audio_vae": ("audio_vae.",),
    "vocoder": ("vocoder.",),
    "text_embedding_projection": (
        "text_embedding_projection.",
        "model.text_embedding_projection.",
    ),
}


def _read_metadata(path: Path) -> tuple[dict, dict]:
    """Read config and quantization metadata from safetensors file."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    if not metadata:
        return {}, {}

    config = {}
    if "config" in metadata:
        config = json.loads(metadata["config"])

    quant_meta = {}
    if "_quantization_metadata" in metadata:
        quant_meta = json.loads(metadata["_quantization_metadata"])

    return config, quant_meta


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    """Load all weights from safetensors file."""
    return load_file(str(path))


def _filter_transformer_config(config: dict) -> dict:
    """Extract transformer config with allowed keys."""
    transformer = config.get("transformer", {})
    allowed = {
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "cross_attention_dim",
        "caption_channels",
        "norm_eps",
        "attention_type",
        "positional_embedding_theta",
        "positional_embedding_max_pos",
        "timestep_scale_multiplier",
        "use_middle_indices_grid",
        "rope_type",
        "frequencies_precision",
        "in_channels",
        "out_channels",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_in_channels",
        "audio_out_channels",
        "audio_cross_attention_dim",
        "audio_positional_embedding_max_pos",
        "av_ca_timestep_scale_multiplier",
        # LTX-2.3 cross-attention AdaLN and gated attention
        "cross_attention_adaln",
        "apply_gated_attention",
        "connector_apply_gated_attention",
    }
    filtered = {k: v for k, v in transformer.items() if k in allowed}
    if "frequencies_precision" in filtered:
        filtered["double_precision_rope"] = (
            filtered["frequencies_precision"] == "float64"
        )
        del filtered["frequencies_precision"]
    # LTX-2.3: when caption_proj_before_connector is True, the transformer
    # receives connector output directly, so caption_channels should match
    # the connector output dimension (num_attention_heads * attention_head_dim)
    if transformer.get("caption_proj_before_connector", False):
        connector_heads = transformer.get("connector_num_attention_heads", 32)
        connector_head_dim = transformer.get("connector_attention_head_dim", 128)
        filtered["caption_channels"] = connector_heads * connector_head_dim
    return filtered


def _build_text_embedding_projection_config(
    gemma_model_path: str = "",
    metadata_config: dict | None = None,
) -> dict:
    """Build config for text embedding projection component."""
    transformer = (metadata_config or {}).get("transformer", {})
    connector_num_heads = transformer.get("connector_num_attention_heads", 32)
    connector_head_dim = transformer.get("connector_attention_head_dim", 128)
    connector_num_layers = transformer.get("connector_num_layers", 8)
    connector_num_registers = transformer.get(
        "connector_num_learnable_registers", 128
    )
    connector_norm_output = transformer.get("connector_norm_output", True)
    connector_apply_gated = transformer.get(
        "connector_apply_gated_attention", True
    )
    connector_registers_std = transformer.get(
        "connector_learnable_registers_std", 1
    )
    caption_proj_before_connector = transformer.get(
        "caption_proj_before_connector", True
    )
    audio_connector_head_dim = transformer.get(
        "audio_connector_attention_head_dim", 64
    )
    audio_connector_num_heads = transformer.get(
        "audio_connector_num_attention_heads", 32
    )

    hidden_size = connector_num_heads * connector_head_dim
    gemma_hidden_size = 3840
    num_gemma_layers = 49

    config = {
        "architectures": ["LTX2GemmaTextEncoderModel"],
        "hidden_size": hidden_size,
        "num_hidden_layers": 48,
        "num_attention_heads": connector_num_heads,
        "text_len": 1024,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "gemma_model_path": gemma_model_path,
        "gemma_dtype": "bfloat16",
        "padding_side": "left",
        "feature_extractor_in_features": gemma_hidden_size * num_gemma_layers,
        "feature_extractor_out_features": hidden_size,
        "connector_num_attention_heads": connector_num_heads,
        "connector_attention_head_dim": connector_head_dim,
        "connector_num_layers": connector_num_layers,
        "connector_positional_embedding_theta": 10000.0,
        "connector_positional_embedding_max_pos": [4096],
        "connector_rope_type": "split",
        "connector_double_precision_rope": True,
        "connector_num_learnable_registers": connector_num_registers,
    }
    if connector_norm_output:
        config["connector_norm_output"] = True
    if connector_apply_gated:
        config["connector_apply_gated_attention"] = True
    if connector_registers_std is not None:
        config["connector_learnable_registers_std"] = connector_registers_std
    if caption_proj_before_connector:
        config["caption_proj_before_connector"] = True
    if audio_connector_head_dim is not None:
        config["audio_connector_attention_head_dim"] = audio_connector_head_dim
    if audio_connector_num_heads is not None:
        config["audio_connector_num_attention_heads"] = audio_connector_num_heads
    return config


def _split_component_weights(
    weights: dict[str, torch.Tensor],
    quant_meta: dict,
) -> tuple[dict[str, OrderedDict], dict[str, dict]]:
    """Split weights by component and track quantization metadata."""
    components: dict[str, OrderedDict] = {
        name: OrderedDict() for name in COMPONENT_PREFIXES
    }
    component_quant_meta: dict[str, dict] = {
        name: {"format_version": "1.0", "layers": {}}
        for name in COMPONENT_PREFIXES
    }

    quantized_layers = quant_meta.get("layers", {})

    for key, value in weights.items():
        # Handle audio/video embeddings connector specially
        if key.startswith("model.diffusion_model.audio_embeddings_connector."):
            new_key = key.replace(
                "model.diffusion_model.audio_embeddings_connector.",
                "audio_embeddings_connector.",
            )
            components["text_embedding_projection"][new_key] = value
            continue
        if key.startswith("model.diffusion_model.video_embeddings_connector."):
            new_key = key.replace(
                "model.diffusion_model.video_embeddings_connector.",
                "embeddings_connector.",
            )
            components["text_embedding_projection"][new_key] = value
            continue

        matched = False
        for component, prefixes in COMPONENT_PREFIXES.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    components[component][new_key] = value

                    # Track quantization metadata for this layer
                    # The original key without .weight/.bias/.weight_scale etc
                    base_key = key
                    for suffix in [
                        ".weight",
                        ".bias",
                        ".weight_scale",
                        ".weight_scale_2",
                    ]:
                        if base_key.endswith(suffix):
                            base_key = base_key[: -len(suffix)]
                            break

                    if base_key in quantized_layers:
                        # Map to new key format
                        new_base_key = base_key[len(prefix):]
                        component_quant_meta[component]["layers"][
                            new_base_key
                        ] = quantized_layers[base_key]

                    matched = True
                    break
            if matched:
                break

    # Filter out empty components
    components = {name: w for name, w in components.items() if w}
    component_quant_meta = {
        name: meta
        for name, meta in component_quant_meta.items()
        if name in components and meta["layers"]
    }

    return components, component_quant_meta


def _apply_transformer_key_mapping(key: str) -> str:
    """Apply key mapping for transformer weights."""
    # Remove model.diffusion_model. prefix if present
    if key.startswith("model.diffusion_model."):
        return key[len("model.diffusion_model."):]
    return key


def _write_component(
    output_dir: Path,
    name: str,
    weights: OrderedDict,
    config: dict | None,
    quant_meta: dict | None = None,
    dir_name: str | None = None,
) -> None:
    """Write component weights and config to output directory."""
    component_dir = output_dir / (dir_name or name)
    component_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata for safetensors
    metadata = {}
    if quant_meta and quant_meta.get("layers"):
        metadata["_quantization_metadata"] = json.dumps(quant_meta)

    output_file = component_dir / "model.safetensors"
    save_file(weights, str(output_file), metadata=metadata if metadata else None)
    print(f"Saved {name} weights to {output_file}")

    if config is not None:
        config_path = component_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Saved {name} config to {config_path}")


def _wrap_component_config(
    component_name: str,
    component_config: dict | None,
    class_name: str | None = None,
) -> dict | None:
    """Wrap component config with class name."""
    if component_config is None:
        return None
    wrapped = {component_name: component_config}
    if class_name is not None:
        wrapped["_class_name"] = class_name
    return wrapped


def _build_model_index(
    transformer_class_name: str,
    vae_class_name: str,
    pipeline_class_name: str,
    diffusers_version: str,
) -> dict:
    """Build model_index.json content."""
    return {
        "_class_name": pipeline_class_name,
        "_diffusers_version": diffusers_version,
        "transformer": ["diffusers", transformer_class_name],
        "vae": ["diffusers", vae_class_name],
        "text_encoder": ["transformers", "LTX2GemmaTextEncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "audio_vae": ["diffusers", "LTX2AudioDecoder"],
        "vocoder": ["diffusers", "LTX2Vocoder"],
    }


def _write_model_index(output_dir: Path, model_index: dict) -> None:
    """Write model_index.json to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_index_path = output_dir / "model_index.json"
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")
    print(f"Saved model_index.json to {model_index_path}")


def copy_gemma_tokenizer(gemma_src: Path, tokenizer_dest: Path) -> None:
    """Copy tokenizer files from Gemma model."""
    tokenizer_dest.mkdir(parents=True, exist_ok=True)
    tokenizer_file_names = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ]
    copied = 0
    for file_name in tokenizer_file_names:
        src_path = gemma_src / file_name
        if src_path.is_file():
            shutil.copy2(src_path, tokenizer_dest / file_name)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(
            f"No tokenizer files found in {gemma_src}. "
            "Expected at least one tokenizer file."
        )
    print(f"Copied {copied} tokenizer files to {tokenizer_dest}")


def convert_nvfp4_weights(
    source_path: Path,
    output_dir: Path,
    transformer_class_name: str = "LTX2Transformer3DModel",
    pipeline_class_name: str = "LTX2Pipeline",
    diffusers_version: str = "0.33.0.dev0",
    gemma_model_path: str = "",
    components_to_write: set[str] | None = None,
) -> None:
    """Convert nvfp4 quantized weights to diffusers format."""
    print(f"Loading weights from {source_path}...")
    config, quant_meta = _read_metadata(source_path)
    weights = _load_weights(source_path)

    print(f"Total tensors: {len(weights)}")
    print(
        f"Quantized layers: {len(quant_meta.get('layers', {}))}"
    )

    # Split weights by component
    split_weights, component_quant_meta = _split_component_weights(
        weights, quant_meta
    )

    if components_to_write is not None:
        split_weights = {
            name: w for name, w in split_weights.items()
            if name in components_to_write
        }

    # Apply transformer key mapping
    if "transformer" in split_weights:
        transformer_weights = split_weights["transformer"]
        converted_transformer = OrderedDict()
        for key, value in transformer_weights.items():
            new_key = _apply_transformer_key_mapping(
                f"model.diffusion_model.{key}"
            )
            converted_transformer[new_key] = value
        split_weights["transformer"] = converted_transformer

        # Also update quantization metadata keys
        if "transformer" in component_quant_meta:
            old_layers = component_quant_meta["transformer"]["layers"]
            new_layers = {}
            for key, meta in old_layers.items():
                new_key = _apply_transformer_key_mapping(
                    f"model.diffusion_model.{key}"
                )
                new_layers[new_key] = meta
            component_quant_meta["transformer"]["layers"] = new_layers

    # Build configs
    transformer_config = _filter_transformer_config(config)
    if transformer_config:
        transformer_config["_class_name"] = transformer_class_name

    component_configs: dict[str, dict | None] = {
        "transformer": transformer_config or None,
        "vae": _wrap_component_config(
            "vae",
            config.get("vae"),
            class_name="CausalVideoAutoencoder",
        ),
        "audio_vae": _wrap_component_config(
            "audio_vae",
            config.get("audio_vae"),
            class_name="LTX2AudioDecoder",
        ),
        "vocoder": _wrap_component_config(
            "vocoder",
            config.get("vocoder"),
            class_name="LTX2Vocoder",
        ),
        "text_embedding_projection": _build_text_embedding_projection_config(
            gemma_model_path=gemma_model_path,
            metadata_config=config,
        ),
    }

    # Write components
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, component_weights in split_weights.items():
        _write_component(
            output_dir,
            name,
            component_weights,
            component_configs.get(name),
            quant_meta=component_quant_meta.get(name),
        )
        # Also write text_encoder directory for diffusers compatibility
        if name == "text_embedding_projection":
            _write_component(
                output_dir,
                name,
                component_weights,
                component_configs.get(name),
                quant_meta=component_quant_meta.get(name),
                dir_name="text_encoder",
            )

    # Write model_index.json
    required_for_index = {
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
        "text_embedding_projection",
    }
    if components_to_write is not None and not required_for_index.issubset(
        components_to_write
    ):
        print(
            "Skipping model_index.json; not all diffusers components were written."
        )
        return
    if not required_for_index.issubset(split_weights.keys()):
        print("Skipping model_index.json; missing diffusers components in weights.")
        return

    vae_class_name = (component_configs.get("vae") or {}).get(
        "_class_name", "CausalVideoAutoencoder"
    )
    model_index = _build_model_index(
        transformer_class_name=transformer_class_name,
        vae_class_name=vae_class_name,
        pipeline_class_name=pipeline_class_name,
        diffusers_version=diffusers_version,
    )
    _write_model_index(output_dir, model_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LTX-2.3 nvfp4 weights to FastVideo/Diffusers format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to nvfp4 safetensors file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="LTX2Transformer3DModel",
        help="Transformer class name",
    )
    parser.add_argument(
        "--pipeline-class-name",
        type=str,
        default="LTX2Pipeline",
        help="Pipeline class name for model_index.json",
    )
    parser.add_argument(
        "--diffusers-version",
        type=str,
        default="0.33.0.dev0",
        help="Diffusers version for model_index.json",
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="",
        help="Path to Gemma model for tokenizer and text encoder",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="",
        help=(
            "Comma-separated component list to write "
            "(transformer,vae,audio_vae,vocoder,text_embedding_projection)"
        ),
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    components_to_write: set[str] | None = None
    if args.components:
        components_to_write = {
            c.strip() for c in args.components.split(",") if c.strip()
        }

    # Copy Gemma model if provided
    gemma_model_path = ""
    if args.gemma_path:
        gemma_src = Path(args.gemma_path)
        if not gemma_src.is_dir():
            raise ValueError(f"--gemma-path must be a directory: {gemma_src}")
        gemma_dest = output_dir / "text_encoder" / "gemma"
        if gemma_dest.exists():
            shutil.rmtree(gemma_dest)
        gemma_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(gemma_src, gemma_dest)
        copy_gemma_tokenizer(gemma_src, output_dir / "tokenizer")
        gemma_model_path = "gemma"

    convert_nvfp4_weights(
        source_path=source_path,
        output_dir=output_dir,
        transformer_class_name=args.class_name,
        pipeline_class_name=args.pipeline_class_name,
        diffusers_version=args.diffusers_version,
        gemma_model_path=gemma_model_path,
        components_to_write=components_to_write,
    )


if __name__ == "__main__":
    main()
