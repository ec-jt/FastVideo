#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert LingBot-World-Fast bf16 weights to a prequantized NVFP4
checkpoint.

Reads the diffusers transformer safetensors (37GB bf16), quantizes the
block linears tagged by ``LingBotWorldNVFP4Config`` (all
blocks.*.{to_q,to_k,to_v,to_out,ffn.fc_in,ffn.fc_out}) to NVFP4 layer
by layer on the GPU, and writes ONE safetensors file containing:

  * ``<module>._nvfp4_weight`` / ``._nvfp4_weight_scale`` /
    ``._weight_global_sf`` / ``._nvfp4_alpha`` for quantized linears
  * plain bf16 tensors (custom FastVideo names) for everything else

Total ~9-11GB. Loaded at startup by ``load_prequantized_nvfp4``
(fsdp_load.py) via ``NVFP4_PREQUANT_PATH``, cutting worker load time
from ~54s (dense read + quantize) to roughly the time to read 10GB.

Run inside the lingbot-world image (needs flashinfer + a Blackwell
GPU for the quantize kernels):

  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v /mnt/nvme0/models:/models \
    -v $PWD/../shared/FastVideo:/app/FastVideo \
    danucore/lingbot-world:torch2.12 \
    /app/FastVideo/scripts/checkpoint_conversion/convert_lingbot_nvfp4.py \
    --model-path /models/FastVideo/LingBot-World-Fast-Diffusers \
    --output /models/FastVideo/LingBot-World-Fast-Diffusers/nvfp4_prequant.safetensors
"""

import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, "/app/FastVideo")

from fastvideo.layers.quantization.nvfp4_config import (  # noqa: E402
    _LINGBOT_FP4_SUFFIXES, _nvfp4_quantize, _require_flashinfer)
from fastvideo.configs.models.dits.lingbotworld import (  # noqa: E402
    CausalLingBotWorldVideoConfig)
from fastvideo.models.loader.utils import (  # noqa: E402
    get_param_names_mapping, hf_to_custom_state_dict_iter)
from fastvideo.models.loader.weight_utils import (  # noqa: E402
    safetensors_weights_iterator)


import re

_BLOCKS_RE = re.compile(r"^blocks\.\d+\.")


def _is_quantized_weight(custom_name: str) -> bool:
    """Match the same layer set LingBotWorldNVFP4Config tags.

    Custom state-dict names have no model prefix (``blocks.0.to_q``),
    unlike construction-time prefixes (``Wan.blocks.0.to_q``), so match
    on the leading ``blocks.<i>.`` instead of ``.blocks.``.
    """
    if not custom_name.endswith(".weight"):
        return False
    module_name = custom_name[:-len(".weight")]
    return (_BLOCKS_RE.match(module_name) is not None
            and module_name.endswith(_LINGBOT_FP4_SUFFIXES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transformer_dir = os.path.join(args.model_path, "transformer")
    safetensors_list = sorted(
        glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    assert safetensors_list, f"no safetensors under {transformer_dir}"

    SfLayout, _, _ = _require_flashinfer()
    device = torch.device("cuda:0")

    # HF -> FastVideo custom name mapping from the arch config.
    mapping_fn = get_param_names_mapping(
        CausalLingBotWorldVideoConfig().param_names_mapping)

    t0 = time.time()
    out_sd: dict[str, torch.Tensor] = {}
    n_quant = 0
    n_dense = 0
    reverse_mapping: dict = {}
    weight_iter = safetensors_weights_iterator(safetensors_list,
                                               to_cpu=True)
    custom_iter = hf_to_custom_state_dict_iter(weight_iter, mapping_fn,
                                               reverse_mapping)
    for custom_name, tensor in custom_iter:
        if _is_quantized_weight(custom_name):
            module_name = custom_name[:-len(".weight")]
            w = tensor.to(device=device, dtype=torch.bfloat16)
            global_sf = (448 * 6) / w.float().abs().nan_to_num().max()
            fp4_w, fp4_s = _nvfp4_quantize(
                w,
                global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            gsf = torch.as_tensor(global_sf,
                                  device=device,
                                  dtype=torch.float32)
            # flashinfer may return the packed payload as
            # float4_e2m1fn_x2; safetensors only stores standard
            # dtypes, so persist as uint8 (bit-identical view).
            if fp4_w.dtype != torch.uint8:
                fp4_w = fp4_w.view(torch.uint8)
            if fp4_s.dtype not in (torch.uint8, torch.float32,
                                   torch.bfloat16):
                fp4_s = fp4_s.view(torch.uint8)
            out_sd[f"{module_name}._nvfp4_weight"] = fp4_w.contiguous().cpu()
            out_sd[f"{module_name}._nvfp4_weight_scale"] = fp4_s.contiguous().cpu()
            out_sd[f"{module_name}._weight_global_sf"] = gsf.to(
                torch.bfloat16).cpu()
            out_sd[f"{module_name}._nvfp4_alpha"] = (1.0 / gsf).to(
                torch.float32).cpu()
            del w
            n_quant += 1
            if n_quant % 40 == 0:
                print(f"quantized {n_quant} linears "
                      f"({time.time()-t0:.0f}s)", flush=True)
        else:
            out_sd[custom_name] = tensor.to(torch.bfloat16).contiguous()
            n_dense += 1

    total_bytes = sum(t.numel() * t.element_size()
                      for t in out_sd.values())
    print(f"quantized {n_quant} linears, kept {n_dense} dense tensors, "
          f"total {total_bytes/1e9:.2f}GB", flush=True)

    from safetensors.torch import save_file
    # safetensors cannot store uint8-viewed fp4 packed dtype metadata
    # beyond standard dtypes; fp4 payloads from flashinfer are uint8
    # already so this is a plain save.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_file(out_sd, args.output,
              metadata={"format": "lingbot_nvfp4_prequant_v1"})
    print(f"wrote {args.output} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
