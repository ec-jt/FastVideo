# LTX-2.3 Single-GPU Optimization Session Summary

## What We Investigated

Starting from the question "How can a 19B parameter model run on 1 GPU?", we traced the entire memory management system in FastVideo's LTX-2.3 pipeline.

### Key Findings

1. **Two offloading mechanisms exist** — FastVideo has its own `dit_layerwise_offload` (custom async prefetch hooks) and PyTorch's `use_fsdp_inference` + `dit_cpu_offload` (FSDP2 with CPUOffloadPolicy). They are mutually exclusive.

2. **Gemma loads twice** — Once for prompt enhancement (optional, `--enhance-prompt`) and once inside the text encoder during pipeline forward (always, lazy-loaded on first call).

3. **Flash Attention doesn't work** — `flash_attn==2.8.3` is installed but the CUDA extension `flash_attn_2_cuda` was compiled for SM 8.0-9.0, not SM 12.0 (RTX 5090 Blackwell). Needs rebuild: `TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --no-build-isolation`.

4. **The pipeline already stays alive** — The `Worker` creates the pipeline once and the `worker_busy_loop()` handles multiple requests. No reload between requests.

5. **Layerwise offload couldn't work on LTX2** — Two bugs: (a) it searched only direct children for `nn.ModuleList` but LTX2 nests it inside `model.model`, and (b) it required loading the full model to GPU first (38GB > 32GB GPU).

---

## All Code Changes Made

### FastVideo Core (`/home/ubuntu/FastVideo/`)

#### 1. `fastvideo/fastvideo_args.py` — Offload config logging + layerwise CPU loading

**Offload logging** (line 677): After `check_fastvideo_args()` validation, logs which offloading mechanism is active:
```
INFO Offload config: transformer=layerwise (FastVideo custom hooks), text_encoder_cpu_offload=True, ...
```

**Layerwise + CPU offload** (line 665): Changed behavior so `dit_layerwise_offload=True` now **enables** `dit_cpu_offload` instead of disabling it. This tells the weight loader to place weights on CPU during initial loading, which is required when the model is larger than GPU VRAM.

#### 2. `fastvideo/models/loader/fsdp_load.py` — CPU loading path for non-FSDP

**Weight loading** (line 321): When `cpu_offload=True` without FSDP (`no device_mesh`), loads weights to CPU instead of GPU. Previously always loaded to GPU regardless of `cpu_offload` flag.

**Zero-init path** (line 364): Same fix for zero-initialized parameters not found in checkpoint.

#### 3. `fastvideo/hooks/layerwise_offload.py` — Recursive ModuleList search + non-block GPU migration

**Recursive search** (line 140): Changed `model.named_children()` to `model.named_modules()` to find `nn.ModuleList` nested inside wrapper models (LTX2 has `model.model.transformer_blocks`).

**Non-block param migration** (line 183): After attaching hooks to ModuleList blocks, moves non-offloaded parameters (embeddings, norms, projection layers ~200MB) from CPU to GPU. Block params stay on CPU managed by the async prefetch hooks.

#### 4. `fastvideo/models/loader/component_loader.py` — Recursive compatibility check + logging

**Recursive check** (line 898): Changed `model.children()` to `model.modules()` for the `nn.ModuleList` compatibility check.

**Transformer offload logging** (line 872): Logs the specific offloading strategy and model size after loading.

#### 5. `tests/helix/test_ltx2_audio_quality.py` — Keep-alive + multi-prompt + layerwise offload

**Layerwise offload** (line 366): Changed from `use_fsdp_inference=True, dit_layerwise_offload=False` to `use_fsdp_inference=False, dit_layerwise_offload=True`.

**Multiple custom prompts** (line 660): `--custom-prompt` now accepts multiple values via `nargs="+"`.

**Repeat mode** (line 690): Added `--repeat N` flag to generate each prompt N times back-to-back using the same pipeline (proves keep-alive works).

**Keep-alive loop** (line 460): After initial prompt loop, if `repeat > 1`, runs additional generations without shutdown.

### Service Worker (`/mnt/nvme0/dc-disc-poc/worker/LTX2.3-Distilled/`)

#### 6. `app/models/pipeline.py` — Warmup + request timing

**Warmup generation** (line 145): Runs a minimal generation (9 frames, 256x256, 2 steps) during startup to pre-load Gemma text encoder and prime CUDA caches. Controlled by `WARMUP_ON_STARTUP` env var.

**Request timing** (line 249): Added per-request counting and timing: `[req #1] Generated video in 42.3s (pipeline warm, no reload)`.

**Removed `torch.cuda.empty_cache()`**: Was called before every request unnecessarily.

#### 7. `app/config.py` — Warmup config

Added `WARMUP_ON_STARTUP` config variable (default: `true`).

#### 8. `.env` — Single GPU config

Changed `NUM_GPUS=1`, `TP_SIZE=1`, added `WARMUP_ON_STARTUP=true`.

#### 9. `docker-compose.yml` — Single GPU deployment

Changed to `CUDA_VISIBLE_DEVICES=0`, `NUM_GPUS=1`, `TP_SIZE=1`, GPU count: 1, added `WARMUP_ON_STARTUP=true`.

### Documentation

#### 10. `plans/ltx23-single-gpu-memory-explanation.md`

Deep-dive explanation of how the 19B model fits on 1 GPU, covering:
- Both offloading mechanisms with visual flows (Mermaid diagrams)
- Gemma double-load analysis
- Provenance of each component (FastVideo custom vs PyTorch native vs official LTX-2 repo)
- Memory budget breakdown
- Full pipeline timeline

---

## Test Commands

### Two prompts with layerwise offload (async prefetch):
```bash
LTX2_TWO_STAGE=1 python tests/helix/test_ltx2_audio_quality.py \
    --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
    --num-gpus 1 --tp-size 1 --frames 241 \
    --custom-prompt \
        "Ariel the little mermaid singing, 'Hello Valeria, I love to sing for you'" \
        "A jazz pianist playing a smooth melody in a dimly lit club" \
    --enhance-prompt
```

### Same prompt twice (keep-alive proof):
```bash
LTX2_TWO_STAGE=1 python tests/helix/test_ltx2_audio_quality.py \
    --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
    --num-gpus 1 --tp-size 1 --frames 241 \
    --prompt-id singer_spotlight --repeat 2
```

### Revert to FSDP2 (if layerwise has issues):
Change line 366-367 in `test_ltx2_audio_quality.py`:
```python
use_fsdp_inference=True,
dit_layerwise_offload=False,
```

---

## Performance Expectations

| Configuration | First Request | Subsequent Requests |
|--------------|---------------|---------------------|
| FSDP2 CPU offload (previous) | ~55s | ~50s |
| Layerwise offload (new) | ~45s (estimated) | ~35-40s (estimated) |

The layerwise offload should be faster because it uses **async prefetching** — copying the next transformer block from CPU to GPU while the current block is computing. FSDP2 does this synchronously.

---

## Remaining Speedup Opportunities

1. **Rebuild flash_attn for Blackwell SM 12.0** — ~5-10% attention speedup
2. **`torch.compile`** — ~10-20% overall (one-time compilation cost)
3. **Multiple GPUs with `sp_size > 1`** — Linear speedup for denoising (2 GPUs = 2x faster)
4. **Fewer frames** — 121 frames (5s) instead of 241 (10s) = ~2x faster
