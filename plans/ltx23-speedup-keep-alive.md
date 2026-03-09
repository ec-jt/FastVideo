# LTX-2.3 Test Speedup: VRAM Analysis & Keep-Alive Plan

## Problem Statement

Running `test_ltx2_audio_quality.py` takes ~350s total, but only ~140s is actual
video generation. The remaining **210s is model loading** — the dominant bottleneck
when iterating on prompts/settings.

## Current Time Breakdown

| Phase | Time | % of Total |
|-------|------|-----------|
| Gemma prompt enhancement | ~22s | 6% |
| **Model loading (8 workers + FSDP + NCCL)** | **~210s** | **60%** |
| Denoising Stage 1 (half-res, 8 steps) | ~77s | 22% |
| Denoising Stage 2 (full-res, 3 steps) | ~29s | 8% |
| Decode + save + shutdown | ~12s | 4% |

---

## Critical Finding: sp_size=1 Means No Compute Parallelism

The test script at line 357 passes **`sp_size=1`**:

```python
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=num_gpus,    # 8
    tp_size=tp_size,       # 8
    sp_size=1,             # ← NO sequence parallelism!
    use_fsdp_inference=True,
    dit_layerwise_offload=False,
)
```

### What Each Parameter Actually Does

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `num_gpus=8` | 8 | Number of worker processes spawned |
| `tp_size=8` | 8 | Sets `hsdp_shard_dim=8` — FSDP weight sharding across 8 GPUs |
| `sp_size=1` | 1 | Sequence parallelism — **NO compute parallelism** |
| `use_fsdp_inference=True` | True | Use FSDP2 for inference (layer-by-layer weight materialization) |

**This is why TP=8 and TP=2 give the same generation time** — `sp_size=1` means
a single GPU does all the compute. The other GPUs are just weight storage.

---

## How FSDP Inference Avoids OOM (Layer-by-Layer Materialization)

This is the key to understanding the VRAM usage. The 18.99B parameter transformer
has **48 transformer blocks**, and each block is individually wrapped with FSDP via
the `_fsdp_shard_conditions` in `LTX2VideoArchConfig`:

```python
# fastvideo/configs/models/dits/ltx2.py
def is_ltx2_blocks(name, _module):
    return re.search(r"transformer_blocks\.\d+$", name) is not None

_fsdp_shard_conditions = [is_ltx2_blocks]
```

With `reshard_after_forward=True` and `hsdp_shard_dim=8`:

### At Rest (Between Forward Passes)

Each of the 48 transformer blocks has its weights **sharded across 8 GPUs**.
Each block is ~395M params (~790 MB in bf16). Sharded 8 ways = ~99 MB per block per GPU.

```
GPU 0: [block_0_shard_0, block_1_shard_0, ..., block_47_shard_0] = 48 × 99 MB ≈ 4.75 GB
GPU 1: [block_0_shard_1, block_1_shard_1, ..., block_47_shard_1] = 48 × 99 MB ≈ 4.75 GB
...
GPU 7: [block_0_shard_7, block_1_shard_7, ..., block_47_shard_7] = 48 × 99 MB ≈ 4.75 GB
```

### During Forward Pass (One Block at a Time)

When the forward pass reaches `transformer_blocks[0]`, FSDP does an **all-gather**
to reconstruct the full block weights on the compute GPU:

```
Step 1: Processing block 0
  - FSDP all-gathers block_0 shards from all 8 GPUs → full block_0 (~790 MB) on GPU 0
  - GPU 0 runs forward pass through block_0
  - FSDP reshards block_0 (frees the ~690 MB of gathered remote shards)
  - Peak VRAM spike: +690 MB (only the non-local shards)

Step 2: Processing block 1
  - Same pattern: all-gather → compute → reshard
  ...

Step 48: Processing block 47
  - Same pattern
```

### VRAM Budget During Forward Pass

At any given moment during the forward pass, GPU 0 holds:

| Component | Size | Persistent? |
|-----------|------|-------------|
| All 48 block shards (1/8 each) | ~4.75 GB | ✅ Always |
| **One fully-gathered block** | ~790 MB | ❌ Temporary (resharded after use) |
| Non-block model params (embeddings, norms, etc.) | ~200 MB | ✅ Always |
| LatentUpsampler (full copy) | ~1 GB | ✅ Always |
| Audio VAE + Vocoder | ~150 MB | ✅ Always |
| **Activations** (intermediate tensors) | ~8-10 GB | ❌ Temporary |
| CUDA context + NCCL buffers | ~1-2 GB | ✅ Always |
| **Peak Total** | **~16-18 GB** | |

### Why It Doesn't OOM

The trick is that FSDP **never materializes all 48 blocks simultaneously**.
Only 1 block (~790 MB) is fully gathered at a time. Without FSDP, you'd need
all 48 blocks on one GPU = ~38 GB just for weights, plus ~10 GB activations = ~48 GB.

With FSDP sharded across 8 GPUs:
- Persistent weight footprint: **4.75 GB** (1/8 of each block)
- Temporary per-block overhead: **~690 MB** (the 7/8 gathered from other GPUs)
- This is why it fits comfortably in ~15 GB per GPU

### With Fewer GPUs (TP=2)

With `hsdp_shard_dim=2`, each GPU holds 1/2 of each block:
- Persistent: 48 × 395 MB = ~19 GB per GPU
- Temporary per-block: ~395 MB (the other half)
- Plus activations: ~10 GB
- **Total: ~30 GB** — fits in 80 GB

### With TP=1 (No FSDP Sharding)

With `hsdp_shard_dim=1`, the single GPU holds ALL weights:
- All 48 blocks: ~38 GB
- Plus activations: ~10 GB
- Plus upsampler + audio: ~1.2 GB
- Plus CUDA context: ~1 GB
- **Total: ~50 GB** — fits in 80 GB but tight

However, with TP=1 and `use_fsdp_inference=True`, FSDP still wraps each block
with `reshard_after_forward=True`. But since there's only 1 GPU, the "shard" is
the full block — no all-gather needed, no reshard needed. It's effectively a no-op
wrapper. The full 38 GB stays on GPU at all times.

---

## nvidia-smi vs Actual Usage

nvidia-smi reports **15,407 MB** but the "real" persistent model footprint is ~6-7 GB.
The difference:

1. **Activation memory (~8-10 GB)**: Created during forward pass, freed after.
   But PyTorch's CUDA memory allocator **caches freed memory** — it doesn't return
   it to CUDA. So nvidia-smi still shows it as "used" even though PyTorch considers
   it free. This is by design for performance (avoids expensive cudaMalloc calls).

2. **NCCL buffers (~500 MB)**: Allocated once for all-gather/reduce operations.

3. **CUDA context (~800 MB)**: Per-GPU overhead from the CUDA runtime.

You can see the "real" usage with `torch.cuda.memory_allocated()` vs
`torch.cuda.memory_reserved()` — "allocated" is what's actually in use,
"reserved" is what nvidia-smi shows.

---

## Recommended Plan

### Step 1: Try TP=1 Now (Confirm Same Gen Time, Faster Load)

```bash
LTX2_TWO_STAGE=1 python tests/helix/test_ltx2_audio_quality.py \
    --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
    --num-gpus 1 --tp-size 1 --frames 241 \
    --prompt-id singer_spotlight --enhance-prompt
```

Expected: Load time drops from 210s to ~40-60s. Generation time stays ~140s.
VRAM on GPU 0: ~50 GB. Other 7 GPUs: 0 MB.

### Step 2: Implement Keep-Alive Mode

Add `--keep-alive` flag to the test script:

1. Load `VideoGenerator` once
2. Run initial prompt(s)
3. Enter interactive loop for subsequent prompts
4. Zero reload cost for each new generation

### Step 3 (Optional): Investigate SP > 1

Change `sp_size` from 1 to `num_gpus` to actually parallelize the compute.
This could reduce generation time from ~140s to ~20-30s with 8 GPUs.

---

## Implementation: Keep-Alive Mode

### Changes to `tests/helix/test_ltx2_audio_quality.py`

1. Add `--keep-alive` CLI argument
2. Refactor `run_audio_quality_tests()` to separate:
   - Generator creation (one-time)
   - Prompt execution (repeatable)
3. Add interactive loop:
   ```
   [keep-alive] Commands: <prompt_id>, custom:<text>, list, quit
   > singer_spotlight
   ✅ singer_spotlight: 142.9s
   > custom:A robot dancing in rain
   ✅ custom: 139.8s
   > quit
   ```
4. Graceful Ctrl+C handling
