# Plan: SP + MMAP for Maximum Performance

## Goal

Combine **Sequence Parallelism (SP)** across multiple GPUs with **memory-mapped weight sharing** to achieve maximum throughput: fast per-request latency AND efficient multi-instance deployment.

## Current Architecture: 8 × 1-GPU Instances

```
8 instances × 1 GPU each = 8 concurrent requests
Per request: ~40-50s (layerwise offload, CPU→GPU streaming)
Throughput: ~8 requests per ~50s = ~0.16 req/s
```

**Bottleneck:** Each GPU spends most of its time waiting for transformer blocks to stream from CPU. The GPU compute utilization is low (~30-40%) because of the CPU→GPU transfer overhead.

## Proposed Architecture: 4 × 2-GPU SP Instances

```
4 instances × 2 GPUs each (SP=2) = 4 concurrent requests
Per request: ~20-25s (SP splits attention across 2 GPUs)
Throughput: ~4 requests per ~25s = ~0.16 req/s
```

**Same throughput, but 2× faster per-request latency.** Each request uses 2 GPUs with sequence parallelism, halving the attention computation time.

## Or: 2 × 4-GPU SP Instances

```
2 instances × 4 GPUs each (SP=4) = 2 concurrent requests
Per request: ~10-15s (SP splits attention across 4 GPUs)
Throughput: ~2 requests per ~15s = ~0.13 req/s
```

**Slightly lower throughput, but 3-4× faster per-request.** Best for latency-sensitive use cases.

## How SP Works with FSDP2

With SP + FSDP2:
- **FSDP2** shards the 38GB transformer weights across N GPUs (each holds 38/N GB)
- **SP** splits the sequence dimension across N GPUs (each processes 1/N of the tokens)
- During attention: each GPU computes attention on its token shard, then all-reduces
- **No CPU offloading needed** — weights fit on GPU when sharded across 2+ GPUs

| Config | Weights per GPU | Fits in 32GB? | CPU Offload? |
|--------|----------------|---------------|--------------|
| 1 GPU, SP=1 | 38GB | ❌ (needs offload) | Yes |
| 2 GPUs, SP=2 | 19GB | ✅ | No |
| 4 GPUs, SP=4 | 9.5GB | ✅ | No |
| 8 GPUs, SP=8 | 4.75GB | ✅ | No |

**With 2+ GPUs per instance, the entire model fits in GPU VRAM — no CPU offloading needed.** This eliminates the CPU→GPU streaming bottleneck entirely.

## Where MMAP Fits In

With the multi-GPU SP approach, each instance still needs to load the model weights during startup. MMAP helps here:

### Without MMAP (current)
```
Instance 0 (GPUs 0-1): Load 38GB from NVMe → CPU → distribute to 2 GPUs
Instance 1 (GPUs 2-3): Load 38GB from NVMe → CPU → distribute to 2 GPUs
Instance 2 (GPUs 4-5): Load 38GB from NVMe → CPU → distribute to 2 GPUs
Instance 3 (GPUs 6-7): Load 38GB from NVMe → CPU → distribute to 2 GPUs
Total CPU RAM for loading: 4 × 38GB = 152GB (temporary, during load)
```

### With MMAP
```
All instances: MMAP the same 38GB safetensors file
OS page cache: 38GB shared across all 4 instances
Each instance: Read from page cache → distribute to GPUs
Total CPU RAM: 38GB shared (not 152GB)
```

MMAP is most beneficial during **startup** — it reduces the total CPU RAM needed for loading. Once weights are on GPU (via FSDP2 sharding), the CPU RAM is freed.

**Key insight:** With SP=2 and FSDP2, weights live on GPU, not CPU. MMAP only helps during the initial load phase. After startup, CPU RAM usage is minimal.

## Performance Comparison

| Config | GPUs/Instance | Instances | Concurrent | Per-Request | Throughput | CPU RAM |
|--------|--------------|-----------|------------|-------------|------------|---------|
| Current: 1 GPU, layerwise | 1 | 8 | 8 | ~45s | ~0.18/s | 8×38GB=304GB |
| **SP=2, FSDP2** | **2** | **4** | **4** | **~20s** | **~0.20/s** | 4×38GB=152GB |
| SP=4, FSDP2 | 4 | 2 | 2 | ~12s | ~0.17/s | 2×38GB=76GB |
| SP=8, FSDP2 | 8 | 1 | 1 | ~8s | ~0.13/s | 38GB |
| SP=2 + MMAP | 2 | 4 | 4 | ~20s | ~0.20/s | **38GB shared** |
| SP=4 + MMAP | 4 | 2 | 2 | ~12s | ~0.17/s | **38GB shared** |

## Recommended: SP=2 with 4 Instances

**Best balance of latency and throughput:**
- 4 instances × 2 GPUs = 8 GPUs total
- ~20s per request (2× faster than current)
- 4 concurrent requests
- ~0.20 req/s throughput (slightly better than current)
- With MMAP: only 38GB shared CPU RAM during loading

### Configuration

```yaml
# docker-compose.yml for SP=2
services:
  ltx2.3-distilled-0:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - NUM_GPUS=2
      - TP_SIZE=2
      - SP_SIZE=2
      - USE_FSDP_INFERENCE=true
      - DIT_LAYERWISE_OFFLOAD=false  # Not needed with 2 GPUs
      - PORT=18000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]

  ltx2.3-distilled-1:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # Docker remaps to physical GPUs 2,3
      - NUM_GPUS=2
      - TP_SIZE=2
      - SP_SIZE=2
      - PORT=18001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2', '3']
              capabilities: [gpu]
  # ... instances 2-3 for GPUs 4-7
```

## Implementation Steps

### Phase 1: Test SP=2 (no MMAP, no code changes)

Just change the docker-compose config:
```bash
NUM_GPUS=2 TP_SIZE=2 SP_SIZE=2
USE_FSDP_INFERENCE=true
DIT_LAYERWISE_OFFLOAD=false
```

This should work out of the box — FSDP2 + SP is already supported in FastVideo.

### Phase 2: Add MMAP for startup optimization

Modify `weight_utils.py` to use memory-mapped loading:
1. Open safetensors with `safe_open` in mmap mode
2. Keep file handles alive during loading
3. After FSDP2 distributes weights to GPUs, close handles (CPU RAM freed)

### Phase 3: Combine with VSA

SP + VSA stack:
- SP halves the sequence per GPU → smaller attention matrices
- VSA prunes attention within each GPU's shard → even less compute
- Combined: ~4× attention speedup over baseline

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| NCCL overhead for SP | ~5-10% overhead per step for all-reduce | Minimal with NVLink/PCIe Gen5 |
| Fewer concurrent requests | 4 instead of 8 | Per-request is 2× faster, throughput similar |
| MMAP + FSDP2 interaction | FSDP2 may copy from mmap (defeating sharing) | Test with `torch.cuda.memory_allocated()` |
| SP + distilled model quality | SP shouldn't affect quality (mathematically equivalent) | Verify with SSIM test |

## Summary

**SP=2 with 4 instances is the sweet spot** — 2× faster per-request, same throughput, and with MMAP the startup CPU RAM drops from 304GB to 38GB shared. This is the maximum performance configuration for 8 GPUs.
