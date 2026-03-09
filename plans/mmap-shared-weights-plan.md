# Plan: Memory-Mapped Shared Weights for Multi-Instance Deployment

## Problem

Running 8 instances of LTX-2.3 (1 GPU each) requires 8 × 38GB = **304GB of pinned CPU RAM** for model weights alone. Each instance loads its own copy of the same safetensors file into separate memory allocations.

## Solution: Memory-Mapped Safetensors

Use `safetensors`' built-in mmap support so all 8 processes share the same physical RAM pages for the model weights. The OS kernel handles the sharing via virtual memory — each process gets its own virtual address space pointing to the same physical pages.

## How It Works Today

```python
# weight_utils.py:134 — current loading path
with safe_open(st_file, framework="pt", device="cpu") as f:
    for name in f.keys():
        param = f.get_tensor(name)  # COPIES tensor into new allocation
        yield name, param
```

Each `f.get_tensor(name)` allocates new memory and copies the data from the file. 8 processes = 8 copies = 304GB.

## How mmap Would Work

```python
# Proposed: mmap-based loading
from safetensors import safe_open

with safe_open(st_file, framework="pt", device="cpu") as f:
    for name in f.keys():
        # safe_open already mmaps the file internally.
        # get_tensor() returns a view into the mmap'd region
        # if we DON'T copy it.
        param = f.get_tensor(name)
        # The tensor's storage points to the mmap'd file.
        # Multiple processes opening the same file share
        # the same physical pages via the OS page cache.
        yield name, param
```

Actually, `safetensors` `safe_open` already uses mmap internally. The issue is that `get_tensor()` returns a **copy** by default. To get a zero-copy view, we need to use the lower-level API or keep the file handle open.

## Detailed Implementation Plan

### Approach: Keep safetensors file handles open for mmap sharing

The `safetensors` library's `safe_open` mmaps the file. `get_tensor()` copies from the mmap. To avoid the copy, we can:

1. Use `safe_open` and keep the handle alive (don't close it)
2. Access tensors as views into the mmap'd region
3. The layerwise offload hooks will pin and copy individual blocks on demand

### Changes Required

#### 1. `fastvideo/models/loader/weight_utils.py` — Add mmap iterator

```python
def safetensors_weights_iterator_mmap(
    hf_weights_files: list[str],
) -> tuple[Generator[tuple[str, torch.Tensor], None, None], list]:
    """Iterate over weights using mmap (zero-copy, shared across processes).
    
    Returns (iterator, file_handles) — caller must keep file_handles alive
    to prevent the mmap from being unmapped.
    """
    handles = []
    for st_file in hf_weights_files:
        f = safe_open(st_file, framework="pt", device="cpu")
        handles.append(f)
    
    def _iter():
        for f in handles:
            for name in f.keys():
                yield name, f.get_tensor(name)
    
    return _iter(), handles
```

**Problem:** `get_tensor()` still copies. The safetensors Python API doesn't expose zero-copy views.

### Alternative Approach: numpy mmap + torch.from_numpy

```python
import numpy as np
from safetensors import safe_open

def load_safetensors_mmap(path: str) -> dict[str, torch.Tensor]:
    """Load safetensors with true zero-copy mmap."""
    # safetensors stores tensors contiguously in the file
    # We can mmap the entire file and create tensor views
    with safe_open(path, framework="numpy") as f:
        tensors = {}
        for name in f.keys():
            np_tensor = f.get_tensor(name)  # numpy array backed by mmap
            tensors[name] = torch.from_numpy(np_tensor)
    return tensors
```

**Problem:** `framework="numpy"` may not support all dtypes (e.g., bfloat16).

### Best Approach: Use safetensors' `torch.load` with mmap

PyTorch 2.1+ supports `torch.load(..., mmap=True)` which memory-maps the file. But safetensors isn't a PyTorch checkpoint format.

### Practical Approach: Pin-on-demand in layerwise hooks

Instead of pinning all 38GB upfront, modify the layerwise offload hooks to:
1. Load weights to regular (non-pinned) CPU memory via mmap
2. Pin each block's memory only when it's about to be prefetched
3. Unpin after the block is released

This way, the mmap'd pages are shared across processes (read-only), and only the currently-active block (~800MB) needs pinned memory per process.

## Downsides of mmap Approach

| Concern | Impact | Mitigation |
|---------|--------|------------|
| **No pinned memory** | Async DMA transfers require pinned memory. mmap'd pages are not pinned. | Pin on demand (copy block to pinned buffer before DMA) |
| **Page faults** | First access to each mmap'd page triggers a page fault (~4KB at a time). Can cause latency spikes during first forward pass. | Pre-fault pages with `madvise(MADV_WILLNEED)` or `mlock()` |
| **Copy-on-write for pinning** | `pin_memory()` on a mmap'd tensor triggers copy-on-write, creating a private copy. Defeats the sharing purpose. | Use a separate pinned buffer and copy into it (double-buffer approach) |
| **bfloat16 support** | numpy doesn't support bfloat16. safetensors' numpy backend can't handle bf16 tensors. | Use PyTorch framework, accept the copy, or use custom mmap |
| **File handle lifetime** | mmap'd tensors are only valid while the file handle is open. Must keep handles alive for the process lifetime. | Store handles in the model/pipeline object |
| **Docker volume mounts** | mmap works across Docker containers if they mount the same host file (`:ro`). | Already using `:ro` mount for models |

## Recommendation

**For 8 instances on the same machine, the mmap approach has limited benefit because of the pinned memory requirement.** The layerwise offload hooks need pinned memory for async DMA, and pinning mmap'd pages triggers copy-on-write, creating per-process copies anyway.

**Better alternatives for reducing RAM usage:**

### Option A: Shared pinned memory via CUDA IPC (complex)
- One "loader" process pins the weights
- Other processes access via CUDA IPC or shared memory
- Complex to implement, fragile

### Option B: Sequential block loading (simple, recommended)
- Don't load all 38GB to CPU upfront
- Load each block from disk on demand (safetensors supports random access)
- Pin only the current + next block (~1.6GB pinned per process)
- 8 processes × 1.6GB = 12.8GB pinned total (vs 304GB)
- Trade-off: slightly slower first pass (disk I/O per block), but subsequent passes use OS page cache

### Option C: Reduce instances
- Run 4 instances instead of 8 (4 × 38GB = 152GB, more manageable)
- Each instance handles 2 requests sequentially
- Simpler, no code changes needed

## Next Steps

1. **First: Test the layerwise offload changes** — verify 1 instance works correctly
2. **Then: Measure actual RAM usage** — `free -h` to see how much is available
3. **If RAM is sufficient (>320GB):** Run 8 instances simultaneously, no mmap needed
4. **If RAM is tight:** Implement Option B (sequential block loading) for minimal RAM usage
