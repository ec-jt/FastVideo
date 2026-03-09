# Plan: torch.compile + Keep Gemma on GPU

## 1. torch.compile — 10-20% Speedup

### What It Does

`torch.compile` traces the PyTorch operations in each transformer block and generates optimized CUDA kernels via Triton. Benefits:
- **Operator fusion**: Combines multiple small ops (layernorm + linear + activation) into single kernels
- **Memory planning**: Reduces intermediate tensor allocations
- **Kernel optimization**: Triton generates GPU-specific kernels for your SM 12.0 (Blackwell)

### How FastVideo Already Supports It

FastVideo has built-in support via `enable_torch_compile=True` in `FastVideoArgs`:

```python
# fastvideo/fastvideo_args.py:139
enable_torch_compile: bool = False
torch_compile_kwargs: dict[str, Any] = field(default_factory=dict)
```

And in `fsdp_load.py:158`:
```python
compile_in_loader = enable_torch_compile and training_mode
if compile_in_loader:
    model = torch.compile(model, **compile_kwargs)
```

**Problem:** The current code only compiles during training (`training_mode=True`). For inference, we need to compile separately.

### Implementation Plan

#### Option A: Compile in component_loader after loading (recommended)

Add to `component_loader.py` after the model is loaded and layerwise offload is set up:

```python
if fastvideo_args.enable_torch_compile and fastvideo_args.inference_mode:
    compile_kwargs = fastvideo_args.torch_compile_kwargs or {}
    # Use 'reduce-overhead' mode for inference (smaller graphs, less compilation time)
    compile_kwargs.setdefault("mode", "reduce-overhead")
    logger.info("Compiling transformer for inference with kwargs=%s", compile_kwargs)
    model = torch.compile(model, **compile_kwargs)
```

#### Option B: Compile individual blocks (better for layerwise offload)

With layerwise offload, the model's forward is intercepted by hooks. Compiling the whole model may conflict with the hook mechanism. Instead, compile each block individually:

```python
if fastvideo_args.enable_torch_compile and fastvideo_args.inference_mode:
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            for idx, block in enumerate(module):
                module[idx] = torch.compile(block, mode="reduce-overhead")
            break
```

### Compatibility with Layerwise Offload

**Concern:** `torch.compile` traces the forward pass and generates static graphs. The layerwise offload hooks dynamically swap parameters in/out. This may cause issues:

1. **Parameter mutation**: The hooks replace `param.data` with placeholders after each block. `torch.compile` may cache the parameter references and not see the swaps.
2. **CUDA stream synchronization**: The async prefetch uses a separate CUDA stream. `torch.compile` may not be aware of this.

**Mitigation:** Use `torch.compile(block, dynamic=True)` to handle dynamic shapes, and test thoroughly. The `@torch.compiler.disable` decorators on the hook methods (already present in `layerwise_offload.py`) should prevent compilation of the hook logic itself.

### Expected Impact

| Phase | Without compile | With compile | Speedup |
|-------|----------------|--------------|---------|
| Per denoising step (half-res) | ~2.0s | ~1.6-1.8s | 10-20% |
| Per denoising step (full-res) | ~3.75s | ~3.0-3.4s | 10-20% |
| First request (compilation) | N/A | +60-120s | One-time cost |
| Subsequent requests | ~50s | ~40-45s | 10-20% |

### Compilation Cache

PyTorch caches compiled kernels in `~/.cache/torch/`. For Docker, mount this as a named volume so restarting the container doesn't trigger recompilation:

```yaml
volumes:
  - torch_compile_cache:/root/.cache/torch
```

### How to Enable

```python
# In test script:
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=1, tp_size=1, sp_size=1,
    use_fsdp_inference=False,
    dit_layerwise_offload=True,
    enable_torch_compile=True,
    torch_compile_kwargs={"mode": "reduce-overhead"},
)
```

Or via environment variable (needs to be added):
```bash
ENABLE_TORCH_COMPILE=true
```

### Risks

- **Blackwell SM 12.0 support**: Triton may not have full SM 12.0 support yet. Could fail during compilation.
- **Layerwise offload interaction**: Parameter swapping may break compiled graphs. Needs testing.
- **First request latency**: +60-120s for compilation. Unacceptable for cold starts unless cached.
- **Memory during compilation**: Compilation itself uses significant GPU memory for tracing.

---

## 2. Keep Gemma on GPU — Eliminate 13.5s Text Encoding Overhead

### The Problem

Currently, the Gemma text encoder (~9B params, ~18GB in bf16) lives on CPU. Every request:
1. Moves Gemma to GPU (~5s for 18GB CPU→GPU transfer)
2. Runs forward pass (~3s)
3. Moves Gemma back to CPU (~5s)
4. Total: **~13.5s per request** just for text encoding

### Why It Doesn't Fit Today

| Component | VRAM | Notes |
|-----------|------|-------|
| Transformer block (layerwise) | ~1.6GB | 2 blocks: current + prefetched |
| Non-block params on GPU | ~200MB | Embeddings, norms, proj |
| Activations (full-res) | ~8-10GB | During denoising |
| CUDA context | ~1GB | Fixed overhead |
| **Available for Gemma** | **~12-19GB** | On 32GB RTX 5090 |
| **Gemma needs** | **~18GB** | Full model in bf16 |

It's tight — Gemma (18GB) + peak activations (10GB) + transformer block (1.6GB) + overhead (2GB) = ~31.6GB on a 32GB GPU. Might just barely fit, or might OOM during full-res denoising.

### Options

#### Option A: Keep Gemma on GPU, reduce activation memory (risky)

Keep Gemma on GPU permanently. Reduce activation memory by:
- Using gradient checkpointing (inference doesn't use this)
- Reducing batch size (already 1)
- Using smaller resolution (reduces quality)

**Risk:** OOM during full-res Stage 2 denoising when activations peak.

#### Option B: Quantize Gemma to FP8 (~9GB instead of 18GB)

Quantize Gemma to FP8 (8-bit float). Halves the memory:
- Gemma FP8: ~9GB
- Total: 9 + 10 + 1.6 + 2 = ~22.6GB — fits comfortably in 32GB

**Implementation:**
```python
# In gemma.py, when loading Gemma:
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
self._gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma_path,
    quantization_config=quantization_config,
    device_map="cuda",
)
```

**Risk:** Slight quality degradation in text encoding. Needs quality testing.

#### Option C: Keep Gemma on GPU only during text encoding, don't move back to CPU

Currently the code does:
```python
model.to(device=get_local_torch_device())  # GPU
outputs = model(...)
model.to(device=orig_device)  # back to CPU
```

Instead, keep Gemma on GPU after first use. It stays there for subsequent requests. The layerwise offload hooks manage the transformer memory, so Gemma and the transformer don't compete for the same VRAM (only 1-2 blocks are on GPU at a time).

**Memory budget with Gemma on GPU:**
- Gemma: 18GB (permanent)
- Transformer block: 1.6GB (temporary, per step)
- Non-block params: 0.2GB (permanent)
- Activations: 8-10GB (temporary, during denoising)
- CUDA context: 1GB
- **Total peak: ~30.8GB** — fits in 32GB with ~1.2GB headroom

**This is the simplest and best option.** Just remove the `.to(orig_device)` line in `gemma.py:723`.

### Recommended: Option C (keep on GPU after first load)

**Change in `fastvideo/models/encoders/gemma.py:698`:**

```python
def forward(self, input_ids, ...):
    model = self.gemma_model
    model.to(device=get_local_torch_device())  # Move to GPU (first time: load from disk + move)
    outputs = model(...)
    # DON'T move back to CPU — keep on GPU for next request
    # model.to(device=orig_device)  ← REMOVE THIS
    ...
```

**Expected impact:**
- First request: ~55s (includes Gemma disk load + GPU move)
- Subsequent requests: **~37-40s** (Gemma already on GPU, saves ~13.5s)

### Risk Assessment for Option C

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OOM during full-res denoising | Medium | High | Monitor with `torch.cuda.memory_allocated()`. If OOM, fall back to CPU offload. |
| OOM during VAE decode | Low | Medium | VAE is small (~200MB). Should fit. |
| Gemma + upsampler conflict | Low | Low | Upsampler is only ~1GB and is used briefly between stages. |

### Implementation Steps

1. **Test Option C first** — just comment out the `.to(orig_device)` line
2. **If OOM** — try Option B (FP8 quantization)
3. **If quality issues with FP8** — stick with CPU offload (current behavior)

---

## Priority Order

1. **First: Test layerwise offload** — verify the CPU loading + layerwise hooks work
2. **Then: Keep Gemma on GPU (Option C)** — simple 1-line change, ~13s savings
3. **Then: torch.compile** — more complex, needs compatibility testing with layerwise offload
4. **Optional: FP8 Gemma** — if Option C OOMs
