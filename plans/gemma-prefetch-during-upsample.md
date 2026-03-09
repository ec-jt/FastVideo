# Plan: Async Gemma Prefetch During Upsampling

## Concept

Hide the ~13s Gemma CPU→GPU transfer by overlapping it with the upsampler step in the two-stage distilled pipeline. For back-to-back API requests, Gemma is already on GPU when the next request's text encoding starts.

## Timeline: Current vs Proposed

### Current (Gemma offloaded after each text encoding)

```
Request 1:
  [Text Enc: Gemma load+fwd 13.5s] [Stage1 16s] [Upsample 0.9s] [Stage2 11s] [Decode 5s]
                                                                                          |
Request 2:                                                                                 |
  [Text Enc: Gemma load+fwd 13.5s] [Stage1 16s] [Upsample 0.9s] [Stage2 11s] [Decode 5s]

Total per request: ~46.4s
```

### Proposed (Gemma prefetched during upsample, stays on GPU for next request)

```
Request 1:
  [Text Enc: Gemma load+fwd 13.5s] [Stage1 16s] [Upsample 0.9s + Gemma prefetch] [Stage2 11s] [Decode 5s]
                                                   ↑ Gemma moves to GPU async       Gemma on GPU ↓
Request 2:                                                                                        |
  [Text Enc: Gemma fwd only 3s] [Stage1 16s] [Upsample 0.9s + Gemma prefetch] [Stage2 11s] [Decode 5s]

Total per request (after first): ~36s  (saves ~10s)
```

## Architecture

### Components Involved

1. **`gemma.py` forward()** — Currently moves Gemma GPU→CPU after forward. Needs to conditionally keep on GPU.
2. **`ltx2_distilled_denoising.py` _run_two_stage()** — The upsample step between Stage 1 and Stage 2. This is where we trigger the async Gemma prefetch.
3. **`LTX2TextEncodingStage`** — Needs to know if Gemma is already on GPU to skip the load.
4. **New: `GEMMA_PREFETCH_MODE` env var** — Controls the behavior:
   - `off` (default): Current behavior, Gemma offloaded after each forward
   - `keep_on_gpu`: Keep Gemma on GPU permanently after first load
   - `prefetch_during_upsample`: Offload after text encoding, prefetch back during upsample

### Memory Budget with Gemma on GPU During Stage 2

| Component | VRAM | When |
|-----------|------|------|
| Gemma | 18GB | Permanent after prefetch |
| Transformer block current | 0.8GB | During each step |
| Transformer block prefetch | 0.8GB | Overlapped |
| Non-block params | 0.2GB | Permanent |
| Activations full-res | 8-10GB | During Stage 2 |
| CUDA context | 1GB | Permanent |
| **Total peak** | **~30.8GB** | During Stage 2 |
| **RTX 5090 VRAM** | **32GB** | |
| **Headroom** | **~1.2GB** | Tight but should work |

### Detailed Implementation

#### Change 1: `gemma.py` — Conditional offload

```python
def forward(self, input_ids, ...):
    model = self.gemma_model
    model.to(device=get_local_torch_device())
    outputs = model(...)
    
    # Controlled by GEMMA_PREFETCH_MODE env var
    mode = os.getenv("GEMMA_PREFETCH_MODE", "off")
    if mode == "off":
        model.to(device="cpu")  # Current behavior
    elif mode == "keep_on_gpu":
        pass  # Stay on GPU permanently
    elif mode == "prefetch_during_upsample":
        model.to(device="cpu")  # Offload now, will be prefetched during upsample
    
    # ... rest of forward
```

#### Change 2: `ltx2_distilled_denoising.py` — Async prefetch during upsample

In `_run_two_stage()`, after the upsample step, start async Gemma prefetch:

```python
# ── Spatial upsample ─────────────────────────
self.spatial_upsampler.to(device)
upsampled_latents = upsample_video_latent(...)
self.spatial_upsampler.to("cpu")
torch.cuda.empty_cache()

# ── Async Gemma prefetch for next request ────
# Start moving Gemma to GPU in background while
# we prepare Stage 2.  By the time the next
# request's text encoding runs, Gemma is already
# on GPU.
if os.getenv("GEMMA_PREFETCH_MODE") == "prefetch_during_upsample":
    self._async_prefetch_gemma()
```

The `_async_prefetch_gemma()` method:

```python
def _async_prefetch_gemma(self):
    """Move Gemma to GPU asynchronously using a background thread.
    
    This overlaps the CPU→GPU transfer with Stage 2 preparation,
    hiding the ~5s transfer time.
    """
    import threading
    
    def _prefetch():
        # Access the text encoder's Gemma model
        # (it's cached in _gemma_model after first use)
        text_encoder = self._get_text_encoder()  # Need reference
        if text_encoder and hasattr(text_encoder, '_gemma_model'):
            gemma = text_encoder._gemma_model
            if gemma is not None and gemma.device.type == "cpu":
                gemma.to(device=get_local_torch_device())
    
    thread = threading.Thread(target=_prefetch, daemon=True)
    thread.start()
    # Don't join — let it run in background during Stage 2
```

#### Challenge: Getting Text Encoder Reference in Denoising Stage

The denoising stage doesn't have a reference to the text encoder. We need to pass it through or store it on the pipeline.

**Option A**: Pass text encoder reference when creating the denoising stage in `ltx2_distilled_pipeline.py`:

```python
self.add_stage(
    stage_name="denoising_stage",
    stage=LTX2DistilledDenoisingStage(
        transformer=self.get_module("transformer"),
        spatial_upsampler=spatial_upsampler,
        per_channel_statistics=per_channel_stats,
        vae=vae,
        text_encoder=self.get_module("text_encoder"),  # NEW
    ),
)
```

**Option B**: Store the Gemma model reference on the ForwardBatch:

```python
# In text encoding stage, after forward:
batch.extra["_gemma_model_ref"] = text_encoder._gemma_model

# In denoising stage, during upsample:
gemma = batch.extra.get("_gemma_model_ref")
if gemma and gemma.device.type == "cpu":
    gemma.to(device)
```

**Option B is simpler** — no constructor changes needed.

### Files to Change

| File | Change |
|------|--------|
| `fastvideo/models/encoders/gemma.py` | Conditional offload based on `GEMMA_PREFETCH_MODE` |
| `fastvideo/pipelines/stages/ltx2_distilled_denoising.py` | Async Gemma prefetch during upsample step |
| `fastvideo/pipelines/stages/text_encoding.py` or `ltx2_text_encoding.py` | Store Gemma ref in batch.extra |
| Service `.env` / `docker-compose.yml` | Add `GEMMA_PREFETCH_MODE=prefetch_during_upsample` |

### Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OOM during Stage 2 with Gemma on GPU | Medium | Monitor VRAM. If OOM, fall back to `off` mode. |
| Thread safety of `.to()` call | Low | PyTorch `.to()` is thread-safe for model movement. |
| Gemma not fully loaded by Stage 2 start | Low | The 18GB transfer takes ~5s. Stage 2 prep takes ~1s. Gemma should be ready. |
| Race condition if next request arrives during prefetch | Low | The `.to()` call is idempotent — if Gemma is already on GPU, it's a no-op. |

### Testing

```bash
# Test with prefetch mode:
GEMMA_PREFETCH_MODE=prefetch_during_upsample \
LTX2_TWO_STAGE=1 python tests/helix/test_ltx2_audio_quality.py \
    --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
    --num-gpus 1 --tp-size 1 --frames 241 \
    --custom-prompt "Prompt A" "Prompt B" \
    --enhance-prompt

# Expected: First video ~55s, second video ~36s (vs ~50s without prefetch)
```

### Priority

This is a **Phase 3 optimization** — implement after:
1. ✅ Layerwise offload with CPU loading (done)
2. Test layerwise offload works
3. Then implement Gemma prefetch
