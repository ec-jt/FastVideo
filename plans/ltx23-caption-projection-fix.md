# LTX-2.3 Caption Projection Fix

## Root Cause: Zero-Initialized Caption Projections

### The Problem

The warning during model loading reveals the critical issue:
```
WARNING: Zero-initializing 8 parameters not found in checkpoint:
  model.caption_projection.linear_1.weight
  model.caption_projection.linear_1.bias
  model.caption_projection.linear_2.weight
  model.caption_projection.linear_2.bias
  model.audio_caption_projection.linear_1.weight
  model.audio_caption_projection.linear_1.bias
  model.audio_caption_projection.linear_2.weight
  model.audio_caption_projection.linear_2.bias
```

These are the **text conditioning projection layers**. When zero-initialized:
- The model receives **zero text conditioning** for both video and audio
- This explains both visual artifacts AND audio buzzing
- The model generates unconditional output (no prompt understanding)

### Architecture Difference: 19B vs 22B

The LTX-2 reference code has two architectures:

| Feature | LTX-2 (19B) | LTX-2.3 (22B) |
|---------|-------------|----------------|
| `caption_proj_before_connector` | `False` | `True` |
| Caption projection location | **Transformer** | **Text Encoder** |
| `caption_projection` in transformer | Created with weights | `None` (not created) |
| `cross_attention_adaln` | `False` | `True` |

Reference code (`model_configurator.py:125-145`):
```python
def _build_caption_projections(config, is_av):
    """19B: projection in transformer. 22B: projection in text encoder."""
    if transformer_config.get("caption_proj_before_connector", False):
        return None, None  # 22B: no projection in transformer
    # 19B: create projection
    caption_projection = create_caption_projection(transformer_config)
    ...
```

Reference code (`model.py:136-137`):
```python
if caption_projection is not None:
    self.caption_projection = caption_projection  # Only set if provided
```

### The Bug in FastVideo

In `fastvideo/models/dits/ltx2.py`, `LTXModel._init_video()` **always** creates
`caption_projection` regardless of model type:

```python
def _init_video(self, ...):
    self.caption_projection = PixArtAlphaTextProjection(
        in_features=caption_channels,  # 3840
        hidden_size=self.inner_dim,    # 4096
    )
```

For LTX-2.3, this should be skipped because:
1. The text encoder already projects to the right dimension (4096 for video, 2048 for audio)
2. The checkpoint doesn't contain these weights
3. Zero-initialized weights zero out all text conditioning

### The Fix

When `cross_attention_adaln=True` (LTX-2.3), skip creating `caption_projection`
and `audio_caption_projection`. The `TransformerArgsPreprocessor` already handles
the `None` case correctly.

**Files to modify:**
1. `fastvideo/models/dits/ltx2.py` - `LTXModel._init_video()` and `_init_audio()`
   - Conditionally create `caption_projection` only when `cross_attention_adaln=False`
2. `fastvideo/models/dits/ltx2.py` - `_init_preprocessors()`
   - Pass `None` for `caption_projection` when it doesn't exist

### Impact

This fix should resolve:
- ✅ Audio buzzing (audio caption projection was zeroing out audio text conditioning)
- ✅ Visual artifacts (video caption projection was zeroing out video text conditioning)
- ✅ The model will properly condition on text prompts

This is **independent** of the two-stage pipeline fix - even the single-stage
pipeline should produce much better results with this fix.
