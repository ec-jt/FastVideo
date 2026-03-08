# LTX-2.3 Audio Content Quality — Investigation Plan

## Problem

The vocoder fix (BigVGAN v2 + BWE) resolved the audio buzzing, but the
generated audio content doesn't match the text prompt.  The audio sounds
like speech in a "weird language" instead of matching the prompt (e.g.
jazz music).  The upstream LTX-2.3 API produces correct audio for the
same prompts.

## What's Working

- Vocoder: ✅ No more buzzing, clean 48 kHz audio via BWE
- Weight loading: ✅ All 3 models load with 0 missing/unexpected keys
- Guidance params: ✅ `cfg_audio=7.0 mod_audio=3.0 stg_audio=1.0 rescale=0.7`
  match upstream exactly
- Guidance formula: ✅ Matches upstream `MultiModalGuider.calculate()`
- Audio-video cross-attention: ✅ Lip-syncing works (A2V/V2A cross-attn functional)
- Text encoder: ✅ Gemma → feature_extractor → audio_connector pipeline exists

## Root Cause: Wrong Feature Extractor Normalization for LTX-2.3

### The Core Bug

FastVideo uses **V1 normalization** (`_norm_and_concat_padded_batch`) for
ALL models, but LTX-2.3 (22B) requires **V2 normalization**
(`norm_and_concat_per_token_rms` + `_rescale_norm`).

This means **both video AND audio text conditioning are wrong** for
LTX-2.3.  The text embeddings fed to the transformer are incorrectly
normalized, causing the model to receive garbled conditioning signals.

### Upstream V1 vs V2 Feature Extractors

The upstream code in
[`feature_extractor.py`](../LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py)
has two distinct classes:

**V1 (`FeatureExtractorV1`)** — used by LTX-2 (19B):
1. Stack hidden states: `[B, T, D, L]`
2. Per-batch normalization: `_norm_and_concat_padded_batch()` — computes
   masked mean/range across `(T, D)` dims, normalizes to `8 * (x - mean) / range`
3. Flatten to `[B, T, D*L]`
4. Single `aggregate_embed` linear (no bias): `D*L → 3840`
5. Returns `(features, features)` for AV or `(features, None)` for video-only

**V2 (`FeatureExtractorV2`)** — used by LTX-2.3 (22B):
1. Stack hidden states: `[B, T, D, L]`
2. **Per-token RMS normalization**: `norm_and_concat_per_token_rms()` —
   computes RMS per token across hidden dim, normalizes each token independently
3. Flatten to `[B, T, D*L]`
4. **Rescale normalization**: `x * sqrt(target_dim / source_dim)` before
   each linear projection
5. Dual `video_aggregate_embed` / `audio_aggregate_embed` linears (**with bias**):
   - Video: `D*L → video_inner_dim` (4096 for 32 heads × 128 dim)
   - Audio: `D*L → audio_inner_dim` (2048 for 32 heads × 64 dim)

### What FastVideo Does Wrong

In [`gemma.py:_run_feature_extractor()`](fastvideo/models/encoders/gemma.py:471):

```python
# WRONG for LTX-2.3: uses V1 normalization for all models
normed_text_features = _norm_and_concat_padded_batch(
    encoded_text_features, sequence_lengths, padding_side=padding_side
)
normed = normed_text_features.to(encoded_text_features_dtype)
video_features, audio_features = self.feature_extractor_linear(normed)
```

This should use `norm_and_concat_per_token_rms` + `_rescale_norm` for
LTX-2.3, but FastVideo has **no implementation** of either function.

### Secondary Bug: Audio Connector Attention Mask

In [`gemma.py:_run_connectors()`](fastvideo/models/encoders/gemma.py:504),
when `audio_encoded_input is not None` (LTX-2.3 path), the audio
connector mask is created from the **already-processed** binary
`attention_mask` (after the video connector converted it). The upstream
code passes the **original** additive mask to both connectors.

```python
# BUG: attention_mask here is already binary (from video connector output)
# but _convert_to_additive_mask expects the original binary input mask
audio_connector_mask = self._convert_to_additive_mask(
    attention_mask.squeeze(-1), audio_input.dtype
) if audio_encoded_input is not None else connector_attention_mask
```

The upstream [`EmbeddingsProcessor.create_embeddings()`](../LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py:49)
passes the same `additive_attention_mask` to both video and audio
connectors.

## Fix Plan

### Fix 1: Implement V2 Feature Extractor Normalization

**File**: [`fastvideo/models/encoders/gemma.py`](fastvideo/models/encoders/gemma.py)

Add two new functions matching upstream:

```python
def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    B, T, D, L = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask_3d, normed, torch.zeros_like(normed))

def _rescale_norm(
    x: torch.Tensor, target_dim: int, source_dim: int
) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)
```

### Fix 2: Add Config Flag and Conditional Logic

**File**: [`fastvideo/configs/models/encoders/gemma.py`](fastvideo/configs/models/encoders/gemma.py)

Add a `feature_extractor_version` field (default `"v1"`, set to `"v2"`
for LTX-2.3 configs).

**File**: [`fastvideo/models/encoders/gemma.py`](fastvideo/models/encoders/gemma.py)

Update `GemmaFeaturesExtractorProjLinear` to store `embedding_dim` for
V2 rescale, and update `_run_feature_extractor()` to branch on the
config flag:

```python
def _run_feature_extractor(self, hidden_states, attention_mask, padding_side):
    encoded_text_features = torch.stack(hidden_states, dim=-1)
    dtype = encoded_text_features.dtype

    if self.feature_extractor_version == "v2":
        normed = _norm_and_concat_per_token_rms(
            encoded_text_features, attention_mask
        ).to(dtype)
        # V2: rescale before each projection
        video_features, audio_features = self.feature_extractor_linear(
            normed, rescale=True
        )
    else:
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side
        ).to(dtype)
        video_features, audio_features = self.feature_extractor_linear(normed)

    return video_features, audio_features
```

### Fix 3: Fix Audio Connector Attention Mask

**File**: [`fastvideo/models/encoders/gemma.py`](fastvideo/models/encoders/gemma.py)

In `_run_connectors()`, save the original `connector_attention_mask`
before the video connector modifies it, and pass it to the audio
connector:

```python
def _run_connectors(self, encoded_input, attention_mask, audio_encoded_input=None):
    connector_attention_mask = self._convert_to_additive_mask(
        attention_mask, encoded_input.dtype
    )
    # Save original mask for audio connector
    original_additive_mask = connector_attention_mask

    encoded, encoded_connector_attention_mask = self.embeddings_connector(
        encoded_input, connector_attention_mask
    )
    # ... video mask processing ...

    audio_input = audio_encoded_input if audio_encoded_input is not None else encoded_input
    # FIX: always use original additive mask for audio connector
    encoded_for_audio, _ = self.audio_embeddings_connector(
        audio_input, original_additive_mask
    )
    return encoded, encoded_for_audio, attention_mask.squeeze(-1)
```

### Fix 4: Update Config Loading for LTX-2.3

Ensure the pipeline config loader sets `feature_extractor_version="v2"`
when loading LTX-2.3 models (detected by `cross_attention_adaln=True`
or `caption_proj_before_connector=True` in the config).

## Impact Assessment

- **Audio content**: The V2 normalization fix will produce correctly
  conditioned audio embeddings, making audio match the text prompt
- **Video quality**: Video conditioning is ALSO wrong with V1 norm on
  LTX-2.3 — this fix will improve video quality too
- **LTX-2 (19B) compatibility**: V1 path is preserved, no regression

## Files to Modify

| File | Change |
|------|--------|
| [`fastvideo/models/encoders/gemma.py`](fastvideo/models/encoders/gemma.py) | Add V2 norm functions, update `_run_feature_extractor`, fix `_run_connectors` mask |
| [`fastvideo/configs/models/encoders/gemma.py`](fastvideo/configs/models/encoders/gemma.py) | Add `feature_extractor_version` config field |
| Pipeline config loader | Auto-detect V2 from model config |

## Verification

After the fix:
1. Run LTX-2.3 with jazz prompt — audio should be music, not speech
2. Compare debug log sums between FastVideo and upstream at each stage
3. Run LTX-2 (19B) to verify no regression
