# Helix Parallelism Test Setup Plan

## Overview

This document outlines the test setup for validating Helix parallelism before and after implementation.

## CRITICAL: VSA Backend Requirement

**The local FastVideo models require `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`**

### Why?

FastVideo models have two attention architectures:
- `WanTransformerBlock` - Standard attention (no `to_gate_compress` layers)
- `WanTransformerBlock_VSA` - Video Sparse Attention (has `to_gate_compress` layers)

The local FastVideo checkpoints at `/mnt/nvme0/models/FastVideo/` were trained with VSA, so they contain `to_gate_compress` weights. The model architecture selection happens in [`fastvideo/models/dits/wanvideo.py`](../fastvideo/models/dits/wanvideo.py:593):

```python
transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock
```

### Error Without VSA Backend

Without setting the environment variable, model loading fails with:
```
ValueError: Parameter blocks.0.to_gate_compress.bias not found in custom model state dict
```

### Solution

The test files automatically set this environment variable:
```python
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
```

For manual testing, set it before running:
```bash
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
```

## Test Models

| Model | HuggingFace ID | Params | Current Status |
|-------|----------------|--------|----------------|
| **Wan 1.3B** | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | 1.3B | ✅ Works with SP |
| **Wan 5B** | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 5B | ⚠️ Tight fit with SP |
| **Wan 14B** | `FastVideo/FastWan2.1-T2V-14B-Diffusers` | 14B | ❌ OOM with SP |

### Alternative Test Models (if needed)

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **Wan 14B VSA** | `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` | 14B | VSA sparse attention |
| **Causal Wan 14B** | `FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers` | 14B | I2V variant |

## Test Matrix

### TP Configurations to Test

For each model, we test multiple TP sizes to validate weight sharding:

| Model | TP=1 | TP=2 | TP=4 | TP=8 |
|-------|------|------|------|------|
| Wan 1.3B | ✅ Baseline | ✅ Test | ✅ Test | ⚠️ May be slow |
| Wan 5B | ✅ Baseline | ✅ Test | ✅ Test | ⚠️ May be slow |
| Wan 14B | ❌ OOM | ✅ Key test | ✅ Test | ⚠️ May be slow |

### Sequence Length Configurations

Different video lengths to test scaling behavior:

| Config | Frames | Resolution | Sequence Length | Memory Impact |
|--------|--------|------------|-----------------|---------------|
| **Short** | 17 | 480×720 | ~2,700 tokens | Low |
| **Medium** | 45 | 480×832 | ~7,500 tokens | Medium |
| **Long** | 81 | 720×1280 | ~27,000 tokens | High |
| **Very Long** | 121 | 720×1280 | ~40,000 tokens | Very High |

### Full Test Matrix

```
For each model (1.3B, 5B, 14B):
  For each TP size (1, 2, 4):
    For each sequence length (short, medium, long):
      - Run generation
      - Record memory usage
      - Record generation time
      - Save output for SSIM comparison
```

Total test configurations: 3 models × 3 TP sizes × 3 seq lengths = **27 tests**

## Test Directory Structure

```
tests/helix/
├── __init__.py
├── README.md
├── conftest.py                    # Shared fixtures
├── test_wan_baseline.py           # Baseline SP tests
├── test_wan_helix.py              # Helix TP+CP tests (after implementation)
├── test_helix_comparison.py       # SSIM comparison tests
└── reference_videos/              # Reference outputs for SSIM
    ├── wan_1.3b/
    ├── wan_5b/
    └── wan_14b/
```

## Test Configurations

### Wan 1.3B Configuration

```python
WAN_1_3B_BASELINE = {
    "model_path": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_1_3B_HELIX = {
    "model_path": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 1,  # No weight sharding needed for small model
    "cp_size": 8,
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

# TP variations for Wan 1.3B
WAN_1_3B_TP_CONFIGS = [
    {"tp_size": 1, "cp_size": 8},  # Baseline equivalent
    {"tp_size": 2, "cp_size": 4},  # 2-way weight sharding
    {"tp_size": 4, "cp_size": 2},  # 4-way weight sharding
]

# Sequence length variations for Wan 1.3B
WAN_1_3B_SEQ_CONFIGS = [
    {"num_frames": 17, "height": 480, "width": 720, "name": "short"},
    {"num_frames": 45, "height": 480, "width": 832, "name": "medium"},
    {"num_frames": 81, "height": 720, "width": 1280, "name": "long"},
]
```

### Wan 5B Configuration

```python
WAN_5B_BASELINE = {
    "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_5B_HELIX = {
    "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 2,  # Split weights across 2 GPUs
    "cp_size": 4,  # 4-way context parallelism
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

# TP variations for Wan 5B
WAN_5B_TP_CONFIGS = [
    {"tp_size": 1, "cp_size": 8},  # Baseline equivalent (may be tight)
    {"tp_size": 2, "cp_size": 4},  # Recommended
    {"tp_size": 4, "cp_size": 2},  # More weight sharding
]

# Sequence length variations for Wan 5B
WAN_5B_SEQ_CONFIGS = [
    {"num_frames": 17, "height": 480, "width": 720, "name": "short"},
    {"num_frames": 45, "height": 480, "width": 720, "name": "medium"},
    {"num_frames": 81, "height": 720, "width": 1280, "name": "long"},
]
```

### Wan 14B Configuration

```python
WAN_14B_BASELINE = {
    "model_path": "FastVideo/FastWan2.1-T2V-14B-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
    # Expected: OOM - this is the test case that proves Helix is needed
}

WAN_14B_HELIX = {
    "model_path": "FastVideo/FastWan2.1-T2V-14B-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 2,  # Split 28GB weights -> 14GB per GPU
    "cp_size": 4,  # 4-way context parallelism
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

# TP variations for Wan 14B - CRITICAL for fitting on 32GB GPUs
WAN_14B_TP_CONFIGS = [
    {"tp_size": 1, "cp_size": 8},  # Expected OOM (28GB weights)
    {"tp_size": 2, "cp_size": 4},  # 14GB weights - should fit!
    {"tp_size": 4, "cp_size": 2},  # 7GB weights - comfortable
]

# Sequence length variations for Wan 14B
WAN_14B_SEQ_CONFIGS = [
    {"num_frames": 17, "height": 480, "width": 720, "name": "short"},
    {"num_frames": 45, "height": 480, "width": 720, "name": "medium"},
    {"num_frames": 81, "height": 720, "width": 1280, "name": "long"},
]
```

### Complete Test Matrix

```python
# All test configurations
TEST_MATRIX = {
    "wan_1.3b": {
        "model_path": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "tp_configs": WAN_1_3B_TP_CONFIGS,
        "seq_configs": WAN_1_3B_SEQ_CONFIGS,
        "expected_baseline": "pass",
    },
    "wan_5b": {
        "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "tp_configs": WAN_5B_TP_CONFIGS,
        "seq_configs": WAN_5B_SEQ_CONFIGS,
        "expected_baseline": "tight",
    },
    "wan_14b": {
        "model_path": "FastVideo/FastWan2.1-T2V-14B-Diffusers",
        "tp_configs": WAN_14B_TP_CONFIGS,
        "seq_configs": WAN_14B_SEQ_CONFIGS,
        "expected_baseline": "oom",
    },
}
```

## Test Files to Create

### 1. `tests/helix/__init__.py`

```python
# SPDX-License-Identifier: Apache-2.0
"""Helix parallelism test suite."""
```

### 2. `tests/helix/conftest.py`

```python
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Helix tests."""

import os
import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.tests.utils import compute_video_ssim_torchvision

# Test prompt used across all tests
TEST_PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "its eyes wide with interest. The playful yet serene atmosphere is "
    "complemented by soft natural light filtering through the petals. "
    "Mid-shot, warm and cheerful tones."
)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality"
)

@pytest.fixture
def test_prompt():
    return TEST_PROMPT

@pytest.fixture
def negative_prompt():
    return NEGATIVE_PROMPT

@pytest.fixture
def reference_dir():
    return os.path.join(os.path.dirname(__file__), "reference_videos")

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            i: torch.cuda.memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        }
    return {}

def get_peak_gpu_memory():
    """Get peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            i: torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        }
    return {}
```

### 3. `tests/helix/test_wan_baseline.py`

```python
# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for Wan models with current SP implementation."""

import json
import os
import time
import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.tests.utils import compute_video_ssim_torchvision

# Model configurations
WAN_1_3B_CONFIG = {
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_5B_CONFIG = {
    "model_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_14B_CONFIG = {
    "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "num_gpus": 8,
    "sp_size": 8,
    "tp_size": 1,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}


class TestWanBaseline:
    """Baseline tests for Wan models with SP."""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_prompt, negative_prompt, reference_dir):
        self.prompt = test_prompt
        self.neg_prompt = negative_prompt
        self.reference_dir = reference_dir
        self.output_dir = "tests/helix/baseline_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _run_generation(self, config, model_name):
        """Run video generation and record metrics."""
        torch.cuda.reset_peak_memory_stats()
        
        generator = VideoGenerator.from_pretrained(
            config["model_path"],
            num_gpus=config["num_gpus"],
            sp_size=config["sp_size"],
            tp_size=config["tp_size"],
        )
        
        start_time = time.time()
        video = generator.generate_video(
            self.prompt,
            negative_prompt=self.neg_prompt,
            height=config["height"],
            width=config["width"],
            num_frames=config["num_frames"],
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            seed=config["seed"],
            output_path=self.output_dir,
            save_video=True,
        )
        generation_time = time.time() - start_time
        
        # Record metrics
        peak_memory = {
            i: torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        }
        
        metrics = {
            "model": model_name,
            "config": config,
            "generation_time_seconds": generation_time,
            "peak_memory_gb": peak_memory,
            "max_peak_memory_gb": max(peak_memory.values()),
        }
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        generator.shutdown()
        return video, metrics
    
    def test_wan_1_3b_baseline(self):
        """Test Wan 1.3B with SP=8."""
        video, metrics = self._run_generation(WAN_1_3B_CONFIG, "wan_1.3b")
        
        assert video is not None
        assert metrics["max_peak_memory_gb"] < 32, "Should fit on 32GB GPU"
        print(f"Wan 1.3B: {metrics['generation_time_seconds']:.2f}s, "
              f"peak memory: {metrics['max_peak_memory_gb']:.2f}GB")
    
    def test_wan_5b_baseline(self):
        """Test Wan 5B with SP=8."""
        video, metrics = self._run_generation(WAN_5B_CONFIG, "wan_5b")
        
        assert video is not None
        # 5B might be tight on 32GB
        print(f"Wan 5B: {metrics['generation_time_seconds']:.2f}s, "
              f"peak memory: {metrics['max_peak_memory_gb']:.2f}GB")
    
    @pytest.mark.xfail(reason="Expected OOM - 14B doesn't fit with SP")
    def test_wan_14b_baseline(self):
        """Test Wan 14B with SP=8 - expected to OOM."""
        video, metrics = self._run_generation(WAN_14B_CONFIG, "wan_14b")
        
        # This should fail with OOM
        assert video is not None
```

### 4. `tests/helix/test_wan_helix.py`

```python
# SPDX-License-Identifier: Apache-2.0
"""Helix parallelism tests for Wan models (after implementation)."""

import json
import os
import time
import pytest
import torch

from fastvideo import VideoGenerator

# Helix configurations
WAN_1_3B_HELIX = {
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 1,
    "cp_size": 8,
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_5B_HELIX = {
    "model_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 2,
    "cp_size": 4,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}

WAN_14B_HELIX = {
    "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "num_gpus": 8,
    "use_helix": True,
    "tp_size": 2,
    "cp_size": 4,
    "height": 480,
    "width": 720,
    "num_frames": 45,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "seed": 1024,
}


class TestWanHelix:
    """Helix parallelism tests for Wan models."""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_prompt, negative_prompt, reference_dir):
        self.prompt = test_prompt
        self.neg_prompt = negative_prompt
        self.reference_dir = reference_dir
        self.output_dir = "tests/helix/helix_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _run_generation(self, config, model_name):
        """Run video generation with Helix and record metrics."""
        torch.cuda.reset_peak_memory_stats()
        
        generator = VideoGenerator.from_pretrained(
            config["model_path"],
            num_gpus=config["num_gpus"],
            use_helix=config["use_helix"],
            tp_size=config["tp_size"],
            cp_size=config["cp_size"],
        )
        
        start_time = time.time()
        video = generator.generate_video(
            self.prompt,
            negative_prompt=self.neg_prompt,
            height=config["height"],
            width=config["width"],
            num_frames=config["num_frames"],
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            seed=config["seed"],
            output_path=self.output_dir,
            save_video=True,
        )
        generation_time = time.time() - start_time
        
        peak_memory = {
            i: torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        }
        
        metrics = {
            "model": model_name,
            "config": config,
            "generation_time_seconds": generation_time,
            "peak_memory_gb": peak_memory,
            "max_peak_memory_gb": max(peak_memory.values()),
        }
        
        metrics_path = os.path.join(self.output_dir, f"{model_name}_helix_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        generator.shutdown()
        return video, metrics
    
    @pytest.mark.skip(reason="Helix not implemented yet")
    def test_wan_1_3b_helix(self):
        """Test Wan 1.3B with Helix (tp=1, cp=8)."""
        video, metrics = self._run_generation(WAN_1_3B_HELIX, "wan_1.3b")
        
        assert video is not None
        assert metrics["max_peak_memory_gb"] < 32
        print(f"Wan 1.3B Helix: {metrics['generation_time_seconds']:.2f}s, "
              f"peak memory: {metrics['max_peak_memory_gb']:.2f}GB")
    
    @pytest.mark.skip(reason="Helix not implemented yet")
    def test_wan_5b_helix(self):
        """Test Wan 5B with Helix (tp=2, cp=4)."""
        video, metrics = self._run_generation(WAN_5B_HELIX, "wan_5b")
        
        assert video is not None
        # With TP=2, memory should be ~half of baseline
        assert metrics["max_peak_memory_gb"] < 20
        print(f"Wan 5B Helix: {metrics['generation_time_seconds']:.2f}s, "
              f"peak memory: {metrics['max_peak_memory_gb']:.2f}GB")
    
    @pytest.mark.skip(reason="Helix not implemented yet")
    def test_wan_14b_helix(self):
        """Test Wan 14B with Helix (tp=2, cp=4) - THE KEY TEST."""
        video, metrics = self._run_generation(WAN_14B_HELIX, "wan_14b")
        
        assert video is not None
        # With TP=2, 14B weights = 14GB per GPU, should fit!
        assert metrics["max_peak_memory_gb"] < 32
        print(f"Wan 14B Helix: {metrics['generation_time_seconds']:.2f}s, "
              f"peak memory: {metrics['max_peak_memory_gb']:.2f}GB")
```

### 5. `tests/helix/test_helix_comparison.py`

```python
# SPDX-License-Identifier: Apache-2.0
"""SSIM comparison tests between baseline SP and Helix outputs."""

import json
import os
import pytest

from fastvideo.tests.utils import compute_video_ssim_torchvision


class TestHelixComparison:
    """Compare Helix outputs to baseline for quality validation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.baseline_dir = "tests/helix/baseline_outputs"
        self.helix_dir = "tests/helix/helix_outputs"
        self.min_ssim = 0.95  # Minimum acceptable SSIM
    
    def _compare_videos(self, model_name):
        """Compare baseline and Helix videos for a model."""
        baseline_video = os.path.join(self.baseline_dir, f"{model_name}.mp4")
        helix_video = os.path.join(self.helix_dir, f"{model_name}.mp4")
        
        if not os.path.exists(baseline_video):
            pytest.skip(f"Baseline video not found: {baseline_video}")
        if not os.path.exists(helix_video):
            pytest.skip(f"Helix video not found: {helix_video}")
        
        ssim = compute_video_ssim_torchvision(baseline_video, helix_video)
        
        return ssim
    
    @pytest.mark.skip(reason="Run after both baseline and Helix tests complete")
    def test_wan_1_3b_ssim(self):
        """Compare Wan 1.3B baseline vs Helix."""
        ssim = self._compare_videos("wan_1.3b")
        assert ssim >= self.min_ssim, f"SSIM {ssim} < {self.min_ssim}"
        print(f"Wan 1.3B SSIM: {ssim:.4f}")
    
    @pytest.mark.skip(reason="Run after both baseline and Helix tests complete")
    def test_wan_5b_ssim(self):
        """Compare Wan 5B baseline vs Helix."""
        ssim = self._compare_videos("wan_5b")
        assert ssim >= self.min_ssim, f"SSIM {ssim} < {self.min_ssim}"
        print(f"Wan 5B SSIM: {ssim:.4f}")
```

## Running the Tests

### Step 1: Run Baseline Tests (Before Helix)

```bash
# Run baseline tests to establish current behavior
pytest tests/helix/test_wan_baseline.py -v --tb=short

# Expected results:
# - test_wan_1_3b_baseline: PASS
# - test_wan_5b_baseline: PASS (or tight on memory)
# - test_wan_14b_baseline: XFAIL (expected OOM)
```

### Step 2: Implement Helix

Follow the implementation plan in `plans/helix-parallelism-implementation.md`.

### Step 3: Run Helix Tests (After Implementation)

```bash
# Remove skip markers and run Helix tests
pytest tests/helix/test_wan_helix.py -v --tb=short

# Expected results:
# - test_wan_1_3b_helix: PASS
# - test_wan_5b_helix: PASS (with lower memory)
# - test_wan_14b_helix: PASS (THE KEY SUCCESS!)
```

### Step 4: Run Comparison Tests

```bash
# Compare outputs for quality validation
pytest tests/helix/test_helix_comparison.py -v

# Expected results:
# - All SSIM scores > 0.95
```

## Success Criteria

| Test | Baseline (SP) | Helix (TP+CP) | Success |
|------|---------------|---------------|---------|
| Wan 1.3B runs | ✅ | ✅ | Both work |
| Wan 5B runs | ✅ | ✅ | Both work |
| Wan 14B runs | ❌ OOM | ✅ | **Helix enables it!** |
| Wan 1.3B SSIM | N/A | >0.95 | Quality preserved |
| Wan 5B SSIM | N/A | >0.95 | Quality preserved |
| Wan 5B memory | ~12GB | ~7GB | Memory reduced |
| Wan 14B memory | N/A | <20GB | Fits on 32GB |

## Next Steps

1. Switch to Code mode to create the actual test files
2. Run baseline tests to establish current behavior
3. Implement Helix parallelism
4. Run Helix tests to validate implementation
5. Run comparison tests to ensure quality
