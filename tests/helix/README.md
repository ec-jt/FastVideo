# Helix Parallelism Test Suite

This directory contains tests for validating Helix parallelism (TP+CP) implementation in FastVideo.

## Overview

Helix parallelism combines **Tensor Parallelism (TP)** with **Context Parallelism (CP)** to enable running large DiT models on GPUs with limited memory. The key insight is:

- **Current SP (Sequence Parallelism)**: Splits activations but NOT weights. Each GPU needs full model weights.
- **Helix (TP+CP)**: Splits BOTH weights (via TP) AND activations (via CP). Enables larger models on smaller GPUs.

### Why Helix Matters

| Model | Weights (BF16) | SP=8 Memory/GPU | Helix TP=2,CP=4 Memory/GPU |
|-------|----------------|-----------------|----------------------------|
| Wan 1.3B | 2.6 GB | 2.6 GB + activations | 1.3 GB + activations |
| Wan 5B | 10 GB | 10 GB + activations | 5 GB + activations |
| Wan 14B | 28 GB | **28 GB + activations** ❌ | **14 GB + activations** ✅ |

**Wan 14B cannot run on 32GB GPUs with SP alone, but CAN run with Helix TP=2.**

## Test Models

All models are located at `/mnt/nvme0/models/FastVideo/`:

### Wan 1.3B (FastWan2.1-T2V-1.3B-Diffusers)
- **Transformer**: 12 heads × 128 dim = 1536 hidden, 30 layers
- **Valid TP sizes**: 1, 2, 3, 4, 6 (must divide 12 heads)
- **Memory**: ~2.6 GB weights + ~9.4 GB text encoder

### Wan 5B (FastWan2.2-TI2V-5B-FullAttn-Diffusers)
- **Transformer**: 24 heads × 128 dim = 3072 hidden, 30 layers
- **Valid TP sizes**: 1, 2, 3, 4, 6, 8 (must divide 24 heads)
- **Memory**: ~10 GB weights + ~9.4 GB text encoder
- **Note**: TI2V model with 48 input channels

### Wan 14B (FastWan2.1-T2V-14B-Diffusers)
- **Transformer**: 40 heads × 128 dim = 5120 hidden, 40 layers
- **Valid TP sizes**: 1, 2, 4, 5, 8 (must divide 40 heads)
- **Memory**: ~28 GB weights + ~9.4 GB text encoder
- **Critical**: Requires Helix to fit on 32GB GPUs

## Test Structure

```
tests/helix/
├── README.md                 # This file
├── __init__.py              # Module docstring
├── conftest.py              # Fixtures and model configurations
├── test_wan_baseline.py     # Baseline SP tests (run BEFORE Helix)
├── test_wan_helix.py        # Helix tests (run AFTER implementation)
├── baseline_outputs/        # Generated baseline videos for SSIM
└── helix_outputs/           # Generated Helix videos for comparison
```

## Running Tests

### Prerequisites

1. Models downloaded to `/mnt/nvme0/models/FastVideo/`
2. CUDA available with sufficient GPUs
3. FastVideo installed: `pip install -e .`

### IMPORTANT: VSA Backend Requirement

The local FastVideo models use **Video Sparse Attention (VSA)** architecture with `to_gate_compress` layers. The tests automatically set the required environment variable:

```bash
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
```

**Why is this needed?**

FastVideo models have two attention architectures:
- `WanTransformerBlock` - Standard attention (no `to_gate_compress`)
- `WanTransformerBlock_VSA` - Video Sparse Attention (has `to_gate_compress`)

The local FastVideo checkpoints at `/mnt/nvme0/models/FastVideo/` were trained with VSA, so they contain `to_gate_compress` weights. Without setting `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`, the model loader will fail with:

```
ValueError: Parameter blocks.0.to_gate_compress.bias not found in custom model state dict
```

The test files (`conftest.py`, `test_wan_baseline.py`, `test_wan_helix.py`) automatically set this environment variable, so you don't need to set it manually when running tests.

### Quick Start

```bash
# Run all unit tests (no GPU required)
pytest tests/helix/ -v -k "not gpu"

# Run baseline tests (requires GPUs)
pytest tests/helix/test_wan_baseline.py -v --tb=short

# Run just 1.3B model tests (faster)
pytest tests/helix/test_wan_baseline.py -v -k "1_3B"

# Run memory profiling tests
pytest tests/helix/test_wan_baseline.py -v -k "memory"

# Run benchmarks
pytest tests/helix/test_wan_baseline.py -v -k "benchmark" --benchmark
```

### Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.gpu` - Requires CUDA GPUs
- `@pytest.mark.slow` - Long-running tests (video generation)
- `@pytest.mark.benchmark` - Performance benchmarks

### Running Specific Test Categories

```bash
# Skip GPU tests
pytest tests/helix/ -v -m "not gpu"

# Only fast tests
pytest tests/helix/ -v -m "not slow"

# Only benchmarks
pytest tests/helix/ -v -m "benchmark"
```

## Test Workflow

### Phase 1: Baseline Tests (Before Helix)

Run these tests to establish baseline behavior:

```bash
# 1. Verify configurations are correct
pytest tests/helix/test_wan_baseline.py::TestModelConfigurations -v

# 2. Test model loading with SP
pytest tests/helix/test_wan_baseline.py::TestModelLoading -v

# 3. Generate baseline videos
pytest tests/helix/test_wan_baseline.py::TestBaselineGeneration -v

# 4. Generate SSIM references
pytest tests/helix/test_wan_baseline.py::TestSSIMReferenceGeneration -v

# 5. Run benchmarks
pytest tests/helix/test_wan_baseline.py::TestBaselineBenchmarks -v
```

### Phase 2: Helix Tests (After Implementation)

After implementing Helix (see `plans/helix-parallelism-implementation.md`):

```bash
# 1. Test Helix model loading
pytest tests/helix/test_wan_helix.py::TestHelixModelLoading -v

# 2. Test Helix generation
pytest tests/helix/test_wan_helix.py::TestHelixGeneration -v

# 3. Compare SSIM with baseline
pytest tests/helix/test_wan_helix.py::TestHelixSSIMComparison -v

# 4. Verify memory reduction
pytest tests/helix/test_wan_helix.py::TestHelixMemoryComparison -v

# 5. Benchmark Helix vs SP
pytest tests/helix/test_wan_helix.py::TestHelixBenchmarks -v
```

## Configuration Reference

### Parallelism Configurations

| Config | TP | SP | CP | GPUs | Description |
|--------|----|----|----|----|-------------|
| sp8 | 1 | 8 | 1 | 8 | Pure SP (baseline) |
| sp4 | 1 | 4 | 1 | 4 | SP=4 |
| sp2 | 1 | 2 | 1 | 2 | SP=2 |
| helix_tp2_cp4 | 2 | 1 | 4 | 8 | Helix: weights/2, seq/4 |
| helix_tp4_cp2 | 4 | 1 | 2 | 8 | Helix: weights/4, seq/2 |
| helix_tp2_cp2 | 2 | 1 | 2 | 4 | Helix: weights/2, seq/2 |

### Sequence Configurations

| Config | Frames | Resolution | Sequence Length* |
|--------|--------|------------|------------------|
| short | 17 | 480×832 | 1,697,280 |
| medium | 45 | 480×832 | 4,492,800 |
| long | 81 | 480×832 | 8,087,040 |
| high_res | 17 | 720×1280 | 3,916,800 |

*Sequence length after patchification with patch_size=(1,2,2)

## Expected Results

### Memory Comparison (Wan 14B)

| Configuration | Weight Memory | Total Memory | Fits 32GB? |
|---------------|---------------|--------------|------------|
| SP=8 | 28 GB | ~37 GB | ❌ No |
| Helix TP=2, CP=4 | 14 GB | ~24 GB | ✅ Yes |
| Helix TP=4, CP=2 | 7 GB | ~17 GB | ✅ Yes |

### Throughput Comparison (Wan 1.3B)

Expected results (actual may vary):

| Configuration | Time (17 frames) | Speedup |
|---------------|------------------|---------|
| SP=1 | ~10s | 1.0x |
| SP=8 | ~2s | 5.0x |
| Helix TP=2, CP=4 | ~2.5s | 4.0x |

Note: Helix may be slightly slower than pure SP due to TP communication overhead, but enables running larger models.

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure models are at `/mnt/nvme0/models/FastVideo/`
2. **Insufficient GPUs**: Tests skip automatically if not enough GPUs
3. **OOM errors**: Try smaller sequence lengths or higher TP
4. **Helix tests skipped**: Helix not implemented yet

### Debug Commands

```bash
# Check GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Check model paths
ls -la /mnt/nvme0/models/FastVideo/

# Run with verbose output
pytest tests/helix/ -v -s --tb=long
```

## Related Documentation

- [Helix Implementation Plan](../../plans/helix-parallelism-implementation.md)
- [Helix Test Setup Plan](../../plans/helix-test-setup.md)
- [FastVideo Model Testing Plan](../../plans/fastvideo-model-testing-plan.md)
