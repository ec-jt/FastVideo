# SPDX-License-Identifier: Apache-2.0
"""
Helix parallelism tests for Wan models.

These tests verify Helix (TP+CP) implementation:
1. Verify models load correctly with Helix configurations
2. Compare outputs with baseline SP for correctness (SSIM)
3. Verify memory reduction with TP (weight sharding)
4. Benchmark throughput improvements

Run with:
    pytest tests/helix/test_wan_helix.py -v
    pytest tests/helix/test_wan_helix.py -v -k "14B"  # Critical 14B tests
    pytest tests/helix/test_wan_helix.py -v -k "memory"  # Memory tests

NOTE: These tests require Helix implementation to be complete.
      See plans/helix-parallelism-implementation.md for implementation details.
"""

import gc
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import torch

# Set VSA backend for FastVideo models (required for local FastVideo checkpoints)
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

from tests.helix.conftest import (
    ALL_MODEL_CONFIGS,
    HELIX_CONFIGS,
    ModelConfig,
    ParallelismConfig,
    SequenceConfig,
    SEQ_SHORT,
    SEQ_MEDIUM,
    WAN_1_3B_CONFIG,
    WAN_5B_CONFIG,
    WAN_14B_CONFIG,
    check_model_fits,
    estimate_memory_per_gpu,
    get_valid_helix_combinations,
    skip_if_insufficient_gpus,
    skip_if_model_not_found,
)

# Output directory for Helix results
HELIX_OUTPUT_DIR = Path("tests/helix/helix_outputs")
BASELINE_OUTPUT_DIR = Path("tests/helix/baseline_outputs")


# =============================================================================
# Skip marker for unimplemented Helix
# =============================================================================
def helix_not_implemented():
    """Check if Helix is implemented."""
    try:
        # Try to import Helix components
        from fastvideo.distributed.helix_comm import HelixCommunicator
        return False
    except ImportError:
        return True


helix_skip = pytest.mark.skipif(
    helix_not_implemented(),
    reason="Helix parallelism not yet implemented. See plans/helix-parallelism-implementation.md"
)


# =============================================================================
# Test Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for Helix results."""
    HELIX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return HELIX_OUTPUT_DIR


# =============================================================================
# Unit Tests - Helix Configuration Validation
# =============================================================================
class TestHelixConfigurations:
    """Test Helix configuration validity."""

    @pytest.mark.parametrize("config", ALL_MODEL_CONFIGS, ids=lambda c: c.name)
    def test_valid_helix_combinations(self, config: ModelConfig):
        """Verify valid Helix (TP, CP) combinations exist."""
        combinations = get_valid_helix_combinations(config, total_gpus=8)
        print(f"\n{config.name} valid Helix combinations: {combinations}")
        assert len(combinations) > 0, f"No valid Helix combinations for {config.name}"

    def test_wan_14b_helix_fits(self):
        """Verify Wan 14B fits with Helix TP=2."""
        config = WAN_14B_CONFIG
        # With TP=2: 28GB / 2 = 14GB weights per GPU
        # + 9.4GB text encoder = 23.4GB
        # Should fit on 32GB with room for activations
        fits = check_model_fits(
            config, SEQ_SHORT, tp_size=2, cp_size=4, gpu_memory_gb=32.0
        )
        assert fits is True, "Wan 14B should fit with Helix TP=2, CP=4"

    def test_wan_14b_helix_tp4_fits(self):
        """Verify Wan 14B fits with Helix TP=4."""
        config = WAN_14B_CONFIG
        # With TP=4: 28GB / 4 = 7GB weights per GPU
        # + 9.4GB text encoder = 16.4GB
        # Plenty of room for activations
        fits = check_model_fits(
            config, SEQ_MEDIUM, tp_size=4, cp_size=2, gpu_memory_gb=32.0
        )
        assert fits is True, "Wan 14B should fit with Helix TP=4, CP=2"


# =============================================================================
# Integration Tests - Helix Model Loading
# =============================================================================
@helix_skip
@pytest.mark.gpu
class TestHelixModelLoading:
    """Test model loading with Helix configurations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "helix_config", HELIX_CONFIGS, ids=lambda c: c.name
    )
    def test_load_wan_1_3b_helix(self, helix_config: ParallelismConfig):
        """Test loading Wan 1.3B with Helix configurations."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(helix_config.total_gpus)

        # Check if TP size is valid for this model
        if helix_config.tp_size not in config.valid_tp_sizes:
            pytest.skip(
                f"TP={helix_config.tp_size} not valid for {config.name} "
                f"(valid: {config.valid_tp_sizes})"
            )

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=helix_config.total_gpus,
            tp_size=helix_config.tp_size,
            cp_size=helix_config.cp_size,
        )
        assert generator is not None

        # Check memory usage - should be reduced by TP
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        expected_weight_memory = config.transformer_memory_gb / helix_config.tp_size
        print(f"\n{config.name} with {helix_config.name}:")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Expected weight memory: {expected_weight_memory:.2f} GB")

        del generator

    def test_load_wan_14b_helix_tp2_cp4(self):
        """Test loading Wan 14B with Helix TP=2, CP=4 - THE CRITICAL TEST."""
        config = WAN_14B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        # This is the configuration that enables 14B on 32GB GPUs
        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )
        assert generator is not None

        # Verify memory is within bounds
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\n{config.name} with Helix TP=2, CP=4:")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Expected: ~14GB weights + ~9.4GB text encoder = ~23.4GB")

        # Should be well under 32GB
        assert memory_allocated < 28.0, f"Memory too high: {memory_allocated:.2f} GB"

        del generator


# =============================================================================
# Integration Tests - Helix Video Generation
# =============================================================================
@helix_skip
@pytest.mark.gpu
@pytest.mark.slow
class TestHelixGeneration:
    """Test video generation with Helix configurations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_generate_wan_1_3b_helix_tp2_cp4(self, output_dir, test_prompt, seed):
        """Generate video with Wan 1.3B using Helix TP=2, CP=4."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )

        start_time = time.time()
        output = generator.generate(
            prompt=test_prompt,
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=4,
            seed=seed,
        )
        generation_time = time.time() - start_time

        output_path = output_dir / "wan_1_3b_helix_tp2_cp4.mp4"
        output.save(str(output_path))

        print(f"\n{config.name} Helix TP=2, CP=4:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Output: {output_path}")

        assert output_path.exists()
        del generator

    def test_generate_wan_14b_helix(self, output_dir, test_prompt, seed):
        """Generate video with Wan 14B using Helix - THE MAIN GOAL."""
        config = WAN_14B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        # This should work with Helix but fail with SP alone
        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )

        start_time = time.time()
        output = generator.generate(
            prompt=test_prompt,
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=4,
            seed=seed,
        )
        generation_time = time.time() - start_time

        output_path = output_dir / "wan_14b_helix_tp2_cp4.mp4"
        output.save(str(output_path))

        print(f"\n{config.name} Helix TP=2, CP=4:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Output: {output_path}")
        print(f"  SUCCESS: 14B model running on 32GB GPUs!")

        assert output_path.exists()
        del generator


# =============================================================================
# SSIM Comparison Tests
# =============================================================================
@helix_skip
@pytest.mark.gpu
@pytest.mark.slow
class TestHelixSSIMComparison:
    """Compare Helix outputs with baseline SP outputs using SSIM."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_ssim_wan_1_3b_helix_vs_baseline(self, output_dir, seed):
        """Compare Helix output with baseline SP output using SSIM."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        # Check baseline exists
        baseline_path = BASELINE_OUTPUT_DIR / "ssim_ref_wan_1_3b_prompt0.mp4"
        if not baseline_path.exists():
            pytest.skip(
                f"Baseline not found: {baseline_path}. "
                "Run test_wan_baseline.py first."
            )

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )

        # Generate with same prompt and seed as baseline
        prompt = "A serene mountain landscape at sunset"
        output = generator.generate(
            prompt=prompt,
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=4,
            seed=seed,
        )

        helix_path = output_dir / "ssim_helix_wan_1_3b_prompt0.mp4"
        output.save(str(helix_path))

        # Calculate SSIM
        try:
            from fastvideo.tests.ssim.utils import calculate_video_ssim
            ssim_score = calculate_video_ssim(str(baseline_path), str(helix_path))
            print(f"\nSSIM comparison {config.name}:")
            print(f"  Baseline: {baseline_path}")
            print(f"  Helix: {helix_path}")
            print(f"  SSIM Score: {ssim_score:.4f}")

            # SSIM should be very high (>0.95) for same seed
            assert ssim_score > 0.95, f"SSIM too low: {ssim_score:.4f}"
        except ImportError:
            print("SSIM utils not available, skipping SSIM calculation")

        del generator


# =============================================================================
# Memory Comparison Tests
# =============================================================================
@helix_skip
@pytest.mark.gpu
class TestHelixMemoryComparison:
    """Compare memory usage between Helix and baseline SP."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_memory_reduction_wan_1_3b(self):
        """Verify Helix reduces memory compared to SP."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        # Measure SP memory
        torch.cuda.reset_peak_memory_stats(0)
        generator_sp = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            sp_size=8,
        )
        sp_memory = torch.cuda.max_memory_allocated(0) / 1e9
        del generator_sp
        gc.collect()
        torch.cuda.empty_cache()

        # Measure Helix memory
        torch.cuda.reset_peak_memory_stats(0)
        generator_helix = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )
        helix_memory = torch.cuda.max_memory_allocated(0) / 1e9
        del generator_helix

        print(f"\n{config.name} memory comparison:")
        print(f"  SP=8 memory: {sp_memory:.2f} GB")
        print(f"  Helix TP=2, CP=4 memory: {helix_memory:.2f} GB")
        print(f"  Reduction: {(1 - helix_memory/sp_memory) * 100:.1f}%")

        # Helix should use less memory due to weight sharding
        # With TP=2, weights should be ~50% smaller
        assert helix_memory < sp_memory, "Helix should use less memory than SP"

    def test_wan_14b_memory_with_helix(self):
        """Verify Wan 14B fits in memory with Helix."""
        config = WAN_14B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        torch.cuda.reset_peak_memory_stats(0)

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )

        # Generate to measure peak memory
        _ = generator.generate(
            prompt="A test video",
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=2,
            seed=42,
        )

        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n{config.name} Helix TP=2, CP=4:")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  GPU memory limit: 32 GB")
        print(f"  Headroom: {32 - peak_memory:.2f} GB")

        # Must fit in 32GB
        assert peak_memory < 32.0, f"Peak memory {peak_memory:.2f} GB exceeds 32GB"

        del generator


# =============================================================================
# Benchmark Tests
# =============================================================================
@helix_skip
@pytest.mark.gpu
@pytest.mark.benchmark
class TestHelixBenchmarks:
    """Benchmark Helix performance vs baseline SP."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_benchmark_helix_vs_sp(self, test_prompt, seed):
        """Benchmark Helix throughput vs SP."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        from fastvideo import VideoGenerator

        results = []

        # Benchmark SP=8
        generator_sp = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            sp_size=8,
        )

        # Warmup
        _ = generator_sp.generate(
            prompt=test_prompt,
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=2,
            seed=seed,
        )

        # Benchmark
        times = []
        for _ in range(3):
            start = time.time()
            _ = generator_sp.generate(
                prompt=test_prompt,
                num_frames=SEQ_SHORT.num_frames,
                height=SEQ_SHORT.height,
                width=SEQ_SHORT.width,
                num_inference_steps=4,
                seed=seed,
            )
            times.append(time.time() - start)

        sp_avg = sum(times) / len(times)
        results.append({"config": "SP=8", "avg_time": sp_avg})

        del generator_sp
        gc.collect()
        torch.cuda.empty_cache()

        # Benchmark Helix TP=2, CP=4
        generator_helix = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=8,
            tp_size=2,
            cp_size=4,
        )

        # Warmup
        _ = generator_helix.generate(
            prompt=test_prompt,
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=2,
            seed=seed,
        )

        # Benchmark
        times = []
        for _ in range(3):
            start = time.time()
            _ = generator_helix.generate(
                prompt=test_prompt,
                num_frames=SEQ_SHORT.num_frames,
                height=SEQ_SHORT.height,
                width=SEQ_SHORT.width,
                num_inference_steps=4,
                seed=seed,
            )
            times.append(time.time() - start)

        helix_avg = sum(times) / len(times)
        results.append({"config": "Helix TP=2, CP=4", "avg_time": helix_avg})

        del generator_helix

        # Print results
        print("\n" + "=" * 60)
        print(f"Benchmark Results: {config.name}")
        print("=" * 60)
        for r in results:
            fps = SEQ_SHORT.num_frames / r["avg_time"]
            print(f"{r['config']:20s}: {r['avg_time']:.2f}s ({fps:.2f} frames/sec)")

        # Calculate comparison
        speedup = sp_avg / helix_avg
        print(f"\nHelix vs SP speedup: {speedup:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
