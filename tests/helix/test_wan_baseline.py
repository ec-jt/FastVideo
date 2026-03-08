# SPDX-License-Identifier: Apache-2.0
"""
Baseline tests for Wan models using current SP (Sequence Parallelism).

These tests establish the baseline behavior before Helix implementation:
1. Verify models load correctly with various SP configurations
2. Measure memory usage per GPU
3. Generate reference outputs for SSIM comparison
4. Benchmark throughput for comparison with Helix

Run with:
    pytest tests/helix/test_wan_baseline.py -v
    pytest tests/helix/test_wan_baseline.py -v -k "1_3B"  # Just 1.3B model
    pytest tests/helix/test_wan_baseline.py -v -k "short"  # Just short sequences
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
    ALL_SEQUENCE_CONFIGS,
    BASELINE_SP_CONFIGS,
    ModelConfig,
    ParallelismConfig,
    SequenceConfig,
    SEQ_SHORT,
    WAN_1_3B_CONFIG,
    WAN_5B_CONFIG,
    WAN_14B_CONFIG,
    check_model_fits,
    estimate_memory_per_gpu,
    skip_if_insufficient_gpus,
    skip_if_model_not_found,
)

# Output directory for baseline results
BASELINE_OUTPUT_DIR = Path("tests/helix/baseline_outputs")


# =============================================================================
# Test Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for baseline results."""
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINE_OUTPUT_DIR


# =============================================================================
# Unit Tests - Configuration Validation
# =============================================================================
class TestModelConfigurations:
    """Test that model configurations are valid."""

    @pytest.mark.parametrize("config", ALL_MODEL_CONFIGS, ids=lambda c: c.name)
    def test_hidden_size_calculation(self, config: ModelConfig):
        """Verify hidden size is calculated correctly."""
        expected = config.transformer.num_attention_heads * config.transformer.attention_head_dim
        assert config.transformer.hidden_size == expected

    @pytest.mark.parametrize("config", ALL_MODEL_CONFIGS, ids=lambda c: c.name)
    def test_valid_tp_sizes(self, config: ModelConfig):
        """Verify TP sizes divide num_attention_heads."""
        num_heads = config.transformer.num_attention_heads
        for tp in config.valid_tp_sizes:
            assert num_heads % tp == 0, f"TP={tp} doesn't divide {num_heads} heads"

    @pytest.mark.parametrize("config", ALL_MODEL_CONFIGS, ids=lambda c: c.name)
    def test_valid_sp_sizes(self, config: ModelConfig):
        """Verify SP sizes divide num_attention_heads."""
        num_heads = config.transformer.num_attention_heads
        for sp in config.valid_sp_sizes:
            assert num_heads % sp == 0, f"SP={sp} doesn't divide {num_heads} heads"

    def test_wan_1_3b_config(self):
        """Verify Wan 1.3B configuration matches local model."""
        config = WAN_1_3B_CONFIG
        assert config.transformer.num_attention_heads == 12
        assert config.transformer.attention_head_dim == 128
        assert config.transformer.num_layers == 30
        assert config.transformer.hidden_size == 1536  # 12 * 128

    def test_wan_5b_config(self):
        """Verify Wan 5B configuration matches local model."""
        config = WAN_5B_CONFIG
        assert config.transformer.num_attention_heads == 24
        assert config.transformer.attention_head_dim == 128
        assert config.transformer.num_layers == 30
        assert config.transformer.hidden_size == 3072  # 24 * 128
        # TI2V has 48 channels
        assert config.transformer.in_channels == 48

    def test_wan_14b_config(self):
        """Verify Wan 14B configuration matches local model."""
        config = WAN_14B_CONFIG
        assert config.transformer.num_attention_heads == 40
        assert config.transformer.attention_head_dim == 128
        assert config.transformer.num_layers == 40
        assert config.transformer.hidden_size == 5120  # 40 * 128


class TestSequenceConfigurations:
    """Test sequence length calculations."""

    @pytest.mark.parametrize("config", ALL_SEQUENCE_CONFIGS, ids=lambda c: c.name)
    def test_sequence_length_calculation(self, config: SequenceConfig):
        """Verify sequence length is calculated correctly for Wan patch_size=(1,2,2)."""
        # For patch_size=(1, 2, 2): seq_len = frames * (H/2) * (W/2)
        expected = config.num_frames * (config.height // 2) * (config.width // 2)
        assert config.sequence_length == expected

    def test_short_sequence_length(self):
        """Verify short sequence length."""
        # 17 frames @ 480x832 with patch_size=(1,2,2)
        # = 17 * 240 * 416 = 1,697,280
        assert SEQ_SHORT.sequence_length == 17 * 240 * 416


class TestMemoryEstimation:
    """Test memory estimation functions."""

    def test_wan_1_3b_fits_single_gpu(self):
        """Wan 1.3B should fit on single 32GB GPU with short sequence."""
        fits = check_model_fits(
            WAN_1_3B_CONFIG, SEQ_SHORT, tp_size=1, sp_size=1, gpu_memory_gb=32.0
        )
        # 2.6GB weights + 9.4GB text encoder + activations should fit
        assert fits is True

    def test_wan_14b_needs_tp(self):
        """Wan 14B should NOT fit on single 32GB GPU."""
        fits = check_model_fits(
            WAN_14B_CONFIG, SEQ_SHORT, tp_size=1, sp_size=1, gpu_memory_gb=32.0
        )
        # 28GB weights + 9.4GB text encoder = 37.4GB > 32GB
        assert fits is False

    def test_wan_14b_fits_with_helix_tp2_cp4(self):
        """Wan 14B should fit with Helix TP=2, CP=4 on 32GB GPUs."""
        fits = check_model_fits(
            WAN_14B_CONFIG, SEQ_SHORT, tp_size=2, cp_size=4, gpu_memory_gb=32.0
        )
        # 14GB weights + 9.4GB text encoder + 1GB activations (split by CP=4) should fit
        # Total: ~24.4GB < 25.6GB (32GB * 0.8)
        assert fits is True

    def test_memory_estimation_decreases_with_tp(self):
        """Memory per GPU should decrease with higher TP."""
        mem_tp1 = estimate_memory_per_gpu(WAN_14B_CONFIG, SEQ_SHORT, tp_size=1)
        mem_tp2 = estimate_memory_per_gpu(WAN_14B_CONFIG, SEQ_SHORT, tp_size=2)
        mem_tp4 = estimate_memory_per_gpu(WAN_14B_CONFIG, SEQ_SHORT, tp_size=4)
        assert mem_tp1 > mem_tp2 > mem_tp4


# =============================================================================
# Integration Tests - Model Loading (requires GPUs)
# =============================================================================
@pytest.mark.gpu
class TestModelLoading:
    """Test model loading with various configurations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("config", [WAN_1_3B_CONFIG], ids=lambda c: c.name)
    def test_load_small_model_single_gpu(self, config: ModelConfig):
        """Test loading 1.3B model on single GPU."""
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(1)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=1,
            sp_size=1,
            tp_size=1,
            text_encoder_cpu_offload=True,
        )
        assert generator is not None

        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\n{config.name} memory allocated: {memory_allocated:.2f} GB")

        # Cleanup
        del generator

    @pytest.mark.parametrize(
        "sp_config", BASELINE_SP_CONFIGS[:2], ids=lambda c: c.name
    )
    def test_load_with_sp(self, sp_config: ParallelismConfig):
        """Test loading model with sequence parallelism."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(sp_config.total_gpus)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=sp_config.total_gpus,
            sp_size=sp_config.sp_size,
            tp_size=sp_config.tp_size,
            text_encoder_cpu_offload=True,
        )
        assert generator is not None

        # Cleanup
        del generator


# =============================================================================
# Integration Tests - Video Generation (requires GPUs)
# =============================================================================
@pytest.mark.gpu
@pytest.mark.slow
class TestBaselineGeneration:
    """Test video generation with baseline SP configurations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_generate_wan_1_3b_sp1(self, output_dir, test_prompt, seed):
        """Generate video with Wan 1.3B, SP=1 (single GPU baseline)."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(1)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=1,
        )

        # Generate with short sequence
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

        # Save output
        output_path = output_dir / f"wan_1_3b_sp1_baseline.mp4"
        output.save(str(output_path))

        # Log metrics
        print(f"\n{config.name} SP=1 generation:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Output: {output_path}")

        assert output_path.exists()
        del generator

    @pytest.mark.parametrize(
        "sp_size,num_gpus",
        [(2, 2), (4, 4), (8, 8)],
        ids=["sp2", "sp4", "sp8"],
    )
    def test_generate_wan_1_3b_sp_scaling(
        self, sp_size, num_gpus, output_dir, test_prompt, seed
    ):
        """Test SP scaling with Wan 1.3B."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(num_gpus)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=num_gpus,
            sp_size=sp_size,
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

        output_path = output_dir / f"wan_1_3b_sp{sp_size}_baseline.mp4"
        output.save(str(output_path))

        print(f"\n{config.name} SP={sp_size} generation:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Speedup vs SP=1: (run sp1 test first for comparison)")

        assert output_path.exists()
        del generator

    @pytest.mark.parametrize(
        "seq_config",
        [SEQ_SHORT],  # Start with short only
        ids=lambda c: c.name,
    )
    def test_generate_wan_5b(self, seq_config, output_dir, test_prompt, seed):
        """Test Wan 5B generation (TI2V model)."""
        config = WAN_5B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(4)  # Need at least 4 GPUs for 5B

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=4,
            sp_size=4,
        )

        start_time = time.time()
        output = generator.generate(
            prompt=test_prompt,
            num_frames=seq_config.num_frames,
            height=seq_config.height,
            width=seq_config.width,
            num_inference_steps=4,
            seed=seed,
        )
        generation_time = time.time() - start_time

        output_path = output_dir / f"wan_5b_sp4_{seq_config.name}_baseline.mp4"
        output.save(str(output_path))

        print(f"\n{config.name} SP=4 {seq_config.name}:")
        print(f"  Time: {generation_time:.2f}s")

        assert output_path.exists()
        del generator

    def test_wan_14b_requires_helix(self, output_dir):
        """Demonstrate that Wan 14B cannot run with SP alone on 32GB GPUs.

        This test documents the limitation that Helix is designed to solve.
        """
        config = WAN_14B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(8)

        # Check memory estimation
        mem_sp8 = estimate_memory_per_gpu(config, SEQ_SHORT, sp_size=8)
        print(f"\n{config.name} estimated memory with SP=8: {mem_sp8:.2f} GB")

        # SP doesn't reduce weight memory, only activation memory
        # 28GB weights + 9.4GB text encoder = 37.4GB per GPU
        # This exceeds 32GB even with SP=8
        fits = check_model_fits(config, SEQ_SHORT, sp_size=8, gpu_memory_gb=32.0)

        if not fits:
            pytest.skip(
                f"{config.name} requires Helix (TP+CP) to fit on 32GB GPUs. "
                f"Estimated memory: {mem_sp8:.2f} GB per GPU."
            )

        # If somehow it fits, try to load (unlikely)
        from fastvideo import VideoGenerator

        try:
            generator = VideoGenerator.from_pretrained(
                config.local_path,
                num_gpus=8,
                sp_size=8,
            )
            del generator
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"{config.name} OOM with SP=8: {e}")
            raise


# =============================================================================
# Benchmark Tests
# =============================================================================
@pytest.mark.gpu
@pytest.mark.benchmark
class TestBaselineBenchmarks:
    """Benchmark tests for baseline comparison with Helix."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_benchmark_wan_1_3b_throughput(self, test_prompt, seed):
        """Benchmark Wan 1.3B throughput with different SP sizes."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)

        results = []

        for sp_size in [1, 2, 4, 8]:
            skip_if_insufficient_gpus(sp_size)

            from fastvideo import VideoGenerator

            generator = VideoGenerator.from_pretrained(
                config.local_path,
                num_gpus=sp_size,
                sp_size=sp_size,
            )

            # Warmup
            _ = generator.generate(
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
                _ = generator.generate(
                    prompt=test_prompt,
                    num_frames=SEQ_SHORT.num_frames,
                    height=SEQ_SHORT.height,
                    width=SEQ_SHORT.width,
                    num_inference_steps=4,
                    seed=seed,
                )
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            results.append(
                {
                    "sp_size": sp_size,
                    "avg_time": avg_time,
                    "frames_per_sec": SEQ_SHORT.num_frames / avg_time,
                }
            )

            del generator
            gc.collect()
            torch.cuda.empty_cache()

        # Print results
        print("\n" + "=" * 60)
        print(f"Benchmark Results: {config.name}")
        print("=" * 60)
        for r in results:
            print(
                f"SP={r['sp_size']:2d}: {r['avg_time']:.2f}s "
                f"({r['frames_per_sec']:.2f} frames/sec)"
            )

        # Calculate speedups
        if len(results) > 1:
            baseline = results[0]["avg_time"]
            print("\nSpeedups vs SP=1:")
            for r in results[1:]:
                speedup = baseline / r["avg_time"]
                print(f"  SP={r['sp_size']}: {speedup:.2f}x")


# =============================================================================
# Memory Profiling Tests
# =============================================================================
@pytest.mark.gpu
class TestMemoryProfiling:
    """Profile memory usage for baseline configurations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_memory_profile_wan_1_3b(self):
        """Profile memory usage for Wan 1.3B."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(1)

        torch.cuda.reset_peak_memory_stats(0)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=1,
        )

        after_load = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n{config.name} memory after load: {after_load:.2f} GB")

        # Generate to measure peak
        _ = generator.generate(
            prompt="A test video",
            num_frames=SEQ_SHORT.num_frames,
            height=SEQ_SHORT.height,
            width=SEQ_SHORT.width,
            num_inference_steps=2,
            seed=42,
        )

        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"{config.name} peak memory during generation: {peak:.2f} GB")

        del generator

    @pytest.mark.parametrize(
        "config",
        [WAN_1_3B_CONFIG, WAN_5B_CONFIG],
        ids=lambda c: c.name,
    )
    def test_memory_vs_sequence_length(self, config: ModelConfig):
        """Test how memory scales with sequence length."""
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(4)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=4,
            sp_size=4,
        )

        results = []
        for seq_config in [SEQ_SHORT]:  # Start with short only
            torch.cuda.reset_peak_memory_stats(0)

            _ = generator.generate(
                prompt="A test video",
                num_frames=seq_config.num_frames,
                height=seq_config.height,
                width=seq_config.width,
                num_inference_steps=2,
                seed=42,
            )

            peak = torch.cuda.max_memory_allocated(0) / 1e9
            results.append(
                {
                    "seq_name": seq_config.name,
                    "seq_len": seq_config.sequence_length,
                    "peak_memory": peak,
                }
            )

        print(f"\n{config.name} memory vs sequence length:")
        for r in results:
            print(f"  {r['seq_name']}: {r['peak_memory']:.2f} GB (seq_len={r['seq_len']:,})")

        del generator


# =============================================================================
# SSIM Reference Generation
# =============================================================================
@pytest.mark.gpu
@pytest.mark.slow
class TestSSIMReferenceGeneration:
    """Generate reference outputs for SSIM comparison after Helix implementation."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up GPU memory after each test."""
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_generate_ssim_reference_wan_1_3b(self, output_dir, seed):
        """Generate SSIM reference for Wan 1.3B."""
        config = WAN_1_3B_CONFIG
        skip_if_model_not_found(config)
        skip_if_insufficient_gpus(1)

        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            config.local_path,
            num_gpus=1,
        )

        # Use fixed prompts for reproducibility
        prompts = [
            "A serene mountain landscape at sunset",
            "A busy city street with cars and pedestrians",
            "Ocean waves crashing on a sandy beach",
        ]

        for i, prompt in enumerate(prompts):
            output = generator.generate(
                prompt=prompt,
                num_frames=SEQ_SHORT.num_frames,
                height=SEQ_SHORT.height,
                width=SEQ_SHORT.width,
                num_inference_steps=4,
                seed=seed,
            )

            output_path = output_dir / f"ssim_ref_wan_1_3b_prompt{i}.mp4"
            output.save(str(output_path))
            print(f"Saved SSIM reference: {output_path}")

        del generator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
