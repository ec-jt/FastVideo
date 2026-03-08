#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test video generation with different parallelism configurations.

Run with:
    FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN python tests/helix/test_video_generation.py
"""

import os
import sys
import torch

# Set VSA backend for FastVideo models
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

from fastvideo import VideoGenerator


def test_wan_1_3b_sp1():
    """Test Wan 1.3B with SP=1 (single GPU)."""
    print(f"\n{'='*60}")
    print(f"Testing Wan 1.3B SP=1 on {torch.cuda.device_count()} GPUs")
    print(f"{'='*60}")

    os.makedirs("tests/helix/baseline_outputs", exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        model_path="/mnt/nvme0/models/FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        sp_size=1,
        tp_size=1,
    )
    print("Model loaded! Generating video...")

    video = generator.generate_video(
        prompt="A cat walking on grass SP1",
        output_path="tests/helix/baseline_outputs",
        save_video=True,
        seed=42,
    )

    video_path = video.get("video_path")
    gen_time = video.get("generation_time")
    peak_mem = video.get("peak_memory_mb")
    print(f"Video path: {video_path}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Peak memory: {peak_mem:.2f} MB")

    del generator
    torch.cuda.empty_cache()
    print("Wan 1.3B SP=1 video generation PASSED!")
    return {"video_path": video_path, "gen_time": gen_time, "peak_mem": peak_mem}


def test_wan_1_3b_sp6():
    """Test Wan 1.3B with SP=6 (6 GPUs)."""
    print(f"\n{'='*60}")
    print(f"Testing Wan 1.3B SP=6 on {torch.cuda.device_count()} GPUs")
    print(f"{'='*60}")

    os.makedirs("tests/helix/baseline_outputs", exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        model_path="/mnt/nvme0/models/FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=6,
        sp_size=6,
        tp_size=1,
    )
    print("Model loaded! Generating video...")

    video = generator.generate_video(
        prompt="A cat walking on grass SP6",
        output_path="tests/helix/baseline_outputs",
        save_video=True,
        seed=42,
    )

    video_path = video.get("video_path")
    gen_time = video.get("generation_time")
    peak_mem = video.get("peak_memory_mb")
    print(f"Video path: {video_path}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Peak memory: {peak_mem:.2f} MB")

    del generator
    torch.cuda.empty_cache()
    print("Wan 1.3B SP=6 video generation PASSED!")
    return {"video_path": video_path, "gen_time": gen_time, "peak_mem": peak_mem}


def test_wan_5b_sp8():
    """Test Wan 5B with SP=8 (8 GPUs)."""
    print(f"\n{'='*60}")
    print(f"Testing Wan 5B SP=8 on {torch.cuda.device_count()} GPUs")
    print(f"{'='*60}")

    os.makedirs("tests/helix/baseline_outputs", exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        model_path="/mnt/nvme0/models/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        num_gpus=8,
        sp_size=8,
        tp_size=1,
    )
    print("Model loaded! Generating video...")

    # Note: TI2V model may need an image input, but let's try T2V first
    video = generator.generate_video(
        prompt="A cat walking on grass SP8 5B",
        output_path="tests/helix/baseline_outputs",
        save_video=True,
        seed=42,
    )

    video_path = video.get("video_path")
    gen_time = video.get("generation_time")
    peak_mem = video.get("peak_memory_mb")
    print(f"Video path: {video_path}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Peak memory: {peak_mem:.2f} MB")

    del generator
    torch.cuda.empty_cache()
    print("Wan 5B SP=8 video generation PASSED!")
    return {"video_path": video_path, "gen_time": gen_time, "peak_mem": peak_mem}


if __name__ == "__main__":
    results = {}

    # Parse command line args
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "sp1":
            results["sp1"] = test_wan_1_3b_sp1()
        elif test_name == "sp6":
            results["sp6"] = test_wan_1_3b_sp6()
        elif test_name == "sp8":
            results["sp8"] = test_wan_5b_sp8()
        else:
            print(f"Unknown test: {test_name}")
            print("Usage: python test_video_generation.py [sp1|sp6|sp8]")
            sys.exit(1)
    else:
        # Run all tests
        print("Running all video generation tests...")
        results["sp1"] = test_wan_1_3b_sp1()
        results["sp6"] = test_wan_1_3b_sp6()
        results["sp8"] = test_wan_5b_sp8()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"{name}: {result['gen_time']:.2f}s, {result['peak_mem']:.2f} MB")
