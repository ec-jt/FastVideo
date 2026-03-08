# SPDX-License-Identifier: Apache-2.0
"""
LTX2 Video Generation Test Script

This script tests LTX2-Distilled video generation with FSDP inference mode
enabled for weight sharding across 8 GPUs.

Usage:
    python tests/helix/test_ltx2_video_generation.py

Requirements:
    - 8x GPUs with 32GB VRAM each
    - FastVideo installed with LTX2 support
    - Model: FastVideo/LTX2-Distilled-Diffusers
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VideoConfig:
    """Configuration for video generation test."""
    prompt: str
    num_frames: int
    height: int
    width: int
    seed: int = 42
    num_inference_steps: int = 8
    
    @property
    def duration_seconds(self) -> float:
        """Calculate video duration at 24fps."""
        return self.num_frames / 24.0
    
    @property
    def resolution(self) -> str:
        """Return resolution string."""
        return f"{self.height}x{self.width}"


def get_peak_memory() -> int:
    """Get peak GPU memory usage in MB."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    mem_values = [int(x.strip()) for x in result.stdout.strip().split('\n')]
    return max(mem_values)


def run_video_generation_test(
    config: VideoConfig,
    num_gpus: int = 8,
    tp_size: int = 8,
    use_fsdp_inference: bool = True,
    model_path: str = "FastVideo/LTX2-Distilled-Diffusers",
) -> dict:
    """
    Run a video generation test with the given configuration.
    
    Args:
        config: Video generation configuration
        num_gpus: Number of GPUs to use
        tp_size: Tensor parallelism size
        use_fsdp_inference: Whether to use FSDP for inference
        model_path: Path to the model
        
    Returns:
        Dictionary with test results
    """
    from fastvideo import VideoGenerator
    
    print(f"\n{'='*70}")
    print(f"Test: {config.num_frames} frames @ {config.resolution}")
    print(f"Duration: ~{config.duration_seconds:.1f}s at 24fps")
    print(f"Prompt: {config.prompt[:50]}...")
    print(f"{'='*70}")
    
    result = {
        "config": {
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "prompt": config.prompt,
            "seed": config.seed,
        },
        "status": "pending",
        "generation_time_seconds": None,
        "peak_memory_mb": None,
        "error": None,
    }
    
    try:
        # Load model
        print("Loading model...")
        load_start = time.time()
        generator = VideoGenerator.from_pretrained(
            model_path,
            num_gpus=num_gpus,
            tp_size=tp_size,
            sp_size=1,
            use_fsdp_inference=use_fsdp_inference,
            dit_layerwise_offload=False,  # Must disable for FSDP
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
        
        # Generate video
        print("Generating video...")
        gen_start = time.time()
        video = generator.generate_video(
            prompt=config.prompt,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            seed=config.seed,
        )
        gen_time = time.time() - gen_start
        
        # Get memory usage
        peak_memory = get_peak_memory()
        
        result["status"] = "passed"
        result["generation_time_seconds"] = gen_time
        result["peak_memory_mb"] = peak_memory
        result["load_time_seconds"] = load_time
        
        print(f"✅ SUCCESS: {config.num_frames} frames @ {config.resolution}")
        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Peak memory: {peak_memory} MB")
        
        generator.shutdown()
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"❌ FAILED: {type(e).__name__}: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="LTX2 Video Generation Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=704, help="Video width")
    parser.add_argument("--model-path", type=str, default="FastVideo/LTX2-Distilled-Diffusers",
                        help="Path to model (local or HuggingFace)")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor parallelism size")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps")
    args = parser.parse_args()
    
    # Define test configurations
    test_configs = [
        # Quick test
        VideoConfig(
            prompt="A jazz band playing live music in a cozy club, saxophone solo, drums, piano",
            num_frames=33,
            height=480,
            width=704,
        ),
    ]
    
    if not args.quick:
        # Add more comprehensive tests
        test_configs.extend([
            # Standard duration tests
            VideoConfig(
                prompt="A beautiful sunset over the ocean with waves crashing on the shore",
                num_frames=121,  # ~5 seconds
                height=480,
                width=704,
            ),
            VideoConfig(
                prompt="A jazz band playing live music in a cozy club, saxophone solo",
                num_frames=241,  # ~10 seconds
                height=480,
                width=704,
            ),
            # Long video test
            VideoConfig(
                prompt="A jazz band playing live music in a cozy club, saxophone solo",
                num_frames=481,  # ~20 seconds
                height=480,
                width=704,
            ),
            # High resolution test
            VideoConfig(
                prompt="A beautiful sunset over the ocean with waves crashing",
                num_frames=121,  # ~5 seconds
                height=1024,
                width=1536,
            ),
        ])
    
    # Override with custom config if provided
    if args.prompt or args.frames:
        test_configs = [
            VideoConfig(
                prompt=args.prompt or "A beautiful nature scene",
                num_frames=args.frames or 121,
                height=args.height,
                width=args.width,
            )
        ]
    
    print("\n" + "="*70)
    print("LTX2-Distilled Video Generation Tests")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Configuration: TP={args.tp_size}, FSDP Inference=True")
    print(f"Number of tests: {len(test_configs)}")
    print("="*70)
    
    results = []
    for config in test_configs:
        result = run_video_generation_test(
            config,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            model_path=args.model_path,
        )
        results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    for r in results:
        cfg = r["config"]
        status = "✅" if r["status"] == "passed" else "❌"
        if r["status"] == "passed":
            print(f"{status} {cfg['num_frames']} frames @ {cfg['height']}x{cfg['width']}: "
                  f"{r['generation_time_seconds']:.2f}s, {r['peak_memory_mb']} MB")
        else:
            print(f"{status} {cfg['num_frames']} frames @ {cfg['height']}x{cfg['width']}: "
                  f"FAILED - {r['error']}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
