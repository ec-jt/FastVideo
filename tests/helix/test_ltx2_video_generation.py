# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 / LTX-2.3 Video Generation Test Script

Supports both dev (non-distilled) and distilled pipelines:

Dev one-stage pipeline (LTX2Pipeline + LTX2DenoisingStage):
  - 30 inference steps with CFG guidance (guidance_scale=3.0)
  - Multi-modal guidance: text CFG, modality isolation, STG
  - Resolution divisible by 32
  - 4 forward passes per denoising step

Distilled pipeline (LTX23DistilledPipeline + LTX2DistilledDenoisingStage):
  - 8 inference steps, no CFG (guidance_scale=1.0)
  - Single forward pass per step
  - Optional two-stage with spatial upsampler (LTX2_TWO_STAGE=1)
  - Resolution divisible by 64 for two-stage

Usage (dev one-stage):
    python tests/helix/test_ltx2_video_generation.py --quick \
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Dev-Diffusers \
        --num-gpus 8 --tp-size 8

Usage (distilled):
    python tests/helix/test_ltx2_video_generation.py --quick \
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
        --num-gpus 8 --tp-size 8

Requirements:
    - 8x GPUs with 32GB VRAM each
    - FastVideo installed with LTX2 support
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch


def _is_distilled_model(model_path: str) -> bool:
    """Detect whether the model path refers to a distilled model."""
    lower = model_path.lower()
    return "distilled" in lower


@dataclass
class VideoConfig:
    """Configuration for video generation test."""
    prompt: str
    num_frames: int
    height: int
    width: int
    seed: int = 42
    num_inference_steps: int = 30
    guidance_scale: float = 3.0

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
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
    )
    mem_values = [
        int(x.strip())
        for x in result.stdout.strip().split('\n')
    ]
    return max(mem_values)


def run_video_generation_test(
    config: VideoConfig,
    num_gpus: int = 8,
    tp_size: int = 8,
    use_fsdp_inference: bool = True,
    model_path: str = (
        "FastVideo/LTX2.3-Dev-Diffusers"),
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

    is_distilled = _is_distilled_model(model_path)
    pipeline_desc = (
        "LTX23DistilledPipeline (distilled)"
        if is_distilled
        else "LTX2Pipeline (dev one-stage, CFG)"
    )

    print(f"\n{'='*70}")
    print(
        f"Test: {config.num_frames} frames "
        f"@ {config.resolution}")
    print(
        f"Duration: ~{config.duration_seconds:.1f}s "
        f"at 24fps")
    print(f"Prompt: {config.prompt[:50]}...")
    print(f"Pipeline: {pipeline_desc}")
    print(
        f"Steps: {config.num_inference_steps}, "
        f"Guidance: {config.guidance_scale}")
    print(f"{'='*70}")

    result = {
        "config": {
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "prompt": config.prompt,
            "seed": config.seed,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
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
            dit_layerwise_offload=False,
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
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
        )
        gen_time = time.time() - gen_start

        # Get memory usage
        peak_memory = get_peak_memory()

        result["status"] = "passed"
        result["generation_time_seconds"] = gen_time
        result["peak_memory_mb"] = peak_memory
        result["load_time_seconds"] = load_time

        print(
            f"✅ SUCCESS: {config.num_frames} frames "
            f"@ {config.resolution}")
        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Peak memory: {peak_memory} MB")

        generator.shutdown()

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LTX-2 / LTX-2.3 Video Generation Tests "
            "(dev and distilled)"))
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test only")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt")
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of frames")
    parser.add_argument(
        "--height", type=int, default=None,
        help="Video height")
    parser.add_argument(
        "--width", type=int, default=None,
        help="Video width")
    parser.add_argument(
        "--model-path", type=str,
        default=(
            "FastVideo/LTX2.3-Dev-Diffusers"),
        help="Path to model (local or HuggingFace)")
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs to use")
    parser.add_argument(
        "--tp-size", type=int, default=8,
        help="Tensor parallelism size")
    parser.add_argument(
        "--steps", type=int, default=None,
        help=(
            "Number of inference steps "
            "(default: 30 for dev, 8 for distilled)"))
    parser.add_argument(
        "--guidance-scale", type=float, default=None,
        help=(
            "Guidance scale "
            "(default: 3.0 for dev, 1.0 for distilled)"))
    args = parser.parse_args()

    # Auto-detect model type from path
    is_distilled = _is_distilled_model(args.model_path)

    # Set defaults based on model type
    if args.steps is None:
        args.steps = 8 if is_distilled else 30
    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if is_distilled else 3.0

    # Default resolution
    default_height = args.height or 480
    default_width = args.width or 704

    # Validate resolution based on pipeline type
    if is_distilled:
        # Two-stage distilled needs divisible by 64
        divisor = 64
        divisor_label = "64 (two-stage distilled)"
    else:
        # One-stage dev needs divisible by 32
        divisor = 32
        divisor_label = "32 (one-stage dev)"

    if default_height % divisor != 0 or default_width % divisor != 0:
        print(
            f"WARNING: Resolution "
            f"{default_height}x{default_width} "
            f"is not divisible by {divisor_label}. "
            f"Adjusting to nearest valid resolution.")
        default_height = (
            (default_height + divisor - 1) // divisor
        ) * divisor
        default_width = (
            (default_width + divisor - 1) // divisor
        ) * divisor
        print(
            f"  Adjusted to: "
            f"{default_height}x{default_width}")

    # Define test configurations
    test_configs = [
        # Quick test — short video
        VideoConfig(
            prompt=(
                "A jazz band playing live music in a cozy "
                "club, saxophone solo, drums, piano"),
            num_frames=33,
            height=default_height,
            width=default_width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ),
    ]

    if not args.quick:
        # Add more comprehensive tests
        test_configs.extend([
            # Standard duration tests
            VideoConfig(
                prompt=(
                    "A beautiful sunset over the ocean with "
                    "waves crashing on the shore"),
                num_frames=121,  # ~5 seconds
                height=default_height,
                width=default_width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ),
            VideoConfig(
                prompt=(
                    "A jazz band playing live music in a "
                    "cozy club, saxophone solo"),
                num_frames=241,  # ~10 seconds
                height=default_height,
                width=default_width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ),
            # Max duration test
            VideoConfig(
                prompt=(
                    "A jazz band playing live music in a "
                    "cozy club, saxophone solo, drums, "
                    "piano"),
                num_frames=481,  # ~20 seconds
                height=default_height,
                width=default_width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ),
        ])

    # Override with custom config if provided
    if args.prompt or args.frames:
        test_configs = [
            VideoConfig(
                prompt=(
                    args.prompt
                    or "A beautiful nature scene"),
                num_frames=args.frames or 121,
                height=default_height,
                width=default_width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            )
        ]

    # Print header
    model_type = "Distilled" if is_distilled else "Dev"
    pipeline_name = (
        "LTX23DistilledPipeline"
        if is_distilled
        else "LTX2Pipeline"
    )

    print("\n" + "=" * 70)
    print(
        f"LTX-2.3 {model_type} Video Generation Tests")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Pipeline: {pipeline_name}")
    if is_distilled:
        print(
            "  Stage 1: 8 steps at half resolution "
            "(DISTILLED_SIGMA_VALUES)")
        print(
            "  Upsample: 2x spatial via "
            "LatentUpsampler")
        print(
            "  Stage 2: 3 steps at full resolution "
            "(STAGE_2_DISTILLED_SIGMA_VALUES)")
        print("Guidance: None (guidance_scale=1.0)")
    else:
        print(
            f"  Steps: {args.steps} with CFG "
            f"(guidance_scale={args.guidance_scale})")
        print(
            "  Multi-modal guidance: text CFG + "
            "modality isolation + STG")
        print(
            "  4 forward passes per denoising step")
    print(
        f"Configuration: TP={args.tp_size}, "
        "FSDP Inference=True")
    print(f"Number of tests: {len(test_configs)}")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(
        1 for r in results if r["status"] == "passed")
    failed = sum(
        1 for r in results if r["status"] == "failed")

    for r in results:
        cfg = r["config"]
        status = (
            "✅" if r["status"] == "passed" else "❌")
        if r["status"] == "passed":
            print(
                f"{status} {cfg['num_frames']} frames "
                f"@ {cfg['height']}x{cfg['width']}: "
                f"{r['generation_time_seconds']:.2f}s, "
                f"{r['peak_memory_mb']} MB")
        else:
            print(
                f"{status} {cfg['num_frames']} frames "
                f"@ {cfg['height']}x{cfg['width']}: "
                f"FAILED - {r['error']}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
