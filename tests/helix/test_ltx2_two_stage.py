# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 Two-Stage Pipeline Test

Tests the two-stage pipeline (CFG Stage 1 + distilled Stage 2)
using the LTX-2.3 Dev model with 8 GPUs and FSDP.

Resolution convention (matches upstream TI2VidTwoStagesPipeline
from LTX-2/packages/ltx-pipelines/src/ltx_pipelines/
ti2vid_two_stages.py):

  --height / --width specify the **full (Stage 2)** resolution.
  Stage 1 automatically runs at **half** that resolution.

  Example: --height 1024 --width 1536
    Stage 1: 512×768  (CFG guided, 30 steps)
    Stage 2: 1024×1536 (distilled LoRA, 3 steps)

Usage:
    python tests/helix/test_ltx2_two_stage.py \\
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Dev-Diffusers \\
        --num-gpus 8 --tp-size 8 --frames 121 \\
        --height 1024 --width 1536
"""

import argparse
import os
import time

import torch


def get_peak_memory() -> int:
    """Get peak GPU memory usage in MB."""
    import subprocess
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    mem_values = [
        int(x.strip())
        for x in result.stdout.strip().split("\n")
    ]
    return max(mem_values)


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2.3 Two-Stage Pipeline Test",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=(
            "/mnt/nvme0/models/FastVideo/"
            "LTX2.3-Dev-Diffusers"
        ),
        help="Path to LTX-2.3 Dev model",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
    )
    parser.add_argument(
        "--tp-size", type=int, default=8,
    )
    parser.add_argument(
        "--frames", type=int, default=121,
        help="Number of frames (~5s at 24fps)",
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help=(
            "Full (Stage 2) height. Stage 1 = half. "
            "Default 1024 → Stage 1 at 512."
        ),
    )
    parser.add_argument(
        "--width", type=int, default=1536,
        help=(
            "Full (Stage 2) width. Stage 1 = half. "
            "Default 1536 → Stage 1 at 768."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of inference steps for Stage 1",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=3.0,
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=(
            "A young woman begins singing alone on a dark "
            "concert stage, standing at center under a "
            "single warm spotlight while holding a vintage "
            "microphone on a stand. She slowly raises the "
            "microphone closer to her lips and sings softly "
            "in a clear soprano voice, gently swaying from "
            "side to side as she breathes between lines."
        ),
        help="Prompt for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/two_stage_test",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1 resolution is half of user-specified (full)
    s1_height = args.height // 2
    s1_width = args.width // 2

    print("\n" + "=" * 70)
    print("LTX-2.3 Two-Stage Pipeline Test")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Pipeline: LTX23TwoStagePipeline")
    print(f"  Stage 1: {s1_height}x{s1_width}, "
          f"{args.steps} steps with CFG "
          f"(guidance_scale={args.guidance_scale})")
    print(f"  Upsample: 2x spatial via LatentUpsampler")
    print(f"  Stage 2: {args.height}x{args.width}, "
          f"3 steps simple (distilled LoRA)")
    print(f"Frames: {args.frames} (~{args.frames/24:.1f}s)")
    print(f"GPUs: {args.num_gpus}, TP: {args.tp_size}")
    print("=" * 70)

    from fastvideo import VideoGenerator

    print("\nLoading model...")
    load_start = time.time()
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_size=1,
        use_fsdp_inference=True,
        dit_layerwise_offload=False,
        override_pipeline_cls_name="LTX23TwoStagePipeline",
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # The user-specified height/width is the full (Stage 2)
    # resolution.  The pipeline internally halves it for
    # Stage 1, then upsamples back to full for Stage 2.
    print(f"\nGenerating video...")
    print(f"Prompt: {args.custom_prompt[:80]}...")
    gen_start = time.time()
    video = generator.generate_video(
        prompt=args.custom_prompt,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    )
    gen_time = time.time() - gen_start
    peak_mem = get_peak_memory()

    print(f"\n{'=' * 70}")
    print(f"✅ SUCCESS")
    print(f"  Stage 1: {s1_height}x{s1_width} (half-res)")
    print(f"  Stage 2: {args.height}x{args.width} (output)")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Peak memory: {peak_mem} MB")
    print(f"{'=' * 70}")

    generator.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
