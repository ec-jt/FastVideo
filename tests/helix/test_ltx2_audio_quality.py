# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 Multi-Prompt Audio Quality Test

Tests the distilled two-stage pipeline across diverse prompts
covering different audio scenarios (music, nature, dialogue,
action, etc.) to evaluate audio content quality after the V2
feature extractor normalization fix.

Usage:
    # Run all prompts (distilled two-stage, ~10s each):
    python tests/helix/test_ltx2_audio_quality.py \
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
        --num-gpus 8 --tp-size 8

    # Run with prompt enhancement:
    python tests/helix/test_ltx2_audio_quality.py \
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers \
        --num-gpus 8 --tp-size 8 --enhance-prompt
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


# ── Test Prompts ──────────────────────────────────────────────

TEST_PROMPTS = [
    {
        "id": "jazz_club",
        "category": "music",
        "prompt": (
            "A warm, intimate jazz club with dim amber lighting. "
            "A saxophone player in a dark suit stands center stage, "
            "eyes closed, playing a smooth melodic solo. Behind him, "
            "a pianist gently comps chords on a grand piano while a "
            "drummer brushes the snare with wire brushes. The rich "
            "sound of the saxophone fills the room, accompanied by "
            "soft piano chords and the gentle swish of brushes on "
            "the snare. The camera slowly dollies in from a medium "
            "shot to a close-up of the saxophonist's fingers moving "
            "over the keys."
        ),
    },
    {
        "id": "forest_stream",
        "category": "nature",
        "prompt": (
            "A serene forest clearing at golden hour. Tall pine "
            "trees surround a small stream that trickles over smooth "
            "rocks. Birds sing in the canopy above, a robin's clear "
            "whistle followed by a woodpecker's rhythmic tapping. "
            "The gentle sound of water flowing over stones mixes "
            "with rustling leaves as a light breeze passes through. "
            "A deer steps cautiously into the clearing, its hooves "
            "crunching softly on dry leaves. The camera holds a "
            "wide establishing shot, slowly panning right to follow "
            "the deer."
        ),
    },
    {
        "id": "city_street",
        "category": "urban",
        "prompt": (
            "A busy New York City intersection at midday. Yellow "
            "taxis honk as they navigate through traffic. Pedestrians "
            "cross the street in a hurried pace, their footsteps "
            "creating a constant rhythm on the asphalt. A street "
            "musician plays acoustic guitar on the corner, strumming "
            "a folk melody that cuts through the urban noise. The "
            "camera tracks a woman in a red coat as she walks past "
            "the musician, pausing briefly to listen before "
            "continuing. Car engines rumble, a distant siren wails, "
            "and the guitarist's clear notes ring out."
        ),
    },
    {
        "id": "kitchen_dialogue",
        "category": "dialogue",
        "prompt": (
            "A cozy kitchen with warm morning light streaming "
            "through the window. A woman in her 30s with short "
            "brown hair sits at a wooden table, holding a coffee "
            "mug. She looks up and says in a cheerful voice, "
            "'Good morning! Did you sleep well?' A man enters the "
            "frame from the right, yawning and stretching. He "
            "replies in a groggy voice, 'Not really, the neighbor's "
            "dog was barking all night.' The sound of coffee being "
            "poured fills the background as the woman stands and "
            "walks to the counter. Soft ambient kitchen sounds, "
            "the hum of a refrigerator, a clock ticking on the wall."
        ),
    },
    {
        "id": "desert_explosion",
        "category": "action",
        "prompt": (
            "A dusty desert road stretches into the distance under "
            "a harsh midday sun. A black SUV races toward the camera "
            "at high speed, kicking up a massive cloud of dust. The "
            "roar of the engine grows louder as it approaches. "
            "Suddenly, the vehicle swerves hard to the left as an "
            "explosion erupts on the road ahead, a thunderous boom "
            "followed by debris flying through the air. The camera "
            "shakes from the blast wave. The SUV skids to a stop, "
            "tires screeching on gravel. Smoke and dust billow "
            "across the frame as the engine idles."
        ),
    },
    {
        "id": "singer_spotlight",
        "category": "vocal",
        "prompt": (
            "A young woman with long dark hair stands alone on a "
            "dimly lit stage, a single spotlight illuminating her "
            "face. She holds a microphone close and begins singing "
            "softly in a clear soprano voice, 'Somewhere over the "
            "rainbow, way up high.' Her voice echoes gently in the "
            "empty concert hall. The camera slowly pushes in from a "
            "medium shot as she closes her eyes and the melody "
            "builds. Faint reverb fills the space. She sways "
            "slightly, her voice growing stronger with each phrase."
        ),
    },
    {
        "id": "factory_floor",
        "category": "industrial",
        "prompt": (
            "Inside a large industrial factory, heavy machinery "
            "operates in a rhythmic pattern. Metal presses stamp "
            "down with loud clanging sounds at regular intervals. "
            "Sparks fly from a welding station in the background, "
            "creating bright orange flashes. A worker in a hard hat "
            "and safety goggles walks along the production line, "
            "inspecting parts. The constant hum of conveyor belts "
            "mixes with the sharp hiss of pneumatic tools. The "
            "camera tracks the worker from a low angle as "
            "fluorescent lights flicker overhead."
        ),
    },
    {
        "id": "ocean_cliffs",
        "category": "water",
        "prompt": (
            "A dramatic coastal cliff at sunset. Massive ocean "
            "waves crash against the dark rocks below, sending "
            "white spray high into the air. The deep rumble of the "
            "ocean fills the soundscape, punctuated by the sharp "
            "crack of each wave impact. Seagulls cry overhead, "
            "circling in the golden light. The camera holds a wide "
            "shot from the cliff edge, looking down at the churning "
            "water. Wind whistles past, and the distant sound of a "
            "foghorn echoes across the bay."
        ),
    },
]


@dataclass
class TestResult:
    prompt_id: str
    category: str
    prompt: str
    enhanced_prompt: str | None
    status: str
    generation_time_seconds: float | None
    peak_memory_mb: int | None
    output_path: str | None
    error: str | None = None


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


def _is_distilled(model_path: str) -> bool:
    """Detect distilled model from path name."""
    return "distilled" in model_path.lower()


def run_audio_quality_tests(
    model_path: str,
    num_gpus: int = 8,
    tp_size: int = 8,
    num_frames: int = 241,
    height: int = 512,
    width: int = 768,
    seed: int = 42,
    enhance_prompt: bool = False,
    output_dir: str = "outputs/audio_quality",
    prompt_ids: list[str] | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> list[TestResult]:
    """Run test prompts and collect results.

    Args:
        prompt_ids: If provided, only run prompts with these IDs.
            Example: ``["singer_spotlight", "jazz_club"]``
    """
    from fastvideo import VideoGenerator

    os.makedirs(output_dir, exist_ok=True)

    # Filter prompts if specific IDs requested
    prompts_to_run = TEST_PROMPTS
    if prompt_ids:
        prompts_to_run = [
            p for p in TEST_PROMPTS if p["id"] in prompt_ids
        ]
        if not prompts_to_run:
            available = [p["id"] for p in TEST_PROMPTS]
            raise ValueError(
                f"No prompts matched IDs {prompt_ids}. "
                f"Available: {available}"
            )

    # Auto-detect model type and set defaults
    is_distilled = _is_distilled(model_path)
    if num_inference_steps is None:
        num_inference_steps = 8 if is_distilled else 30
    if guidance_scale is None:
        guidance_scale = 1.0 if is_distilled else 3.0
    model_type = "Distilled" if is_distilled else "Dev"

    print("\n" + "=" * 70)
    print("LTX-2.3 Multi-Prompt Audio Quality Test")
    print("=" * 70)
    print(f"Model: {model_path} ({model_type})")
    print(f"Frames: {num_frames} (~{num_frames/24:.1f}s)")
    print(f"Resolution: {height}x{width}")
    print(f"Steps: {num_inference_steps}, "
          f"Guidance: {guidance_scale}")
    print(f"Enhance prompt: {enhance_prompt}")
    print(f"Prompts: {len(prompts_to_run)}")
    if prompt_ids:
        print(f"  Filter: {prompt_ids}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # If enhance_prompt is requested, load Gemma FIRST (before
    # the video generator) to enhance all prompts, then free it
    # to reclaim GPU memory before loading the heavy pipeline.
    enhanced_prompts: dict[str, str] = {}
    if enhance_prompt:
        try:
            from transformers import (
                AutoTokenizer as _AT,
                Gemma3ForConditionalGeneration as _G3,
            )
            gemma_path = os.path.join(
                model_path, "text_encoder", "gemma",
            )
            if not os.path.isdir(gemma_path):
                raise FileNotFoundError(
                    f"Gemma model not found at {gemma_path}"
                )
            print(f"\nLoading Gemma for prompt enhancement "
                  f"from {gemma_path}...")
            _tokenizer = _AT.from_pretrained(
                gemma_path, local_files_only=True,
            )
            _model = _G3.from_pretrained(
                gemma_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            ).to("cuda").eval()
            sys_prompt_path = (
                Path(__file__).resolve().parents[2]
                / "assets" / "prompts"
                / "gemma_t2v_system_prompt.txt"
            )
            with open(sys_prompt_path, encoding="utf-8") as f:
                _sys_prompt = f.read()
            print("Gemma loaded ✅ — enhancing prompts...\n")

            for test in prompts_to_run:
                pid = test["id"]
                raw_prompt = test["prompt"]
                print(f"Enhancing [{pid}]...")
                messages = [
                    {"role": "system", "content": _sys_prompt},
                    {"role": "user",
                     "content": f"user prompt: {raw_prompt}"},
                ]
                inputs = _tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(_model.device)
                torch.manual_seed(seed)
                outputs = _model.generate(
                    inputs,
                    max_new_tokens=512,
                    do_sample=True,
                )
                raw = _tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                # Clean: remove leading non-alpha chars
                for ci, ch in enumerate(raw):
                    if ch.isalpha():
                        raw = raw[ci:]
                        break
                enhanced_prompts[pid] = raw
                print(f"\n{'='*60}")
                print(f"ENHANCED PROMPT [{pid}] "
                      f"({len(raw)} chars):")
                print(f"{'='*60}")
                print(raw)
                print(f"{'='*60}\n")

            # Free Gemma to reclaim GPU memory
            del _model, _tokenizer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("Gemma freed — loading video pipeline...\n")

        except Exception as e:
            print(
                f"⚠️  Could not enhance prompts: {e}. "
                "Running with original prompts."
            )
            import traceback
            traceback.print_exc()
            enhance_prompt = False

    # Load video generator
    print("Loading model...")
    load_start = time.time()
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=1,
        use_fsdp_inference=True,
        dit_layerwise_offload=False,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    results: list[TestResult] = []

    for i, test in enumerate(prompts_to_run):
        prompt_id = test["id"]
        category = test["category"]
        prompt = test["prompt"]
        enhanced = None

        print(f"\n{'─' * 70}")
        print(
            f"[{i+1}/{len(prompts_to_run)}] "
            f"{prompt_id} ({category})"
        )
        print(f"Prompt: {prompt[:80]}...")

        # Use pre-enhanced prompt if available
        if prompt_id in enhanced_prompts:
            enhanced = enhanced_prompts[prompt_id]
            print(f"Using enhanced prompt ({len(enhanced)} chars)")
            prompt = enhanced

        try:
            gen_start = time.time()
            video = generator.generate_video(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=seed,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            gen_time = time.time() - gen_start
            peak_mem = get_peak_memory()

            # The output path is auto-generated by VideoGenerator
            # Find the most recent file in outputs/
            output_files = sorted(
                Path("outputs").glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            output_path = (
                str(output_files[0]) if output_files else None
            )

            # Copy to our organized output dir
            if output_path:
                suffix = "_enhanced" if enhance_prompt else ""
                dest = os.path.join(
                    output_dir,
                    f"{prompt_id}{suffix}.mp4",
                )
                import shutil
                shutil.copy2(output_path, dest)
                output_path = dest

            result = TestResult(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt,
                enhanced_prompt=enhanced,
                status="passed",
                generation_time_seconds=gen_time,
                peak_memory_mb=peak_mem,
                output_path=output_path,
            )
            print(
                f"✅ {prompt_id}: {gen_time:.1f}s, "
                f"{peak_mem} MB → {output_path}"
            )

        except Exception as e:
            result = TestResult(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt,
                enhanced_prompt=enhanced,
                status="failed",
                generation_time_seconds=None,
                peak_memory_mb=None,
                output_path=None,
                error=str(e),
            )
            print(f"❌ {prompt_id}: {e}")
            import traceback
            traceback.print_exc()

        results.append(result)

    # Shutdown
    generator.shutdown()

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    return results


def print_summary(results: list[TestResult]) -> None:
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")

    for r in results:
        status = "✅" if r.status == "passed" else "❌"
        if r.status == "passed":
            print(
                f"{status} {r.prompt_id:20s} "
                f"({r.category:10s}): "
                f"{r.generation_time_seconds:.1f}s, "
                f"{r.peak_memory_mb} MB"
            )
        else:
            print(
                f"{status} {r.prompt_id:20s} "
                f"({r.category:10s}): "
                f"FAILED - {r.error}"
            )

    print(f"\nTotal: {passed} passed, {failed} failed")
    if passed > 0:
        avg_time = sum(
            r.generation_time_seconds
            for r in results
            if r.status == "passed"
        ) / passed
        print(f"Average generation time: {avg_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2.3 Multi-Prompt Audio Quality Test",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=(
            "/mnt/nvme0/models/FastVideo/"
            "LTX2.3-Distilled-Diffusers"
        ),
        help="Path to model",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
    )
    parser.add_argument(
        "--tp-size", type=int, default=8,
    )
    parser.add_argument(
        "--frames", type=int, default=241,
        help="Number of frames (~10s at 24fps)",
    )
    parser.add_argument(
        "--height", type=int, default=512,
    )
    parser.add_argument(
        "--width", type=int, default=768,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Use Gemma to enhance prompts before generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/audio_quality",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Run only specific prompt IDs. "
            "Example: --prompt-id singer_spotlight jazz_club"
        ),
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help=(
            "Run a single custom prompt instead of the "
            "built-in test prompts. "
            "Example: --custom-prompt 'A cat playing piano'"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of inference steps. "
            "Default: 8 for distilled, 30 for dev."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Guidance scale. "
            "Default: 1.0 for distilled, 3.0 for dev."
        ),
    )
    args = parser.parse_args()

    # If custom prompt provided, inject it as a test prompt
    prompt_ids = args.prompt_id
    if args.custom_prompt:
        TEST_PROMPTS.insert(0, {
            "id": "custom",
            "category": "custom",
            "prompt": args.custom_prompt,
        })
        prompt_ids = ["custom"]

    results = run_audio_quality_tests(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        enhance_prompt=args.enhance_prompt,
        output_dir=args.output_dir,
        prompt_ids=prompt_ids,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    print_summary(results)
    failed = sum(1 for r in results if r.status == "failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
