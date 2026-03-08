#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode saved latent tensors to video files for debugging.

Usage:
    python scripts/debug/decode_latents.py \
        --debug-dir /tmp/ltx2_debug \
        --model-path /mnt/nvme0/models/FastVideo/LTX2.3-Distilled-Diffusers

This reads .pt files saved by the denoising stage when
LTX2_DEBUG_DIR is set, decodes them with the VAE, and
saves as .mp4 files.
"""

import argparse
import glob
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Decode latent .pt files to video")
    parser.add_argument(
        "--debug-dir", type=str, required=True,
        help="Directory containing *_latent.pt files")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model directory (for VAE)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as debug-dir)")
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Output video FPS")
    args = parser.parse_args()

    output_dir = args.output_dir or args.debug_dir
    os.makedirs(output_dir, exist_ok=True)

    # Find all latent files
    pt_files = sorted(
        glob.glob(os.path.join(args.debug_dir, "*_latent.pt")))
    if not pt_files:
        print("No *_latent.pt files found in %s" % args.debug_dir)
        return

    print("Found %d latent files:" % len(pt_files))
    for f in pt_files:
        data = torch.load(f, map_location="cpu")
        vl = data.get("video_latent")
        al = data.get("audio_latent")
        print("  %s: video=%s audio=%s" % (
            os.path.basename(f),
            tuple(vl.shape) if vl is not None else None,
            tuple(al.shape) if al is not None else None,
        ))

    # Load VAE
    print("\nLoading VAE from %s..." % args.model_path)
    vae_path = os.path.join(args.model_path, "vae")

    from fastvideo.models.loader.component_loader import (
        PipelineComponentLoader)
    from fastvideo.fastvideo_args import FastVideoArgs

    # Minimal args for loading
    fastvideo_args = FastVideoArgs(
        model=args.model_path,
        num_gpus=1,
    )

    vae = PipelineComponentLoader.load_module(
        module_name="vae",
        component_model_path=vae_path,
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    )
    vae = vae.to("cuda", dtype=torch.bfloat16)
    vae.eval()
    print("VAE loaded")

    # Decode each latent
    for pt_file in pt_files:
        label = os.path.basename(pt_file).replace(
            "_latent.pt", "")
        data = torch.load(pt_file, map_location="cpu")
        video_latent = data.get("video_latent")
        if video_latent is None:
            print("Skipping %s (no video_latent)" % label)
            continue

        print("\nDecoding %s: shape=%s" % (
            label, tuple(video_latent.shape)))
        video_latent = video_latent.to(
            "cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            decoded = vae.decode(video_latent)

        # Convert to uint8 video
        decoded = decoded.float().clamp(-1, 1)
        decoded = ((decoded + 1) / 2 * 255).to(torch.uint8)
        # Shape: [B, C, F, H, W] -> [F, H, W, C]
        video = decoded[0].permute(1, 2, 3, 0).cpu().numpy()

        # Save as mp4
        out_path = os.path.join(output_dir, "%s.mp4" % label)
        try:
            import imageio
            writer = imageio.get_writer(
                out_path, fps=args.fps, codec="libx264")
            for frame in video:
                writer.append_data(frame)
            writer.close()
            print("  Saved: %s (%d frames, %dx%d)" % (
                out_path, video.shape[0],
                video.shape[2], video.shape[1]))
        except ImportError:
            # Fallback: save as individual frames
            frame_dir = os.path.join(
                output_dir, "%s_frames" % label)
            os.makedirs(frame_dir, exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(video):
                img = Image.fromarray(frame)
                img.save(os.path.join(
                    frame_dir, "frame_%04d.png" % i))
            print("  Saved %d frames to %s" % (
                len(video), frame_dir))


if __name__ == "__main__":
    main()
