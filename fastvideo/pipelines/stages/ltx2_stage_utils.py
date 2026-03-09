# SPDX-License-Identifier: Apache-2.0
"""
Shared utility functions for LTX-2.3 multi-stage pipelines.

These are standalone functions extracted from the existing
``LTX2DistilledDenoisingStage`` and ``LTX2DenoisingStage`` so
that new pipelines can reuse them without modifying existing
working code.
"""

from __future__ import annotations

import os

import torch

from fastvideo.logger import init_logger
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler, upsample_video_latent)

logger = init_logger(__name__)

# Debug output directory (set LTX2_DEBUG_DIR env var to enable)
_DEBUG_DIR = os.environ.get("LTX2_DEBUG_DIR", "")


def run_spatial_upsample(
    half_latents: torch.Tensor,
    spatial_upsampler: LatentUpsampler,
    per_channel_statistics: torch.Tensor,
    target_h: int,
    target_w: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """Upsample latents 2× spatially, managing GPU memory.

    Moves the upsampler to GPU, runs the upsample, then offloads
    back to CPU to free VRAM for the next stage.

    Args:
        half_latents: Latents at half spatial resolution.
        spatial_upsampler: The ``LatentUpsampler`` model.
        per_channel_statistics: VAE per-channel stats for
            latent normalization.
        target_h: Target latent height (full resolution).
        target_w: Target latent width (full resolution).
        batch_size: Batch size (repeats upsampled latent if > 1).

    Returns:
        Upsampled latents at full spatial resolution.
    """
    device = half_latents.device
    spatial_upsampler.to(device)
    pcs = per_channel_statistics.to(device)

    with torch.no_grad():
        upsampled = upsample_video_latent(
            latent=half_latents[:1],
            upsampler=spatial_upsampler,
            per_channel_statistics=pcs,
        )

    # Offload to CPU to free VRAM
    spatial_upsampler.to("cpu")
    per_channel_statistics.to("cpu")
    torch.cuda.empty_cache()

    if batch_size > 1:
        upsampled = upsampled.repeat(batch_size, 1, 1, 1, 1)

    logger.info(
        "[LTX2] Upsampled latents: %s → %s",
        tuple(half_latents.shape),
        tuple(upsampled.shape),
    )
    return upsampled


def noise_latent_at_sigma(
    clean_latent: torch.Tensor,
    start_sigma: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add noise to a clean latent at a given sigma level.

    Implements: ``x_noisy = (1 - σ) * x_clean + σ * noise``

    Args:
        clean_latent: The clean (denoised) latent tensor.
        start_sigma: The sigma level to noise at.
        generator: Optional RNG for reproducibility.

    Returns:
        Noised latent tensor.
    """
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    sigma = float(start_sigma)
    noised = (
        (1.0 - sigma) * clean_latent.float()
        + sigma * noise.float()
    ).to(clean_latent.dtype)
    return noised


def debug_save_latent(
    latent: torch.Tensor,
    label: str,
    audio_latent: torch.Tensor | None = None,
) -> None:
    """Save latent tensor for debugging when LTX2_DEBUG_DIR is set.

    No-op if the environment variable is not set.
    """
    if not _DEBUG_DIR:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, f"{label}_latent.pt")
    save_dict: dict[str, torch.Tensor] = {
        "video_latent": latent.detach().cpu(),
    }
    if audio_latent is not None:
        save_dict["audio_latent"] = audio_latent.detach().cpu()
    torch.save(save_dict, path)
    logger.info(
        "[DEBUG] Saved %s latent: shape=%s to %s",
        label, tuple(latent.shape), path,
    )
