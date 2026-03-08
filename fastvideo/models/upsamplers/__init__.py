# SPDX-License-Identifier: Apache-2.0
"""Latent upsampler models for two-stage pipelines."""

from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler,
    upsample_video_latent,
)

__all__ = [
    "LatentUpsampler",
    "upsample_video_latent",
]
