# SPDX-License-Identifier: Apache-2.0
"""
Latent spatial upsampler for two-stage video generation pipelines.

Ported from LTX-2 reference implementation:
  ltx_core/model/upsampler/model.py

The upsampler operates in latent space, performing 2x spatial
upsampling via a learned convolutional network with residual blocks
and pixel-shuffle upsampling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────


class PixelShuffleND(nn.Module):
    """N-dimensional pixel shuffle for upsampling tensors.

    Args:
        dims: Number of spatial dimensions to shuffle (2 or 3).
        upscale_factors: Per-dimension upscale factors.
    """

    def __init__(
        self,
        dims: int,
        upscale_factors: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        assert dims in (1, 2, 3), "dims must be 1, 2, or 3"
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        elif self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        elif self.dims == 1:
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )
        else:
            raise ValueError(f"Unsupported dims: {self.dims}")


class ResBlock(nn.Module):
    """Residual block with two convolutions, group norm, and SiLU.

    Args:
        channels: Number of input/output channels.
        mid_channels: Intermediate channel count (defaults to
            *channels*).
        dims: Convolution dimensionality (2 or 3).
    """

    def __init__(
        self,
        channels: int,
        mid_channels: int | None = None,
        dims: int = 3,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.conv1 = conv(
            channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(
            mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


# ─────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────


class LatentUpsampler(nn.Module):
    """Learned 2× spatial upsampler operating in VAE latent space.

    Architecture:
        initial_conv → GroupNorm → SiLU
        → N ResBlocks
        → Conv2d + PixelShuffle (2× spatial)
        → N ResBlocks
        → final_conv

    For ``dims=3`` the upsampler conv is applied per-frame (the
    temporal axis is untouched).

    Args:
        in_channels: Latent channel count (default 128).
        mid_channels: Hidden channel count (default 512).
        num_blocks_per_stage: ResBlocks before/after upsampling.
        dims: Convolution dimensionality (2 or 3).
        spatial_upsample: Enable spatial 2× upsampling.
        temporal_upsample: Enable temporal 2× upsampling.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.initial_conv = conv(
            in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims)
             for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(
                    mid_channels, 8 * mid_channels,
                    kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            # Spatial-only: use Conv2d + 2D pixel shuffle
            self.upsampler = nn.Sequential(
                nn.Conv2d(
                    mid_channels, 4 * mid_channels,
                    kernel_size=3, padding=1),
                PixelShuffleND(2),
            )
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(
                    mid_channels, 2 * mid_channels,
                    kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError(
                "Either spatial_upsample or temporal_upsample "
                "must be True")

        self.post_upsample_res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims)
             for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(
            mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample latent tensor.

        Args:
            latent: ``[B, C, F, H, W]`` latent tensor.

        Returns:
            Upsampled tensor with doubled spatial dimensions.
        """
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(
                x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # Remove the first frame after temporal
                # upsampling (encodes one pixel frame).
                x = x[:, :, 1:, :, :]
            else:
                # Spatial-only: rearrange to 2D, upsample,
                # rearrange back.
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(
                    x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LatentUpsampler":
        """Load a LatentUpsampler from a safetensors file.

        The file must contain a ``config`` key in its metadata
        with the JSON-serialised constructor arguments.

        Args:
            path: Path to ``.safetensors`` file.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Initialised and weight-loaded ``LatentUpsampler``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Spatial upsampler weights not found: {path}")

        # Read metadata to get config
        import struct
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))

        metadata = header.get("__metadata__", {})
        config_str = metadata.get("config", "{}")
        config = json.loads(config_str)

        # Remove _class_name if present
        config.pop("_class_name", None)
        # Remove spatial_scale and rational_resampler
        # (not needed for our simplified version)
        config.pop("spatial_scale", None)
        config.pop("rational_resampler", None)

        logger.info(
            "Loading LatentUpsampler from %s with config: %s",
            path, config,
        )

        model = cls(**config)

        # Load weights
        state_dict = load_file(str(path), device=str(device))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(dtype=dtype, device=device)
        model.eval()

        logger.info(
            "LatentUpsampler loaded: %d parameters",
            sum(p.numel() for p in model.parameters()),
        )
        return model


# ─────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────


def upsample_video_latent(
    latent: torch.Tensor,
    upsampler: LatentUpsampler,
    per_channel_statistics: nn.Module,
) -> torch.Tensor:
    """Apply spatial upsampling with proper normalization.

    The reference implementation un-normalizes the latent using
    the VAE encoder's per-channel statistics, runs the upsampler,
    then re-normalizes.  This function replicates that flow using
    the ``per_channel_statistics`` module from the VAE.

    Args:
        latent: Normalized latent ``[B, C, F, H, W]``.
        upsampler: ``LatentUpsampler`` instance.
        per_channel_statistics: Module with ``un_normalize`` and
            ``normalize`` methods (from the VAE encoder/decoder).

    Returns:
        Upsampled and re-normalized latent.
    """
    latent = per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = per_channel_statistics.normalize(latent)
    return latent
