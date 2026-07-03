# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.models.dits.lingbotworld import (
    CausalLingBotWorldVideoConfig, LingBotWorldVideoConfig)
from fastvideo.configs.pipelines.wan import (Wan2_2_I2V_A14B_Config,
                                             WanT2V480PConfig)


@dataclass
class LingBotWorldI2V480PConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947


@dataclass
class LingBotWorldFastI2VConfig(WanT2V480PConfig):
    """Config for FastVideo/LingBot-World-Fast-Diffusers.

    Causal DMD-distilled I2V world model. Single transformer
    (transformer_2 is null in model_index.json), UMT5 text encoder,
    Wan2.1 16-channel VAE.

    The official reference (generate_fast.py) samples with 4 DMD
    timesteps picked from a shift-5.0 flow-match schedule at indices
    [0, 179, 358, 679], which corresponds to
    dmd_denoising_steps=[1000, 821, 642, 321] with
    warp_denoising_step=True.
    """

    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 821, 642, 321])
    warp_denoising_step: bool = True
    dit_config: DiTConfig = field(
        default_factory=CausalLingBotWorldVideoConfig)

    def __post_init__(self) -> None:
        # Need the VAE encoder for image conditioning and the decoder
        # for final latent decode.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
