# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.sample.base import SamplingParam

# Default negative prompt shared across LTX-2 and LTX-2.3 models.
DEFAULT_LTX_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, "
    "washed out colors, excessive noise, grainy texture, poor lighting, "
    "flickering, motion blur, distorted proportions, unnatural skin "
    "tones, deformed facial features, asymmetrical face, missing facial "
    "features, extra limbs, disfigured hands, wrong hand count, "
    "artifacts around text, inconsistent perspective, camera shake, "
    "incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect, incorrect ethnicity, "
    "wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect "
    "dialogue, added dialogue, repetitive speech, jittery movement, "
    "awkward pauses, incorrect timing, unnatural transitions, "
    "inconsistent framing, tilted camera, flat lighting, inconsistent "
    "tone, cinematic oversaturation, stylized filters, or AI artifacts.")


@dataclass
class LTX2BaseSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2 base one-stage T2V.

    Values follow the official LTX-2 one-stage defaults.
    Multi-modal CFG params are read by ``LTX2DenoisingStage``.
    """

    seed: int = 10
    num_frames: int = 121
    height: int = 512
    width: int = 768
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    # Copied/following official LTX-2 DEFAULT_NEGATIVE_PROMPT.
    negative_prompt: str = DEFAULT_LTX_NEGATIVE_PROMPT
    # Official LTX-2 multi-modal CFG defaults.
    ltx2_cfg_scale_video: float = 3.0
    ltx2_cfg_scale_audio: float = 7.0
    ltx2_modality_scale_video: float = 3.0
    ltx2_modality_scale_audio: float = 3.0
    ltx2_rescale_scale: float = 0.7
    # STG (Spatio-Temporal Guidance) defaults from official LTX-2.
    ltx2_stg_scale_video: float = 1.0
    ltx2_stg_scale_audio: float = 1.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=lambda: [29])
    ltx2_stg_blocks_audio: list[int] = field(default_factory=lambda: [29])


@dataclass
class LTX2DistilledSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2 distilled one-stage T2V."""

    seed: int = 10
    num_frames: int = 121
    height: int = 1024
    width: int = 1536
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    # No default negative_prompt for distilled models
    negative_prompt: str = ""


# =============================================================================
# LTX-2.3 Sampling Parameters
# =============================================================================
# Based on official LTX-2.3 constants from:
# https://github.com/ec-jt/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py


@dataclass
class LTX23BaseSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2.3 dev/base one-stage T2V.

    Values follow the official LTX-2.3 defaults which differ from LTX-2:
    - num_inference_steps: 30 (vs 40 for LTX-2)
    - stg_blocks: [28] (vs [29] for LTX-2)
    """

    seed: int = 10
    num_frames: int = 121
    height: int = 512
    width: int = 768
    fps: int = 24
    num_inference_steps: int = 30  # LTX-2.3 uses 30 steps (vs 40 for LTX-2)
    guidance_scale: float = 3.0
    negative_prompt: str = DEFAULT_LTX_NEGATIVE_PROMPT
    # Official LTX-2.3 multi-modal CFG defaults.
    ltx2_cfg_scale_video: float = 3.0
    ltx2_cfg_scale_audio: float = 7.0
    ltx2_modality_scale_video: float = 3.0
    ltx2_modality_scale_audio: float = 3.0
    ltx2_rescale_scale: float = 0.7
    # STG (Spatio-Temporal Guidance) defaults from official LTX-2.3.
    # Note: LTX-2.3 uses block 28 instead of 29.
    ltx2_stg_scale_video: float = 1.0
    ltx2_stg_scale_audio: float = 1.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=lambda: [28])
    ltx2_stg_blocks_audio: list[int] = field(default_factory=lambda: [28])


@dataclass
class LTX23DistilledSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2.3 distilled one-stage T2V.

    Uses 8 denoising steps with CFG=1.0 for fast inference.
    """

    seed: int = 10
    num_frames: int = 121
    height: int = 1024
    width: int = 1536
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    negative_prompt: str = ""


@dataclass
class LTX23HQSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2.3 HQ two-stage pipeline.

    Used for high-quality generation with spatial upscaling.
    Stage 1 generates at half resolution, then upscaled in stage 2.
    """

    seed: int = 10
    num_frames: int = 121
    # Stage 1 resolution (half of final 1920x1088)
    height: int = 544  # 1088 // 2
    width: int = 960   # 1920 // 2
    fps: int = 24
    num_inference_steps: int = 15
    guidance_scale: float = 3.0
    negative_prompt: str = DEFAULT_LTX_NEGATIVE_PROMPT
    # HQ pipeline uses different rescale values
    ltx2_cfg_scale_video: float = 3.0
    ltx2_cfg_scale_audio: float = 7.0
    ltx2_modality_scale_video: float = 3.0
    ltx2_modality_scale_audio: float = 3.0
    ltx2_rescale_scale: float = 0.45  # Lower rescale for HQ
    # STG disabled for HQ pipeline
    ltx2_stg_scale_video: float = 0.0
    ltx2_stg_scale_audio: float = 0.0
    ltx2_stg_blocks_video: list[int] = field(default_factory=list)
    ltx2_stg_blocks_audio: list[int] = field(default_factory=list)


@dataclass
class LTX23FP8SamplingParam(LTX23BaseSamplingParam):
    """Default sampling parameters for LTX-2.3 FP8 quantized model.

    Inherits from LTX23BaseSamplingParam with same defaults.
    FP8 quantization is handled at the model loading level.
    """
    pass


# Backward compatibility aliases.
LTX2SamplingParam = LTX2DistilledSamplingParam
LTX23SamplingParam = LTX23BaseSamplingParam
