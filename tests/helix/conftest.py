# SPDX-License-Identifier: Apache-2.0
"""
Pytest fixtures and configurations for Helix parallelism tests.

Model configurations are derived from local models at /mnt/nvme0/models/FastVideo/

IMPORTANT: FastVideo models require FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
The local FastVideo checkpoints use Video Sparse Attention (VSA) architecture with
`to_gate_compress` layers. Without this environment variable, the model will fail
to load with: "ValueError: Parameter blocks.0.to_gate_compress.bias not found"

The VSA backend is set in test_wan_baseline.py and test_wan_helix.py.
"""

import os
from dataclasses import dataclass
from typing import Optional

import pytest

# Ensure VSA backend is set for all tests using this conftest
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

# =============================================================================
# Model Configurations (from local /mnt/nvme0/models/FastVideo/)
# =============================================================================


@dataclass
class TransformerConfig:
    """Transformer model configuration."""

    num_attention_heads: int
    attention_head_dim: int
    num_layers: int
    ffn_dim: int
    in_channels: int
    out_channels: int
    text_dim: int
    freq_dim: int
    patch_size: tuple[int, int, int]
    rope_max_seq_len: int

    @property
    def hidden_size(self) -> int:
        """Calculate hidden size from heads * head_dim."""
        return self.num_attention_heads * self.attention_head_dim


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""

    name: str
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    vocab_size: int


@dataclass
class ModelConfig:
    """Complete model configuration for testing."""

    name: str
    hf_repo: str
    local_path: str
    transformer: TransformerConfig
    text_encoder: TextEncoderConfig
    # Estimated memory in GB (BF16)
    transformer_memory_gb: float
    text_encoder_memory_gb: float
    # Valid TP sizes (must divide num_attention_heads)
    valid_tp_sizes: list[int]
    # Valid SP sizes (must divide num_attention_heads)
    valid_sp_sizes: list[int]


# =============================================================================
# Wan 1.3B Configuration
# From: /mnt/nvme0/models/FastVideo/FastWan2.1-T2V-1.3B-Diffusers/
# =============================================================================
WAN_1_3B_CONFIG = ModelConfig(
    name="Wan-1.3B",
    hf_repo="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    local_path="/mnt/nvme0/models/FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    transformer=TransformerConfig(
        num_attention_heads=12,
        attention_head_dim=128,
        num_layers=30,
        ffn_dim=8960,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        patch_size=(1, 2, 2),
        rope_max_seq_len=1024,
    ),
    text_encoder=TextEncoderConfig(
        name="umt5-xxl",
        d_model=4096,
        num_heads=64,
        num_layers=24,
        d_ff=10240,
        vocab_size=256384,
    ),
    # 1.3B params * 2 bytes (BF16) = ~2.6GB
    transformer_memory_gb=2.6,
    # UMT5-XXL ~4.7B params = ~9.4GB
    text_encoder_memory_gb=9.4,
    # 12 heads: divisible by 1, 2, 3, 4, 6, 12
    valid_tp_sizes=[1, 2, 3, 4, 6],
    valid_sp_sizes=[1, 2, 3, 4, 6],
)

# =============================================================================
# Wan 5B Configuration (TI2V - Text+Image to Video)
# From: /mnt/nvme0/models/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers/
# =============================================================================
WAN_5B_CONFIG = ModelConfig(
    name="Wan-5B",
    hf_repo="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    local_path="/mnt/nvme0/models/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    transformer=TransformerConfig(
        num_attention_heads=24,
        attention_head_dim=128,
        num_layers=30,
        ffn_dim=14336,
        in_channels=48,  # Note: 48 channels for TI2V
        out_channels=48,
        text_dim=4096,
        freq_dim=256,
        patch_size=(1, 2, 2),
        rope_max_seq_len=1024,
    ),
    text_encoder=TextEncoderConfig(
        name="umt5-xxl",
        d_model=4096,
        num_heads=64,
        num_layers=24,
        d_ff=10240,
        vocab_size=256384,
    ),
    # 5B params * 2 bytes (BF16) = ~10GB
    transformer_memory_gb=10.0,
    # UMT5-XXL ~4.7B params = ~9.4GB
    text_encoder_memory_gb=9.4,
    # 24 heads: divisible by 1, 2, 3, 4, 6, 8, 12, 24
    valid_tp_sizes=[1, 2, 3, 4, 6, 8],
    valid_sp_sizes=[1, 2, 3, 4, 6, 8],
)

# =============================================================================
# Wan 14B Configuration
# From: /mnt/nvme0/models/FastVideo/FastWan2.1-T2V-14B-Diffusers/
# =============================================================================
WAN_14B_CONFIG = ModelConfig(
    name="Wan-14B",
    hf_repo="FastVideo/FastWan2.1-T2V-14B-Diffusers",
    local_path="/mnt/nvme0/models/FastVideo/FastWan2.1-T2V-14B-Diffusers",
    transformer=TransformerConfig(
        num_attention_heads=40,
        attention_head_dim=128,
        num_layers=40,
        ffn_dim=13824,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        patch_size=(1, 2, 2),
        rope_max_seq_len=1024,
    ),
    text_encoder=TextEncoderConfig(
        name="umt5-xxl",
        d_model=4096,
        num_heads=64,
        num_layers=24,
        d_ff=10240,
        vocab_size=256384,
    ),
    # 14B params * 2 bytes (BF16) = ~28GB
    transformer_memory_gb=28.0,
    # UMT5-XXL ~4.7B params = ~9.4GB
    text_encoder_memory_gb=9.4,
    # 40 heads: divisible by 1, 2, 4, 5, 8, 10, 20, 40
    valid_tp_sizes=[1, 2, 4, 5, 8],
    valid_sp_sizes=[1, 2, 4, 5, 8],
)

# All model configs for parametrized tests
ALL_MODEL_CONFIGS = [WAN_1_3B_CONFIG, WAN_5B_CONFIG, WAN_14B_CONFIG]


# =============================================================================
# Sequence Length Configurations
# =============================================================================
@dataclass
class SequenceConfig:
    """Video sequence configuration for testing."""

    name: str
    num_frames: int
    height: int
    width: int
    description: str

    @property
    def sequence_length(self) -> int:
        """Calculate sequence length after patchification.

        For Wan models with patch_size=(1, 2, 2):
        seq_len = (frames / 1) * (height / 2) * (width / 2)
                = frames * (height * width / 4)
        """
        return self.num_frames * (self.height // 2) * (self.width // 2)


# Standard test sequence configurations
SEQ_SHORT = SequenceConfig(
    name="short",
    num_frames=17,
    height=480,
    width=832,
    description="17 frames @ 480x832 - Quick smoke test",
)

SEQ_MEDIUM = SequenceConfig(
    name="medium",
    num_frames=45,
    height=480,
    width=832,
    description="45 frames @ 480x832 - Standard generation",
)

SEQ_LONG = SequenceConfig(
    name="long",
    num_frames=81,
    height=480,
    width=832,
    description="81 frames @ 480x832 - Long video stress test",
)

SEQ_HIGH_RES = SequenceConfig(
    name="high_res",
    num_frames=17,
    height=720,
    width=1280,
    description="17 frames @ 720p - High resolution test",
)

ALL_SEQUENCE_CONFIGS = [SEQ_SHORT, SEQ_MEDIUM, SEQ_LONG, SEQ_HIGH_RES]


# =============================================================================
# Parallelism Configurations
# =============================================================================
@dataclass
class ParallelismConfig:
    """Parallelism configuration for testing."""

    name: str
    tp_size: int
    sp_size: int
    cp_size: int  # Context parallelism (for Helix)
    total_gpus: int
    description: str

    @property
    def is_helix(self) -> bool:
        """Check if this is a Helix configuration (TP + CP)."""
        return self.tp_size > 1 and self.cp_size > 1


# Baseline SP-only configurations (current FastVideo behavior)
BASELINE_SP_CONFIGS = [
    ParallelismConfig(
        name="sp8",
        tp_size=1,
        sp_size=8,
        cp_size=1,
        total_gpus=8,
        description="Pure SP - Full weights on each GPU",
    ),
    ParallelismConfig(
        name="sp4",
        tp_size=1,
        sp_size=4,
        cp_size=1,
        total_gpus=4,
        description="SP=4 - Full weights on each GPU",
    ),
    ParallelismConfig(
        name="sp2",
        tp_size=1,
        sp_size=2,
        cp_size=1,
        total_gpus=2,
        description="SP=2 - Full weights on each GPU",
    ),
]

# TP-only configurations (for comparison)
TP_ONLY_CONFIGS = [
    ParallelismConfig(
        name="tp2",
        tp_size=2,
        sp_size=1,
        cp_size=1,
        total_gpus=2,
        description="TP=2 - Weights split across 2 GPUs",
    ),
    ParallelismConfig(
        name="tp4",
        tp_size=4,
        sp_size=1,
        cp_size=1,
        total_gpus=4,
        description="TP=4 - Weights split across 4 GPUs",
    ),
    ParallelismConfig(
        name="tp8",
        tp_size=8,
        sp_size=1,
        cp_size=1,
        total_gpus=8,
        description="TP=8 - Weights split across 8 GPUs",
    ),
]

# Helix configurations (TP + CP combined)
HELIX_CONFIGS = [
    ParallelismConfig(
        name="helix_tp2_cp4",
        tp_size=2,
        sp_size=1,
        cp_size=4,
        total_gpus=8,
        description="Helix TP=2, CP=4 - Weights/2, Sequence/4",
    ),
    ParallelismConfig(
        name="helix_tp4_cp2",
        tp_size=4,
        sp_size=1,
        cp_size=2,
        total_gpus=8,
        description="Helix TP=4, CP=2 - Weights/4, Sequence/2",
    ),
    ParallelismConfig(
        name="helix_tp2_cp2",
        tp_size=2,
        sp_size=1,
        cp_size=2,
        total_gpus=4,
        description="Helix TP=2, CP=2 - Weights/2, Sequence/2",
    ),
]


# =============================================================================
# Pytest Fixtures
# =============================================================================
@pytest.fixture(params=ALL_MODEL_CONFIGS, ids=lambda c: c.name)
def model_config(request) -> ModelConfig:
    """Parametrized fixture for all model configurations."""
    return request.param


@pytest.fixture(params=[WAN_1_3B_CONFIG], ids=lambda c: c.name)
def small_model_config(request) -> ModelConfig:
    """Fixture for small model (1.3B) - quick tests."""
    return request.param


@pytest.fixture(params=[WAN_14B_CONFIG], ids=lambda c: c.name)
def large_model_config(request) -> ModelConfig:
    """Fixture for large model (14B) - Helix critical tests."""
    return request.param


@pytest.fixture(params=ALL_SEQUENCE_CONFIGS, ids=lambda c: c.name)
def sequence_config(request) -> SequenceConfig:
    """Parametrized fixture for all sequence configurations."""
    return request.param


@pytest.fixture(params=[SEQ_SHORT], ids=lambda c: c.name)
def short_sequence_config(request) -> SequenceConfig:
    """Fixture for short sequence - quick tests."""
    return request.param


@pytest.fixture(params=BASELINE_SP_CONFIGS, ids=lambda c: c.name)
def baseline_parallelism(request) -> ParallelismConfig:
    """Parametrized fixture for baseline SP configurations."""
    return request.param


@pytest.fixture(params=HELIX_CONFIGS, ids=lambda c: c.name)
def helix_parallelism(request) -> ParallelismConfig:
    """Parametrized fixture for Helix configurations."""
    return request.param


@pytest.fixture
def models_dir() -> str:
    """Return the local models directory path."""
    return "/mnt/nvme0/models/FastVideo"


@pytest.fixture
def test_prompt() -> str:
    """Standard test prompt for video generation."""
    return (
        "A serene mountain landscape at sunset, with golden light "
        "reflecting off a calm lake surrounded by pine trees."
    )


@pytest.fixture
def num_inference_steps() -> int:
    """Default number of inference steps for testing."""
    return 4  # Fast for testing


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


# =============================================================================
# Helper Functions
# =============================================================================
def get_valid_tp_sp_combinations(
    model_config: ModelConfig, total_gpus: int = 8
) -> list[tuple[int, int]]:
    """Get valid (tp_size, sp_size) combinations for a model.

    Args:
        model_config: Model configuration
        total_gpus: Total number of GPUs available

    Returns:
        List of valid (tp_size, sp_size) tuples where tp * sp <= total_gpus
    """
    combinations = []
    for tp in model_config.valid_tp_sizes:
        for sp in model_config.valid_sp_sizes:
            if tp * sp <= total_gpus:
                combinations.append((tp, sp))
    return combinations


def get_valid_helix_combinations(
    model_config: ModelConfig, total_gpus: int = 8
) -> list[tuple[int, int]]:
    """Get valid (tp_size, cp_size) combinations for Helix.

    Args:
        model_config: Model configuration
        total_gpus: Total number of GPUs available

    Returns:
        List of valid (tp_size, cp_size) tuples where tp * cp <= total_gpus
    """
    combinations = []
    for tp in model_config.valid_tp_sizes:
        if tp < 2:
            continue  # Helix requires TP >= 2
        for cp in [2, 4, 8]:  # Common CP sizes
            if tp * cp <= total_gpus:
                combinations.append((tp, cp))
    return combinations


def estimate_memory_per_gpu(
    model_config: ModelConfig,
    sequence_config: SequenceConfig,
    tp_size: int = 1,
    sp_size: int = 1,
    cp_size: int = 1,
) -> float:
    """Estimate memory usage per GPU in GB.

    Args:
        model_config: Model configuration
        sequence_config: Sequence configuration
        tp_size: Tensor parallelism size
        sp_size: Sequence parallelism size
        cp_size: Context parallelism size

    Returns:
        Estimated memory per GPU in GB
    """
    # Weight memory (split by TP)
    weight_memory = model_config.transformer_memory_gb / tp_size

    # Activation memory estimation (rough)
    # For inference, we only need activations for current layer (not all layers)
    # Split by SP or CP
    parallel_factor = max(sp_size, cp_size)

    # Rough activation estimate for inference:
    # - Input activations: seq_len * hidden_size * 2 bytes (BF16)
    # - Attention intermediate: computed in chunks
    # - Typical inference activation memory: ~2-6GB depending on sequence length
    # This is a simplified estimate; actual depends on batch size and implementation
    activation_memory_gb = 4.0 / parallel_factor  # Base ~4GB, split by parallelism

    # Text encoder (not split by TP, but can be offloaded)
    text_encoder_memory = model_config.text_encoder_memory_gb

    return weight_memory + activation_memory_gb + text_encoder_memory


def check_model_fits(
    model_config: ModelConfig,
    sequence_config: SequenceConfig,
    tp_size: int = 1,
    sp_size: int = 1,
    cp_size: int = 1,
    gpu_memory_gb: float = 32.0,
) -> bool:
    """Check if a model configuration fits in GPU memory.

    Args:
        model_config: Model configuration
        sequence_config: Sequence configuration
        tp_size: Tensor parallelism size
        sp_size: Sequence parallelism size
        cp_size: Context parallelism size
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        True if the configuration fits in memory
    """
    estimated = estimate_memory_per_gpu(
        model_config, sequence_config, tp_size, sp_size, cp_size
    )
    # Leave 20% headroom for CUDA overhead
    return estimated < gpu_memory_gb * 0.8


def skip_if_model_not_found(model_config: ModelConfig) -> None:
    """Skip test if model is not found locally."""
    if not os.path.exists(model_config.local_path):
        pytest.skip(f"Model not found: {model_config.local_path}")


def skip_if_insufficient_gpus(required_gpus: int) -> None:
    """Skip test if insufficient GPUs available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    available = torch.cuda.device_count()
    if available < required_gpus:
        pytest.skip(f"Requires {required_gpus} GPUs, only {available} available")
