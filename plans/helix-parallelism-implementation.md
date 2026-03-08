# Helix Parallelism Implementation Plan for FastVideo

## Executive Summary

This plan outlines the implementation of Helix-style parallelism (combined Tensor Parallelism + Context Parallelism with overlapped communication) in FastVideo. This enables running large DiT models (14B+) on multi-GPU setups where individual GPUs cannot hold full model weights.

**Key Benefits:**
- **No weight redundancy**: Weights sharded across TP ranks (1/TP per GPU)
- **Higher throughput**: Sequence/activation sharding via CP
- **Hidden communication**: Ring-based communication overlapped with compute

---

## CRITICAL: VSA Backend Requirement

**The local FastVideo models require `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`**

FastVideo models have two attention architectures:
- `WanTransformerBlock` - Standard attention (no `to_gate_compress` layers)
- `WanTransformerBlock_VSA` - Video Sparse Attention (has `to_gate_compress` layers)

The local FastVideo checkpoints at `/mnt/nvme0/models/FastVideo/` were trained with VSA. The model architecture selection happens in [`fastvideo/models/dits/wanvideo.py`](../fastvideo/models/dits/wanvideo.py:593):

```python
transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock
```

Without setting the environment variable, model loading fails with:
```
ValueError: Parameter blocks.0.to_gate_compress.bias not found in custom model state dict
```

**Solution**: Set before running any tests or inference:
```bash
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
```

---

## Reference Implementations

This plan draws from three key sources:

### 1. Helix Paper ([arXiv:2507.07120](https://arxiv.org/pdf/2507.07120))
Theoretical foundation for TP+CP overlap with ring communication patterns.

### 2. TensorRT-LLM Helix Implementation

TensorRT-LLM has a production Helix implementation with these key components:

| Component | File | Purpose |
|-----------|------|---------|
| **CpType.HELIX** | `tensorrt_llm/mapping.py` | Enum for Helix CP mode |
| **helixAllToAll.cu** | `cpp/tensorrt_llm/kernels/helixAllToAll.cu` | FIFO-based all-to-all with MNNVL |
| **helixKernels.cu** | `cpp/tensorrt_llm/kernels/helixKernels.cu` | Post-processing to combine partial outputs |
| **alltoall_helix()** | `tensorrt_llm/_torch/distributed/ops.py` | Python wrapper for NCCL all-to-all |
| **_helix_post_process()** | `tensorrt_llm/_torch/distributed/ops.py` | Combines partial outputs after exchange |

**Key TensorRT-LLM Design Decisions:**
- **Two communication modes**: NCCL-based and FIFO-based (for MNNVL)
- **LSE correction**: Uses `warpReduceCorrectedSum()` for numerical stability
- **Blackwell optimizations**: Programmatic dependent launch (PDL) for kernel overlap
- **MLA support**: Optimized for DeepSeek-V3 style attention with `kv_lora_rank`

### 3. SGLang DCP Implementation

SGLang's Decode-Context Parallelism provides patterns for:

| Concept | Description |
|---------|-------------|
| **Interleaved KV sharding** | `token_idx % dcp_world_size == dcp_rank` |
| **LSE correction kernel** | `cp_lse_ag_out_rs` + `correct_attn_out` |
| **DCPTokenToKVPoolAllocator** | Virtual capacity expansion with alignment |
| **Chunked prefill** | Ensures `extend_seq_len % dcp_world_size == 0` |

---

## Current State Analysis

### What FastVideo Has Today

```
fastvideo/distributed/
├── parallel_state.py      # TP, SP, DP group management
├── communication_op.py    # All-to-all, all-gather, all-reduce ops
├── utils.py               # Padding, sharding utilities
└── device_communicators/  # NCCL/CPU communicator implementations

fastvideo/attention/
├── layer.py               # DistributedAttention (Ulysses-style SP)
└── backends/              # FlashAttention, STA, VSA implementations

fastvideo/layers/
├── linear.py              # TP-aware linear layers (from vLLM)
└── ...                    # Other layers
```

### Current Parallelism Support

| Feature | Status | Location |
|---------|--------|----------|
| **SP (Ulysses)** | ✅ Implemented | `fastvideo/attention/layer.py` |
| **TP (Text Encoders)** | ✅ Implemented | `fastvideo/layers/linear.py` |
| **TP (DiT)** | ❌ Not implemented | DiT models use `nn.Linear` |
| **FSDP Inference** | ✅ Available | `fastvideo/models/loader/fsdp_load.py` |
| **Ring Attention** | ❌ Not implemented | - |
| **Helix (TP+CP)** | ❌ Not implemented | - |

### Current SP Implementation (Ulysses-style)

From `fastvideo/attention/layer.py`:

```python
def forward(self, q, k, v, ...):
    # Stack QKV
    qkv = torch.cat([q, k, v], dim=0)
    
    # All-to-all: scatter heads, gather sequence
    qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
    
    # Compute attention (each rank has full seq, subset of heads)
    output = self.attn_impl.forward(q, k, v, ...)
    
    # All-to-all back: scatter sequence, gather heads
    output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
```

**Problem**: Each GPU needs full weights because the linear projections (Q, K, V, O) happen BEFORE the all-to-all.

---

## FSDP Inference: Current Workaround and Its Limitations

### What is FSDP?

**FSDP (Fully Sharded Data Parallel)** is PyTorch's built-in mechanism for distributing model weights across GPUs. Originally designed for training, it can also be used for inference via `use_fsdp_inference=True`.

### How FSDP Works

```
FSDP Weight Sharding (8 GPUs):
┌──────────────────────────────────────────────────────────────┐
│ Full Model Weights: [W1, W2, W3, W4, W5, W6, W7, W8]       │
│                                                              │
│ GPU 0: [W1]  ──┐                                            │
│ GPU 1: [W2]  ──┤  Each GPU holds 1/8 of weights             │
│ GPU 2: [W3]  ──┤                                            │
│ GPU 3: [W4]  ──┤  Before each layer's forward pass:         │
│ GPU 4: [W5]  ──┤  → All-gather to reconstruct full weights  │
│ GPU 5: [W6]  ──┤  → Compute forward pass                    │
│ GPU 6: [W7]  ──┤  → Discard gathered weights                │
│ GPU 7: [W8]  ──┘                                            │
└──────────────────────────────────────────────────────────────┘
```

### FSDP in FastVideo Code

FSDP is enabled via [`fastvideo/models/loader/fsdp_load.py`](../fastvideo/models/loader/fsdp_load.py):

```python
# Line 101: FSDP only used for training OR explicit fsdp_inference
use_fsdp = training_mode or fsdp_inference

# Line 168-257: shard_model() applies PyTorch fully_shard() to matching modules
def shard_model(model, *, fsdp_shard_conditions, ...):
    for n, m in reversed(list(model.named_modules())):
        if any([cond(n, m) for cond in fsdp_shard_conditions]):
            fully_shard(m, **fsdp_kwargs)

# Line 327-331: Weight loading distributes tensors across device mesh
sharded_tensor = distribute_tensor(
    full_tensor,
    meta_sharded_param.device_mesh,
    meta_sharded_param.placements,
)
```

### FSDP Configuration Gotcha

```python
# fastvideo/fastvideo_args.py:665-670
if self.dit_layerwise_offload:  # Default: True
    if self.use_fsdp_inference:
        logger.warning("dit_layerwise_offload is enabled, "
                       "automatically disabling use_fsdp_inference.")
        self.use_fsdp_inference = False  # FSDP gets disabled!
```

**To use FSDP inference, you MUST set both:**
```python
use_fsdp_inference=True
dit_layerwise_offload=False  # Otherwise FSDP gets auto-disabled
```

### FSDP Test Results

| Model | Without FSDP | With FSDP | Improvement |
|-------|-------------|-----------|-------------|
| LTX2 TP=8 | OOM (29.63GB/GPU) | 5.1GB at load, 31.5GB peak | ✅ Fits! |
| LTX2 121 frames 480x704 | OOM | 96.80s, 30.7GB peak | ✅ 5s video |
| LTX2 481 frames 480x704 | OOM | 105.86s, 31.5GB peak | ✅ 20s video |
| LTX2 121 frames 1024x1536 | OOM | 99.72s, 31.5GB peak | ✅ High-res |

### Why FSDP is a Workaround, Not a Solution

| Aspect | FSDP | Helix (TP+CP) |
|--------|------|---------------|
| **Weight sharding** | ✅ Automatic via `fully_shard()` | ✅ Via TP-aware linear layers |
| **Communication** | All-gather full weights before EVERY layer | All-reduce partial results (smaller) |
| **Memory during forward** | Temporarily holds full weights per layer | Only holds 1/TP of weights always |
| **Sequence parallelism** | ❌ Not supported | ✅ CP splits sequence across groups |
| **Designed for** | Training (gradient accumulation) | Inference (minimal communication) |
| **Code changes needed** | None (works with any model) | Requires TP-aware layers in DiT |
| **Performance** | Slower (all-gather overhead per layer) | Faster (all-reduce is smaller) |
| **Scaling** | Good for training, suboptimal for inference | Optimized for inference |

### The Key Problem FSDP Has

During each layer's forward pass, FSDP must:
1. **All-gather** the full weight tensor from all GPUs (communication)
2. **Compute** the forward pass with full weights (compute)
3. **Discard** the gathered weights (memory)

This means **every layer** requires a full all-gather of weights, which is:
- **O(model_size)** communication per layer
- **Temporary peak memory** equals full layer weights

### How Helix Improves Upon FSDP

With Helix (TP-aware layers), each layer:
1. **Computes** with local 1/TP weight shard (no gathering needed)
2. **All-reduce** the partial output (much smaller than full weights)

This means:
- **O(activation_size)** communication per layer (much smaller than weights)
- **No temporary peak memory** from weight gathering
- **Can combine with CP** for sequence parallelism (FSDP cannot)

```
FSDP Forward Pass (per layer):
  All-gather weights (SLOW) → Compute → Discard weights
  Communication: O(weight_size) = O(hidden_dim × hidden_dim)

Helix Forward Pass (per layer):
  Compute with local shard → All-reduce output (FAST)
  Communication: O(activation_size) = O(batch × seq_len × hidden_dim)
```

For a typical DiT layer with hidden_dim=5120:
- **FSDP all-gather**: 5120 × 5120 × 2 bytes = 50MB per weight matrix
- **Helix all-reduce**: batch × seq_len × 5120 × 2 bytes = much smaller for short sequences

### Can We Just Extend FSDP into Helix?

**No.** FSDP and Helix are fundamentally different approaches:

| | FSDP | Helix |
|---|------|-------|
| **Where sharding happens** | At the PyTorch runtime level (transparent to model code) | At the model architecture level (TP-aware layers) |
| **How weights are used** | Gathered on-demand, then discarded | Permanently split, never gathered |
| **Communication pattern** | All-gather (reconstruct full weights) | All-reduce (combine partial outputs) |
| **Sequence parallelism** | Not supported | Built-in via CP |
| **Implementation** | PyTorch `fully_shard()` wrapper | Custom `ColumnParallelLinear`, `RowParallelLinear` |

**Why you can't extend FSDP:**
1. FSDP operates at the **tensor distribution** level — it wraps existing modules and handles weight gathering/scattering automatically
2. Helix operates at the **computation** level — each GPU computes a different partial result using its weight shard
3. FSDP's all-gather reconstructs the full weight before computation; Helix never reconstructs it
4. FSDP has no concept of sequence/context parallelism; Helix combines TP with CP

**However, FSDP is useful as a fallback:**
- For models that don't yet have Helix support (like LTX2 today)
- For training where FSDP's gradient sharding is needed
- As a quick way to test if a model fits in memory before implementing Helix

**The implementation path:**
1. **Today**: Use FSDP inference for LTX2 (works but suboptimal)
2. **Phase 1**: Add TP-aware layers to DiT models (replaces FSDP's weight sharding with more efficient TP)
3. **Phase 2**: Add CP for sequence parallelism (something FSDP cannot do)
4. **Phase 3**: Combine TP+CP = Helix (optimal performance)

---

## Helix Architecture for FastVideo

### Core Concept: TP + CP with Overlapped Communication

```
Helix Communication Pattern (TP=2, CP=4, 8 GPUs total):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  GPU Layout:                                                    │
│                                                                 │
│     CP Ring 0          CP Ring 1                                │
│     ┌─────────┐        ┌─────────┐                              │
│     │ GPU 0 ←→ GPU 2 ←→ GPU 4 ←→ GPU 6 │  (TP rank 0)          │
│     └─────────┘        └─────────┘                              │
│         ↕                  ↕                                    │
│     ┌─────────┐        ┌─────────┐                              │
│     │ GPU 1 ←→ GPU 3 ←→ GPU 5 ←→ GPU 7 │  (TP rank 1)          │
│     └─────────┘        └─────────┘                              │
│                                                                 │
│  Horizontal arrows (←→): CP all-to-all (partial outputs)       │
│  Vertical arrows (↕): TP all-reduce (weight sharding)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Helix Forward Pass (Following TensorRT-LLM Pattern)

```
Step 1: Sequence Sharding (CP dimension)
┌─────────────────────────────────────────────────────────────────┐
│  Each CP rank gets 1/CP of the sequence (latent patches)       │
│  GPU 0,1: patches [0, 4, 8, ...]                               │
│  GPU 2,3: patches [1, 5, 9, ...]                               │
│  GPU 4,5: patches [2, 6, 10, ...]                              │
│  GPU 6,7: patches [3, 7, 11, ...]                              │
└─────────────────────────────────────────────────────────────────┘

Step 2: TP-Sharded QKV Projection
┌─────────────────────────────────────────────────────────────────┐
│  Each TP rank has 1/TP of QKV projection weights               │
│  GPU 0: Q₀, K₀, V₀ = Linear_TP0(X_chunk0)                      │
│  GPU 1: Q₁, K₁, V₁ = Linear_TP1(X_chunk0)  ← same chunk!       │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘

Step 3: Local Attention Computation
┌─────────────────────────────────────────────────────────────────┐
│  Each GPU computes attention with LOCAL KV only                 │
│  Output: partial_attn_out, partial_lse (log-sum-exp)           │
└─────────────────────────────────────────────────────────────────┘

Step 4: CP All-to-All Exchange (TensorRT-LLM style)
┌─────────────────────────────────────────────────────────────────┐
│  Exchange partial outputs and LSE stats across CP ranks         │
│  gathered_o: [cp_size, num_tokens, num_heads, head_dim]        │
│  gathered_stats: [cp_size, num_tokens, num_heads, 2]           │
└─────────────────────────────────────────────────────────────────┘

Step 5: Helix Post-Processing (LSE Correction)
┌─────────────────────────────────────────────────────────────────┐
│  Combine partial outputs using corrected softmax weights        │
│  global_lse = log(Σ exp(local_lse - max_lse)) + max_lse        │
│  output = Σ (exp(local_lse - global_lse) * partial_out)        │
└─────────────────────────────────────────────────────────────────┘

Step 6: TP-Sharded Output Projection + All-Reduce
┌─────────────────────────────────────────────────────────────────┐
│  Each GPU has 1/TP of output projection weights                 │
│  O_partial = Linear_TP(attention_output)                        │
│  TP All-Reduce → O_full                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: CP Group Management and Communication Primitives

#### 1.1 Add CP Group to Parallel State

**Modify**: `fastvideo/distributed/parallel_state.py`

```python
# Add new global for CP group
_CP: GroupCoordinator | None = None

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    sequence_model_parallel_size: int = 1,  # Keep for backward compat
    context_parallel_size: int = 1,  # NEW - Helix CP
    ...
):
    """Initialize TP, SP, CP, and DP groups.
    
    For Helix mode (cp_size > 1):
    - TP groups: GPUs that share the same sequence chunk
    - CP groups: GPUs that share the same TP rank
    
    Example with 8 GPUs, TP=2, CP=4:
    - TP groups: [0,1], [2,3], [4,5], [6,7]
    - CP groups: [0,2,4,6], [1,3,5,7]
    """
    global _CP
    
    if context_parallel_size > 1:
        # Helix mode: CP is orthogonal to TP
        cp_ranks = []
        for tp_rank in range(tensor_model_parallel_size):
            cp_group_ranks = [
                tp_rank + i * tensor_model_parallel_size 
                for i in range(context_parallel_size)
            ]
            cp_ranks.append(cp_group_ranks)
        
        _CP = init_model_parallel_group(
            group_ranks=cp_ranks,
            local_rank=local_rank,
            backend=backend,
            group_name="cp",
        )

def get_cp_group() -> GroupCoordinator:
    assert _CP is not None, "CP group not initialized"
    return _CP

def get_cp_world_size() -> int:
    if _CP is None:
        return 1
    return _CP.world_size

def get_cp_rank() -> int:
    if _CP is None:
        return 0
    return _CP.rank_in_group
```

#### 1.2 Helix All-to-All Communication

**New file**: `fastvideo/distributed/helix_comm.py`

```python
"""Helix communication primitives following TensorRT-LLM patterns."""

import torch
import torch.distributed as dist
from fastvideo.distributed.parallel_state import get_cp_group, get_cp_world_size

def helix_alltoall(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-to-all exchange of partial attention outputs and LSE stats.
    
    Args:
        local_output: [batch, seq_len, num_heads, head_dim] - partial attention output
        local_lse: [batch, num_heads, seq_len] - log-sum-exp statistics
        
    Returns:
        gathered_output: [cp_size, batch, seq_len, num_heads, head_dim]
        gathered_lse: [cp_size, batch, num_heads, seq_len]
    """
    cp_group = get_cp_group()
    cp_size = get_cp_world_size()
    
    # Allocate output buffers
    gathered_output = torch.empty(
        cp_size, *local_output.shape,
        dtype=local_output.dtype, device=local_output.device
    )
    gathered_lse = torch.empty(
        cp_size, *local_lse.shape,
        dtype=local_lse.dtype, device=local_lse.device
    )
    
    # All-gather outputs and LSE
    dist.all_gather_into_tensor(
        gathered_output, local_output, group=cp_group.device_group
    )
    dist.all_gather_into_tensor(
        gathered_lse, local_lse, group=cp_group.device_group
    )
    
    return gathered_output, gathered_lse


def helix_post_process(
    gathered_output: torch.Tensor,
    gathered_lse: torch.Tensor,
) -> torch.Tensor:
    """Combine partial attention outputs using LSE correction.
    
    Following TensorRT-LLM's helixKernels.cu pattern:
    1. Compute global max LSE across CP ranks
    2. Compute correction factors: exp(local_lse - global_lse)
    3. Weighted sum of partial outputs
    
    Args:
        gathered_output: [cp_size, batch, seq_len, num_heads, head_dim]
        gathered_lse: [cp_size, batch, num_heads, seq_len]
        
    Returns:
        combined_output: [batch, seq_len, num_heads, head_dim]
    """
    cp_size = gathered_output.shape[0]
    
    # Compute global max LSE for numerical stability
    # gathered_lse: [cp_size, batch, num_heads, seq_len]
    max_lse = gathered_lse.max(dim=0).values  # [batch, num_heads, seq_len]
    
    # Compute correction factors
    # exp(local_lse - max_lse) for each CP rank
    correction = torch.exp(gathered_lse - max_lse.unsqueeze(0))  # [cp_size, batch, num_heads, seq_len]
    
    # Normalize correction factors
    correction_sum = correction.sum(dim=0)  # [batch, num_heads, seq_len]
    correction = correction / correction_sum.unsqueeze(0)  # [cp_size, batch, num_heads, seq_len]
    
    # Reshape for broadcasting: [cp_size, batch, seq_len, num_heads, 1]
    correction = correction.permute(0, 1, 3, 2).unsqueeze(-1)
    
    # Weighted sum of partial outputs
    combined_output = (gathered_output * correction).sum(dim=0)  # [batch, seq_len, num_heads, head_dim]
    
    return combined_output
```

### Phase 2: TP-Sharded DiT Layers

#### 2.1 TP-Aware QKV Projection

**New file**: `fastvideo/layers/tp_linear.py`

```python
"""Tensor-parallel linear layers for DiT models."""

import torch
import torch.nn as nn
from fastvideo.distributed import (
    get_tp_rank, get_tp_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)

class ColumnParallelLinear(nn.Module):
    """Linear layer with column-parallel weight sharding.
    
    Weight shape: [output_size / tp_size, input_size]
    Each TP rank holds a shard of the output dimension.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.gather_output = gather_output
        
        assert output_size % self.tp_size == 0
        self.output_size_per_partition = output_size // self.tp_size
        
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul with sharded weights
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather to get full output
            output = tensor_model_parallel_all_gather(output, dim=-1)
        
        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel weight sharding.
    
    Weight shape: [output_size, input_size / tp_size]
    Each TP rank holds a shard of the input dimension.
    Requires all-reduce after forward.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.input_is_parallel = input_is_parallel
        
        assert input_size % self.tp_size == 0
        self.input_size_per_partition = input_size // self.tp_size
        
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
        if bias:
            # Bias is NOT sharded - only rank 0 adds it after all-reduce
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Shard input along last dimension
            x = x.chunk(self.tp_size, dim=-1)[self.tp_rank]
        
        # Local matmul with sharded weights
        output = torch.nn.functional.linear(x, self.weight)
        
        # All-reduce to sum partial results
        output = tensor_model_parallel_all_reduce(output)
        
        # Add bias (only after all-reduce)
        if self.bias is not None:
            output = output + self.bias
        
        return output
```

### Phase 3: Helix Attention Layer

#### 3.1 Helix Distributed Attention

**New file**: `fastvideo/attention/helix_attention.py`

```python
"""Helix-style distributed attention combining TP and CP."""

import torch
import torch.nn as nn
from fastvideo.attention.backends.flash_attn import FlashAttentionImpl
from fastvideo.distributed.parallel_state import (
    get_tp_group, get_cp_group, get_tp_world_size, get_cp_world_size, get_cp_rank
)
from fastvideo.distributed.helix_comm import helix_alltoall, helix_post_process
from fastvideo.layers.tp_linear import ColumnParallelLinear, RowParallelLinear


class HelixDistributedAttention(nn.Module):
    """
    Helix-style attention combining TP (weight sharding) and CP (sequence sharding).
    
    Following TensorRT-LLM's Helix implementation:
    1. Each CP rank processes a chunk of the sequence
    2. Each TP rank holds a shard of the weights
    3. Local attention computes partial outputs with LSE
    4. All-to-all exchanges partial outputs across CP ranks
    5. Post-processing combines outputs using LSE correction
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        
        self.tp_size = get_tp_world_size()
        self.cp_size = get_cp_world_size()
        
        # Validate head divisibility
        assert num_heads % self.tp_size == 0, \
            f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})"
        
        self.num_heads_per_tp = num_heads // self.tp_size
        self.num_kv_heads_per_tp = self.num_kv_heads // self.tp_size
        
        # TP-sharded QKV projection (column parallel)
        qkv_size = (self.num_heads + 2 * self.num_kv_heads) * head_dim
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            qkv_size,
            bias=qkv_bias,
            gather_output=False,  # Keep sharded for attention
        )
        
        # TP-sharded output projection (row parallel)
        self.out_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=out_bias,
            input_is_parallel=True,  # Input is already TP-sharded
        )
        
        # Local attention implementation (FlashAttention)
        self.attn_impl = FlashAttentionImpl(
            num_heads=self.num_heads_per_tp,
            head_size=head_dim,
            num_kv_heads=self.num_kv_heads_per_tp,
            return_lse=True,  # Need LSE for Helix post-processing
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
                Note: seq_len is already sharded by CP (1/cp_size of full sequence)
            freqs_cis: RoPE frequencies (cos, sin)
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. TP-sharded QKV projection
        qkv = self.qkv_proj(hidden_states)  # [batch, seq_len, (q+2kv)*head_dim/tp]
        
        # Split into Q, K, V
        q_size = self.num_heads_per_tp * self.head_dim
        kv_size = self.num_kv_heads_per_tp * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads_per_tp, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_tp, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_tp, self.head_dim)
        
        # 2. Apply RoPE if provided
        if freqs_cis is not None:
            cos, sin = freqs_cis
            # Apply rotary embedding to Q and K
            q, k = self._apply_rope(q, k, cos, sin)
        
        # 3. Local attention with LSE
        # Each CP rank computes attention with its LOCAL KV only
        attn_output, lse = self.attn_impl.forward_with_lse(q, k, v)
        # attn_output: [batch, seq_len, num_heads_per_tp, head_dim]
        # lse: [batch, num_heads_per_tp, seq_len]
        
        # 4. Helix all-to-all exchange across CP ranks
        if self.cp_size > 1:
            gathered_output, gathered_lse = helix_alltoall(attn_output, lse)
            
            # 5. Helix post-processing (LSE correction)
            attn_output = helix_post_process(gathered_output, gathered_lse)
        
        # 6. Reshape for output projection
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # 7. TP-sharded output projection (includes all-reduce)
        output = self.out_proj(attn_output)
        
        return output
    
    def _apply_rope(self, q, k, cos, sin):
        """Apply rotary position embedding."""
        # Implementation depends on RoPE style (neox vs original)
        # Simplified version:
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
```

### Phase 4: Model Integration

#### 4.1 Helix-Enabled Transformer Block

**New file**: `fastvideo/models/dits/helix_blocks.py`

```python
"""Helix-enabled transformer blocks for DiT models."""

import torch
import torch.nn as nn
from fastvideo.attention.helix_attention import HelixDistributedAttention
from fastvideo.layers.tp_linear import ColumnParallelLinear, RowParallelLinear
from fastvideo.layers.layernorm import RMSNorm


class HelixTransformerBlock(nn.Module):
    """Transformer block with Helix parallelism (TP + CP)."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        head_dim = head_dim or hidden_size // num_heads
        
        # Pre-attention norm
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        
        # Helix attention
        self.attn = HelixDistributedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
        
        # Pre-FFN norm
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        
        # TP-sharded FFN
        self.ffn = HelixMLP(hidden_size, ffn_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, freqs_cis=freqs_cis)
        hidden_states = residual + hidden_states
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class HelixMLP(nn.Module):
    """MLP with TP-sharded weights (SwiGLU style)."""
    
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        
        # Gate + Up projection (column parallel)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * ffn_dim,
            bias=False,
            gather_output=False,
        )
        
        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            ffn_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x
```

#### 4.2 Sequence Sharding for CP

**New file**: `fastvideo/distributed/cp_utils.py`

```python
"""Context parallelism utilities for sequence sharding."""

import torch
from fastvideo.distributed.parallel_state import get_cp_rank, get_cp_world_size


def shard_sequence_for_cp(
    hidden_states: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Shard sequence across CP ranks.
    
    Uses interleaved sharding (like SGLang DCP):
    token_idx % cp_world_size == cp_rank
    
    Args:
        hidden_states: [batch, seq_len, hidden_size]
        seq_dim: Dimension to shard (default: 1)
        
    Returns:
        Sharded tensor: [batch, seq_len // cp_size, hidden_size]
    """
    cp_rank = get_cp_rank()
    cp_size = get_cp_world_size()
    
    if cp_size == 1:
        return hidden_states
    
    seq_len = hidden_states.shape[seq_dim]
    
    # Pad sequence to be divisible by cp_size
    pad_len = (cp_size - seq_len % cp_size) % cp_size
    if pad_len > 0:
        pad_shape = list(hidden_states.shape)
        pad_shape[seq_dim] = pad_len
        padding = torch.zeros(pad_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.cat([hidden_states, padding], dim=seq_dim)
    
    # Interleaved sharding: take every cp_size-th token starting at cp_rank
    indices = torch.arange(cp_rank, hidden_states.shape[seq_dim], cp_size, device=hidden_states.device)
    return hidden_states.index_select(seq_dim, indices)


def gather_sequence_from_cp(
    hidden_states: torch.Tensor,
    original_seq_len: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Gather sequence from all CP ranks.
    
    Args:
        hidden_states: [batch, seq_len // cp_size, hidden_size]
        original_seq_len: Original sequence length before sharding
        seq_dim: Dimension to gather (default: 1)
        
    Returns:
        Full tensor: [batch, original_seq_len, hidden_size]
    """
    cp_size = get_cp_world_size()
    
    if cp_size == 1:
        return hidden_states
    
    # All-gather across CP ranks
    gathered = get_cp_group().all_gather(hidden_states, dim=seq_dim)
    
    # Reorder from interleaved to sequential
    # gathered is [batch, seq_len, hidden_size] but in wrong order
    batch_size = gathered.shape[0]
    hidden_size = gathered.shape[-1]
    seq_len_per_rank = hidden_states.shape[seq_dim]
    
    # Reshape to [batch, cp_size, seq_len_per_rank, hidden_size]
    gathered = gathered.view(batch_size, cp_size, seq_len_per_rank, hidden_size)
    
    # Transpose to [batch, seq_len_per_rank, cp_size, hidden_size]
    gathered = gathered.transpose(1, 2)
    
    # Reshape to [batch, seq_len_per_rank * cp_size, hidden_size]
    gathered = gathered.reshape(batch_size, -1, hidden_size)
    
    # Trim to original length
    return gathered[:, :original_seq_len, :]
```

### Phase 5: CLI and Pipeline Integration

#### 5.1 CLI Arguments

**Modify**: `fastvideo/fastvideo_args.py`

```python
@dataclass
class FastVideoArgs:
    # ... existing args ...
    
    # Helix parallelism
    use_helix: bool = False
    cp_size: int = 1  # Context parallel size
    
    @classmethod
    def add_cli_args(cls, parser):
        # ... existing args ...
        
        parser.add_argument(
            "--use-helix",
            action="store_true",
            help="Enable Helix parallelism (TP + CP) for large models",
        )
        parser.add_argument(
            "--cp-size",
            type=int,
            default=1,
            help="Context parallelism size for Helix mode",
        )
    
    def __post_init__(self):
        # ... existing validation ...
        
        if self.use_helix:
            assert self.tp_size > 1, "Helix requires TP > 1 for weight sharding"
            assert self.cp_size > 1, "Helix requires CP > 1 for sequence sharding"
            # SP is replaced by CP in Helix mode
            if self.sp_size > 1:
                logger.warning("Helix mode enabled, setting sp_size=1 (using CP instead)")
                self.sp_size = 1
```

#### 5.2 Pipeline Initialization

**Modify**: `fastvideo/pipelines/composed_pipeline_base.py`

```python
def _init_distributed(self, fastvideo_args):
    if fastvideo_args.use_helix:
        # Initialize Helix parallelism (TP + CP)
        initialize_model_parallel(
            tensor_model_parallel_size=fastvideo_args.tp_size,
            context_parallel_size=fastvideo_args.cp_size,
            sequence_model_parallel_size=1,  # Disabled in Helix mode
        )
        warmup_helix_communication()
        logger.info(
            f"Helix parallelism initialized: TP={fastvideo_args.tp_size}, CP={fastvideo_args.cp_size}"
        )
    else:
        # Existing SP initialization
        initialize_model_parallel(
            tensor_model_parallel_size=fastvideo_args.tp_size,
            sequence_model_parallel_size=fastvideo_args.sp_size,
        )
        warmup_sequence_parallel_communication()
```

#### 5.3 Model Loading with TP Weight Sharding

**Modify**: `fastvideo/models/loader/fsdp_load.py`

```python
def load_helix_model(
    model_cls: type[nn.Module],
    model_path: str,
    tp_rank: int,
    tp_size: int,
    **kwargs,
) -> nn.Module:
    """Load model with TP-sharded weights for Helix mode.
    
    Args:
        model_cls: Model class to instantiate
        model_path: Path to model weights
        tp_rank: Current TP rank
        tp_size: Total TP size
        
    Returns:
        Model with sharded weights loaded
    """
    # Create model with Helix-enabled blocks
    model = model_cls(use_helix=True, **kwargs)
    
    # Load full state dict
    state_dict = load_state_dict(model_path)
    
    # Shard weights according to TP rank
    sharded_state_dict = {}
    for name, param in state_dict.items():
        if is_column_parallel_weight(name):
            # Shard along output dimension (dim 0)
            shard_size = param.shape[0] // tp_size
            sharded_state_dict[name] = param[tp_rank * shard_size:(tp_rank + 1) * shard_size]
        elif is_row_parallel_weight(name):
            # Shard along input dimension (dim 1)
            shard_size = param.shape[1] // tp_size
            sharded_state_dict[name] = param[:, tp_rank * shard_size:(tp_rank + 1) * shard_size]
        else:
            # Replicated weight (norms, embeddings, etc.)
            sharded_state_dict[name] = param
    
    model.load_state_dict(sharded_state_dict)
    return model


def is_column_parallel_weight(name: str) -> bool:
    """Check if weight should be column-parallel sharded."""
    column_parallel_patterns = [
        "qkv_proj.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "gate_up_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
    ]
    return any(pattern in name for pattern in column_parallel_patterns)


def is_row_parallel_weight(name: str) -> bool:
    """Check if weight should be row-parallel sharded."""
    row_parallel_patterns = [
        "out_proj.weight",
        "o_proj.weight",
        "down_proj.weight",
    ]
    return any(pattern in name for pattern in row_parallel_patterns)
```

---

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `fastvideo/distributed/helix_comm.py` | Helix all-to-all and post-processing |
| `fastvideo/distributed/cp_utils.py` | CP sequence sharding utilities |
| `fastvideo/layers/tp_linear.py` | TP-aware linear layers |
| `fastvideo/attention/helix_attention.py` | Helix distributed attention |
| `fastvideo/models/dits/helix_blocks.py` | Helix transformer blocks |

### Modified Files

| File | Changes |
|------|---------|
| `fastvideo/distributed/parallel_state.py` | Add CP group management |
| `fastvideo/distributed/communication_op.py` | Add Helix comm warmup |
| `fastvideo/fastvideo_args.py` | Add `--use-helix`, `--cp-size` args |
| `fastvideo/models/dits/wanvideo.py` | Integrate Helix blocks option |
| `fastvideo/models/dits/hunyuanvideo.py` | Integrate Helix blocks option |
| `fastvideo/models/dits/ltx2.py` | Integrate Helix blocks option |
| `fastvideo/models/loader/fsdp_load.py` | Add TP weight sharding |
| `fastvideo/pipelines/composed_pipeline_base.py` | Helix initialization |

---

## Testing Strategy

### Unit Tests

```python
# tests/distributed/test_helix_comm.py
def test_helix_alltoall():
    """Test all-to-all exchange of partial outputs."""
    
def test_helix_post_process():
    """Test LSE correction produces correct combined output."""

# tests/distributed/test_cp_utils.py
def test_shard_sequence_for_cp():
    """Test interleaved sequence sharding."""
    
def test_gather_sequence_from_cp():
    """Test sequence gathering and reordering."""

# tests/attention/test_helix_attention.py
def test_helix_attention_correctness():
    """Compare Helix attention output to single-GPU attention."""
```

### Integration Tests

```python
# tests/ssim/test_helix_similarity.py
def test_wan_helix_vs_sp():
    """Compare Helix output to SP output for Wan model."""
    # Should produce identical results (within numerical tolerance)
    
def test_wan_helix_memory():
    """Verify memory usage is reduced with Helix."""
```

### Multi-GPU Tests

```bash
# Run with 8 GPUs: TP=2, CP=4
torchrun --nproc_per_node=8 tests/distributed/test_helix_e2e.py \
    --tp-size 2 --cp-size 4 --use-helix
```

---

## Usage Example

```bash
# Run Wan 14B with Helix parallelism on 8 GPUs
fastvideo generate \
    --model Wan-AI/Wan2.1-T2V-14B \
    --prompt "A cat playing piano" \
    --use-helix \
    --tp-size 2 \
    --cp-size 4 \
    --num-gpus 8

# Memory per GPU:
# - Without Helix (SP=8): 28GB weights + ~5GB activations = 33GB (doesn't fit on 24GB!)
# - With Helix (TP=2, CP=4): 14GB weights + ~5GB activations = 19GB ✓
```

---

## Comparison: FastVideo SP vs Helix

| Aspect | Current SP (Ulysses) | Helix (TP+CP) |
|--------|---------------------|---------------|
| **Weight per GPU** | 100% (full model) | 1/TP (sharded) |
| **Activations per GPU** | 1/SP | 1/CP |
| **Communication** | All-to-all (blocking) | All-gather + post-process |
| **Memory for 14B** | ~28GB weights | ~14GB weights (TP=2) |
| **Throughput** | ~SP× | ~CP× |
| **Complexity** | Simple | More complex |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Numerical differences** | Extensive SSIM testing against SP baseline |
| **Communication overhead** | Profile and optimize all-to-all patterns |
| **Memory fragmentation** | Pre-allocate buffers for CP exchange |
| **Weight loading complexity** | Clear TP sharding logic with tests |
| **Backward pass** | Implement gradients for all custom ops |

---

## References

1. **Helix Paper**: [arXiv:2507.07120](https://arxiv.org/pdf/2507.07120)
2. **TensorRT-LLM Helix**: `tensorrt_llm/mapping.py`, `helixAllToAll.cu`, `helixKernels.cu`
3. **SGLang DCP**: Decode-Context Parallelism for DeepSeek-v2
4. **Ring Attention**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
5. **Blockwise Attention**: [arXiv:2305.19370](https://arxiv.org/abs/2305.19370)
