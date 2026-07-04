# SPDX-License-Identifier: Apache-2.0
"""Triton flash-attention prefill over an fp8 (e4m3) KV cache.

Pattern from vLLM commit c9e50123 (Triton MLA decode fp8 KV support),
adapted from paged MLA decode to LingBot-World's dense MHA prefill
shapes: the cache stays float8_e4m3fn in HBM, each K/V tile is loaded
as fp8 (1 byte/elem) and upconverted IN REGISTERS with a per-head
scale before the dot products. No dequantized copy is ever
materialized in global memory.

Shapes (rank-local, Ulysses head-sharded):
  q:   [L_q, H, D] bf16, RoPE'd
  k/v: [L_kv, H, D] float8_e4m3fn (the valid cache window)
  k_scale/v_scale: [H] fp32 per-head dequant scales
  out: [L_q, H, D] bf16

Non-causal full attention over the window (matches the block-causal
scheme: queries of the current chunk attend to the whole cached
window including themselves).
"""

import torch
import triton
import triton.language as tl


# AOT-selected launch config: swept offline on RTX 5090 (SM120) over
# {64,128,256} x {64,128} tiles, 4/8 warps, 2-4 stages at kv_len 4680 /
# 14040 / 32760 - (M=128, N=128, warps=8, stages=2) won at EVERY
# length. Runtime @triton.autotune is deliberately avoided: the causal
# rollout calls with a different (growing) L_kv every chunk, and
# autotuning inside the generation loop costs seconds per generation
# for a realtime target.
_BLOCK_M = 128
_BLOCK_N = 128
_NUM_WARPS = 8
_NUM_STAGES = 2


@triton.jit
def _fp8_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    L_q,
    L_kv,
    sm_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < L_q

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + pid_h * stride_qh +
        offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    k_scale = tl.load(k_scale_ptr + pid_h)
    v_scale = tl.load(v_scale_ptr + pid_h)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, L_kv, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L_kv

        k = tl.load(
            k_ptr + offs_n[:, None] * stride_kn + pid_h * stride_kh +
            offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        if KV_IS_FP8:
            # In-register dequant: fp8 tile -> fp32 * per-head scale,
            # then down to q's dtype for the tensor-core dot.
            k = (k.to(tl.float32) * k_scale).to(q_ptr.dtype.element_ty)

        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vn + pid_h * stride_vh +
            offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        if KV_IS_FP8:
            v = (v.to(tl.float32) * v_scale).to(q_ptr.dtype.element_ty)

        acc += tl.dot(p.to(q_ptr.dtype.element_ty), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + offs_m[:, None] * stride_om + pid_h * stride_oh +
        offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def triton_fp8_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Non-causal prefill attention with fp8 KV dequantized in-kernel.

    q: [L_q, H, D] bf16/fp16; k/v: [L_kv, H, D] float8_e4m3fn (or the
    same dtype as q, in which case scales are ignored at compile
    time); k_scale/v_scale: [H] fp32. Returns [L_q, H, D] in q.dtype.
    """
    L_q, H, D = q.shape
    L_kv = k.shape[0]
    assert k.shape == (L_kv, H, D) and v.shape == (L_kv, H, D)
    assert q.is_contiguous()
    if sm_scale is None:
        sm_scale = D**-0.5

    kv_is_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_is_fp8:
        # Triton loads fp8 via uint8-view + bitcast-free `.to(f32)` is
        # not supported for e4m3 on all versions; reinterpret through
        # the element type directly (triton >= 3.0 supports fp8e4nv
        # pointers).
        pass

    out = torch.empty_like(q)
    grid = (triton.cdiv(L_q, _BLOCK_M), H)
    _fp8_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        L_q,
        L_kv,
        sm_scale,
        H=H,
        D=D,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        KV_IS_FP8=kv_is_fp8,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
