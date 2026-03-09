# SPDX-License-Identifier: Apache-2.0
"""
Res2s second-order sampler for LTX-2.3 HQ pipeline.

Ported from upstream:
  LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py
  LTX-2/packages/ltx-core/src/ltx_core/components/diffusion_steps.py

The Res2s sampler uses a two-stage Runge-Kutta step with SDE noise
injection, allowing fewer steps for comparable quality vs Euler.
"""

from __future__ import annotations

import math

import torch


# ─────────────────────────────────────────────────────────────
# Phi functions for Runge-Kutta coefficients
# ─────────────────────────────────────────────────────────────


def phi(j: int, neg_h: float) -> float:
    r"""Compute φⱼ(z) where z = -h (negative step size in log-space).

    φ₁(z) = (e^z - 1) / z
    φ₂(z) = (e^z - 1 - z) / z²
    φⱼ(z) = (e^z - Σₖ₌₀^(j-1) zᵏ/k!) / zʲ

    These functions naturally appear when solving:
        dx/dt = A*x + g(x,t)  (linear drift + nonlinear part)
    """
    if abs(neg_h) < 1e-10:
        # Taylor series for small h to avoid division by zero
        # φⱼ(0) = 1/j!
        return 1.0 / math.factorial(j)

    # Compute the "remainder" sum: Σₖ₌₀^(j-1) z^k/k!
    remainder = sum(
        neg_h**k / math.factorial(k) for k in range(j))

    # φⱼ(z) = (e^z - remainder) / z^j
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(
    h: float,
    phi_cache: dict,
    c2: float = 0.5,
) -> tuple[float, float, float]:
    """Compute res_2s Runge-Kutta coefficients for a step.

    Args:
        h: Step size in log-space = log(sigma / sigma_next).
        phi_cache: Dictionary to cache phi function results.
            Cache key: ``(j, neg_h)``.
        c2: Substep position (default 0.5 = midpoint).

    Returns:
        ``(a21, b1, b2)`` — RK coefficients.
    """

    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    # Substep coefficient
    neg_h_c2 = -h * c2
    phi_1_c2 = get_phi(1, neg_h_c2)
    a21 = c2 * phi_1_c2

    # Final combination weights
    neg_h_full = -h
    phi_2_full = get_phi(2, neg_h_full)
    b2 = phi_2_full / c2

    phi_1_full = get_phi(1, neg_h_full)
    b1 = phi_1_full - b2

    return a21, b1, b2


# ─────────────────────────────────────────────────────────────
# Res2s diffusion step (SDE noise injection)
# ─────────────────────────────────────────────────────────────


def res2s_get_sde_coeff(
    sigma_next: torch.Tensor,
    sigma_up: torch.Tensor | None = None,
    sigma_down: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SDE coefficients for variance-preserving noise.

    Given either ``sigma_down`` or ``sigma_up``, returns the
    mixing coefficients ``(alpha_ratio, sigma_down, sigma_up)``
    used for variance-preserving noise injection.

    Ported from ``Res2sDiffusionStep.get_sde_coeff`` in upstream.
    """
    if sigma_down is not None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        sigma_up = (
            sigma_next**2
            - sigma_down**2 * alpha_ratio**2
        ).clamp(min=0) ** 0.5
    elif sigma_up is not None:
        sigma_up = sigma_up.clamp(max=sigma_next * 0.9999)
        sigma_signal = 1.0 - sigma_next
        sigma_residual = (
            sigma_next**2 - sigma_up**2
        ).clamp(min=0) ** 0.5
        alpha_ratio = sigma_signal + sigma_residual
        sigma_down = sigma_residual / alpha_ratio
    else:
        alpha_ratio = torch.ones_like(sigma_next)
        sigma_down = sigma_next
        sigma_up = torch.zeros_like(sigma_next)

    sigma_up = torch.nan_to_num(sigma_up, 0.0)
    nan_mask = torch.isnan(sigma_down)
    sigma_down[nan_mask] = sigma_next[nan_mask].to(
        sigma_down.dtype)
    alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

    return alpha_ratio, sigma_down, sigma_up


def res2s_step(
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Advance one step with SDE noise injection.

    Ported from ``Res2sDiffusionStep.step`` in upstream.
    """
    alpha_ratio, sigma_down, sigma_up = res2s_get_sde_coeff(
        sigma_next, sigma_up=sigma_next * 0.5)
    output_dtype = denoised_sample.dtype

    if torch.any(sigma_up == 0) or torch.any(sigma_next == 0):
        return denoised_sample

    # Extract epsilon prediction
    eps_next = (sample - denoised_sample) / (sigma - sigma_next)
    denoised_next = sample - sigma * eps_next

    # Mix deterministic and stochastic components
    x_noised = (
        alpha_ratio
        * (denoised_next + sigma_down * eps_next)
        + sigma_up * noise
    )
    return x_noised.to(output_dtype)
