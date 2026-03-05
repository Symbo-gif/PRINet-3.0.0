"""Sweep utilities, phase-to-rate conversion, and oscillation detection.

Provides :func:`phase_to_rate`, :func:`detect_oscillation`, and
:func:`sweep_coupling_params` for parameter-space exploration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .oscillator_state import OscillatorState, _wrap_phase
from .oscillator_models import KuramotoOscillator, OscillatorModel
from .networks import DeltaThetaGammaNetwork

def detect_oscillation(
    r_history: list[float],
    window: int = 20,
    threshold: float = 0.01,
) -> bool:
    """Detect destabilizing oscillations in order-parameter history.

    Computes the windowed variance of the last ``window`` values in
    ``r_history``. If variance exceeds ``threshold``, oscillation is
    flagged.

    Args:
        r_history: List of order-parameter values over time.
        window: Number of recent values to inspect.
        threshold: Variance threshold above which oscillation is detected.

    Returns:
        ``True`` if destabilizing oscillations are detected.

    Example:
        >>> detect_oscillation([0.8, 0.2, 0.9, 0.1] * 5, window=10)
        True
    """
    if len(r_history) < window:
        return False
    recent = r_history[-window:]
    mean = sum(recent) / len(recent)
    var = sum((v - mean) ** 2 for v in recent) / len(recent)
    return var > threshold




def phase_to_rate(
    phase: Tensor,
    amplitude: Tensor,
    mode: str = "soft",
    sparsity: float = 0.1,
    temperature: float = 1.0,
) -> Tensor:
    """Convert oscillatory phase-amplitude representations to rate codes.

    Implements feedforward inhibition (FFI) gating followed by winner-
    take-all (WTA) competition to produce sparse rate-coded outputs.

    The conversion pipeline:
        1. Compute instantaneous rate: ``r_i = A_i · (1 + cos(φ_i)) / 2``
        2. Apply WTA sparsity (soft or hard).

    Args:
        phase: Phase tensor ``(..., N)``.
        amplitude: Amplitude tensor ``(..., N)``.
        mode: WTA mode — ``"soft"`` (differentiable softmax-based),
            ``"hard"`` (top-k selection), or ``"annealed"`` (temperature-
            dependent interpolation between soft and hard).
        sparsity: Target fraction of active units (for hard/annealed
            modes). E.g. ``0.1`` means ~10% active.
        temperature: Temperature for soft/annealed modes. Lower values
            produce sharper (more sparse) outputs.

    Returns:
        Rate-coded output tensor of same shape as input. Non-negative.

    Raises:
        ValueError: If ``mode`` is unknown.

    Example:
        >>> phase = torch.rand(32, 64) * 2 * 3.14159
        >>> amp = torch.ones(32, 64)
        >>> rates = phase_to_rate(phase, amp, mode="soft")
        >>> assert rates.shape == (32, 64)
    """
    # Step 1: Instantaneous rate via FFI gating
    rate = amplitude * (1.0 + torch.cos(phase)) / 2.0

    # Step 2: Winner-take-all sparsity
    if mode == "soft":
        rate = torch.softmax(rate / temperature, dim=-1)
    elif mode == "hard":
        k = max(1, int(rate.shape[-1] * sparsity))
        topk_vals, topk_idx = torch.topk(rate, k, dim=-1)
        sparse_rate = torch.zeros_like(rate)
        sparse_rate.scatter_(-1, topk_idx, topk_vals)
        rate = sparse_rate
    elif mode == "annealed":
        # Interpolate: at high temperature → soft; low → hard
        soft = torch.softmax(rate / temperature, dim=-1)
        k = max(1, int(rate.shape[-1] * sparsity))
        topk_vals, topk_idx = torch.topk(rate, k, dim=-1)
        hard = torch.zeros_like(rate)
        hard.scatter_(-1, topk_idx, topk_vals)
        # Blend factor: sigmoid(1/temp - 1) ∈ (0, 1)
        blend = torch.sigmoid(
            torch.tensor(1.0 / max(temperature, 1e-6) - 1.0)
        )
        rate = (1.0 - blend) * soft + blend * hard
    else:
        raise ValueError(
            f"Unknown phase_to_rate mode '{mode}'. "
            f"Use 'soft', 'hard', or 'annealed'."
        )

    return rate


# =========================================================================
# Q3: Feedforward Inhibition, Feedback Inhibition, DG-Inspired Pipeline
# =========================================================================




def sweep_coupling_params(
    n_oscillators: int = 64,
    k_values: Optional[list[float]] = None,
    m_values: Optional[list[float]] = None,
    n_steps: int = 100,
    dt: float = 0.01,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> list[dict[str, float]]:
    """Sweep coupling strength K and PAC depth m for DeltaThetaGammaNetwork.

    Runs a grid search over ``(K, m)`` pairs, recording the final
    per-band order parameters for each configuration.

    Args:
        n_oscillators: Total oscillators split evenly across bands.
        k_values: Coupling strengths to sweep. Default ``[0.5, 1, 2, 4]``.
        m_values: PAC depths to sweep. Default ``[0.1, 0.3, 0.5, 0.7]``.
        n_steps: Integration steps per configuration.
        dt: Timestep.
        seed: Random seed.
        device: Torch device.

    Returns:
        List of dicts with keys ``K``, ``m``, ``r_delta``, ``r_theta``,
        ``r_gamma``.

    Example:
        >>> results = sweep_coupling_params(n_oscillators=24, n_steps=50)
        >>> assert len(results) > 0
    """
    if k_values is None:
        k_values = [0.5, 1.0, 2.0, 4.0]
    if m_values is None:
        m_values = [0.1, 0.3, 0.5, 0.7]

    n_per_band = max(2, n_oscillators // 3)
    results: list[dict[str, float]] = []

    for K in k_values:
        for m in m_values:
            net = DeltaThetaGammaNetwork(
                n_delta=n_per_band,
                n_theta=n_per_band,
                n_gamma=n_per_band,
                coupling_strength=K,
                pac_depth_dt=m,
                pac_depth_tg=m,
                device=device,
            )
            init = net.create_initial_state(seed=seed)
            final, _ = net.integrate(init, n_steps=n_steps, dt=dt)
            r_d, r_t, r_g = net.order_parameters(final)
            results.append(
                {
                    "K": K,
                    "m": m,
                    "r_delta": r_d.item(),
                    "r_theta": r_t.item(),
                    "r_gamma": r_g.item(),
                }
            )
    return results
