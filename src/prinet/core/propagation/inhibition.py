"""Inhibitory circuits for oscillatory binding (DG conversion).

Provides :class:`FeedforwardInhibition`, :class:`FeedbackInhibition`,
and :class:`DentateGyrusConverter` (FFI → integration → FBI pipeline).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .oscillator_state import OscillatorState, OscillatorSyncError, _wrap_phase


class FeedforwardInhibition:
    """Feedforward Inhibition (FFI) for phase-to-rate conversion.

    Implements fast inhibition with a configurable delay that gates
    phase signals through an exponential decay envelope. The delay
    is expressed in integration steps and controls which temporal
    window of the oscillatory signal is allowed through.

    The gating operation:
        1. Compute instantaneous rate: ``r = A · (1 + cos(φ)) / 2``
        2. Apply delay via phase shift: ``φ_delayed = φ + 2π · delay_frac``
        3. Gate: ``output = r · exp(-α / τ)`` where ``α`` is the
           normalised time since the inhibition window.

    Args:
        delay_steps: Inhibition delay in integration steps.
            Default ``1`` (≈1ms equivalent at dt=0.001).
        tau: Decay time constant controlling the sharpness of
            the inhibition window. Smaller → sharper. Default ``0.05``.
        delay_fraction: Fraction of a full cycle to shift the
            inhibition window (0 to 1). Default ``0.1``.

    Example:
        >>> ffi = FeedforwardInhibition(delay_steps=1, tau=0.05)
        >>> phase = torch.rand(32, 64) * 2 * 3.14159
        >>> amp = torch.ones(32, 64)
        >>> gated = ffi.gate(phase, amp)
        >>> assert gated.shape == (32, 64)
    """

    def __init__(
        self,
        delay_steps: int = 1,
        tau: float = 0.05,
        delay_fraction: float = 0.1,
    ) -> None:
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be non-negative, got {delay_steps}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self._delay_steps = delay_steps
        self._tau = tau
        self._delay_fraction = delay_fraction

    @property
    def delay_steps(self) -> int:
        """Inhibition delay in integration steps."""
        return self._delay_steps

    @property
    def tau(self) -> float:
        """Decay time constant."""
        return self._tau

    def gate(
        self,
        phase: Tensor,
        amplitude: Tensor,
    ) -> Tensor:
        """Apply feedforward inhibition gating.

        Args:
            phase: Phase tensor ``(..., N)``.
            amplitude: Amplitude tensor ``(..., N)``.

        Returns:
            Gated rate signal ``(..., N)``. Non-negative.
        """
        # Instantaneous rate via cosine readout
        rate = amplitude * (1.0 + torch.cos(phase)) / 2.0

        # Apply phase delay: shift by delay_fraction cycles
        delayed_phase = phase + 2.0 * math.pi * self._delay_fraction
        delayed_phase = _wrap_phase(delayed_phase)

        # Exponential decay envelope based on phase alignment
        # When delayed phase aligns with original → max transmission
        alignment = torch.cos(delayed_phase - phase)
        # Map [-1, 1] → [0, 1] for gating strength
        gate_strength = torch.exp((alignment - 1.0) / max(self._tau, 1e-8))

        return rate * gate_strength


class FeedbackInhibition:
    """Feedback Inhibition (FBI) enforcing Winner-Take-All sparsity.

    Implements slow inhibition with a configurable delay that enforces
    top-k sparse selection. Uses a Straight-Through Estimator (STE)
    for gradient flow through the hard selection operation.

    The inhibition pipeline:
        1. Compute soft scores via temperature-scaled softmax.
        2. Select top-k winners (hard selection in forward pass).
        3. Pass gradients through soft scores (STE for backward pass).

    Args:
        k: Number of winners to select. If ``None``, computed from
            ``sparsity`` and input dimension.
        sparsity: Target sparsity level (fraction of active units).
            Used when ``k is None``. Default ``0.1``.
        delay_steps: Feedback delay in integration steps (controls
            how many steps of activity are accumulated before
            competition). Default ``20``.
        temperature: Softmax temperature for soft score computation.
            Default ``1.0``.

    Example:
        >>> fbi = FeedbackInhibition(k=6, temperature=1.0)
        >>> rates = torch.randn(32, 64).abs()
        >>> sparse = fbi.compete(rates)
        >>> active = (sparse > 0).float().mean()
    """

    def __init__(
        self,
        k: Optional[int] = None,
        sparsity: float = 0.1,
        delay_steps: int = 20,
        temperature: float = 1.0,
    ) -> None:
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
        self._k = k
        self._sparsity = sparsity
        self._delay_steps = delay_steps
        self._temperature = max(temperature, 1e-8)

    @property
    def delay_steps(self) -> int:
        """Feedback delay in steps."""
        return self._delay_steps

    @property
    def temperature(self) -> float:
        """Softmax temperature."""
        return self._temperature

    def compete(self, rates: Tensor) -> Tensor:
        """Apply feedback inhibition to enforce sparse WTA.

        Uses Straight-Through Estimator: forward pass uses hard
        top-k mask, backward pass uses soft (differentiable) scores.

        Args:
            rates: Input rate tensor ``(..., N)``. Non-negative.

        Returns:
            Sparse rate tensor ``(..., N)`` with only top-k active.
        """
        N = rates.shape[-1]
        k = self._k if self._k is not None else max(1, int(N * self._sparsity))
        k = min(k, N)

        # Soft scores for gradient flow
        soft_scores = torch.softmax(rates / self._temperature, dim=-1)

        # Hard top-k selection
        topk_vals, topk_idx = torch.topk(rates, k, dim=-1)
        hard_mask = torch.zeros_like(rates)
        hard_mask.scatter_(-1, topk_idx, 1.0)

        # STE: forward = hard mask × rates, backward flows through soft scores
        # Detach the (hard - soft) difference so gradients flow through soft only
        soft_selected = soft_scores * rates
        hard_selected = hard_mask * rates
        return hard_selected + (soft_selected - soft_selected.detach())


class DentateGyrusConverter:
    """Dentate-Gyrus-inspired sparse phase-to-rate conversion pipeline.

    Implements the bio-inspired conversion from dense oscillatory phase
    codes to sparse rate codes, modelled on the dentate gyrus circuit:

        phase_input → FFI gating → temporal integration → FBI competition
        → sparse rate output

    The pipeline is fully differentiable (FBI uses Straight-Through
    Estimator for hard sparsity).

    Args:
        n_oscillators: Number of input oscillators.
        k: Number of active units in output. If ``None``, computed
            from ``target_sparsity``.
        target_sparsity: Target fraction of active units. Default ``0.1``.
        ffi_delay: FFI delay in integration steps. Default ``1``.
        ffi_tau: FFI decay time constant. Default ``0.05``.
        fbi_delay: FBI delay in integration steps. Default ``20``.
        fbi_temperature: FBI softmax temperature. Default ``1.0``.
        integration_alpha: EMA decay for temporal integration.
            Higher values → more smoothing. Default ``0.95``.

    Example:
        >>> dg = DentateGyrusConverter(64, k=6)
        >>> phase = torch.rand(32, 64) * 2 * 3.14159
        >>> amp = torch.ones(32, 64)
        >>> sparse_rates = dg.convert(phase, amp)
        >>> assert sparse_rates.shape == (32, 64)
        >>> assert (sparse_rates > 0).float().mean() < 0.2
    """

    def __init__(
        self,
        n_oscillators: int,
        k: Optional[int] = None,
        target_sparsity: float = 0.1,
        ffi_delay: int = 1,
        ffi_tau: float = 0.05,
        fbi_delay: int = 20,
        fbi_temperature: float = 1.0,
        integration_alpha: float = 0.95,
    ) -> None:
        self._n_oscillators = n_oscillators
        self._ffi = FeedforwardInhibition(
            delay_steps=ffi_delay,
            tau=ffi_tau,
        )
        self._fbi = FeedbackInhibition(
            k=k,
            sparsity=target_sparsity,
            delay_steps=fbi_delay,
            temperature=fbi_temperature,
        )
        self._alpha = integration_alpha

    @property
    def ffi(self) -> FeedforwardInhibition:
        """Feedforward inhibition component."""
        return self._ffi

    @property
    def fbi(self) -> FeedbackInhibition:
        """Feedback inhibition component."""
        return self._fbi

    def convert(
        self,
        phase: Tensor,
        amplitude: Tensor,
        n_integration_steps: int = 5,
    ) -> Tensor:
        """Run the full DG conversion pipeline.

        Args:
            phase: Phase tensor ``(B, N)`` or ``(N,)``.
            amplitude: Amplitude tensor, same shape.
            n_integration_steps: Number of temporal integration EMA
                steps. More steps → smoother rate estimates.

        Returns:
            Sparse rate tensor of same shape as input.
        """
        was_1d = phase.dim() == 1
        if was_1d:
            phase = phase.unsqueeze(0)
            amplitude = amplitude.unsqueeze(0)

        # Step 1: FFI gating — fast inhibition gates the phase signal
        gated = self._ffi.gate(phase, amplitude)

        # Step 2: Temporal integration via EMA
        # Simulate multiple integration steps with the FFI-gated signal
        integrated = gated.clone()
        phase_step = phase.clone()
        for _ in range(n_integration_steps - 1):
            # Advance phase by one notional step
            phase_step = _wrap_phase(phase_step + 0.1 * 2.0 * math.pi)
            new_gated = self._ffi.gate(phase_step, amplitude)
            integrated = self._alpha * integrated + (1.0 - self._alpha) * new_gated

        # Step 3: FBI competition — slow inhibition enforces WTA sparsity
        sparse_rate = self._fbi.compete(integrated)

        if was_1d:
            return sparse_rate.squeeze(0)
        return sparse_rate
