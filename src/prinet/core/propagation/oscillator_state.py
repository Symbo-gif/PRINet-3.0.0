"""Oscillator state containers and numerical helpers.

Provides :class:`OscillatorState`, :class:`OscillatorSyncError`, and
the low-level helper functions (``_wrap_phase``, ``_safe_phase_diff``,
``_clamp_finite``, ``_build_phase_knn_index``) used across all
oscillator dynamics submodules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants for numerical stability (P0 NaN fix)
# ---------------------------------------------------------------------------

_TWO_PI: float = 2.0 * math.pi

_SPARSE_EPS: float = 1e-8
"""Epsilon guard for sparse coupling numerical stability."""

_DERIV_CLAMP: float = 1e4
"""Upper bound for derivative magnitudes to prevent NaN propagation."""


def _wrap_phase(phase: Tensor) -> Tensor:
    """Wrap phase tensor to [0, 2π).

    Args:
        phase: Phase tensor of any shape.

    Returns:
        Wrapped phase in [0, 2π).
    """
    return phase % _TWO_PI


def _safe_phase_diff(phi_j: Tensor, phi_i: Tensor) -> Tensor:
    """Compute wrapped phase difference on the circle.

    Uses ``atan2(sin, cos)`` to produce numerically stable differences
    in ``(-π, π]`` instead of raw subtraction which can accumulate
    floating-point error at large N.

    Args:
        phi_j: Target phases, any shape.
        phi_i: Source phases, broadcastable to ``phi_j``.

    Returns:
        Wrapped phase differences in ``(-π, π]``.
    """
    raw = phi_j - phi_i
    return torch.atan2(torch.sin(raw), torch.cos(raw))


def _clamp_finite(t: Tensor, limit: float = _DERIV_CLAMP) -> Tensor:
    """Clamp tensor values and replace any NaN/Inf with zero.

    Args:
        t: Input tensor.
        limit: Symmetric clamp bound.

    Returns:
        Clamped, finite tensor.
    """
    t = torch.clamp(t, -limit, limit)
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _build_phase_knn_index(
    flat_phase: Tensor,
    k: int,
) -> Tensor:
    """Build k-nearest-phase-neighbour index on the phase circle.

    Uses sort-based O(N log N) algorithm: sort phases, then for each
    oscillator take k/2 left and k - k//2 right neighbours in sorted
    order (wrapping around the ring).

    Args:
        flat_phase: Phases of shape ``(B, N)`` already flattened.
        k: Number of neighbours per oscillator.  Must satisfy
           ``1 <= k < N``.

    Returns:
        Neighbour index tensor of shape ``(B, N, k)`` mapping each
        oscillator to its k nearest phase neighbours (original indices).
    """
    B, N = flat_phase.shape
    device = flat_phase.device

    sorted_phase, sort_idx = torch.sort(flat_phase, dim=-1)  # (B, N)

    # Inverse map: inv_idx[b, sort_idx[b, j]] = j
    inv_idx = torch.empty_like(sort_idx)
    inv_idx.scatter_(
        1,
        sort_idx,
        torch.arange(N, device=device).expand(B, -1),
    )
    positions = inv_idx  # (B, N)

    half_k = k // 2
    left_offsets = -torch.arange(1, half_k + 1, device=device)
    right_offsets = torch.arange(1, k - half_k + 1, device=device)
    all_offsets = torch.cat([left_offsets, right_offsets])  # (k,)

    # Neighbour positions in sorted array, wrapped mod N
    nbr_sorted_pos = (
        positions.unsqueeze(-1) + all_offsets.unsqueeze(0).unsqueeze(0)
    ) % N  # (B, N, k)

    # Map sorted positions back to original oscillator indices
    nbr_idx = sort_idx.gather(1, nbr_sorted_pos.reshape(B, -1)).reshape(B, N, k)

    return nbr_idx


class OscillatorSyncError(Exception):
    """Raised when oscillator synchronization drops below safe thresholds."""

    pass


@dataclass
class OscillatorState:
    """Container for the full state of a coupled oscillator system.

    Attributes:
        phase: Phase of each oscillator, shape ``(N,)`` or ``(B, N)``.
        amplitude: Amplitude of each oscillator, shape ``(N,)`` or ``(B, N)``.
        frequency: Natural frequency of each oscillator, same shape.
        freq_band: Optional integer labels per oscillator indicating
            frequency band membership (0=Delta, 1=Theta, 2=Gamma).
            Shape matches ``phase``. Default ``None``.
    """

    phase: Tensor
    amplitude: Tensor
    frequency: Tensor
    freq_band: Optional[Tensor] = None

    @property
    def n_bands(self) -> int:
        """Number of distinct frequency bands.

        Returns 0 if ``freq_band`` is ``None``.
        """
        if self.freq_band is None:
            return 0
        return int(torch.unique(self.freq_band).numel())

    @staticmethod
    def create_random(
        n_oscillators: int,
        batch_size: Optional[int] = None,
        freq_range: Tuple[float, float] = (0.1, 10.0),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> "OscillatorState":
        """Create a random initial oscillator state.

        Args:
            n_oscillators: Number of oscillators ``N``.
            batch_size: If not ``None``, create batched state ``(B, N)``.
            freq_range: ``(min_freq, max_freq)`` for uniform initialization.
            device: Torch device.
            dtype: Data type.
            seed: Random seed for reproducibility.

        Returns:
            Randomly initialized ``OscillatorState``.
        """
        if seed is not None:
            gen = torch.Generator(device=device or torch.device("cpu"))
            gen.manual_seed(seed)
        else:
            gen = None

        shape = (
            (batch_size, n_oscillators) if batch_size is not None else (n_oscillators,)
        )

        phase = torch.rand(shape, device=device, dtype=dtype, generator=gen)
        phase = phase * 2.0 * math.pi  # Uniform [0, 2π)

        amplitude = torch.ones(shape, device=device, dtype=dtype)

        freq_lo, freq_hi = freq_range
        frequency = torch.rand(shape, device=device, dtype=dtype, generator=gen)
        frequency = frequency * (freq_hi - freq_lo) + freq_lo

        return OscillatorState(phase=phase, amplitude=amplitude, frequency=frequency)

    @staticmethod
    def create_synchronized(
        n_oscillators: int,
        base_frequency: float = 1.0,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "OscillatorState":
        """Create a fully-synchronized initial state.

        All oscillators begin at phase 0, unit amplitude, and the
        same base frequency. Useful for testing synchronization
        stability.

        Args:
            n_oscillators: Number of oscillators.
            base_frequency: Natural frequency for all oscillators.
            batch_size: Optional batch dimension.
            device: Torch device.
            dtype: Data type.

        Returns:
            Synchronized ``OscillatorState``.
        """
        shape = (
            (batch_size, n_oscillators) if batch_size is not None else (n_oscillators,)
        )
        phase = torch.zeros(shape, device=device, dtype=dtype)
        amplitude = torch.ones(shape, device=device, dtype=dtype)
        frequency = torch.full(shape, base_frequency, device=device, dtype=dtype)
        return OscillatorState(phase=phase, amplitude=amplitude, frequency=frequency)

    def clone(self) -> "OscillatorState":
        """Create a deep copy of this state.

        Returns:
            A new ``OscillatorState`` with cloned tensors.
        """
        return OscillatorState(
            phase=self.phase.clone(),
            amplitude=self.amplitude.clone(),
            frequency=self.frequency.clone(),
            freq_band=self.freq_band.clone() if self.freq_band is not None else None,
        )

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators in the system."""
        return self.phase.shape[-1]

    @property
    def device(self) -> torch.device:
        """Device of the state tensors."""
        return self.phase.device
