"""Phase-Amplitude Coupling between oscillator bands.

Provides :class:`PhaseAmplitudeCoupling` for cross-frequency coupling
where slow-band phase modulates fast-band amplitude.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .oscillator_state import OscillatorState


class PhaseAmplitudeCoupling:
    """Phase-Amplitude Coupling (PAC) between oscillator bands.

    Implements cross-frequency coupling where the phase of a slow
    oscillator modulates the amplitude of a fast oscillator:

        A_fast(t) = A_0 · [1 + m · cos(φ_slow(t) + φ_offset)]

    where ``m`` is the modulation depth and ``φ_offset`` is a
    learnable phase offset.

    Args:
        modulation_depth: Modulation depth ``m`` ∈ [0, 1].
        amplitude_clamp: Tuple ``(min, max)`` for amplitude clamping.

    Example:
        >>> pac = PhaseAmplitudeCoupling(modulation_depth=0.5)
        >>> slow_phase = torch.zeros(10)
        >>> fast_amp = torch.ones(20)
        >>> modulated = pac.modulate(slow_phase, fast_amp)
    """

    def __init__(
        self,
        modulation_depth: float = 0.3,
        amplitude_clamp: Tuple[float, float] = (1e-6, 10.0),
    ) -> None:
        if not 0.0 <= modulation_depth <= 1.0:
            raise ValueError(
                f"modulation_depth must be in [0, 1], got {modulation_depth}"
            )
        self._m = modulation_depth
        self._amp_min, self._amp_max = amplitude_clamp

    @property
    def modulation_depth(self) -> float:
        """Current modulation depth."""
        return self._m

    @modulation_depth.setter
    def modulation_depth(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"modulation_depth must be in [0, 1], got {value}")
        self._m = value

    def modulate(
        self,
        slow_phase: Tensor,
        fast_amplitude: Tensor,
        phase_offset: float = 0.0,
    ) -> Tensor:
        """Apply PAC modulation to fast-band amplitudes.

        Each fast oscillator's amplitude is modulated by the mean phase
        of the slow band:

            A_out = A_in · [1 + m · cos(mean(φ_slow) + offset)]

        Args:
            slow_phase: Phase of slow oscillators ``(..., N_slow)``.
            fast_amplitude: Amplitude of fast oscillators ``(..., N_fast)``.
            phase_offset: Phase offset for modulation.

        Returns:
            Modulated fast amplitudes, clamped to ``amplitude_clamp``.
        """
        # Mean slow phase (over oscillator dim)
        mean_slow = slow_phase.mean(dim=-1, keepdim=True)  # (..., 1)
        modulation = 1.0 + self._m * torch.cos(mean_slow + phase_offset)
        # Broadcast modulation across fast oscillators
        modulated = fast_amplitude * modulation
        return torch.clamp(modulated, min=self._amp_min, max=self._amp_max)
