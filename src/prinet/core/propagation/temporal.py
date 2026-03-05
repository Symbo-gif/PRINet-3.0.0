"""Temporal phase propagation across video / sequence frames.

Provides :class:`TemporalPhasePropagator` for multi-frame phase
coherence tracking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .oscillator_state import OscillatorState, _wrap_phase


class TemporalPhasePropagator:
    """Temporal phase propagation for multi-frame oscillatory binding.

    Implements E.2: initializes oscillator phases at frame *t* from
    the final phases at frame *t−1*, enabling persistent object
    binding across temporal sequences.

    The propagation rule blends prior phase information with fresh
    input-driven initialization:

        φ_init(t) = α · φ_final(t−1) + (1 − α) · φ_input(t)

    where ``α`` (``carry_strength``) controls the temporal inertia.
    Values near 1.0 produce strong phase continuity (persistent
    binding); values near 0.0 produce fresh re-binding each frame.

    Amplitude carry uses exponential decay:

        A_init(t) = β · A_final(t−1) + (1 − β) · A_input(t)

    Args:
        carry_strength: Phase carry-over strength α ∈ [0, 1].
            Default ``0.8`` (strong temporal binding).
        amplitude_decay: Amplitude carry-over β ∈ [0, 1].
            Default ``0.5`` (moderate persistence).

    Example:
        >>> prop = TemporalPhasePropagator(carry_strength=0.8)
        >>> prev_phase = torch.rand(16, 44) * 2 * 3.14159
        >>> prev_amp = torch.ones(16, 44)
        >>> input_phase = torch.rand(16, 44) * 2 * 3.14159
        >>> input_amp = torch.ones(16, 44)
        >>> new_phase, new_amp = prop.propagate(
        ...     prev_phase, prev_amp, input_phase, input_amp
        ... )
    """

    def __init__(
        self,
        carry_strength: float = 0.8,
        amplitude_decay: float = 0.5,
    ) -> None:
        if not 0.0 <= carry_strength <= 1.0:
            raise ValueError(f"carry_strength must be in [0, 1], got {carry_strength}")
        if not 0.0 <= amplitude_decay <= 1.0:
            raise ValueError(
                f"amplitude_decay must be in [0, 1], got {amplitude_decay}"
            )
        self._alpha = carry_strength
        self._beta = amplitude_decay

    @property
    def carry_strength(self) -> float:
        """Phase carry-over strength α."""
        return self._alpha

    @property
    def amplitude_decay(self) -> float:
        """Amplitude carry-over β."""
        return self._beta

    def propagate(
        self,
        prev_phase: Tensor,
        prev_amplitude: Tensor,
        input_phase: Tensor,
        input_amplitude: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate phase state from frame t−1 to frame t.

        Blends carried-over phase/amplitude from the previous frame
        with input-driven initialization for the current frame.

        Args:
            prev_phase: Final phases from frame t−1, ``(B, N)`` or ``(N,)``.
            prev_amplitude: Final amplitudes from frame t−1.
            input_phase: Input-driven phase init for frame t.
            input_amplitude: Input-driven amplitude init for frame t.

        Returns:
            ``(init_phase, init_amplitude)`` — initial state for frame t.
        """
        # Phase blending using circular weighted mean
        # Convert to complex phasors, blend, recover angle
        alpha = self._alpha
        z_prev = torch.exp(1j * prev_phase.to(torch.float64))
        z_input = torch.exp(1j * input_phase.to(torch.float64))
        z_blend = alpha * z_prev + (1.0 - alpha) * z_input
        # Recover phase from blended phasor (handles wraparound correctly)
        blended_phase = torch.angle(z_blend).to(prev_phase.dtype)
        # Wrap to [0, 2π)
        blended_phase = _wrap_phase(blended_phase)

        # Amplitude blending: simple linear interpolation
        beta = self._beta
        blended_amp = beta * prev_amplitude + (1.0 - beta) * input_amplitude
        blended_amp = torch.clamp(blended_amp, min=1e-6, max=10.0)

        return blended_phase, blended_amp

    def propagate_sequence(
        self,
        dynamics: "DiscreteDeltaThetaGamma",
        input_phases: Tensor,
        input_amplitudes: Tensor,
        n_steps: int = 10,
        dt: float = 0.01,
    ) -> Tuple[Tensor, Tensor, list[Tensor]]:
        """Process a sequence of frames with temporal phase propagation.

        For each frame in the sequence:
        1. Blend previous final phase with current input phase.
        2. Run dynamics integration for ``n_steps``.
        3. Store final phase as carry-over for next frame.

        Args:
            dynamics: :class:`DiscreteDeltaThetaGamma` instance for stepping.
            input_phases: ``(B, T, N)`` — input phases for T frames.
            input_amplitudes: ``(B, T, N)`` — input amplitudes.
            n_steps: Integration steps per frame.
            dt: Timestep for dynamics.

        Returns:
            ``(final_phases, final_amplitudes, all_correlations)`` where:
                - ``final_phases``: ``(B, T, N)`` — output phases per frame.
                - ``final_amplitudes``: ``(B, T, N)`` — output amplitudes.
                - ``all_correlations``: list of ``(B,)`` inter-frame
                  correlation tensors (length ``T-1``).
        """
        from prinet.core.measurement import inter_frame_phase_correlation

        B, T, N = input_phases.shape
        device = input_phases.device
        dtype = input_phases.dtype

        out_phases = torch.zeros(B, T, N, device=device, dtype=dtype)
        out_amps = torch.zeros(B, T, N, device=device, dtype=dtype)
        correlations: list[Tensor] = []

        prev_phase: Optional[Tensor] = None
        prev_amp: Optional[Tensor] = None

        for t in range(T):
            ip = input_phases[:, t, :]  # (B, N)
            ia = input_amplitudes[:, t, :]  # (B, N)

            if prev_phase is not None:
                # Blend with previous frame's final state
                init_phase, init_amp = self.propagate(
                    prev_phase, prev_amp, ip, ia  # type: ignore[arg-type]
                )
            else:
                init_phase = _wrap_phase(ip)
                init_amp = torch.clamp(ia, min=1e-6, max=10.0)

            # Run dynamics
            final_phase, final_amp = dynamics.integrate(
                init_phase, init_amp, n_steps=n_steps, dt=dt
            )

            out_phases[:, t, :] = final_phase
            out_amps[:, t, :] = final_amp

            # Compute inter-frame correlation
            if prev_phase is not None:
                corr = inter_frame_phase_correlation(final_phase, prev_phase)
                correlations.append(corr)

            prev_phase = final_phase.detach()
            prev_amp = final_amp.detach()

        return out_phases, out_amps, correlations
