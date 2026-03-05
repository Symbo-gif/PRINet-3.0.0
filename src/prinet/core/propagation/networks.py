"""Hierarchical oscillatory networks for PRINet.

Provides :class:`ThetaGammaNetwork` (2-band),
:class:`DeltaThetaGammaNetwork` (3-band continuous ODE), and
:class:`DiscreteDeltaThetaGamma` (3-band trainable discrete-time).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .coupling import PhaseAmplitudeCoupling
from .integrators import ExponentialIntegrator, MultiRateIntegrator
from .oscillator_models import KuramotoOscillator, OscillatorModel
from .oscillator_state import (
    _TWO_PI,
    OscillatorState,
    OscillatorSyncError,
    _wrap_phase,
)


class ThetaGammaNetwork:
    """Two-frequency (Theta-Gamma) hierarchical oscillator network.

    Composes Theta-band (~6 Hz) and Gamma-band (~40 Hz) oscillator
    populations with phase-amplitude coupling (PAC) between them.
    The Theta phase modulates Gamma amplitudes.

    This serves as a 2-frequency baseline for binding-capacity
    experiments. Expected binding capacity ≈ 7 items.

    Args:
        n_theta: Number of Theta-band oscillators.
        n_gamma: Number of Gamma-band oscillators.
        coupling_strength: Intra-band coupling K.
        pac_depth: PAC modulation depth m.
        theta_freq: Center frequency for Theta band (Hz).
        gamma_freq: Center frequency for Gamma band (Hz).
        sparse_k: k for sparse k-NN coupling (``None`` for auto).
        device: Torch device.
        dtype: Data type.

    Example:
        >>> net = ThetaGammaNetwork(n_theta=16, n_gamma=64)
        >>> state = net.create_initial_state(seed=42)
        >>> final = net.step(state, dt=0.01)
    """

    def __init__(
        self,
        n_theta: int = 16,
        n_gamma: int = 64,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
        theta_freq: float = 6.0,
        gamma_freq: float = 40.0,
        sparse_k: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._n_theta = n_theta
        self._n_gamma = n_gamma
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        self._theta_model = KuramotoOscillator(
            n_oscillators=n_theta,
            coupling_strength=coupling_strength,
            coupling_mode="sparse_knn",
            sparse_k=sparse_k,
            device=self._device,
            dtype=dtype,
        )
        self._gamma_model = KuramotoOscillator(
            n_oscillators=n_gamma,
            coupling_strength=coupling_strength,
            coupling_mode="sparse_knn",
            sparse_k=sparse_k,
            device=self._device,
            dtype=dtype,
        )
        self._pac = PhaseAmplitudeCoupling(modulation_depth=pac_depth)
        self._gamma_integrator = MultiRateIntegrator(
            sub_steps=max(1, int(gamma_freq / max(theta_freq, 1e-6))),
        )

    @property
    def n_theta(self) -> int:
        """Number of Theta-band oscillators."""
        return self._n_theta

    @property
    def n_gamma(self) -> int:
        """Number of Gamma-band oscillators."""
        return self._n_gamma

    @property
    def n_total(self) -> int:
        """Total number of oscillators."""
        return self._n_theta + self._n_gamma

    def create_initial_state(
        self, seed: Optional[int] = None
    ) -> Tuple[OscillatorState, OscillatorState]:
        """Create initial states for both bands.

        Args:
            seed: Random seed.

        Returns:
            Tuple ``(theta_state, gamma_state)``.
        """
        theta_state = OscillatorState.create_random(
            self._n_theta,
            freq_range=(4.0, 8.0),
            device=self._device,
            dtype=self._dtype,
            seed=seed,
        )
        gamma_seed = seed + 1000 if seed is not None else None
        gamma_state = OscillatorState.create_random(
            self._n_gamma,
            freq_range=(30.0, 50.0),
            device=self._device,
            dtype=self._dtype,
            seed=gamma_seed,
        )
        return theta_state, gamma_state

    def step(
        self,
        state: Tuple[OscillatorState, OscillatorState],
        dt: float = 0.01,
    ) -> Tuple[OscillatorState, OscillatorState]:
        """Advance both bands by one outer timestep with PAC.

        Args:
            state: Tuple ``(theta_state, gamma_state)``.
            dt: Outer timestep.

        Returns:
            Updated ``(theta_state, gamma_state)``.
        """
        theta_state, gamma_state = state

        # Step Theta band (slow, single step)
        new_theta = self._theta_model.step(theta_state, dt=dt)

        # Apply PAC: Theta phase modulates Gamma amplitude
        gamma_state = OscillatorState(
            phase=gamma_state.phase,
            amplitude=self._pac.modulate(new_theta.phase, gamma_state.amplitude),
            frequency=gamma_state.frequency,
        )

        # Step Gamma band (fast, multi-rate sub-stepping)
        new_gamma = self._gamma_integrator.step(self._gamma_model, gamma_state, dt=dt)

        return new_theta, new_gamma

    def integrate(
        self,
        state: Tuple[OscillatorState, OscillatorState],
        n_steps: int,
        dt: float = 0.01,
        record_trajectory: bool = False,
    ) -> Tuple[
        Tuple[OscillatorState, OscillatorState],
        Optional[list[Tuple[OscillatorState, OscillatorState]]],
    ]:
        """Integrate both bands for multiple outer steps.

        Args:
            state: Initial ``(theta_state, gamma_state)``.
            n_steps: Number of outer steps.
            dt: Outer timestep.
            record_trajectory: Record intermediates.

        Returns:
            ``(final_states, trajectory)``.
        """
        trajectory: Optional[list[Tuple[OscillatorState, OscillatorState]]] = (
            [] if record_trajectory else None
        )
        current = state
        for _ in range(n_steps):
            current = self.step(current, dt=dt)
            if trajectory is not None:
                trajectory.append((current[0].clone(), current[1].clone()))
        return current, trajectory

    def order_parameters(
        self, state: Tuple[OscillatorState, OscillatorState]
    ) -> Tuple[Tensor, Tensor]:
        """Compute per-band order parameters.

        Args:
            state: ``(theta_state, gamma_state)``.

        Returns:
            ``(r_theta, r_gamma)``.
        """
        from prinet.core.measurement import kuramoto_order_parameter

        r_theta = kuramoto_order_parameter(state[0].phase)
        r_gamma = kuramoto_order_parameter(state[1].phase)
        return r_theta, r_gamma


class DeltaThetaGammaNetwork:
    """Three-frequency (Delta-Theta-Gamma) hierarchical oscillator network.

    Composes Delta (~2 Hz), Theta (~6 Hz), and Gamma (~40 Hz) oscillator
    populations with cascaded phase-amplitude coupling:

        Delta phase → modulates Theta amplitude (PAC₁)
        Theta phase → modulates Gamma amplitude (PAC₂)

    This three-level hierarchy is inspired by neural oscillatory binding
    and is hypothesized to achieve binding capacity > 7 items (the
    two-frequency limit).

    Args:
        n_delta: Number of Delta-band oscillators.
        n_theta: Number of Theta-band oscillators.
        n_gamma: Number of Gamma-band oscillators.
        coupling_strength: Intra-band coupling K.
        pac_depth_dt: Delta→Theta PAC modulation depth.
        pac_depth_tg: Theta→Gamma PAC modulation depth.
        delta_freq: Center frequency for Delta band (Hz).
        theta_freq: Center frequency for Theta band (Hz).
        gamma_freq: Center frequency for Gamma band (Hz).
        sparse_k: k for sparse k-NN coupling (``None`` for auto).
        device: Torch device.
        dtype: Data type.

    Example:
        >>> net = DeltaThetaGammaNetwork(n_delta=8, n_theta=16, n_gamma=64)
        >>> state = net.create_initial_state(seed=42)
        >>> final = net.step(state, dt=0.01)
    """

    def __init__(
        self,
        n_delta: int = 8,
        n_theta: int = 16,
        n_gamma: int = 64,
        coupling_strength: float = 2.0,
        pac_depth_dt: float = 0.3,
        pac_depth_tg: float = 0.3,
        delta_freq: float = 2.0,
        theta_freq: float = 6.0,
        gamma_freq: float = 40.0,
        sparse_k: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._n_delta = n_delta
        self._n_theta = n_theta
        self._n_gamma = n_gamma
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        self._delta_model = KuramotoOscillator(
            n_oscillators=n_delta,
            coupling_strength=coupling_strength,
            coupling_mode="sparse_knn",
            sparse_k=sparse_k,
            device=self._device,
            dtype=dtype,
        )
        self._theta_model = KuramotoOscillator(
            n_oscillators=n_theta,
            coupling_strength=coupling_strength,
            coupling_mode="sparse_knn",
            sparse_k=sparse_k,
            device=self._device,
            dtype=dtype,
        )
        self._gamma_model = KuramotoOscillator(
            n_oscillators=n_gamma,
            coupling_strength=coupling_strength,
            coupling_mode="sparse_knn",
            sparse_k=sparse_k,
            device=self._device,
            dtype=dtype,
        )

        self._pac_dt = PhaseAmplitudeCoupling(modulation_depth=pac_depth_dt)
        self._pac_tg = PhaseAmplitudeCoupling(modulation_depth=pac_depth_tg)

        # Multi-rate: Theta takes ~3 sub-steps per Delta step,
        # Gamma takes ~20 sub-steps per Delta step
        self._theta_integrator = MultiRateIntegrator(
            sub_steps=max(1, int(theta_freq / max(delta_freq, 1e-6))),
        )
        self._gamma_integrator = MultiRateIntegrator(
            sub_steps=max(1, int(gamma_freq / max(delta_freq, 1e-6))),
        )

    @property
    def n_delta(self) -> int:
        """Number of Delta-band oscillators."""
        return self._n_delta

    @property
    def n_theta(self) -> int:
        """Number of Theta-band oscillators."""
        return self._n_theta

    @property
    def n_gamma(self) -> int:
        """Number of Gamma-band oscillators."""
        return self._n_gamma

    @property
    def n_total(self) -> int:
        """Total number of oscillators."""
        return self._n_delta + self._n_theta + self._n_gamma

    def create_initial_state(
        self, seed: Optional[int] = None
    ) -> Tuple[OscillatorState, OscillatorState, OscillatorState]:
        """Create initial states for all three bands.

        Args:
            seed: Random seed.

        Returns:
            Tuple ``(delta_state, theta_state, gamma_state)``.
        """
        delta_state = OscillatorState.create_random(
            self._n_delta,
            freq_range=(1.0, 4.0),
            device=self._device,
            dtype=self._dtype,
            seed=seed,
        )
        theta_seed = seed + 1000 if seed is not None else None
        theta_state = OscillatorState.create_random(
            self._n_theta,
            freq_range=(4.0, 8.0),
            device=self._device,
            dtype=self._dtype,
            seed=theta_seed,
        )
        gamma_seed = seed + 2000 if seed is not None else None
        gamma_state = OscillatorState.create_random(
            self._n_gamma,
            freq_range=(30.0, 50.0),
            device=self._device,
            dtype=self._dtype,
            seed=gamma_seed,
        )
        return delta_state, theta_state, gamma_state

    def step(
        self,
        state: Tuple[OscillatorState, OscillatorState, OscillatorState],
        dt: float = 0.01,
    ) -> Tuple[OscillatorState, OscillatorState, OscillatorState]:
        """Advance all three bands by one outer timestep with cascaded PAC.

        Args:
            state: ``(delta_state, theta_state, gamma_state)``.
            dt: Outer timestep.

        Returns:
            Updated ``(delta_state, theta_state, gamma_state)``.
        """
        delta_state, theta_state, gamma_state = state

        # Step Delta band (slowest, single step)
        new_delta = self._delta_model.step(delta_state, dt=dt)

        # PAC₁: Delta phase → modulates Theta amplitude
        theta_state = OscillatorState(
            phase=theta_state.phase,
            amplitude=self._pac_dt.modulate(new_delta.phase, theta_state.amplitude),
            frequency=theta_state.frequency,
        )

        # Step Theta band (multi-rate)
        new_theta = self._theta_integrator.step(self._theta_model, theta_state, dt=dt)

        # PAC₂: Theta phase → modulates Gamma amplitude
        gamma_state = OscillatorState(
            phase=gamma_state.phase,
            amplitude=self._pac_tg.modulate(new_theta.phase, gamma_state.amplitude),
            frequency=gamma_state.frequency,
        )

        # Step Gamma band (fastest, multi-rate)
        new_gamma = self._gamma_integrator.step(self._gamma_model, gamma_state, dt=dt)

        return new_delta, new_theta, new_gamma

    def integrate(
        self,
        state: Tuple[OscillatorState, OscillatorState, OscillatorState],
        n_steps: int,
        dt: float = 0.01,
        record_trajectory: bool = False,
    ) -> Tuple[
        Tuple[OscillatorState, OscillatorState, OscillatorState],
        Optional[list[Tuple[OscillatorState, OscillatorState, OscillatorState]]],
    ]:
        """Integrate all three bands for multiple outer steps.

        Args:
            state: Initial ``(delta, theta, gamma)`` states.
            n_steps: Number of outer steps.
            dt: Outer timestep.
            record_trajectory: Record intermediates.

        Returns:
            ``(final_states, trajectory)``.
        """
        trajectory: Optional[
            list[Tuple[OscillatorState, OscillatorState, OscillatorState]]
        ] = ([] if record_trajectory else None)
        current = state
        for _ in range(n_steps):
            current = self.step(current, dt=dt)
            if trajectory is not None:
                trajectory.append(
                    (
                        current[0].clone(),
                        current[1].clone(),
                        current[2].clone(),
                    )
                )
        return current, trajectory

    def order_parameters(
        self,
        state: Tuple[OscillatorState, OscillatorState, OscillatorState],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute per-band order parameters.

        Args:
            state: ``(delta_state, theta_state, gamma_state)``.

        Returns:
            ``(r_delta, r_theta, r_gamma)``.
        """
        from prinet.core.measurement import kuramoto_order_parameter

        return (
            kuramoto_order_parameter(state[0].phase),
            kuramoto_order_parameter(state[1].phase),
            kuramoto_order_parameter(state[2].phase),
        )


# =========================================================================
# Year 2 Q1 — Workstream A: Discrete-Time Multi-Rate Dynamics
# =========================================================================


class DiscreteDeltaThetaGamma(nn.Module):
    """Discrete-time multi-rate hierarchical oscillator network.

    Replaces :class:`DeltaThetaGammaNetwork` with learned discrete update
    rules that approximate hierarchical dynamics in a **fixed number of
    steps** — no inner ODE loop, no multi-rate sub-stepping.

    Architecture per macro step:

    1. **Phase advance:** ``φ_new = φ + 2π·f·dt + W_coupling @ sin(φ_j − φ_i)``
       using learned coupling weights per band.
    2. **PAC gating (multiplicative):**

       - Theta amplitudes gated by delta phase:
         ``a_θ *= σ(W_dt @ [cos(φ_δ), sin(φ_δ)] + b_dt)``
       - Gamma amplitudes gated by theta phase:
         ``a_γ *= σ(W_tg @ [cos(φ_θ), sin(φ_θ)] + b_tg)``

    3. **Amplitude update:** Soft-clamped Stuart-Landau-like decay/growth
       ``a_new = a + dt · a · (μ − |a|²)`` with learnable ``μ`` per band.

    All operations are batched: ``(B, N)`` tensors throughout, no per-sample
    loop.

    Args:
        n_delta: Number of Delta-band oscillators.
        n_theta: Number of Theta-band oscillators.
        n_gamma: Number of Gamma-band oscillators.
        coupling_strength: Initial intra-band coupling magnitude.
        pac_depth: Initial PAC gate bias (higher → stronger gating).
        delta_freq: Center frequency for Delta band (Hz).
        theta_freq: Center frequency for Theta band (Hz).
        gamma_freq: Center frequency for Gamma band (Hz).

    Example:
        >>> net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=32)
        >>> phase = torch.rand(16, 44) * 2 * 3.14159
        >>> amp = torch.ones(16, 44)
        >>> new_phase, new_amp = net.step(phase, amp, dt=0.01)
        >>> assert new_phase.shape == (16, 44)
    """

    def __init__(
        self,
        n_delta: int = 8,
        n_theta: int = 16,
        n_gamma: int = 64,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
        delta_freq: float = 2.0,
        theta_freq: float = 6.0,
        gamma_freq: float = 40.0,
    ) -> None:
        super().__init__()

        self._n_delta = n_delta
        self._n_theta = n_theta
        self._n_gamma = n_gamma
        self._n_total = n_delta + n_theta + n_gamma

        # Per-band natural frequencies (learnable around center)
        self.delta_freq = nn.Parameter(torch.full((n_delta,), delta_freq))
        self.theta_freq = nn.Parameter(torch.full((n_theta,), theta_freq))
        self.gamma_freq = nn.Parameter(torch.full((n_gamma,), gamma_freq))

        # Intra-band coupling: small learned matrix per band
        # Initialized near Kuramoto all-to-all with K/N scaling
        self.W_delta = nn.Parameter(
            torch.randn(n_delta, n_delta) * coupling_strength / n_delta
        )
        self.W_theta = nn.Parameter(
            torch.randn(n_theta, n_theta) * coupling_strength / n_theta
        )
        self.W_gamma = nn.Parameter(
            torch.randn(n_gamma, n_gamma) * coupling_strength / n_gamma
        )

        # PAC gating projections: slow_phase (2D cos/sin) → fast gate
        # Delta → Theta
        self.W_pac_dt = nn.Linear(2 * n_delta, n_theta)
        nn.init.xavier_uniform_(self.W_pac_dt.weight, gain=0.5)
        nn.init.constant_(self.W_pac_dt.bias, pac_depth)

        # Theta → Gamma
        self.W_pac_tg = nn.Linear(2 * n_theta, n_gamma)
        nn.init.xavier_uniform_(self.W_pac_tg.weight, gain=0.5)
        nn.init.constant_(self.W_pac_tg.bias, pac_depth)

        # Stuart-Landau growth parameter μ per band (learnable)
        self.mu_delta = nn.Parameter(torch.tensor(1.0))
        self.mu_theta = nn.Parameter(torch.tensor(1.0))
        self.mu_gamma = nn.Parameter(torch.tensor(1.0))

    @property
    def n_delta(self) -> int:
        """Number of Delta-band oscillators."""
        return self._n_delta

    @property
    def n_theta(self) -> int:
        """Number of Theta-band oscillators."""
        return self._n_theta

    @property
    def n_gamma(self) -> int:
        """Number of Gamma-band oscillators."""
        return self._n_gamma

    @property
    def n_total(self) -> int:
        """Total number of oscillators across all bands."""
        return self._n_total

    def _phase_coupling(self, phase: Tensor, W: Tensor) -> Tensor:
        """Compute Kuramoto-like coupling correction.

        Args:
            phase: Phase tensor ``(B, N)``.
            W: Coupling weight matrix ``(N, N)``.

        Returns:
            Coupling correction ``(B, N)``.
        """
        # sin(φ_j - φ_i) for all pairs, weighted by W
        # phase: (B, N) → phase_j: (B, 1, N), phase_i: (B, N, 1)
        diff = phase.unsqueeze(-2) - phase.unsqueeze(-1)  # (B, N, N)
        sin_diff = torch.sin(diff)  # (B, N, N)
        # W: (N, N), broadcast: coupling = sum_j W_ij * sin(φ_j - φ_i)
        coupling = (W.unsqueeze(0) * sin_diff).sum(dim=-1)  # (B, N)
        return coupling

    def _amplitude_update(self, amp: Tensor, mu: Tensor, dt: float) -> Tensor:
        """Stuart-Landau amplitude dynamics.

        Args:
            amp: Current amplitudes ``(B, N)``.
            mu: Growth parameter (scalar).
            dt: Timestep.

        Returns:
            Updated amplitudes, clamped to ``[1e-6, 10.0]``.
        """
        da = dt * amp * (mu - amp * amp)
        new_amp = amp + da
        return torch.clamp(new_amp, min=1e-6, max=10.0)

    def step(
        self,
        phase: Tensor,
        amplitude: Tensor,
        dt: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """Advance all three bands by one discrete macro step.

        This is the core operation: one forward call = one step with
        learned coupling and PAC gating. No inner loop.

        Args:
            phase: Concatenated phases ``(B, n_total)`` or ``(n_total,)``.
            amplitude: Concatenated amplitudes, same shape.
            dt: Timestep (controls phase advance speed).

        Returns:
            ``(new_phase, new_amplitude)`` with same shape as inputs.
        """
        was_1d = phase.dim() == 1
        if was_1d:
            phase = phase.unsqueeze(0)
            amplitude = amplitude.unsqueeze(0)

        nd, nt, ng = self._n_delta, self._n_theta, self._n_gamma

        # Split into bands
        p_d = phase[:, :nd]
        p_t = phase[:, nd : nd + nt]
        p_g = phase[:, nd + nt :]
        a_d = amplitude[:, :nd]
        a_t = amplitude[:, nd : nd + nt]
        a_g = amplitude[:, nd + nt :]

        # --- Phase advance with intra-band coupling ---
        two_pi = 2.0 * math.pi
        new_p_d = _wrap_phase(
            p_d
            + two_pi * self.delta_freq * dt
            + dt * self._phase_coupling(p_d, self.W_delta)
        )
        new_p_t = _wrap_phase(
            p_t
            + two_pi * self.theta_freq * dt
            + dt * self._phase_coupling(p_t, self.W_theta)
        )
        new_p_g = _wrap_phase(
            p_g
            + two_pi * self.gamma_freq * dt
            + dt * self._phase_coupling(p_g, self.W_gamma)
        )

        # --- PAC gating (multiplicative) ---
        # Delta phase → gate on Theta amplitude
        delta_repr = torch.cat(
            [torch.cos(new_p_d), torch.sin(new_p_d)], dim=-1
        )  # (B, 2*nd)
        gate_dt = torch.sigmoid(self.W_pac_dt(delta_repr))  # (B, nt)
        a_t = a_t * gate_dt

        # Theta phase → gate on Gamma amplitude
        theta_repr = torch.cat(
            [torch.cos(new_p_t), torch.sin(new_p_t)], dim=-1
        )  # (B, 2*nt)
        gate_tg = torch.sigmoid(self.W_pac_tg(theta_repr))  # (B, ng)
        a_g = a_g * gate_tg

        # --- Amplitude dynamics (Stuart-Landau) ---
        new_a_d = self._amplitude_update(a_d, self.mu_delta, dt)
        new_a_t = self._amplitude_update(a_t, self.mu_theta, dt)
        new_a_g = self._amplitude_update(a_g, self.mu_gamma, dt)

        # Reassemble
        new_phase = torch.cat([new_p_d, new_p_t, new_p_g], dim=-1)
        new_amp = torch.cat([new_a_d, new_a_t, new_a_g], dim=-1)

        if was_1d:
            return new_phase.squeeze(0), new_amp.squeeze(0)
        return new_phase, new_amp

    def integrate(
        self,
        phase: Tensor,
        amplitude: Tensor,
        n_steps: int = 10,
        dt: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate for multiple discrete steps.

        Args:
            phase: Initial phases ``(B, n_total)`` or ``(n_total,)``.
            amplitude: Initial amplitudes, same shape.
            n_steps: Number of macro steps.
            dt: Timestep per step.

        Returns:
            ``(final_phase, final_amplitude)``.
        """
        p, a = phase, amplitude
        for _ in range(n_steps):
            p, a = self.step(p, a, dt=dt)
        return p, a

    def order_parameters(
        self,
        phase: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute per-band Kuramoto order parameters.

        Args:
            phase: Concatenated phases ``(B, n_total)`` or ``(n_total,)``.

        Returns:
            ``(r_delta, r_theta, r_gamma)`` — each a scalar tensor.
        """
        from prinet.core.measurement import kuramoto_order_parameter

        was_1d = phase.dim() == 1
        if was_1d:
            phase = phase.unsqueeze(0)

        nd, nt = self._n_delta, self._n_theta
        # Average over batch
        r_d = kuramoto_order_parameter(phase[:, :nd].mean(dim=0))
        r_t = kuramoto_order_parameter(phase[:, nd : nd + nt].mean(dim=0))
        r_g = kuramoto_order_parameter(phase[:, nd + nt :].mean(dim=0))
        return r_d, r_t, r_g

    def pac_index(
        self,
        phase: Tensor,
        amplitude: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute PAC modulation index for both couplings.

        Measures how strongly fast-band amplitude covaries with
        slow-band phase. Returns values ∈ [0, 1] where 0 means
        no coupling and 1 means perfect modulation.

        Args:
            phase: Concatenated phases ``(B, n_total)``.
            amplitude: Concatenated amplitudes, same shape.

        Returns:
            ``(pac_dt, pac_tg)`` — modulation indices for
            Delta→Theta and Theta→Gamma couplings.
        """
        nd, nt = self._n_delta, self._n_theta

        p_d = phase[:, :nd]  # (B, nd)
        a_t = amplitude[:, nd : nd + nt]  # (B, nt)
        p_t = phase[:, nd : nd + nt]  # (B, nt)
        a_g = amplitude[:, nd + nt :]  # (B, ng)

        # Mean slow-band phase per sample
        mean_p_d = p_d.mean(dim=-1, keepdim=True)  # (B, 1)
        mean_p_t = p_t.mean(dim=-1, keepdim=True)  # (B, 1)

        # Mean fast-band amplitude per sample
        mean_a_t = a_t.mean(dim=-1, keepdim=True)  # (B, 1)
        mean_a_g = a_g.mean(dim=-1, keepdim=True)  # (B, 1)

        # PAC index: |mean(a_fast * exp(i * phi_slow))| / mean(a_fast)
        # Using cos correlation as proxy
        cos_d = torch.cos(mean_p_d)  # (B, 1)
        pac_dt_val = (
            ((a_t * cos_d).mean(dim=-1) / (mean_a_t.squeeze(-1) + 1e-8)).abs().mean()
        )

        cos_t = torch.cos(mean_p_t)  # (B, 1)
        pac_tg_val = (
            ((a_g * cos_t).mean(dim=-1) / (mean_a_g.squeeze(-1) + 1e-8)).abs().mean()
        )

        return pac_dt_val, pac_tg_val
