"""Oscillator model implementations for PRINet.

Provides the abstract :class:`OscillatorModel` base class and three
concrete implementations: :class:`KuramotoOscillator`,
:class:`StuartLandauOscillator`, and :class:`HopfOscillator`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.linalg as LA
from torch import Tensor

from .oscillator_state import (
    OscillatorState,
    OscillatorSyncError,
    _TWO_PI,
    _SPARSE_EPS,
    _DERIV_CLAMP,
    _wrap_phase,
    _safe_phase_diff,
    _clamp_finite,
    _build_phase_knn_index,
)

class OscillatorModel(ABC):
    """Abstract base class for oscillator dynamics models.

    All concrete oscillator models (Kuramoto, Stuart-Landau, Hopf, etc.)
    must subclass this and implement the ``compute_derivatives`` method.
    The ``step`` and ``integrate`` methods provide numerical integration
    using Euler or RK4 methods.

    Args:
        n_oscillators: Number of oscillators ``N``.
        coupling_strength: Global coupling constant ``K``.
        device: Torch device.
        dtype: Data type.
    """

    def __init__(
        self,
        n_oscillators: int,
        coupling_strength: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_oscillators < 1:
            raise ValueError(
                f"n_oscillators must be positive, got {n_oscillators}."
            )
        self._n = n_oscillators
        self._K = coupling_strength
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._coupling_matrix: Optional[Tensor] = None

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators."""
        return self._n

    @property
    def coupling_strength(self) -> float:
        """Global coupling constant K."""
        return self._K

    @coupling_strength.setter
    def coupling_strength(self, value: float) -> None:
        """Set the global coupling strength."""
        self._K = value

    @property
    def coupling_matrix(self) -> Tensor:
        """Coupling matrix of shape ``(N, N)``.

        Returns all-to-all coupling scaled by ``K/N`` by default. Can be
        overridden via ``set_coupling_matrix``.
        """
        if self._coupling_matrix is not None:
            return self._coupling_matrix
        # Default: all-to-all uniform coupling
        mat = torch.full(
            (self._n, self._n),
            self._K / self._n,
            device=self._device,
            dtype=self._dtype,
        )
        mat.fill_diagonal_(0.0)
        return mat

    def set_coupling_matrix(self, matrix: Tensor) -> None:
        """Set a custom coupling matrix.

        Args:
            matrix: Coupling matrix of shape ``(N, N)``.

        Raises:
            ValueError: If shape is incorrect.
        """
        if matrix.shape != (self._n, self._n):
            raise ValueError(
                f"Expected coupling matrix shape ({self._n}, {self._n}), "
                f"got {tuple(matrix.shape)}."
            )
        self._coupling_matrix = matrix.to(
            device=self._device, dtype=self._dtype
        )

    @abstractmethod
    def compute_derivatives(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute time derivatives of the oscillator state.

        Args:
            state: Current oscillator state.

        Returns:
            Tuple ``(dphase/dt, damplitude/dt, dfrequency/dt)``.
        """
        ...

    def step(
        self,
        state: OscillatorState,
        dt: float = 0.01,
        method: str = "rk4",
    ) -> OscillatorState:
        """Advance the oscillator state by one timestep.

        Args:
            state: Current state.
            dt: Timestep size.
            method: Integration method, ``"euler"`` or ``"rk4"``.

        Returns:
            Updated ``OscillatorState``.

        Raises:
            ValueError: If method is unknown.
        """
        if method == "euler":
            return self._step_euler(state, dt)
        elif method == "rk4":
            return self._step_rk4(state, dt)
        else:
            raise ValueError(
                f"Unknown integration method '{method}'. "
                f"Use 'euler' or 'rk4'."
            )

    def _step_euler(
        self, state: OscillatorState, dt: float
    ) -> OscillatorState:
        """Forward Euler integration step.

        Args:
            state: Current state.
            dt: Timestep.

        Returns:
            Updated state with wrapped phase.
        """
        dphi, dr, domega = self.compute_derivatives(state)
        return OscillatorState(
            phase=_wrap_phase(state.phase + dt * dphi),
            amplitude=torch.clamp(state.amplitude + dt * dr, min=0.0),
            frequency=state.frequency + dt * domega,
        )

    def _step_rk4(
        self, state: OscillatorState, dt: float
    ) -> OscillatorState:
        """4th-order Runge-Kutta integration step.

        Args:
            state: Current state.
            dt: Timestep.

        Returns:
            Updated state.
        """

        def _make_state(
            s: OscillatorState,
            dphi: Tensor,
            dr: Tensor,
            domega: Tensor,
            scale: float,
        ) -> OscillatorState:
            return OscillatorState(
                phase=s.phase + scale * dphi,
                amplitude=torch.clamp(
                    s.amplitude + scale * dr, min=0.0
                ),
                frequency=s.frequency + scale * domega,
            )

        k1_phi, k1_r, k1_omega = self.compute_derivatives(state)

        s2 = _make_state(state, k1_phi, k1_r, k1_omega, 0.5 * dt)
        k2_phi, k2_r, k2_omega = self.compute_derivatives(s2)

        s3 = _make_state(state, k2_phi, k2_r, k2_omega, 0.5 * dt)
        k3_phi, k3_r, k3_omega = self.compute_derivatives(s3)

        s4 = _make_state(state, k3_phi, k3_r, k3_omega, dt)
        k4_phi, k4_r, k4_omega = self.compute_derivatives(s4)

        phase = _wrap_phase(
            state.phase
            + (dt / 6.0)
            * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)
        )
        amplitude = torch.clamp(
            state.amplitude
            + (dt / 6.0)
            * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r),
            min=0.0,
        )
        frequency = state.frequency + (dt / 6.0) * (
            k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega
        )

        return OscillatorState(
            phase=phase, amplitude=amplitude, frequency=frequency
        )

    def integrate(
        self,
        state: OscillatorState,
        n_steps: int,
        dt: float = 0.01,
        method: str = "rk4",
        record_trajectory: bool = False,
    ) -> Tuple[OscillatorState, Optional[list["OscillatorState"]]]:
        """Integrate the oscillator system for multiple timesteps.

        Args:
            state: Initial state.
            n_steps: Number of integration steps.
            dt: Timestep size.
            method: ``"euler"`` or ``"rk4"``.
            record_trajectory: If ``True``, return list of intermediate
                states.

        Returns:
            Tuple of ``(final_state, trajectory)``. ``trajectory`` is
            ``None`` if ``record_trajectory`` is ``False``.
        """
        trajectory: list[OscillatorState] | None = [] if record_trajectory else None
        current = state

        for _ in range(n_steps):
            current = self.step(current, dt=dt, method=method)
            if trajectory is not None:
                trajectory.append(current.clone())

        return current, trajectory


class KuramotoOscillator(OscillatorModel):
    """Kuramoto coupled oscillator model.

    Implements the extended Kuramoto equations with amplitude and
    frequency modulation as described in the mathematical foundations:

        dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) · |rⱼ|
        drᵢ/dt = -λᵢ rᵢ + Σⱼ Kᵢⱼ cos(φⱼ - φᵢ) · |rⱼ|
        dωᵢ/dt = γ · Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) · |rⱼ|/N

    Args:
        n_oscillators: Number of oscillators.
        coupling_strength: Global coupling constant K.
        decay_rate: Amplitude decay rate λ (scalar or per-oscillator).
        freq_adaptation_rate: Frequency adaptation γ.
        device: Torch device.
        dtype: Data type.

    Example:
        >>> model = KuramotoOscillator(n_oscillators=50, coupling_strength=2.0)
        >>> state = OscillatorState.create_random(50, seed=42)
        >>> final, _ = model.integrate(state, n_steps=100, dt=0.01)
    """

    def __init__(
        self,
        n_oscillators: int,
        coupling_strength: float = 1.0,
        decay_rate: float = 0.1,
        freq_adaptation_rate: float = 0.01,
        mean_field: bool = False,
        coupling_mode: str = "auto",
        sparse_k: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_oscillators, coupling_strength, device, dtype)
        self._decay = decay_rate
        self._gamma = freq_adaptation_rate
        self._mean_field = mean_field
        self._coupling_mode = coupling_mode  # "auto", "full", "mean_field", "sparse_knn"
        self._sparse_k = sparse_k  # k nearest neighbours; None → ceil(log2(N))

    @property
    def decay_rate(self) -> float:
        """Amplitude decay rate λ."""
        return self._decay

    @property
    def freq_adaptation_rate(self) -> float:
        """Frequency adaptation rate γ."""
        return self._gamma

    @property
    def mean_field(self) -> bool:
        """Whether mean-field approximation is active."""
        return self._mean_field

    @property
    def coupling_mode(self) -> str:
        """Active coupling computation mode."""
        return self._coupling_mode

    @property
    def sparse_k(self) -> int:
        """Number of nearest-phase neighbours for sparse coupling."""
        if self._sparse_k is not None:
            return self._sparse_k
        return max(1, math.ceil(math.log2(self._n)))

    def compute_derivatives(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute Kuramoto dynamics derivatives.

        Dispatches based on ``coupling_mode``:

        * ``"mean_field"`` / legacy ``mean_field=True``: O(N) mean-field.
        * ``"sparse_knn"``: O(N log N) k-nearest-phase-neighbour coupling.
        * ``"full"``: O(N²) dense pairwise coupling.
        * ``"auto"`` (default): ``mean_field`` flag decides full vs mean-field
          for backward compatibility.

        Args:
            state: Current oscillator state.

        Returns:
            Tuple ``(dphase/dt, damplitude/dt, dfrequency/dt)``.
        """
        mode = self._coupling_mode
        if mode == "sparse_knn":
            return self._compute_derivatives_sparse_knn(state)
        if mode == "mean_field" or (mode == "auto" and self._mean_field):
            return self._compute_derivatives_mean_field(state)
        return self._compute_derivatives_full(state)

    # ------------------------------------------------------------------
    # Mean-field O(N) path
    # ------------------------------------------------------------------

    def _compute_derivatives_mean_field(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mean-field Kuramoto: O(N) per evaluation."""
        phase = state.phase  # (..., N)
        amp = state.amplitude
        freq = state.frequency

        # Complex order parameter: Z = (1/N) Σⱼ rⱼ e^{iφⱼ}
        z = (amp * torch.exp(1j * phase.to(torch.complex64))).mean(dim=-1)
        R = z.abs().float()  # scalar or (...,)
        psi = z.angle().float()

        K = self._K

        # dφᵢ/dt = ωᵢ + K·R·sin(ψ − φᵢ)
        dphi = freq + K * R.unsqueeze(-1) * torch.sin(psi.unsqueeze(-1) - phase)

        # drᵢ/dt ≈ -λ rᵢ + K·R·cos(ψ − φᵢ)
        dr = -self._decay * amp + K * R.unsqueeze(-1) * torch.cos(
            psi.unsqueeze(-1) - phase
        )

        # dωᵢ/dt = γ · K · R · sin(ψ − φᵢ) / N
        domega = self._gamma * K * R.unsqueeze(-1) * torch.sin(
            psi.unsqueeze(-1) - phase
        ) / self._n

        return dphi, dr, domega

    # ------------------------------------------------------------------
    # Full pairwise O(N²) path
    # ------------------------------------------------------------------

    def _compute_derivatives_full(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Full pairwise Kuramoto: O(N²) per evaluation."""
        coupling = self.coupling_matrix  # (N, N)
        phase = state.phase  # (..., N)
        amp = state.amplitude  # (..., N)
        freq = state.frequency  # (..., N)

        # Phase differences: φⱼ - φᵢ for all pairs
        # For unbatched: phase_diff[i, j] = phase[j] - phase[i]
        phase_diff = phase.unsqueeze(-2) - phase.unsqueeze(-1)  # (..., N, N)

        sin_diff = torch.sin(phase_diff)  # (..., N, N)
        cos_diff = torch.cos(phase_diff)  # (..., N, N)

        # Weighted by amplitudes: Kᵢⱼ · sin(φⱼ - φᵢ) · |rⱼ|
        amp_weight = amp.unsqueeze(-2)  # (..., 1, N) = rⱼ along last dim
        weighted_sin = coupling * sin_diff * amp_weight  # (..., N, N)
        weighted_cos = coupling * cos_diff * amp_weight

        # Phase derivative: dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) |rⱼ|
        dphi = freq + weighted_sin.sum(dim=-1)

        # Amplitude derivative: drᵢ/dt = -λᵢ rᵢ + Σⱼ Kᵢⱼ cos(φⱼ - φᵢ) |rⱼ|
        dr = -self._decay * amp + weighted_cos.sum(dim=-1)

        # Frequency derivative: dωᵢ/dt = γ · Σⱼ ... / N
        domega = (
            self._gamma * weighted_sin.sum(dim=-1) / self._n
        )

        return dphi, dr, domega

    # ------------------------------------------------------------------
    # Sparse k-NN O(N log N) path
    # ------------------------------------------------------------------

    def _compute_derivatives_sparse_knn(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sparse k-nearest-phase-neighbour Kuramoto: O(N k) ≈ O(N log N).

        For each oscillator *i*, we couple only to its *k* nearest
        neighbours in **phase space** (wrapped distance on the circle).
        With ``k = ceil(log₂ N)`` the total edge count is ``N·k = N log N``.

        Unlike mean-field, this preserves local coupling structure and can
        capture chimera states and cluster synchronization that mean-field
        cannot represent.  Unlike full pairwise, memory is O(N·k) not O(N²).

        The coupling strength per edge is ``K / k`` (analogous to ``K / N``
        in the full model), keeping total coupling energy per oscillator
        constant regardless of sparsity level.

        Hardened with:
        - Edge-case guards for N=1 and k boundary conditions.
        - ``_safe_phase_diff`` (atan2-based) for wrapped phase differences.
        - ``_clamp_finite`` on every output derivative to catch NaN/Inf.
        - Factored ``_build_phase_knn_index`` for reusability.
        """
        phase = state.phase            # (..., N)
        amp = state.amplitude          # (..., N)
        freq = state.frequency         # (..., N)

        N = phase.shape[-1]

        # --- Edge-case guard: N=1 → no coupling possible ---
        if N <= 1:
            zeros = torch.zeros_like(phase)
            return freq.clone(), -self._decay * amp, zeros

        k = self.sparse_k
        k = min(k, N - 1)  # can't have more neighbours than oscillators

        # --- Build sparse neighbour index from wrapped phase distance ---
        flat_phase = phase.reshape(-1, N)  # (B, N)
        B = flat_phase.shape[0]

        nbr_idx = _build_phase_knn_index(flat_phase, k)  # (B, N, k)

        # --- Compute coupling terms only for neighbours ---
        flat_amp = amp.reshape(-1, N)
        flat_freq = freq.reshape(-1, N)

        # Gather neighbour phases and amplitudes
        nbr_phase = flat_phase.gather(
            1, nbr_idx.reshape(B, -1)
        ).reshape(B, N, k)
        nbr_amp = flat_amp.gather(
            1, nbr_idx.reshape(B, -1)
        ).reshape(B, N, k)

        # Wrapped phase differences: (φⱼ − φᵢ) ∈ (-π, π]
        phase_i = flat_phase.unsqueeze(-1)  # (B, N, 1)
        phase_diff = _safe_phase_diff(nbr_phase, phase_i)  # (B, N, k)

        sin_diff = torch.sin(phase_diff)
        cos_diff = torch.cos(phase_diff)

        # Coupling weight = K / k (constant total coupling per oscillator)
        K_eff = self._K / k

        # Weighted by neighbour amplitudes
        weighted_sin = K_eff * sin_diff * nbr_amp  # (B, N, k)
        weighted_cos = K_eff * cos_diff * nbr_amp

        sin_sum = weighted_sin.sum(dim=-1)  # (B, N)
        cos_sum = weighted_cos.sum(dim=-1)

        # Phase: dφᵢ/dt = ωᵢ + Σ_neighbours ...
        dphi = flat_freq + sin_sum

        # Amplitude: drᵢ/dt = -λ rᵢ + Σ_neighbours ...
        dr = -self._decay * flat_amp + cos_sum

        # Frequency: dωᵢ/dt = γ · Σ_neighbours ... / k
        domega = self._gamma * sin_sum / k

        # --- Finite-guard: clamp and replace NaN/Inf ---
        dphi = _clamp_finite(dphi.reshape(phase.shape))
        dr = _clamp_finite(dr.reshape(amp.shape))
        domega = _clamp_finite(domega.reshape(freq.shape))

        return dphi, dr, domega


class StuartLandauOscillator(OscillatorModel):
    """Stuart-Landau (Hopf normal form) coupled oscillator model.

    Implements complex-valued dynamics where each oscillator has a
    limit cycle. The model uses:

        dzᵢ/dt = (μ + iωᵢ)zᵢ - |zᵢ|²zᵢ + K/N Σⱼ (zⱼ - zᵢ)

    where zᵢ = rᵢ exp(iφᵢ) and μ controls the bifurcation parameter.

    Args:
        n_oscillators: Number of oscillators.
        coupling_strength: Global coupling constant K.
        bifurcation_param: Hopf bifurcation parameter μ. Positive values
            create limit cycles; negative values create stable fixed points.
        device: Torch device.
        dtype: Data type.

    Example:
        >>> model = StuartLandauOscillator(
        ...     n_oscillators=50,
        ...     coupling_strength=1.5,
        ...     bifurcation_param=1.0,
        ... )
        >>> state = OscillatorState.create_random(50, seed=42)
        >>> final, _ = model.integrate(state, n_steps=200, dt=0.01)
    """

    def __init__(
        self,
        n_oscillators: int,
        coupling_strength: float = 1.0,
        bifurcation_param: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_oscillators, coupling_strength, device, dtype)
        self._mu = bifurcation_param

    @property
    def bifurcation_param(self) -> float:
        """Hopf bifurcation parameter μ."""
        return self._mu

    def compute_derivatives(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute Stuart-Landau dynamics derivatives.

        Converts to complex form, computes dz/dt, and extracts
        the real-valued phase and amplitude derivatives.

        Args:
            state: Current oscillator state.

        Returns:
            Tuple ``(dphase/dt, damplitude/dt, dfrequency/dt)``.
        """
        phase = state.phase  # (..., N)
        amp = state.amplitude  # (..., N)
        freq = state.frequency  # (..., N)

        # Convert to complex representation: z = r * exp(iφ)
        z = amp * torch.exp(1j * phase.to(torch.float64)).to(
            torch.complex64
        )

        # Coupling term: K/N Σⱼ (zⱼ - zᵢ)
        coupling = self.coupling_matrix  # (N, N)
        # z_diff[i, j] = z[j] - z[i]
        z_diff = z.unsqueeze(-2) - z.unsqueeze(-1)  # (..., N, N)
        coupling_term = (coupling * z_diff).sum(dim=-1)

        # Stuart-Landau: dz/dt = (μ + iω)z - |z|²z + coupling
        dz = (
            (self._mu + 1j * freq.to(torch.complex64)) * z
            - (amp**2).to(torch.complex64) * z
            + coupling_term
        )

        # Extract amplitude and phase derivatives from complex dz/dt
        # dz/dt = (dr/dt + i r dφ/dt) exp(iφ)
        # So: dr/dt = Re(dz/dt * exp(-iφ)), r*dφ/dt = Im(dz/dt * exp(-iφ))
        dz_rotated = dz * torch.exp(
            -1j * phase.to(torch.float64)
        ).to(torch.complex64)

        dr = dz_rotated.real.to(self._dtype)
        dphi_raw = dz_rotated.imag.to(self._dtype)

        # Avoid division by zero for amplitude
        safe_amp = torch.clamp(amp, min=1e-8)
        dphi = dphi_raw / safe_amp

        # Frequency does not adapt in basic Stuart-Landau
        domega = torch.zeros_like(freq)

        return dphi, dr, domega


class HopfOscillator(OscillatorModel):
    """Hopf bifurcation oscillator with explicit amplitude-phase dynamics.

    Implements oscillator dynamics in polar coordinates with a
    supercritical Hopf bifurcation:

        drᵢ/dt = μ rᵢ - rᵢ³ + Σⱼ Kᵢⱼ cos(φⱼ - φᵢ) rⱼ
        dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) rⱼ / rᵢ
        dωᵢ/dt = γ · Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) rⱼ / N

    When μ > 0, each oscillator has a stable limit cycle at r = √μ.
    When μ < 0, the origin is a stable fixed point.
    The cubic term -r³ provides natural amplitude saturation unlike
    the Kuramoto model, preventing amplitude explosion.

    This model is required for bifurcation analysis (Q2 Task 1.1)
    and provides more physically grounded dynamics than pure Kuramoto.

    Args:
        n_oscillators: Number of oscillators.
        coupling_strength: Global coupling constant K.
        bifurcation_param: Hopf parameter μ. Positive → limit cycle.
        freq_adaptation_rate: Frequency adaptation γ.
        mean_field: If ``True``, use O(N) mean-field approximation.
        device: Torch device.
        dtype: Data type.

    Example:
        >>> model = HopfOscillator(
        ...     n_oscillators=50,
        ...     coupling_strength=1.0,
        ...     bifurcation_param=1.0,
        ... )
        >>> state = OscillatorState.create_random(50, seed=42)
        >>> final, _ = model.integrate(state, n_steps=200, dt=0.01)
        >>> # Amplitudes should converge near √μ = 1.0
        >>> print(f"Mean amplitude: {final.amplitude.mean():.2f}")
    """

    def __init__(
        self,
        n_oscillators: int,
        coupling_strength: float = 1.0,
        bifurcation_param: float = 1.0,
        freq_adaptation_rate: float = 0.01,
        mean_field: bool = False,
        coupling_mode: str = "auto",
        sparse_k: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_oscillators, coupling_strength, device, dtype)
        self._mu = bifurcation_param
        self._gamma = freq_adaptation_rate
        self._mean_field = mean_field
        self._coupling_mode = coupling_mode
        self._sparse_k = sparse_k

    @property
    def bifurcation_param(self) -> float:
        """Hopf bifurcation parameter μ."""
        return self._mu

    @bifurcation_param.setter
    def bifurcation_param(self, value: float) -> None:
        """Set the bifurcation parameter."""
        self._mu = value

    @property
    def freq_adaptation_rate(self) -> float:
        """Frequency adaptation rate γ."""
        return self._gamma

    @property
    def mean_field(self) -> bool:
        """Whether mean-field approximation is active."""
        return self._mean_field

    @property
    def coupling_mode(self) -> str:
        """Active coupling computation mode."""
        return self._coupling_mode

    @property
    def sparse_k(self) -> int:
        """Number of nearest-phase neighbours for sparse coupling."""
        if self._sparse_k is not None:
            return self._sparse_k
        return max(1, math.ceil(math.log2(self._n)))

    @property
    def limit_cycle_amplitude(self) -> float:
        """Theoretical limit cycle amplitude √μ (only meaningful if μ > 0)."""
        if self._mu <= 0.0:
            return 0.0
        return math.sqrt(self._mu)

    def compute_derivatives(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute Hopf oscillator dynamics derivatives.

        Args:
            state: Current oscillator state.

        Returns:
            Tuple ``(dphase/dt, damplitude/dt, dfrequency/dt)``.
        """
        mode = self._coupling_mode
        if mode == "sparse_knn":
            return self._compute_derivatives_sparse_knn(state)
        if mode == "mean_field" or (mode == "auto" and self._mean_field):
            return self._compute_derivatives_mean_field(state)
        return self._compute_derivatives_full(state)

    def _compute_derivatives_mean_field(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mean-field Hopf dynamics: O(N) per evaluation."""
        phase = state.phase
        amp = state.amplitude
        freq = state.frequency

        # Complex order parameter: Z = (1/N) Σⱼ rⱼ e^{iφⱼ}
        z = (amp * torch.exp(1j * phase.to(torch.complex64))).mean(dim=-1)
        R = z.abs().float()
        psi = z.angle().float()

        K = self._K

        # Phase derivative: dφ/dt = ω + K·R·sin(ψ-φ)/r
        safe_amp = torch.clamp(amp, min=1e-8)
        dphi = freq + K * R.unsqueeze(-1) * torch.sin(
            psi.unsqueeze(-1) - phase
        ) / safe_amp

        # Amplitude derivative: dr/dt = μr - r³ + K·R·cos(ψ-φ)
        dr = (
            self._mu * amp
            - amp ** 3
            + K * R.unsqueeze(-1) * torch.cos(psi.unsqueeze(-1) - phase)
        )

        # Frequency adaptation
        domega = (
            self._gamma
            * K
            * R.unsqueeze(-1)
            * torch.sin(psi.unsqueeze(-1) - phase)
            / self._n
        )

        return dphi, dr, domega

    def _compute_derivatives_full(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Full pairwise Hopf dynamics: O(N²) per evaluation."""
        coupling = self.coupling_matrix
        phase = state.phase
        amp = state.amplitude
        freq = state.frequency

        # Phase differences: φⱼ - φᵢ
        phase_diff = phase.unsqueeze(-2) - phase.unsqueeze(-1)
        sin_diff = torch.sin(phase_diff)
        cos_diff = torch.cos(phase_diff)

        # Amplitude weights: rⱼ along last dimension
        amp_j = amp.unsqueeze(-2)

        # Weighted coupling sums
        weighted_sin = coupling * sin_diff * amp_j
        weighted_cos = coupling * cos_diff * amp_j

        sin_sum = weighted_sin.sum(dim=-1)
        cos_sum = weighted_cos.sum(dim=-1)

        # Phase: dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ-φᵢ) rⱼ / rᵢ
        safe_amp = torch.clamp(amp, min=1e-8)
        dphi = freq + sin_sum / safe_amp

        # Amplitude: drᵢ/dt = μrᵢ - rᵢ³ + Σⱼ Kᵢⱼ cos(φⱼ-φᵢ) rⱼ
        dr = self._mu * amp - amp ** 3 + cos_sum

        # Frequency: dωᵢ/dt = γ · Σⱼ Kᵢⱼ sin(φⱼ-φᵢ) rⱼ / N
        domega = self._gamma * sin_sum / self._n

        return dphi, dr, domega

    # ------------------------------------------------------------------
    # Sparse k-NN O(N log N) path
    # ------------------------------------------------------------------

    def _compute_derivatives_sparse_knn(
        self, state: OscillatorState
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sparse k-nearest-phase-neighbour Hopf dynamics: O(N log N).

        Same neighbour-selection strategy as
        :meth:`KuramotoOscillator._compute_derivatives_sparse_knn` but
        with Hopf cubic saturation and amplitude-relative phase coupling.

        Hardened with:
        - Edge-case guard for N=1.
        - ``_safe_phase_diff`` (atan2-based) wrapped differences.
        - ``_SPARSE_EPS``-clamped amplitude divisor to prevent division
          by zero in the phase equation.
        - ``_clamp_finite`` on all outputs.
        - Factored ``_build_phase_knn_index`` for reusability.
        """
        phase = state.phase
        amp = state.amplitude
        freq = state.frequency

        N = phase.shape[-1]

        # --- Edge-case guard: N=1 → no coupling ---
        if N <= 1:
            zeros = torch.zeros_like(phase)
            dr = self._mu * amp - amp ** 3
            return freq.clone(), dr, zeros

        k = self.sparse_k
        k = min(k, N - 1)

        flat_phase = phase.reshape(-1, N)
        flat_amp = amp.reshape(-1, N)
        flat_freq = freq.reshape(-1, N)
        B = flat_phase.shape[0]

        # Sort-based k-NN on the phase circle (factored helper)
        nbr_idx = _build_phase_knn_index(flat_phase, k)  # (B, N, k)

        nbr_phase = flat_phase.gather(
            1, nbr_idx.reshape(B, -1)
        ).reshape(B, N, k)
        nbr_amp = flat_amp.gather(
            1, nbr_idx.reshape(B, -1)
        ).reshape(B, N, k)

        # Wrapped phase differences: (φⱼ − φᵢ) ∈ (-π, π]
        phase_i = flat_phase.unsqueeze(-1)
        phase_diff = _safe_phase_diff(nbr_phase, phase_i)

        sin_diff = torch.sin(phase_diff)
        cos_diff = torch.cos(phase_diff)

        K_eff = self._K / k
        weighted_sin = K_eff * sin_diff * nbr_amp
        weighted_cos = K_eff * cos_diff * nbr_amp

        sin_sum = weighted_sin.sum(dim=-1)
        cos_sum = weighted_cos.sum(dim=-1)

        # Safe amplitude divisor for Hopf phase equation
        safe_amp = torch.clamp(flat_amp, min=_SPARSE_EPS)

        # Phase: dφᵢ/dt = ωᵢ + Σ_nbr sin(Δφ)·rⱼ / rᵢ
        dphi = flat_freq + sin_sum / safe_amp

        # Amplitude: drᵢ/dt = μrᵢ − rᵢ³ + Σ_nbr cos(Δφ)·rⱼ
        dr = self._mu * flat_amp - flat_amp ** 3 + cos_sum

        # Frequency adaptation
        domega = self._gamma * sin_sum / k

        # --- Finite-guard: clamp and replace NaN/Inf ---
        dphi = _clamp_finite(dphi.reshape(phase.shape))
        dr = _clamp_finite(dr.reshape(amp.shape))
        domega = _clamp_finite(domega.reshape(freq.shape))

        return dphi, dr, domega


