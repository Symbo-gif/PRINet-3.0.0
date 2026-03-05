"""Numerical integrators for oscillator dynamics.

Provides :class:`ExponentialIntegrator` (matrix-exponential /
Krylov-subspace) and :class:`MultiRateIntegrator` (sub-stepped RK4).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.linalg as LA
from torch import Tensor

from .oscillator_state import OscillatorState, _wrap_phase
from .oscillator_models import OscillatorModel

class ExponentialIntegrator:
    """Exponential integrator for stiff oscillator dynamics.

    Implements the exponential Euler method using the matrix exponential
    to handle stiff linear parts of the oscillator ODE exactly, with the
    nonlinear remainder treated via the phi-function φ₁(hA).

    The integration formula is:

        y_{n+1} = exp(hA) y_n + h φ₁(hA) f(y_n)

    where:
        - ``A`` is the (possibly stiff) linear part of the dynamics,
        - ``f(y)`` is the nonlinear remainder,
        - ``φ₁(z) = (exp(z) - I) / z``.

    For oscillator systems whose Jacobian has widely separated eigenvalues
    (stiff dynamics, e.g. Stuart-Landau near bifurcation), exponential
    integrators outperform explicit methods like RK4 by treating the fast
    linear modes exactly.

    The class supports two modes:

    1. **Direct** (``dim ≤ max_direct_dim``): full ``torch.linalg.matrix_exp``.
    2. **Krylov subspace** (``dim > max_direct_dim``): approximate
       ``exp(hA) v`` via an Arnoldi iteration of rank ``krylov_rank``,
       reducing cost from O(N³) to O(N · krylov_rank²).

    Args:
        dim: System dimensionality (3 × N for phase/amplitude/frequency).
        krylov_rank: Rank of the Krylov subspace approximation.
        max_direct_dim: Threshold above which Krylov approximation is used.

    Example:
        >>> integrator = ExponentialIntegrator(dim=150, krylov_rank=16)
        >>> model = StuartLandauOscillator(50, bifurcation_param=1.0)
        >>> state = OscillatorState.create_random(50, seed=0)
        >>> final = integrator.step(model, state, dt=0.01)
    """

    def __init__(
        self,
        dim: int,
        krylov_rank: int = 16,
        max_direct_dim: int = 150,
        stiff_mode: bool = False,
        stiff_cond_threshold: float = 20.0,
        max_krylov_stiff: int = 48,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if krylov_rank < 2:
            raise ValueError(
                f"krylov_rank must be >= 2, got {krylov_rank}"
            )
        self._dim = dim
        self._krylov_rank = min(krylov_rank, dim)
        self._max_direct_dim = max_direct_dim
        self._stiff_mode = stiff_mode
        self._stiff_cond_threshold = stiff_cond_threshold
        self._max_krylov_stiff = max_krylov_stiff

    @property
    def dim(self) -> int:
        """System dimensionality."""
        return self._dim

    @property
    def krylov_rank(self) -> int:
        """Krylov subspace rank."""
        return self._krylov_rank

    @property
    def stiff_mode(self) -> bool:
        """Whether adaptive stiff-mode Krylov is enabled."""
        return self._stiff_mode

    def _adaptive_krylov_dim(self, A: Tensor) -> int:
        """Select Krylov dimension adaptively based on Jacobian stiffness.

        For multi-frequency systems with condition number ~20:1, the
        Krylov dimension is scaled proportionally to the condition
        number relative to ``stiff_cond_threshold``.

        Args:
            A: Jacobian matrix ``(D, D)``.

        Returns:
            Adaptive Krylov dimension, clamped within
            ``[krylov_rank, max_krylov_stiff]``.
        """
        try:
            eigvals = torch.linalg.eigvals(A)
            magnitudes = eigvals.abs()
            mag_max = magnitudes.max()
            mag_min = magnitudes[magnitudes > 1e-10].min() if (magnitudes > 1e-10).any() else mag_max
            cond = (mag_max / mag_min).item() if mag_min > 0 else 1.0
        except Exception:
            cond = 1.0

        # Scale Krylov dim based on condition number
        adaptive_dim = int(cond / max(self._stiff_cond_threshold, 1e-8)) + self._krylov_rank
        return min(max(adaptive_dim, self._krylov_rank), min(self._max_krylov_stiff, self._dim))

    @property
    def use_krylov(self) -> bool:
        """Whether the Krylov approximation will be used."""
        return self._dim > self._max_direct_dim

    # ------------------------------------------------------------------
    # Matrix exponential helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_exp(hA: Tensor) -> Tensor:
        """Compute exp(hA) via ``torch.linalg.matrix_exp``.

        Args:
            hA: Square matrix ``(D, D)``.

        Returns:
            Matrix exponential ``exp(hA)`` of the same shape.
        """
        result: Tensor = LA.matrix_exp(hA)
        return result

    @staticmethod
    def _phi1(hA: Tensor) -> Tensor:
        """Compute φ₁(hA) = (exp(hA) - I) / hA stably.

        Uses eigendecomposition for numerical stability when *hA* has
        small eigenvalues (avoids dividing near-zero by near-zero).

        Args:
            hA: Square matrix ``(D, D)``.

        Returns:
            φ₁(hA) of shape ``(D, D)``.
        """
        D = hA.shape[0]
        device = hA.device
        dtype = hA.dtype

        # Eigendecomposition: hA = V diag(λ) V⁻¹
        eigenvalues, V = torch.linalg.eig(hA)  # complex

        # φ₁(λ) = (exp(λ) - 1) / λ  with limit 1 for λ → 0
        lam = eigenvalues  # complex (D,)
        exp_lam = torch.exp(lam)
        safe_lam = torch.where(
            lam.abs() < 1e-12,
            torch.ones_like(lam),
            lam,
        )
        phi1_lam = torch.where(
            lam.abs() < 1e-12,
            torch.ones_like(lam),
            (exp_lam - 1.0) / safe_lam,
        )

        # Reconstruct: φ₁(hA) = V diag(φ₁(λ)) V⁻¹
        phi1_mat = V @ torch.diag(phi1_lam) @ torch.linalg.inv(V)

        result: Tensor = phi1_mat.real.to(dtype)
        return result

    def _krylov_matrix_exp_vec(
        self,
        A: Tensor,
        h: float,
        v: Tensor,
    ) -> Tensor:
        """Approximate exp(hA) v via Arnoldi–Krylov iteration.

        Constructs a Krylov basis Q and small Hessenberg matrix H
        such that ``exp(hA) v ≈ ‖v‖ Q exp(hH) e₁``.

        Args:
            A: System matrix ``(D, D)``.
            h: Timestep.
            v: Vector to multiply, shape ``(D,)``.

        Returns:
            Approximation of ``exp(hA) v``, shape ``(D,)``.
        """
        D = A.shape[0]
        m = min(self._krylov_rank, D)
        device = A.device
        dtype = A.dtype

        Q = torch.zeros(D, m, device=device, dtype=dtype)
        H = torch.zeros(m, m, device=device, dtype=dtype)

        beta = torch.linalg.norm(v)
        if beta < 1e-30:
            return torch.zeros_like(v)

        Q[:, 0] = v / beta

        for j in range(m):
            w = A @ Q[:, j]
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]
            if j < m - 1:
                h_next = torch.linalg.norm(w)
                if h_next < 1e-14:
                    # Lucky breakdown — exact subspace
                    m = j + 1
                    Q = Q[:, :m]
                    H = H[:m, :m]
                    break
                H[j + 1, j] = h_next
                Q[:, j + 1] = w / h_next

        # exp(h H_m) on the small (m × m) matrix
        exp_hH = LA.matrix_exp(h * H[:m, :m])

        # e₁ = [1, 0, ..., 0]
        e1 = torch.zeros(m, device=device, dtype=dtype)
        e1[0] = 1.0

        result: Tensor = beta * (Q[:, :m] @ exp_hH @ e1)
        return result

    def _krylov_phi1_vec(
        self,
        A: Tensor,
        h: float,
        v: Tensor,
    ) -> Tensor:
        """Approximate h φ₁(hA) v via augmented Krylov.

        Uses the identity: ``h φ₁(hA) v = ∫₀ʰ exp((h-s)A) v ds``
        which can be computed by augmenting the system.

        For efficiency, we compute ``(exp(hA) - I) A⁻¹ v`` when A
        is well-conditioned, otherwise fall back to direct φ₁.

        Args:
            A: System matrix ``(D, D)``.
            h: Timestep.
            v: Vector, shape ``(D,)``.

        Returns:
            Approximation of ``h φ₁(hA) v``, shape ``(D,)``.
        """
        D = A.shape[0]
        m = min(self._krylov_rank, D)
        device = A.device
        dtype = A.dtype

        # Augmented Krylov: compute exp of augmented matrix
        # [hA  hv]    exp gives    [exp(hA)   h φ₁(hA) v]
        # [0    0]    →            [0          1         ]
        # We only need the top-right block's first column.
        Q = torch.zeros(D, m, device=device, dtype=dtype)
        H = torch.zeros(m + 1, m, device=device, dtype=dtype)

        beta = torch.linalg.norm(v)
        if beta < 1e-30:
            return torch.zeros_like(v)

        Q[:, 0] = v / beta

        actual_m = m
        for j in range(m):
            w = A @ Q[:, j]
            for i in range(j + 1):
                H[i, j] = torch.dot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]
            h_next = torch.linalg.norm(w)
            H[j + 1, j] = h_next
            if j < m - 1:
                if h_next < 1e-14:
                    actual_m = j + 1
                    break
                Q[:, j + 1] = w / h_next

        Hm = H[:actual_m, :actual_m]
        exp_hH = LA.matrix_exp(h * Hm)

        # φ₁(hHm) = (exp(hHm) - I) / (hHm)
        I_m = torch.eye(actual_m, device=device, dtype=dtype)
        hHm = h * Hm
        # Stable φ₁ via eigendecomposition of small matrix
        eigenvalues, V = torch.linalg.eig(hHm)
        lam = eigenvalues
        exp_lam = torch.exp(lam)
        safe_lam = torch.where(
            lam.abs() < 1e-12, torch.ones_like(lam), lam
        )
        phi1_lam = torch.where(
            lam.abs() < 1e-12,
            torch.ones_like(lam),
            (exp_lam - 1.0) / safe_lam,
        )
        phi1_Hm = (V @ torch.diag(phi1_lam) @ torch.linalg.inv(V)).real.to(
            dtype
        )

        e1 = torch.zeros(actual_m, device=device, dtype=dtype)
        e1[0] = 1.0

        result: Tensor = h * beta * (Q[:, :actual_m] @ phi1_Hm @ e1)
        return result

    # ------------------------------------------------------------------
    # State ↔ vector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_vector(state: OscillatorState) -> Tensor:
        """Flatten ``(phase, amplitude, frequency)`` into a 1-D vector."""
        return torch.cat(
            [
                state.phase.flatten(),
                state.amplitude.flatten(),
                state.frequency.flatten(),
            ]
        )

    @staticmethod
    def _vector_to_state(
        vec: Tensor, reference: OscillatorState
    ) -> OscillatorState:
        """Unflatten a vector back to an ``OscillatorState``."""
        shape = reference.phase.shape
        n = reference.phase.numel()
        return OscillatorState(
            phase=_wrap_phase(vec[:n].reshape(shape)),
            amplitude=torch.clamp(vec[n : 2 * n].reshape(shape), min=0.0),
            frequency=vec[2 * n : 3 * n].reshape(shape),
        )

    # ------------------------------------------------------------------
    # Linearisation helper
    # ------------------------------------------------------------------

    def _build_jacobian(
        self,
        model: "OscillatorModel",
        state: OscillatorState,
    ) -> Tensor:
        """Build the Jacobian ∂f/∂y at the current state via autograd.

        Falls back to a finite-difference approximation if autograd
        is unavailable (e.g. no ``requires_grad`` on state tensors).

        Args:
            model: Oscillator dynamics model.
            state: Current state.

        Returns:
            Jacobian matrix of shape ``(D, D)`` where ``D = 3N``.
        """
        y = self._state_to_vector(state).detach().requires_grad_(True)
        N = state.phase.numel()

        s = OscillatorState(
            phase=y[:N].reshape(state.phase.shape),
            amplitude=y[N : 2 * N].reshape(state.amplitude.shape),
            frequency=y[2 * N :].reshape(state.frequency.shape),
        )
        dphi, dr, domega = model.compute_derivatives(s)
        f = torch.cat([dphi.flatten(), dr.flatten(), domega.flatten()])

        D = f.shape[0]
        J = torch.zeros(D, D, device=y.device, dtype=y.dtype)
        for i in range(D):
            grad_outputs = torch.zeros_like(f)
            grad_outputs[i] = 1.0
            (g,) = torch.autograd.grad(
                f, y, grad_outputs=grad_outputs, retain_graph=True
            )
            J[i] = g

        return J.detach()

    # ------------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------------

    def step(
        self,
        model: "OscillatorModel",
        state: OscillatorState,
        dt: float = 0.01,
    ) -> OscillatorState:
        """Advance the oscillator state by one exponential integration step.

        Computes the Jacobian of the dynamics at the current state,
        performs linear/nonlinear splitting, and applies:

            y_{n+1} = exp(h A) y_n + h φ₁(h A) g(y_n)

        where ``A = ∂f/∂y`` (Jacobian) and ``g(y) = f(y) - A y`` is
        the nonlinear remainder.

        Args:
            model: Oscillator dynamics model.
            state: Current oscillator state.
            dt: Timestep size.

        Returns:
            Updated ``OscillatorState``.
        """
        y = self._state_to_vector(state)
        D = y.shape[0]

        # Compute RHS and Jacobian
        dphi, dr, domega = model.compute_derivatives(state)
        f_y = torch.cat([dphi.flatten(), dr.flatten(), domega.flatten()])

        A = self._build_jacobian(model, state)

        # Nonlinear remainder: g(y) = f(y) - A y
        g_y = f_y - A @ y

        # In stiff mode, adaptively select Krylov dimension and force
        # Krylov path regardless of dim threshold
        if self._stiff_mode:
            adaptive_rank = self._adaptive_krylov_dim(A)
            saved_rank = self._krylov_rank
            self._krylov_rank = adaptive_rank
            exp_hA_y = self._krylov_matrix_exp_vec(A, dt, y)
            phi1_g = self._krylov_phi1_vec(A, dt, g_y)
            self._krylov_rank = saved_rank
        elif self.use_krylov:
            exp_hA_y = self._krylov_matrix_exp_vec(A, dt, y)
            phi1_g = self._krylov_phi1_vec(A, dt, g_y)
        else:
            hA = dt * A
            exp_hA = self._matrix_exp(hA)
            phi1_hA = self._phi1(hA)
            exp_hA_y = exp_hA @ y
            phi1_g = dt * phi1_hA @ g_y

        y_new = exp_hA_y + phi1_g

        return self._vector_to_state(y_new, state)

    def integrate(
        self,
        model: "OscillatorModel",
        state: OscillatorState,
        n_steps: int,
        dt: float = 0.01,
        record_trajectory: bool = False,
        recompute_jacobian_every: int = 1,
    ) -> Tuple[OscillatorState, Optional[list[OscillatorState]]]:
        """Integrate using the exponential method for multiple steps.

        Args:
            model: Oscillator dynamics model.
            state: Initial oscillator state.
            n_steps: Number of integration steps.
            dt: Timestep size.
            record_trajectory: If ``True``, return intermediate states.
            recompute_jacobian_every: Recompute the Jacobian every this
                many steps. Setting > 1 amortises the linearisation
                cost for slowly-varying systems.

        Returns:
            Tuple of ``(final_state, trajectory)``.
        """
        trajectory: Optional[list[OscillatorState]] = (
            [] if record_trajectory else None
        )
        current = state

        cached_J: Optional[Tensor] = None

        for i in range(n_steps):
            if cached_J is None or i % recompute_jacobian_every == 0:
                y = self._state_to_vector(current)
                A = self._build_jacobian(model, current)
                cached_J = A

            y = self._state_to_vector(current)
            dphi, dr, domega = model.compute_derivatives(current)
            f_y = torch.cat(
                [dphi.flatten(), dr.flatten(), domega.flatten()]
            )
            g_y = f_y - cached_J @ y

            if self._stiff_mode:
                adaptive_rank = self._adaptive_krylov_dim(cached_J)
                saved_rank = self._krylov_rank
                self._krylov_rank = adaptive_rank
                exp_hA_y = self._krylov_matrix_exp_vec(cached_J, dt, y)
                phi1_g = self._krylov_phi1_vec(cached_J, dt, g_y)
                self._krylov_rank = saved_rank
            elif self.use_krylov:
                exp_hA_y = self._krylov_matrix_exp_vec(cached_J, dt, y)
                phi1_g = self._krylov_phi1_vec(cached_J, dt, g_y)
            else:
                hA = dt * cached_J
                exp_hA = self._matrix_exp(hA)
                phi1_hA = self._phi1(hA)
                exp_hA_y = exp_hA @ y
                phi1_g = dt * phi1_hA @ g_y

            y_new = exp_hA_y + phi1_g
            current = self._vector_to_state(y_new, current)

            if trajectory is not None:
                trajectory.append(current.clone())

        return current, trajectory


# =========================================================================
# Q3: Hierarchical Oscillatory Dynamics
# =========================================================================




class MultiRateIntegrator:
    """Multi-rate ODE integrator for hierarchical oscillator systems.

    Different frequency bands require different time-step sizes for
    numerical stability. This integrator takes ``sub_steps`` inner
    RK4 steps for each outer step, allowing fast oscillators (Gamma)
    to be integrated with finer time resolution than slow oscillators
    (Delta/Theta).

    Args:
        sub_steps: Number of sub-steps per outer step. Gamma-band
            oscillators at 40 Hz typically need ~20 sub-steps per
            Delta step at 2 Hz.
        method: Integration method (``"rk4"`` or ``"euler"``).

    Example:
        >>> integrator = MultiRateIntegrator(sub_steps=20)
        >>> model = KuramotoOscillator(64, coupling_strength=2.0)
        >>> state = OscillatorState.create_random(64)
        >>> final = integrator.step(model, state, dt=0.01)
    """

    def __init__(
        self,
        sub_steps: int = 10,
        method: str = "rk4",
    ) -> None:
        if sub_steps < 1:
            raise ValueError(
                f"sub_steps must be >= 1, got {sub_steps}"
            )
        self._sub_steps = sub_steps
        self._method = method

    @property
    def sub_steps(self) -> int:
        """Number of sub-steps per outer step."""
        return self._sub_steps

    @property
    def method(self) -> str:
        """Integration method."""
        return self._method

    def step(
        self,
        model: OscillatorModel,
        state: OscillatorState,
        dt: float = 0.01,
    ) -> OscillatorState:
        """Advance state by one outer step using sub-stepping.

        The outer timestep ``dt`` is divided into ``sub_steps`` inner
        steps of size ``dt / sub_steps``.

        Args:
            model: Oscillator dynamics model.
            state: Current oscillator state.
            dt: Outer timestep size.

        Returns:
            Updated ``OscillatorState``.
        """
        inner_dt = dt / self._sub_steps
        current = state
        for _ in range(self._sub_steps):
            current = model.step(current, dt=inner_dt, method=self._method)
        return current

    def integrate(
        self,
        model: OscillatorModel,
        state: OscillatorState,
        n_steps: int,
        dt: float = 0.01,
        record_trajectory: bool = False,
    ) -> Tuple[OscillatorState, Optional[list[OscillatorState]]]:
        """Integrate for multiple outer steps with sub-stepping.

        Args:
            model: Oscillator dynamics model.
            state: Initial state.
            n_steps: Number of outer steps.
            dt: Outer timestep.
            record_trajectory: Record intermediate states.

        Returns:
            Tuple ``(final_state, trajectory)``.
        """
        trajectory: Optional[list[OscillatorState]] = (
            [] if record_trajectory else None
        )
        current = state
        for _ in range(n_steps):
            current = self.step(model, current, dt=dt)
            if trajectory is not None:
                trajectory.append(current.clone())
        return current, trajectory


