"""OscilloSim v2.2 — Large-scale oscillator simulation with sparse coupling.

Provides a high-level simulation API for 1M+ oscillator systems using
CSR sparse coupling, k-NN topologies, ring/small-world topologies,
and GPU-accelerated integration. Supports chimera state detection via
local order parameter, bimodality index, strength of incoherence,
discontinuity measure, and chimera index (Year 4 Q1–Q1.3).

Public API:
    - :class:`OscilloSim` — Main simulation class for large-scale oscillator dynamics.
    - :class:`SimulationResult` — Dataclass holding simulation output.
    - :func:`quick_simulate` — One-call convenience function.
    - :func:`ring_topology` — 1D ring lattice neighbour index builder.
    - :func:`small_world_topology` — Watts-Strogatz small-world builder.
    - :func:`cosine_coupling_kernel` — Nonlocal cosine kernel weights (Abrams & Strogatz).
    - :func:`local_order_parameter` — Per-oscillator local phase coherence.
    - :func:`bimodality_index` — Bimodality coefficient for chimera detection.
    - :func:`strength_of_incoherence` — SI metric (Gopal et al. 2014).
    - :func:`discontinuity_measure` — Local curvature chimera metric.
    - :func:`chimera_index` — Fraction of incoherent oscillators.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import Tensor

# Suppress PyTorch sparse CSR beta warnings
warnings.filterwarnings(
    "ignore",
    message=".*Sparse CSR tensor support is in beta.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*sparse csr tensor.*",
    category=UserWarning,
)


@dataclass
class SimulationResult:
    """Result container for OscilloSim simulations.

    Attributes:
        final_phase: Final oscillator phases ``(N,)`` or ``(B, N)``.
        final_amplitude: Final oscillator amplitudes.
        order_parameter: Kuramoto order parameter r at each recorded step.
        wall_time_s: Total simulation wall-clock time in seconds.
        n_oscillators: Number of oscillators simulated.
        n_steps: Total integration steps.
        coupling_mode: Coupling topology used.
        device: Device used for simulation.
        throughput: Oscillator-steps per second.
        trajectory_phase: Phase trajectory if ``record_trajectory=True``,
            shape ``(n_record, N)`` or ``(n_record, B, N)``.
    """

    final_phase: Tensor
    final_amplitude: Tensor
    order_parameter: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    n_oscillators: int = 0
    n_steps: int = 0
    coupling_mode: str = ""
    device: str = "cpu"
    throughput: float = 0.0
    trajectory_phase: Optional[Tensor] = None


def _compute_order_parameter(phase: Tensor) -> float:
    """Compute Kuramoto order parameter r = |mean(exp(i*phase))|."""
    if phase.dim() > 1:
        phase = phase.reshape(-1)
    z = torch.exp(1j * phase.to(torch.complex64))
    return float(torch.abs(z.mean()).item())


class OscilloSim:
    """Large-scale oscillator simulator with sparse coupling.

    Supports up to 1M+ oscillators on GPU via sparse CSR coupling
    matrices and efficient k-NN topologies. Implements Kuramoto dynamics
    with Stuart-Landau amplitude control.

    Coupling modes:
        - ``"mean_field"``: O(N) global coupling via order parameter.
        - ``"sparse_knn"``: O(N·k) k-nearest-neighbor coupling.
        - ``"csr"``: O(nnz) arbitrary sparse coupling via CSR matrix.
        - ``"ring"``: O(N·k) 1D ring lattice spatial neighbours (chimera-capable).
        - ``"small_world"``: O(N·k) Watts-Strogatz small-world topology.
        - ``"auto"``: Selects based on N (mean_field for N ≥ 100K,
          sparse_knn for N ≥ 1K, full pairwise otherwise).

    Args:
        n_oscillators: Total number of oscillators.
        coupling_strength: Global coupling constant K.
        coupling_mode: One of ``"mean_field"``, ``"sparse_knn"``,
            ``"csr"``, ``"ring"``, ``"small_world"``, ``"auto"``.
        k_neighbors: Number of neighbours for sparse_knn/ring/small_world.
        sparsity: Sparsity level for CSR mode (0.0 = dense, 0.99 = very sparse).
        mu: Stuart-Landau growth parameter.
        freq_mean: Mean natural frequency.
        freq_std: Standard deviation of natural frequencies.
        phase_lag: Phase lag α for coupling function sin(φ_j − φ_i − α).
            Required for chimera states (α ≈ 1.457, Abrams & Strogatz 2004).
            Default 0.0 (standard Kuramoto).
        p_rewire: Rewiring probability for small_world mode (default 0.1).
        integrator: Integration method — ``"euler"`` (1st order, default) or
            ``"rk4"`` (4th order Runge-Kutta, recommended for chimera studies).
        coupling_weights: Optional per-edge weight tensor ``(N, k)`` for
            distance-dependent coupling kernels. Use :func:`cosine_coupling_kernel`
            to generate. If ``None``, uniform weights are used.
        device: PyTorch device string.
        seed: Random seed for reproducibility.
        dtype: Tensor dtype (default float32).

    Example:
        >>> sim = OscilloSim(n_oscillators=1_000_000, coupling_mode="mean_field",
        ...                   device="cuda")
        >>> result = sim.run(n_steps=100, dt=0.01)
        >>> print(f"Final r = {result.order_parameter[-1]:.3f}")
        >>> print(f"Throughput: {result.throughput:.1e} osc·steps/s")
    """

    def __init__(
        self,
        n_oscillators: int = 1000,
        coupling_strength: float = 2.0,
        coupling_mode: Literal[
            "mean_field", "sparse_knn", "csr", "ring", "small_world", "auto"
        ] = "auto",
        k_neighbors: int = 8,
        sparsity: float = 0.99,
        mu: float = 1.0,
        freq_mean: float = 5.0,
        freq_std: float = 0.5,
        phase_lag: float = 0.0,
        p_rewire: float = 0.1,
        integrator: Literal["euler", "rk4"] = "euler",
        coupling_weights: Optional[Tensor] = None,
        device: str = "cpu",
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_oscillators = n_oscillators
        self.coupling_strength = coupling_strength
        self.k_neighbors = k_neighbors
        self.sparsity = sparsity
        self.mu = mu
        self.phase_lag = phase_lag
        self.p_rewire = p_rewire
        self.integrator = integrator
        self.device = torch.device(device)
        self.seed = seed
        self.dtype = dtype

        # Auto-select coupling mode
        if coupling_mode == "auto":
            if n_oscillators >= 100_000:
                self._coupling_mode = "mean_field"
            elif n_oscillators >= 1_000:
                self._coupling_mode = "sparse_knn"
            else:
                self._coupling_mode = "csr"
        else:
            self._coupling_mode = coupling_mode

        # Setup RNG
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # Natural frequencies (Lorentzian-like distribution)
        self.frequencies = (
            torch.randn(n_oscillators, generator=gen, dtype=dtype) * freq_std
            + freq_mean
        ).to(self.device)

        # Build coupling structure
        self._coupling_csr: Optional[Tensor] = None
        self._neighbors: Optional[Tensor] = None
        self._coupling_weights: Optional[Tensor] = None
        self._build_coupling(gen)

        # Apply user-supplied coupling weights (e.g. cosine kernel)
        if coupling_weights is not None:
            self._coupling_weights = coupling_weights.to(
                device=self.device, dtype=self.dtype
            )

    def _build_coupling(self, gen: torch.Generator) -> None:
        """Build coupling matrices/topologies based on mode."""
        N = self.n_oscillators

        if self._coupling_mode == "sparse_knn":
            # k-NN topology: random neighbors
            k = min(self.k_neighbors, N - 1)
            # Build random neighbor indices
            neighbors = torch.zeros(N, k, dtype=torch.long)
            for i in range(N):
                candidates = torch.cat(
                    [torch.arange(0, i), torch.arange(i + 1, N)]
                )
                perm = torch.randperm(len(candidates), generator=gen)[:k]
                neighbors[i] = candidates[perm]
            self._neighbors = neighbors.to(self.device)

        elif self._coupling_mode == "ring":
            # 1D ring lattice: each oscillator connects to k nearest
            # spatial neighbours. Base topology for chimera states.
            k = min(self.k_neighbors, N - 1)
            if k % 2 != 0:
                k = max(k - 1, 2)
            self._neighbors = ring_topology(N, k, device=str(self.device))

        elif self._coupling_mode == "small_world":
            # Watts-Strogatz: ring lattice + random rewiring
            k = min(self.k_neighbors, N - 1)
            if k % 2 != 0:
                k = max(k - 1, 2)
            self._neighbors = small_world_topology(
                N, k, p_rewire=self.p_rewire,
                device=str(self.device), seed=self.seed,
            )

        elif self._coupling_mode == "csr":
            # Build sparse CSR coupling matrix
            nnz_per_row = max(1, int((1 - self.sparsity) * N))
            crow_indices = [0]
            col_indices = []
            values = []
            running = 0
            for i in range(N):
                candidates = torch.cat(
                    [torch.arange(0, i), torch.arange(i + 1, N)]
                )
                perm = torch.randperm(len(candidates), generator=gen)[:nnz_per_row]
                cols = candidates[perm].sort().values
                col_indices.append(cols)
                vals = (
                    torch.randn(len(cols), generator=gen, dtype=self.dtype) * 0.1
                    + self.coupling_strength / max(nnz_per_row, 1)
                )
                values.append(vals)
                running += len(cols)
                crow_indices.append(running)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._coupling_csr = torch.sparse_csr_tensor(
                    torch.tensor(crow_indices, dtype=torch.int64),
                    torch.cat(col_indices).to(torch.int64),
                    torch.cat(values).to(self.dtype),
                    size=(N, N),
                ).to(self.device)

        # mean_field needs no precomputation

    def _step_mean_field(
        self, phase: Tensor, amplitude: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Mean-field O(N) coupling step."""
        # Complex order parameter
        z = torch.exp(1j * phase.to(torch.complex64))
        Z = z.mean(dim=-1, keepdim=True)
        R = Z.abs().to(self.dtype)
        psi = Z.angle().to(self.dtype)

        # Phase update: dφ/dt = ω + K·R·sin(ψ - φ - α)
        dphi = self.frequencies * dt + self.coupling_strength * R * torch.sin(
            psi - phase - self.phase_lag
        ) * dt
        new_phase = (phase + 2 * math.pi * self.frequencies * dt + dphi) % (
            2 * math.pi
        )

        # Stuart-Landau amplitude
        da = dt * amplitude * (self.mu - amplitude * amplitude)
        new_amp = torch.clamp(amplitude + da, min=1e-6, max=10.0)

        return new_phase, new_amp

    def _compute_sparse_coupling(self, phase: Tensor) -> Tensor:
        """Compute coupling term for sparse k-NN / ring / small-world modes.

        Supports optional per-edge weights (e.g. cosine kernel).

        Args:
            phase: Current oscillator phases ``(N,)`` or ``(B, N)``.

        Returns:
            Coupling contribution ``(N,)`` or ``(B, N)``.
        """
        assert self._neighbors is not None
        k = self._neighbors.shape[1]

        # Gather neighbor phases
        if phase.dim() == 1:
            nbr_phase = phase[self._neighbors]  # (N, k)
            diff = torch.sin(nbr_phase - phase.unsqueeze(-1) - self.phase_lag)
        else:
            nbr_phase = phase[:, self._neighbors.reshape(-1)].reshape(
                phase.shape[0], self.n_oscillators, k
            )
            diff = torch.sin(
                nbr_phase - phase.unsqueeze(-1) - self.phase_lag
            )

        # Apply per-edge coupling weights if available
        if self._coupling_weights is not None:
            # Weights shape (N, k) — broadcast over batch if needed
            weighted = diff * self._coupling_weights
            coupling = self.coupling_strength * weighted.sum(dim=-1)
        else:
            coupling = self.coupling_strength / k * diff.sum(dim=-1)

        return coupling

    def _phase_derivative_sparse(self, phase: Tensor) -> Tensor:
        """Compute dφ/dt for sparse coupling modes (ring/knn/small_world).

        Args:
            phase: Current phases ``(N,)`` or ``(B, N)``.

        Returns:
            Phase time derivative ``(N,)`` or ``(B, N)``.
        """
        coupling = self._compute_sparse_coupling(phase)
        return 2 * math.pi * self.frequencies + coupling

    def _step_sparse_knn(
        self, phase: Tensor, amplitude: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Sparse k-NN / ring / small-world O(N·k) coupling step (Euler)."""
        dphi_dt = self._phase_derivative_sparse(phase)
        new_phase = (phase + dphi_dt * dt) % (2 * math.pi)

        da = dt * amplitude * (self.mu - amplitude * amplitude)
        new_amp = torch.clamp(amplitude + da, min=1e-6, max=10.0)

        return new_phase, new_amp

    def _step_sparse_knn_rk4(
        self, phase: Tensor, amplitude: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Sparse k-NN / ring / small-world O(N·k) coupling step (RK4).

        4th-order Runge-Kutta integration improves chimera detection
        accuracy by 4-8 orders of magnitude vs Euler (literature validated).
        """
        k1 = self._phase_derivative_sparse(phase)
        k2 = self._phase_derivative_sparse(phase + 0.5 * dt * k1)
        k3 = self._phase_derivative_sparse(phase + 0.5 * dt * k2)
        k4 = self._phase_derivative_sparse(phase + dt * k3)

        new_phase = (phase + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % (
            2 * math.pi
        )

        # Stuart-Landau amplitude (Euler is fine for amplitude)
        da = dt * amplitude * (self.mu - amplitude * amplitude)
        new_amp = torch.clamp(amplitude + da, min=1e-6, max=10.0)

        return new_phase, new_amp

    def _step_csr(
        self, phase: Tensor, amplitude: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """CSR sparse matrix O(nnz) coupling step."""
        assert self._coupling_csr is not None
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if phase.dim() == 1:
                coupling = (
                    self._coupling_csr @ sin_phase * cos_phase
                    - self._coupling_csr @ cos_phase * sin_phase
                )
            else:
                coupling = (
                    torch.sparse.mm(self._coupling_csr, sin_phase.T).T * cos_phase
                    - torch.sparse.mm(self._coupling_csr, cos_phase.T).T * sin_phase
                )

        new_phase = (
            phase + 2 * math.pi * self.frequencies * dt + coupling * dt
        ) % (2 * math.pi)

        da = dt * amplitude * (self.mu - amplitude * amplitude)
        new_amp = torch.clamp(amplitude + da, min=1e-6, max=10.0)

        return new_phase, new_amp

    def _step(
        self, phase: Tensor, amplitude: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Dispatch to coupling-mode-specific step function."""
        if self._coupling_mode == "mean_field":
            return self._step_mean_field(phase, amplitude, dt)
        elif self._coupling_mode in ("sparse_knn", "ring", "small_world"):
            if self.integrator == "rk4":
                return self._step_sparse_knn_rk4(phase, amplitude, dt)
            return self._step_sparse_knn(phase, amplitude, dt)
        elif self._coupling_mode == "csr":
            return self._step_csr(phase, amplitude, dt)
        else:
            raise ValueError(f"Unknown coupling mode: {self._coupling_mode}")

    def run(
        self,
        n_steps: int = 100,
        dt: float = 0.01,
        record_trajectory: bool = False,
        record_interval: int = 10,
        initial_phase: Optional[Tensor] = None,
        initial_amplitude: Optional[Tensor] = None,
    ) -> SimulationResult:
        """Run the simulation.

        Args:
            n_steps: Number of integration steps.
            dt: Timestep per step.
            record_trajectory: Whether to record phase trajectory.
            record_interval: Record every N-th step (if trajectory enabled).
            initial_phase: Initial phases ``(N,)``. Random uniform [0, 2π)
                if not provided.
            initial_amplitude: Initial amplitudes ``(N,)``. Ones if not provided.

        Returns:
            :class:`SimulationResult` with final state, order parameters,
            timing, and optional trajectory.
        """
        N = self.n_oscillators
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 1)

        # Initial state
        if initial_phase is None:
            phase = (
                torch.rand(N, generator=gen, dtype=self.dtype).to(self.device)
                * 2
                * math.pi
            )
        else:
            phase = initial_phase.to(device=self.device, dtype=self.dtype)

        if initial_amplitude is None:
            amplitude = torch.ones(N, dtype=self.dtype, device=self.device)
        else:
            amplitude = initial_amplitude.to(device=self.device, dtype=self.dtype)

        order_params: list[float] = []
        trajectory: list[Tensor] = []

        # CUDA synchronization before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        t0 = time.perf_counter()

        with torch.no_grad():
            for step in range(n_steps):
                phase, amplitude = self._step(phase, amplitude, dt)

                # Record periodically
                if step % record_interval == 0 or step == n_steps - 1:
                    order_params.append(_compute_order_parameter(phase))
                    if record_trajectory:
                        trajectory.append(phase.detach().cpu())

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        wall_time = time.perf_counter() - t0
        throughput = (N * n_steps) / max(wall_time, 1e-12)

        traj_tensor: Optional[Tensor] = None
        if trajectory:
            traj_tensor = torch.stack(trajectory)

        return SimulationResult(
            final_phase=phase.detach().cpu(),
            final_amplitude=amplitude.detach().cpu(),
            order_parameter=order_params,
            wall_time_s=wall_time,
            n_oscillators=N,
            n_steps=n_steps,
            coupling_mode=self._coupling_mode,
            device=str(self.device),
            throughput=throughput,
            trajectory_phase=traj_tensor,
        )

    @property
    def coupling_mode(self) -> str:
        """Return the active coupling mode."""
        return self._coupling_mode

    def state_summary(self) -> dict[str, object]:
        """Return a summary dict of the simulator configuration.

        Returns:
            Dict with n_oscillators, coupling_mode, coupling_strength,
            device, and mode-specific info.
        """
        info: dict[str, object] = {
            "n_oscillators": self.n_oscillators,
            "coupling_mode": self._coupling_mode,
            "coupling_strength": self.coupling_strength,
            "mu": self.mu,
            "phase_lag": self.phase_lag,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        if self._coupling_mode == "sparse_knn":
            info["k_neighbors"] = self.k_neighbors
        elif self._coupling_mode in ("ring", "small_world"):
            info["k_neighbors"] = self.k_neighbors
            if self._coupling_mode == "small_world":
                info["p_rewire"] = self.p_rewire
        elif self._coupling_mode == "csr":
            info["sparsity"] = self.sparsity
            if self._coupling_csr is not None:
                info["nnz"] = int(self._coupling_csr._nnz())
        return info


def quick_simulate(
    n_oscillators: int = 10_000,
    n_steps: int = 100,
    coupling_strength: float = 2.0,
    device: str = "cpu",
    seed: int = 42,
) -> SimulationResult:
    """One-call convenience function for quick simulations.

    Auto-selects coupling mode based on oscillator count.

    Args:
        n_oscillators: Number of oscillators.
        n_steps: Integration steps.
        coupling_strength: Coupling constant K.
        device: Device string.
        seed: Random seed.

    Returns:
        :class:`SimulationResult`.

    Example:
        >>> result = quick_simulate(100_000, n_steps=50, device="cuda")
        >>> print(f"r = {result.order_parameter[-1]:.3f}")
    """
    sim = OscilloSim(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        coupling_mode="auto",
        device=device,
        seed=seed,
    )
    return sim.run(n_steps=n_steps)


# =========================================================================
# Year 4 Q1 — T.1: Ring & Small-World Topology Builders
# =========================================================================


def ring_topology(
    N: int,
    k: int,
    device: str = "cpu",
) -> Tensor:
    """Build a 1D ring lattice neighbour index tensor.

    Each oscillator i is connected to its k nearest spatial neighbours
    on a ring (k/2 on each side). This is the base topology required
    for chimera state emergence (Abrams & Strogatz 2004).

    Args:
        N: Number of oscillators (nodes on the ring).
        k: Number of neighbours per oscillator (must be even).
        device: Target device for the output tensor.

    Returns:
        Neighbour index tensor of shape ``(N, k)`` with dtype int64.

    Raises:
        ValueError: If k is odd, k < 2, or k >= N.

    Example:
        >>> nbr = ring_topology(256, k=10)
        >>> print(nbr.shape)
        torch.Size([256, 10])
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k % 2 != 0:
        raise ValueError(f"k must be even for ring lattice, got {k}")
    if k >= N:
        raise ValueError(f"k must be < N, got k={k}, N={N}")

    half = k // 2
    idx = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
    offsets = torch.cat([
        torch.arange(-half, 0, device=device),
        torch.arange(1, half + 1, device=device),
    ])  # (k,)
    nbr = (idx + offsets) % N  # (N, k)
    return nbr.to(torch.int64)


def small_world_topology(
    N: int,
    k: int,
    p_rewire: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
) -> Tensor:
    """Build a Watts-Strogatz small-world neighbour index tensor.

    Starts from a ring lattice of k neighbours and randomly rewires
    each edge with probability ``p_rewire``. This produces networks
    with high clustering (like a lattice) and short path lengths
    (like a random graph).

    Args:
        N: Number of oscillators.
        k: Neighbours per oscillator in the base ring (must be even).
        p_rewire: Probability of rewiring each edge (0 = pure ring,
            1 = fully random).
        device: Target device.
        seed: Random seed for reproducibility.

    Returns:
        Neighbour index tensor of shape ``(N, k)`` with dtype int64.

    Raises:
        ValueError: If k is odd, k < 2, or k >= N.

    Example:
        >>> nbr = small_world_topology(256, k=10, p_rewire=0.1)
        >>> print(nbr.shape)
        torch.Size([256, 10])
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")
    if k >= N:
        raise ValueError(f"k must be < N, got k={k}, N={N}")

    # Start with ring lattice
    nbr = ring_topology(N, k, device=device)

    # Rewire with probability p (avoid self-loops)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.rand(N, k, generator=gen) < p_rewire
    mask = mask.to(device)
    random_targets = torch.randint(
        0, N, (N, k), generator=gen, device="cpu"
    ).to(device)
    nbr = torch.where(mask, random_targets, nbr)

    # Fix self-loops: replace any self-referencing neighbor
    node_ids = torch.arange(N, device=device).unsqueeze(1).expand(N, k)
    self_loop_mask = nbr == node_ids
    if self_loop_mask.any():
        # Replace self-loops with the next neighbor in ring order
        replacement = (node_ids + 1) % N
        nbr = torch.where(self_loop_mask, replacement, nbr)

    return nbr.to(torch.int64)


# =========================================================================
# Year 4 Q1 — T.2: Chimera State Detection Utilities
# =========================================================================


def local_order_parameter(
    phase: Tensor,
    nbr_idx: Tensor,
) -> Tensor:
    """Compute the local Kuramoto order parameter for each oscillator.

    For each oscillator i, computes:

    .. math::

        r_i = \\left| \\frac{1}{k} \\sum_{j \\in \\text{nbr}(i)}
              \\exp(i \\theta_j) \\right|

    where nbr(i) are the spatial neighbours of oscillator i on the
    coupling topology. In a chimera state, coherent oscillators have
    r_i ≈ 1 while incoherent oscillators have r_i ≈ 0, producing
    a bimodal distribution.

    Args:
        phase: Oscillator phases ``(N,)``.
        nbr_idx: Neighbour indices ``(N, k)`` from a ring or
            small-world topology.

    Returns:
        Local order parameter ``(N,)`` in [0, 1].

    Example:
        >>> phase = torch.rand(256) * 2 * 3.14159
        >>> nbr = ring_topology(256, k=10)
        >>> r_local = local_order_parameter(phase, nbr)
        >>> print(r_local.shape, r_local.min().item(), r_local.max().item())
    """
    # Gather neighbour phases
    nbr_phase = phase[nbr_idx]  # (N, k)
    # Complex order parameter per oscillator
    z = torch.exp(1j * nbr_phase.to(torch.complex64))  # (N, k)
    r = torch.abs(z.mean(dim=-1))  # (N,)
    return r.float()


def bimodality_index(values: Tensor) -> float:
    """Compute Sarle's bimodality coefficient.

    .. math::

        BC = \\frac{\\gamma^2 + 1}{\\kappa}

    where γ is the skewness and κ is the kurtosis (excess kurtosis + 3).
    A uniform distribution has BC = 5/9 ≈ 0.555. Values above this
    threshold suggest bimodality, indicating chimera state coexistence
    of coherent and incoherent domains.

    Args:
        values: 1D tensor of values (e.g., local order parameters).

    Returns:
        Bimodality coefficient (float). BC > 0.555 suggests bimodality.

    Example:
        >>> r_local = local_order_parameter(phase, nbr)
        >>> bc = bimodality_index(r_local)
        >>> if bc > 0.555:
        ...     print("Chimera state detected!")
    """
    v = values.float().cpu()
    n = v.numel()
    if n < 4:
        return 0.0

    mean = v.mean()
    diff = v - mean
    var = diff.pow(2).mean()
    if var < 1e-12:
        return 0.0

    std = var.sqrt()
    skew = (diff.pow(3).mean()) / (std.pow(3))
    # Excess kurtosis + 3 = kurtosis
    kurt = (diff.pow(4).mean()) / (var.pow(2))

    if kurt < 1e-12:
        return 0.0

    bc = float((skew.pow(2) + 1.0) / kurt)
    return bc


# ---------------------------------------------------------------------------
# Cosine coupling kernel (Abrams & Strogatz nonlocal coupling)
# ---------------------------------------------------------------------------


def cosine_coupling_kernel(
    n: int,
    k: int,
    A: float = 0.995,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate per-edge cosine coupling weights for a ring topology.

    Implements the nonlocal coupling kernel from Abrams & Strogatz (2004):

    .. math::

        G(i - j) = \\frac{1}{2\\pi}\\left[1 + A \\cos\\!\\left(
            \\frac{2\\pi (i - j)}{N}\\right)\\right]

    where *A* controls the asymmetry.  The function returns a weight matrix
    of shape ``(N, k)`` that can be passed directly to
    :class:`OscilloSim` via the ``coupling_weights`` parameter.

    Args:
        n: Number of oscillators.
        k: Number of neighbours per oscillator (as in the ring topology).
        A: Asymmetry parameter in [0, 1]. ``A → 1`` gives strongly
            nonlocal coupling, ``A = 0`` recovers uniform weights.
        device: Device string.
        dtype: Tensor dtype.

    Returns:
        Tensor of shape ``(N, k)`` with normalised coupling weights.

    Example:
        >>> from prinet.utils.oscillosim import cosine_coupling_kernel, ring_topology
        >>> nbr = ring_topology(256, 90)
        >>> weights = cosine_coupling_kernel(256, 90, A=0.995)
        >>> sim = OscilloSim(256, coupling_strength=100.0,
        ...     coupling_mode="ring", k_neighbors=90, phase_lag=1.521,
        ...     integrator="rk4", coupling_weights=weights)
    """
    half_k = k // 2
    # Offsets: -half_k, ..., -1, 1, ..., half_k  (skip 0)
    left = torch.arange(-half_k, 0, dtype=dtype, device=device)
    right = torch.arange(1, half_k + 1, dtype=dtype, device=device)
    offsets = torch.cat([left, right])  # (k,)
    if offsets.numel() < k:
        # Odd k — include one extra on the right
        offsets = torch.cat(
            [offsets, torch.tensor([half_k + 1], dtype=dtype, device=device)]
        )
    offsets = offsets[:k]

    raw = (1.0 + A * torch.cos(2.0 * math.pi * offsets / n)) / (2.0 * math.pi)
    # Normalise so each row sums to 1.0 (K * Σ w·sin matches K/k * Σ sin for uniform)
    weights = raw / raw.sum()
    # Broadcast to (N, k)
    return weights.unsqueeze(0).expand(n, -1).contiguous()


# ---------------------------------------------------------------------------
# Strength of Incoherence (SI) — Gopal et al. 2014
# ---------------------------------------------------------------------------


def strength_of_incoherence(
    phase: Tensor,
    window_size: int = 10,
) -> Tensor:
    """Compute the Strength of Incoherence (SI) profile.

    The SI metric introduced by Gopal *et al.* (Phys. Rev. E, 2014) measures
    local coherence by comparing the smoothed and unsmoothed finite-difference
    series of the phase field.  For each oscillator *m*:

    .. math::

        z_m = \\theta_m - \\theta_{m+1}

    SI is computed as:

    .. math::

        \\mathrm{SI} = 1 - \\frac{\\langle |\\bar{z}_m| \\rangle}
                                  {\\langle |z_m| \\rangle}

    where :math:`\\bar{z}_m` is the spatial running average of :math:`z_m`
    over a window of ``window_size`` neighbours.  SI → 0 for coherent
    regions and SI → 1 for incoherent regions.

    Args:
        phase: Phase vector ``(N,)`` (radians, may be unwrapped or wrapped).
        window_size: Sliding-window width for spatial smoothing.

    Returns:
        Scalar SI value (float tensor).

    Example:
        >>> si = strength_of_incoherence(phase, window_size=10)
        >>> print(f"SI = {si.item():.3f}")  # ~0 coherent, ~1 chimera
    """
    phase = phase.detach().float()
    N = phase.numel()
    if N < 3:
        return torch.tensor(0.0)

    # Finite difference on the ring
    z = torch.remainder(
        phase - torch.roll(phase, -1), 2 * math.pi
    )
    # Centre to [-π, π]
    z = z - math.pi

    abs_z = z.abs()
    denom = abs_z.mean()
    if denom < 1e-12:
        return torch.tensor(0.0)

    # Spatial running average via circular conv1d
    kernel = torch.ones(1, 1, window_size, device=phase.device) / window_size
    z_padded = torch.cat([z[-window_size:], z, z[:window_size]])
    z_smooth = torch.nn.functional.conv1d(
        z_padded.unsqueeze(0).unsqueeze(0), kernel
    ).squeeze()
    z_smooth = z_smooth[:N]

    numer = z_smooth.abs().mean()
    si = 1.0 - numer / denom
    return si.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Discontinuity Measure (η) — local curvature chimera metric
# ---------------------------------------------------------------------------


def discontinuity_measure(
    phase: Tensor,
    threshold_ratio: float = 0.01,
) -> tuple[Tensor, int]:
    """Compute the discontinuity measure of a phase snapshot.

    Measures the second-order finite difference (local curvature) of
    the phase field on a ring:

    .. math::

        D_m = |\\theta_{m-1} - 2\\theta_m + \\theta_{m+1}|

    Oscillators with :math:`D_m > \\delta` (where :math:`\\delta` is
    ``threshold_ratio * 2π``) are marked *incoherent*.  The chimera
    number *η* counts the number of coherent-to-incoherent transitions
    divided by 2.

    Args:
        phase: Phase vector ``(N,)`` (radians).
        threshold_ratio: Fraction of 2π used as incoherence threshold.

    Returns:
        Tuple ``(coherent_mask, eta)`` where ``coherent_mask`` is a bool
        tensor ``(N,)`` (True = coherent) and ``eta`` is the chimera
        number (0 = fully coherent, 1 = single chimera, ≥2 = multi-chimera).

    Example:
        >>> mask, eta = discontinuity_measure(phase)
        >>> print(f"Chimera number η = {eta}")
    """
    phase = phase.detach().float()
    N = phase.numel()
    if N < 3:
        return torch.ones(N, dtype=torch.bool), 0

    prev_p = torch.roll(phase, 1)
    next_p = torch.roll(phase, -1)

    # Second-order finite difference (handle wrapping via atan2-like approach)
    diff = prev_p - 2 * phase + next_p
    # Wrap to [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    D = diff.abs()

    threshold = threshold_ratio * 2 * math.pi
    coherent_mask = D < threshold

    # Count transitions (coherent ↔ incoherent) on the ring
    transitions = (coherent_mask.int() - torch.roll(coherent_mask.int(), 1)).abs().sum()
    eta = int(transitions.item()) // 2
    return coherent_mask, eta


# ---------------------------------------------------------------------------
# Chimera Index (χ) — fraction of incoherent oscillators
# ---------------------------------------------------------------------------


def chimera_index(
    phase: Tensor,
    nbr_idx: Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute the chimera index χ ∈ [0, 1].

    The chimera index is the fraction of oscillators whose local order
    parameter :math:`r_i` falls below a coherence ``threshold``.

    .. math::

        \\chi = \\frac{N_{\\text{incoherent}}}{N}

    where an oscillator is *incoherent* when :math:`r_i < \\text{threshold}`.
    Typical chimera states yield :math:`\\chi \\in [0.3, 0.6]`.

    Args:
        phase: Phase vector ``(N,)`` (radians).
        nbr_idx: Neighbour index tensor ``(N, k)`` (e.g. from ring_topology).
        threshold: Local order parameter cutoff for incoherence.

    Returns:
        Chimera index as a float.

    Example:
        >>> chi = chimera_index(phase, nbr, threshold=0.5)
        >>> print(f"χ = {chi:.3f}")  # ~0.4-0.6 for chimera
    """
    r_local = local_order_parameter(phase, nbr_idx)
    n_incoherent = (r_local < threshold).sum().item()
    return float(n_incoherent) / phase.numel()


# ---------------------------------------------------------------------------
# Temporal Strength of Incoherence (time-averaged SI)
# ---------------------------------------------------------------------------


def strength_of_incoherence_temporal(
    trajectory: Tensor,
    window_size: int = 10,
    discard_transient: int = 0,
) -> float:
    """Time-averaged Strength of Incoherence over a phase trajectory.

    Computes :func:`strength_of_incoherence` for each snapshot in
    ``trajectory`` (after discarding transient frames) and returns the
    temporal mean.

    Args:
        trajectory: Phase trajectory ``(T, N)`` with T snapshots.
        window_size: Spatial smoothing window for SI computation.
        discard_transient: Number of initial snapshots to discard.

    Returns:
        Mean SI value (float).

    Example:
        >>> si_avg = strength_of_incoherence_temporal(traj, window_size=10,
        ...     discard_transient=100)
    """
    traj = trajectory[discard_transient:]
    T = traj.shape[0]
    if T == 0:
        return 0.0

    si_vals = torch.stack(
        [strength_of_incoherence(traj[t], window_size) for t in range(T)]
    )
    return float(si_vals.mean().item())


# =========================================================================
# Y4 Q1.8: Extended Topology & Dynamics Functions
# =========================================================================


def heterogeneous_natural_frequencies(
    N: int,
    distribution: str = "gaussian",
    spread: float = 1.0,
    center: float = 0.0,
    seed: int = 42,
) -> Tensor:
    """Generate heterogeneous natural frequencies for oscillators.

    Args:
        N: Number of oscillators.
        distribution: One of ``"gaussian"``, ``"lorentzian"``, ``"uniform"``.
        spread: Width parameter (sigma for gaussian, gamma for lorentzian,
            half-width for uniform).
        center: Central frequency.
        seed: Random seed.

    Returns:
        Frequency tensor ``(N,)``.
    """
    gen = torch.Generator().manual_seed(seed)

    if distribution == "gaussian":
        omega = torch.randn(N, generator=gen) * spread + center
    elif distribution == "lorentzian":
        # Lorentzian via inverse CDF: omega = gamma * tan(pi*(u - 0.5))
        u = torch.rand(N, generator=gen)
        u = u.clamp(0.01, 0.99)  # avoid tails
        omega = spread * torch.tan(math.pi * (u - 0.5)) + center
        omega = omega.clamp(center - 10 * spread, center + 10 * spread)
    elif distribution == "uniform":
        omega = (torch.rand(N, generator=gen) - 0.5) * 2 * spread + center
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return omega


def conduction_delay_matrix(
    N: int,
    nbr_idx: Tensor,
    delay_type: str = "distance_proportional",
    max_delay: int = 5,
    seed: int = 42,
) -> Tensor:
    """Generate conduction delay matrix for neighbours.

    Args:
        N: Number of oscillators.
        nbr_idx: Neighbour indices ``(N, k)``.
        delay_type: ``"distance_proportional"``, ``"random_uniform"``,
            or ``"constant"``.
        max_delay: Maximum delay in timesteps.
        seed: Random seed.

    Returns:
        Delay matrix ``(N, k)`` in timestep units (integers).
    """
    k = nbr_idx.shape[1]

    if delay_type == "constant":
        return torch.full((N, k), max_delay, dtype=torch.long)

    if delay_type == "random_uniform":
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, max_delay + 1, (N, k), generator=gen)

    if delay_type == "distance_proportional":
        # Distance on ring topology (indices wrap around)
        positions = torch.arange(N, dtype=torch.float32)
        nbr_positions = positions[nbr_idx.clamp(0, N - 1)]  # (N, k)
        src_positions = positions.unsqueeze(1).expand(N, k)
        dist = torch.min(
            (nbr_positions - src_positions).abs(),
            N - (nbr_positions - src_positions).abs(),
        )
        # Normalise to [0, max_delay]
        max_dist = dist.max()
        if max_dist > 0:
            delays = (dist / max_dist * max_delay).long()
        else:
            delays = torch.zeros(N, k, dtype=torch.long)
        return delays

    raise ValueError(f"Unknown delay_type: {delay_type}")


def community_topology(
    N: int,
    n_communities: int = 2,
    k_intra: int = 20,
    k_inter: int = 5,
    seed: int = 42,
) -> tuple[Tensor, list[list[int]]]:
    """Build modular community topology.

    Dense coupling within communities, sparse coupling between.

    Args:
        N: Total oscillators (divided evenly among communities).
        n_communities: Number of communities.
        k_intra: Neighbours within community.
        k_inter: Neighbours across communities.
        seed: Random seed.

    Returns:
        Tuple of (nbr_idx ``(N, k_intra + k_inter)``,
        community_assignments ``list[list[int]]``).
    """
    gen = torch.Generator().manual_seed(seed)
    community_size = N // n_communities
    communities = []
    for c in range(n_communities):
        start = c * community_size
        end = start + community_size if c < n_communities - 1 else N
        communities.append(list(range(start, end)))

    k_total = k_intra + k_inter
    nbr_idx = torch.zeros(N, k_total, dtype=torch.long)

    for c_idx, community in enumerate(communities):
        cs = len(community)
        for i_local, i_global in enumerate(community):
            # Intra-community: ring-like neighbours
            intra_nbrs = []
            for offset in range(1, k_intra + 1):
                j_local = (i_local + offset) % cs
                intra_nbrs.append(community[j_local])
            if len(intra_nbrs) > k_intra:
                intra_nbrs = intra_nbrs[:k_intra]

            # Inter-community: random from other communities
            other_indices = []
            for oc_idx, oc in enumerate(communities):
                if oc_idx != c_idx:
                    other_indices.extend(oc)
            inter_nbrs = []
            if other_indices and k_inter > 0:
                perm = torch.randperm(len(other_indices), generator=gen)
                for p in range(min(k_inter, len(other_indices))):
                    inter_nbrs.append(other_indices[int(perm[p].item())])

            # Pad if needed
            while len(intra_nbrs) < k_intra:
                intra_nbrs.append(community[0])
            while len(inter_nbrs) < k_inter:
                inter_nbrs.append(other_indices[0] if other_indices else 0)

            nbrs = intra_nbrs[:k_intra] + inter_nbrs[:k_inter]
            nbr_idx[i_global, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)

    return nbr_idx, communities


def hierarchical_topology(
    N: int,
    n_groups: int = 4,
    k_intra: int = 15,
    k_inter: int = 5,
    seed: int = 42,
) -> tuple[Tensor, list[list[int]]]:
    """Build 2-level hierarchical topology.

    Groups form the first level; inter-group links form the second.

    Args:
        N: Total oscillators.
        n_groups: Number of groups at level 1.
        k_intra: Intra-group neighbours.
        k_inter: Inter-group neighbours.
        seed: Random seed.

    Returns:
        Tuple of (nbr_idx, group_assignments).
    """
    return community_topology(N, n_groups, k_intra, k_inter, seed)


def evolutionary_coupling_update(
    coupling_weights: Tensor,
    phase: Tensor,
    nbr_idx: Tensor,
    payoff_type: str = "coordination",
    mutation_rate: float = 0.01,
    seed: int = 42,
) -> Tensor:
    """Update coupling weights via replicator dynamics.

    Oscillators with higher local synchronisation are "fitter" and
    their coupling patterns are replicated.

    Args:
        coupling_weights: Current weights ``(N, k)``.
        phase: Current phase ``(N,)``.
        nbr_idx: Neighbour indices ``(N, k)``.
        payoff_type: ``"coordination"``, ``"prisoners_dilemma"``,
            or ``"hawk_dove"``.
        mutation_rate: Random mutation amplitude.
        seed: Random seed.

    Returns:
        Updated coupling weights ``(N, k)``.
    """
    N, k = coupling_weights.shape
    gen = torch.Generator(device=coupling_weights.device)
    gen.manual_seed(seed)

    # Compute local fitness = local synchronisation
    nbr_phase = phase[nbr_idx.clamp(0, N - 1)]  # (N, k)
    phase_diff = nbr_phase - phase.unsqueeze(1)
    cos_diff = torch.cos(phase_diff)

    # Payoff-weighted fitness
    if payoff_type == "coordination":
        fitness = (coupling_weights * cos_diff).sum(dim=1)  # (N,)
    elif payoff_type == "prisoners_dilemma":
        # Defection bonus for low coupling
        fitness = (coupling_weights * cos_diff).sum(dim=1) - 0.5 * coupling_weights.sum(dim=1)
    elif payoff_type == "hawk_dove":
        # Mixed strategy payoff
        fitness = (coupling_weights * cos_diff).sum(dim=1) - 0.3 * coupling_weights.pow(2).sum(dim=1)
    else:
        fitness = (coupling_weights * cos_diff).sum(dim=1)

    # Replicator: high-fitness oscillators increase coupling
    fitness_norm = torch.softmax(fitness, dim=0)  # (N,)
    # Each oscillator adopts weighted average of neighbours' weights
    nbr_fitness = fitness_norm[nbr_idx.clamp(0, N - 1)]  # (N, k)

    # Weighted evolution
    new_weights = coupling_weights + 0.1 * coupling_weights * (nbr_fitness - fitness_norm.unsqueeze(1))

    # Mutation
    noise = torch.randn_like(new_weights) * mutation_rate
    new_weights = (new_weights + noise).clamp(0.0, 10.0)

    return new_weights


def directed_weighted_topology(
    N: int,
    k: int = 20,
    asymmetry: float = 0.0,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Build directed weighted topology with tuneable asymmetry.

    Args:
        N: Number of oscillators.
        k: Number of neighbours per oscillator.
        asymmetry: 0.0 = symmetric (reciprocal), 1.0 = fully directed.
        seed: Random seed.

    Returns:
        Tuple of (nbr_idx ``(N, k)``, weights ``(N, k)``).
    """
    gen = torch.Generator().manual_seed(seed)

    # Start with ring topology
    nbr_idx = torch.zeros(N, k, dtype=torch.long)
    for i in range(N):
        offsets = torch.arange(1, k + 1)
        nbr_idx[i] = (i + offsets) % N

    # Weights: base symmetric + asymmetric component
    base_weights = torch.ones(N, k)
    asym_noise = torch.rand(N, k, generator=gen) * 2 - 1  # [-1, 1]
    weights = base_weights + asymmetry * asym_noise
    weights = weights.clamp(0.01, 5.0)

    return nbr_idx, weights
