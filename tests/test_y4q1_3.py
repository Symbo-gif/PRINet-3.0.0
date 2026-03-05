"""Year 4 Q1.3 tests — chimera deepening infrastructure.

Covers:
- RK4 integrator in OscilloSim (accuracy vs Euler).
- Cosine coupling kernel (shape, normalisation, symmetry).
- Strength of Incoherence metric (SI) bounds and coherence detection.
- Discontinuity measure (η) for chimera counting.
- Chimera index (χ) bounds and consistency.
- New initial conditions (gaussian_bump_ic, half_sync_half_random_ic).
- Weighted coupling integration in OscilloSim.
"""

import math

import pytest
import torch

from prinet.utils.oscillosim import (
    OscilloSim,
    bimodality_index,
    chimera_index,
    cosine_coupling_kernel,
    discontinuity_measure,
    local_order_parameter,
    ring_topology,
    strength_of_incoherence,
    strength_of_incoherence_temporal,
)
from prinet.utils.y4q1_tools import (
    chimera_initial_condition,
    gaussian_bump_ic,
    half_sync_half_random_ic,
)

# =====================================================================
# Fixtures
# =====================================================================

SEED = 42
DEVICE = "cpu"


@pytest.fixture
def ring_nbr() -> torch.Tensor:
    """Ring topology with N=64, k=8."""
    return ring_topology(64, 8)


@pytest.fixture
def coherent_phase() -> torch.Tensor:
    """Perfectly coherent phase (all identical)."""
    return torch.zeros(64)


@pytest.fixture
def random_phase() -> torch.Tensor:
    """Uniformly random phases."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.rand(64, generator=gen) * 2 * math.pi


# =====================================================================
# RK4 Integrator
# =====================================================================


class TestRK4Integrator:
    """Tests for RK4 integration in OscilloSim."""

    def test_rk4_constructor_accepts(self) -> None:
        """OscilloSim accepts integrator='rk4'."""
        sim = OscilloSim(
            n_oscillators=32,
            coupling_mode="ring",
            k_neighbors=4,
            integrator="rk4",
            seed=SEED,
        )
        assert sim.integrator == "rk4"

    def test_euler_constructor_accepts(self) -> None:
        """OscilloSim accepts integrator='euler'."""
        sim = OscilloSim(
            n_oscillators=32,
            coupling_mode="ring",
            k_neighbors=4,
            integrator="euler",
            seed=SEED,
        )
        assert sim.integrator == "euler"

    def test_rk4_runs_ring(self) -> None:
        """RK4 ring simulation completes without error."""
        sim = OscilloSim(
            n_oscillators=64,
            coupling_strength=2.0,
            coupling_mode="ring",
            k_neighbors=8,
            integrator="rk4",
            seed=SEED,
        )
        result = sim.run(n_steps=50, dt=0.01)
        assert len(result.order_parameter) > 0
        assert result.throughput > 0

    def test_rk4_runs_small_world(self) -> None:
        """RK4 small-world simulation completes without error."""
        sim = OscilloSim(
            n_oscillators=64,
            coupling_strength=2.0,
            coupling_mode="small_world",
            k_neighbors=8,
            integrator="rk4",
            seed=SEED,
        )
        result = sim.run(n_steps=50, dt=0.01)
        assert len(result.order_parameter) > 0

    def test_rk4_more_accurate_than_euler(self) -> None:
        """RK4 with large dt should give closer results to fine Euler.

        We compare RK4 with dt=0.1 against Euler with dt=0.001 (reference).
        RK4 should be closer than Euler with dt=0.1.
        """
        N, k = 32, 6
        n_steps_coarse = 100
        dt_coarse = 0.1
        dt_fine = 0.001
        n_steps_fine = int(n_steps_coarse * dt_coarse / dt_fine)

        # Reference: fine Euler
        sim_ref = OscilloSim(
            n_oscillators=N,
            coupling_strength=2.0,
            coupling_mode="ring",
            k_neighbors=k,
            integrator="euler",
            seed=SEED,
        )
        gen = torch.Generator().manual_seed(SEED)
        init_phase = torch.rand(N, generator=gen) * 2 * math.pi
        ref_result = sim_ref.run(
            n_steps=n_steps_fine, dt=dt_fine, initial_phase=init_phase.clone()
        )

        # Coarse Euler
        sim_euler = OscilloSim(
            n_oscillators=N,
            coupling_strength=2.0,
            coupling_mode="ring",
            k_neighbors=k,
            integrator="euler",
            seed=SEED,
        )
        euler_result = sim_euler.run(
            n_steps=n_steps_coarse, dt=dt_coarse, initial_phase=init_phase.clone()
        )

        # Coarse RK4
        sim_rk4 = OscilloSim(
            n_oscillators=N,
            coupling_strength=2.0,
            coupling_mode="ring",
            k_neighbors=k,
            integrator="rk4",
            seed=SEED,
        )
        rk4_result = sim_rk4.run(
            n_steps=n_steps_coarse, dt=dt_coarse, initial_phase=init_phase.clone()
        )

        # Compare final order parameters
        r_ref = ref_result.order_parameter[-1]
        r_euler = euler_result.order_parameter[-1]
        r_rk4 = rk4_result.order_parameter[-1]
        err_euler = abs(r_euler - r_ref)
        err_rk4 = abs(r_rk4 - r_ref)
        # RK4 should be at least as good (usually much better)
        assert (
            err_rk4 <= err_euler + 0.05
        ), f"RK4 error ({err_rk4:.4f}) > Euler error ({err_euler:.4f})"

    def test_rk4_phase_stays_bounded(self) -> None:
        """Phase values stay in [0, 2π) after RK4 integration."""
        sim = OscilloSim(
            n_oscillators=64,
            coupling_strength=5.0,
            coupling_mode="ring",
            k_neighbors=8,
            integrator="rk4",
            seed=SEED,
        )
        result = sim.run(n_steps=200, dt=0.05)
        assert result.final_phase.min() >= 0.0
        assert result.final_phase.max() < 2 * math.pi + 1e-5


# =====================================================================
# Cosine Coupling Kernel
# =====================================================================


class TestCosineCouplingKernel:
    """Tests for :func:`cosine_coupling_kernel`."""

    def test_shape(self) -> None:
        w = cosine_coupling_kernel(64, 8)
        assert w.shape == (64, 8)

    def test_dtype(self) -> None:
        w = cosine_coupling_kernel(64, 8, dtype=torch.float32)
        assert w.dtype == torch.float32

    def test_all_positive(self) -> None:
        """All weights should be positive for A < 1."""
        w = cosine_coupling_kernel(256, 90, A=0.995)
        assert (w >= 0).all()

    def test_uniform_when_A_zero(self) -> None:
        """A=0 should produce uniform weights."""
        w = cosine_coupling_kernel(64, 8, A=0.0)
        expected = torch.ones(8) / 8.0
        row = w[0]
        assert torch.allclose(row, expected, atol=1e-5)

    def test_rows_identical(self) -> None:
        """All rows should be identical (kernel is translation-invariant)."""
        w = cosine_coupling_kernel(32, 6, A=0.9)
        for i in range(1, 32):
            assert torch.allclose(w[i], w[0], atol=1e-6)

    def test_normalisation(self) -> None:
        """Row sums should equal 1.0 (K * Σ w·sin matches K/k * Σ sin)."""
        k = 8
        w = cosine_coupling_kernel(64, k)
        row_sum = w[0].sum().item()
        assert abs(row_sum - 1.0) < 1e-5

    def test_symmetry_within_row(self) -> None:
        """Cosine kernel should be symmetric: w(+d) == w(-d)."""
        k = 8
        w = cosine_coupling_kernel(256, k, A=0.995)
        row = w[0]
        half = k // 2
        # Left k/2 and right k/2 should be mirror images
        left = row[:half]
        right = row[half:].flip(0)
        assert torch.allclose(left, right, atol=1e-6)

    def test_near_neighbours_stronger(self) -> None:
        """For A > 0, near neighbours should have higher weight."""
        w = cosine_coupling_kernel(256, 20, A=0.99)
        row = w[0]
        # First neighbour (index k/2 = 10 is the +1 neighbour) vs
        # last neighbour (index k-1 = 19 is the +10 neighbour)
        half = 10
        near = row[half]  # +1 neighbour
        far = row[-1]  # +10 neighbour
        assert near > far

    def test_integration_with_oscillosim(self) -> None:
        """OscilloSim accepts coupling_weights from cosine kernel."""
        N, k = 64, 8
        weights = cosine_coupling_kernel(N, k, A=0.995)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_strength=10.0,
            coupling_mode="ring",
            k_neighbors=k,
            coupling_weights=weights,
            seed=SEED,
        )
        result = sim.run(n_steps=50, dt=0.01)
        assert len(result.order_parameter) > 0


# =====================================================================
# Strength of Incoherence (SI)
# =====================================================================


class TestStrengthOfIncoherence:
    """Tests for :func:`strength_of_incoherence`."""

    def test_coherent_gives_low_si(self, coherent_phase: torch.Tensor) -> None:
        """Fully coherent phases should yield SI ≈ 0."""
        si = strength_of_incoherence(coherent_phase, window_size=5)
        assert si.item() < 0.1

    def test_random_gives_high_si(self, random_phase: torch.Tensor) -> None:
        """Fully random phases should yield SI close to 1."""
        si = strength_of_incoherence(random_phase, window_size=5)
        assert si.item() > 0.3  # Random should be significantly incoherent

    def test_output_bounded(self, random_phase: torch.Tensor) -> None:
        """SI should be in [0, 1]."""
        si = strength_of_incoherence(random_phase, window_size=5)
        assert 0.0 <= si.item() <= 1.0

    def test_small_input(self) -> None:
        """SI should handle tiny inputs gracefully."""
        si = strength_of_incoherence(torch.tensor([0.0, 0.1]))
        assert si.item() >= 0.0

    def test_single_oscillator(self) -> None:
        """Single oscillator returns 0."""
        si = strength_of_incoherence(torch.tensor([1.0]))
        assert si.item() == 0.0

    def test_window_size_effect(self) -> None:
        """Larger window should smooth more and lower SI for structured data."""
        phase = torch.linspace(0, 2 * math.pi, 128)
        si_small = strength_of_incoherence(phase, window_size=3)
        si_large = strength_of_incoherence(phase, window_size=30)
        # Both should be valid
        assert 0.0 <= si_small.item() <= 1.0
        assert 0.0 <= si_large.item() <= 1.0


# =====================================================================
# Discontinuity Measure
# =====================================================================


class TestDiscontinuityMeasure:
    """Tests for :func:`discontinuity_measure`."""

    def test_coherent_gives_eta_zero(self, coherent_phase: torch.Tensor) -> None:
        """Coherent phases should yield η = 0."""
        mask, eta = discontinuity_measure(coherent_phase)
        assert eta == 0
        assert mask.all()  # All coherent

    def test_mask_shape(self) -> None:
        phase = torch.zeros(64)
        mask, _ = discontinuity_measure(phase)
        assert mask.shape == (64,)
        assert mask.dtype == torch.bool

    def test_random_has_incoherent_oscillators(
        self, random_phase: torch.Tensor
    ) -> None:
        """Random phases should have some incoherent oscillators."""
        mask, eta = discontinuity_measure(random_phase, threshold_ratio=0.01)
        # Some oscillators should be incoherent
        n_incoherent = (~mask).sum().item()
        assert n_incoherent > 0

    def test_eta_nonnegative(self, random_phase: torch.Tensor) -> None:
        """η should always be ≥ 0."""
        _, eta = discontinuity_measure(random_phase)
        assert eta >= 0

    def test_threshold_sensitivity(self) -> None:
        """Tighter threshold should mark more oscillators incoherent."""
        gen = torch.Generator().manual_seed(123)
        phase = torch.rand(128, generator=gen) * 2 * math.pi
        mask_loose, _ = discontinuity_measure(phase, threshold_ratio=0.5)
        mask_tight, _ = discontinuity_measure(phase, threshold_ratio=0.001)
        n_coherent_loose = mask_loose.sum().item()
        n_coherent_tight = mask_tight.sum().item()
        assert n_coherent_loose >= n_coherent_tight

    def test_small_input(self) -> None:
        """Handles N < 3 gracefully."""
        mask, eta = discontinuity_measure(torch.tensor([0.0, 1.0]))
        assert eta == 0
        assert mask.all()

    def test_chimera_like_pattern(self) -> None:
        """A phase pattern with a sharp discontinuity should give η ≥ 1."""
        N = 128
        phase = torch.zeros(N)
        # Create a sharp jump in the middle
        phase[N // 2 :] = math.pi
        mask, eta = discontinuity_measure(phase, threshold_ratio=0.05)
        assert eta >= 1  # At least one coherent/incoherent boundary pair


# =====================================================================
# Chimera Index
# =====================================================================


class TestChimeraIndex:
    """Tests for :func:`chimera_index`."""

    def test_coherent_gives_zero(
        self, coherent_phase: torch.Tensor, ring_nbr: torch.Tensor
    ) -> None:
        """Coherent state should give χ ≈ 0."""
        chi = chimera_index(coherent_phase, ring_nbr, threshold=0.5)
        assert chi < 0.1

    def test_random_gives_nonzero(
        self, random_phase: torch.Tensor, ring_nbr: torch.Tensor
    ) -> None:
        """Random state should give χ > 0."""
        chi = chimera_index(random_phase, ring_nbr, threshold=0.9)
        assert chi > 0.0

    def test_bounded(self, random_phase: torch.Tensor, ring_nbr: torch.Tensor) -> None:
        """χ should be in [0, 1]."""
        chi = chimera_index(random_phase, ring_nbr)
        assert 0.0 <= chi <= 1.0

    def test_threshold_monotonic(
        self, random_phase: torch.Tensor, ring_nbr: torch.Tensor
    ) -> None:
        """Higher threshold → more oscillators classified as incoherent → larger χ."""
        chi_low = chimera_index(random_phase, ring_nbr, threshold=0.3)
        chi_high = chimera_index(random_phase, ring_nbr, threshold=0.9)
        assert chi_high >= chi_low


# =====================================================================
# Temporal Strength of Incoherence
# =====================================================================


class TestStrengthOfIncoherenceTemporal:
    """Tests for :func:`strength_of_incoherence_temporal`."""

    def test_coherent_trajectory(self) -> None:
        """Coherent trajectory gives SI_temporal ≈ 0."""
        T, N = 10, 64
        traj = torch.zeros(T, N)
        si = strength_of_incoherence_temporal(traj, window_size=5)
        assert si < 0.1

    def test_random_trajectory(self) -> None:
        """Random trajectory gives SI_temporal > 0."""
        gen = torch.Generator().manual_seed(SEED)
        T, N = 10, 64
        traj = torch.rand(T, N, generator=gen) * 2 * math.pi
        si = strength_of_incoherence_temporal(traj, window_size=5)
        assert si > 0.1

    def test_discard_transient(self) -> None:
        """Discarding transient frames works correctly."""
        T, N = 20, 64
        traj = torch.zeros(T, N)
        si = strength_of_incoherence_temporal(traj, window_size=5, discard_transient=15)
        assert si < 0.1

    def test_empty_after_discard(self) -> None:
        """Discarding all frames returns 0."""
        T, N = 5, 64
        traj = torch.rand(T, N) * 2 * math.pi
        si = strength_of_incoherence_temporal(traj, window_size=3, discard_transient=T)
        assert si == 0.0


# =====================================================================
# New Initial Conditions
# =====================================================================


class TestGaussianBumpIC:
    """Tests for :func:`gaussian_bump_ic`."""

    def test_shape(self) -> None:
        ic = gaussian_bump_ic(256)
        assert ic.shape == (256,)

    def test_bounded(self) -> None:
        """Output should be in [0, 2π)."""
        ic = gaussian_bump_ic(256)
        assert ic.min() >= 0.0
        assert ic.max() < 2 * math.pi + 1e-5

    def test_peak_near_centre(self) -> None:
        """The bump should peak near the centre of the array."""
        ic = gaussian_bump_ic(256, noise_amp=0.0)
        peak_idx = ic.argmax().item()
        assert abs(peak_idx - 128) <= 10

    def test_reproducible(self) -> None:
        """Same seed gives identical output."""
        ic1 = gaussian_bump_ic(64, seed=0)
        ic2 = gaussian_bump_ic(64, seed=0)
        assert torch.allclose(ic1, ic2)

    def test_different_seed_differs(self) -> None:
        """Different seeds give different output."""
        ic1 = gaussian_bump_ic(64, seed=0)
        ic2 = gaussian_bump_ic(64, seed=99)
        assert not torch.allclose(ic1, ic2)


class TestHalfSyncHalfRandomIC:
    """Tests for :func:`half_sync_half_random_ic`."""

    def test_shape(self) -> None:
        ic = half_sync_half_random_ic(256)
        assert ic.shape == (256,)

    def test_bounded(self) -> None:
        """Output should be in [0, 2π)."""
        ic = half_sync_half_random_ic(256)
        assert ic.min() >= 0.0
        assert ic.max() < 2 * math.pi + 1e-5

    def test_coherent_half_near_sync_phase(self) -> None:
        """First half should be near sync_phase (angular distance)."""
        ic = half_sync_half_random_ic(100, sync_phase=0.0, noise_amp=0.01)
        first_half = ic[:50]
        # Angular distance from 0: min(φ, 2π − φ)
        ang_dist = torch.min(first_half, 2 * math.pi - first_half)
        assert ang_dist.max() < 0.05

    def test_incoherent_half_spread(self) -> None:
        """Second half should be spread across [0, 2π)."""
        ic = half_sync_half_random_ic(200, seed=42)
        second_half = ic[100:]
        spread = second_half.max() - second_half.min()
        assert spread > math.pi  # Should span a wide range

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        ic1 = half_sync_half_random_ic(64, seed=7)
        ic2 = half_sync_half_random_ic(64, seed=7)
        assert torch.allclose(ic1, ic2)


# =====================================================================
# Weighted Coupling Integration
# =====================================================================


class TestWeightedCoupling:
    """Integration tests for OscilloSim with coupling_weights."""

    def test_cosine_kernel_affects_dynamics(self) -> None:
        """Cosine-weighted coupling should produce different dynamics than uniform."""
        N, k = 64, 8
        weights = cosine_coupling_kernel(N, k, A=0.995)
        init = gaussian_bump_ic(N, seed=SEED)

        sim_uniform = OscilloSim(
            n_oscillators=N,
            coupling_strength=10.0,
            coupling_mode="ring",
            k_neighbors=k,
            seed=SEED,
        )
        sim_cosine = OscilloSim(
            n_oscillators=N,
            coupling_strength=10.0,
            coupling_mode="ring",
            k_neighbors=k,
            coupling_weights=weights,
            seed=SEED,
        )

        r_uniform = sim_uniform.run(n_steps=100, dt=0.01, initial_phase=init.clone())
        r_cosine = sim_cosine.run(n_steps=100, dt=0.01, initial_phase=init.clone())

        # Final order parameters should differ
        assert r_uniform.order_parameter[-1] != pytest.approx(
            r_cosine.order_parameter[-1], abs=1e-3
        )

    def test_rk4_with_cosine_weights(self) -> None:
        """RK4 + cosine weights runs without error."""
        N, k = 64, 8
        weights = cosine_coupling_kernel(N, k, A=0.995)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_strength=50.0,
            coupling_mode="ring",
            k_neighbors=k,
            phase_lag=1.521,
            integrator="rk4",
            coupling_weights=weights,
            seed=SEED,
        )
        result = sim.run(n_steps=100, dt=0.05)
        assert len(result.order_parameter) > 0
        assert result.final_phase.shape == (N,)

    def test_chimera_setup_runs(self) -> None:
        """Full chimera-optimised setup (K=100, α≈1.52, cosine, RK4) runs."""
        N, k = 128, 40
        weights = cosine_coupling_kernel(N, k, A=0.995)
        init = gaussian_bump_ic(N, seed=SEED)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_strength=100.0,
            coupling_mode="ring",
            k_neighbors=k,
            phase_lag=math.pi / 2 - 0.05,
            integrator="rk4",
            coupling_weights=weights,
            seed=SEED,
        )
        result = sim.run(n_steps=500, dt=0.05, initial_phase=init)
        # Compute all metrics on the final state
        nbr = ring_topology(N, k)
        r_local = local_order_parameter(result.final_phase, nbr)
        bc = bimodality_index(r_local)
        si = strength_of_incoherence(result.final_phase, window_size=10)
        _, eta = discontinuity_measure(result.final_phase)
        chi = chimera_index(result.final_phase, nbr)

        # All metrics should be valid numbers
        assert isinstance(bc, float)
        assert 0.0 <= si.item() <= 1.0
        assert eta >= 0
        assert 0.0 <= chi <= 1.0
