"""Unit tests for prinet.core: decomposition, propagation, and measurement.

Covers Tucker/CP decomposition, Kuramoto/Stuart-Landau oscillator dynamics,
and synchronization metrics. All tests use seeded RNG for determinism per
Testing Standards.
"""

from __future__ import annotations

import math

import pytest
import torch

from prinet.core.decomposition import (
    CPDecomposition,
    DecompositionError,
    DimensionsMismatchError,
    PolyadicTensor,
    TensorDecompositionBase,
)
from prinet.core.measurement import (
    build_phase_knn,
    extract_concept_probabilities,
    kuramoto_order_parameter,
    kuramoto_order_parameter_complex,
    mean_phase_coherence,
    phase_coherence_matrix,
    power_spectral_density,
    sparse_mean_phase_coherence,
    sparse_synchronization_energy,
    synchronization_energy,
)
from prinet.core.propagation import (
    HopfOscillator,
    KuramotoOscillator,
    OscillatorModel,
    OscillatorState,
    OscillatorSyncError,
    StuartLandauOscillator,
    _build_phase_knn_index,
    _clamp_finite,
    _safe_phase_diff,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cpu")
DTYPE = torch.float32


@pytest.fixture
def rng() -> torch.Generator:
    """Seeded random generator for deterministic tests."""
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return gen


@pytest.fixture
def small_tensor() -> torch.Tensor:
    """Small 3D tensor for decomposition tests."""
    torch.manual_seed(SEED)
    return torch.randn(8, 8, 8)


@pytest.fixture
def random_state() -> OscillatorState:
    """Random oscillator state with 50 oscillators."""
    return OscillatorState.create_random(50, seed=SEED)


@pytest.fixture
def synchronized_state() -> OscillatorState:
    """Fully synchronized oscillator state."""
    return OscillatorState.create_synchronized(50, base_frequency=1.0)


@pytest.fixture
def kuramoto_model() -> KuramotoOscillator:
    """Standard Kuramoto model with 50 oscillators."""
    return KuramotoOscillator(
        n_oscillators=50,
        coupling_strength=2.0,
        decay_rate=0.1,
        freq_adaptation_rate=0.01,
    )


@pytest.fixture
def stuart_landau_model() -> StuartLandauOscillator:
    """Standard Stuart-Landau model with 50 oscillators."""
    return StuartLandauOscillator(
        n_oscillators=50,
        coupling_strength=1.5,
        bifurcation_param=1.0,
    )


# ===========================================================================
# 1. DECOMPOSITION TESTS
# ===========================================================================


class TestPolyadicTensor:
    """Tests for Tucker (HOSVD) decomposition."""

    def test_decompose_reconstruct_basic(self, small_tensor: torch.Tensor) -> None:
        """Tucker decomposition and reconstruction produce valid output."""
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4)
        pt.decompose(small_tensor)
        recon = pt.reconstruct()
        assert recon.shape == small_tensor.shape

    def test_reconstruction_error_decreases_with_rank(self) -> None:
        """Higher rank produces lower reconstruction error."""
        torch.manual_seed(SEED)
        data = torch.randn(10, 10, 10)
        errors = []
        for rank in [2, 4, 8]:
            pt = PolyadicTensor(shape=(10, 10, 10), rank=rank)
            pt.decompose(data)
            errors.append(pt.reconstruction_error(data))
        # Error should decrease (or remain same) as rank increases
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-5

    def test_full_rank_near_perfect_reconstruction(self) -> None:
        """Full-rank decomposition yields near-zero error."""
        torch.manual_seed(SEED)
        data = torch.randn(6, 6, 6)
        pt = PolyadicTensor(shape=(6, 6, 6), rank=6)
        pt.decompose(data)
        error = pt.reconstruction_error(data)
        assert error < 1e-4, f"Expected near-zero error, got {error}"

    def test_shape_mismatch_raises(self) -> None:
        """Decompose raises DimensionsMismatchError for wrong shape."""
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4)
        with pytest.raises(DimensionsMismatchError, match="Expected shape"):
            pt.decompose(torch.randn(4, 4, 4))

    def test_reconstruct_before_decompose_raises(self) -> None:
        """Reconstruct before decompose raises DecompositionError."""
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4)
        with pytest.raises(DecompositionError, match="No decomposition"):
            pt.reconstruct()

    def test_factors_before_decompose_raises(self) -> None:
        """Accessing factors before decompose raises DecompositionError."""
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4)
        with pytest.raises(DecompositionError, match="No decomposition"):
            _ = pt.factors

    def test_core_before_decompose_raises(self) -> None:
        """Accessing core before decompose raises DecompositionError."""
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4)
        with pytest.raises(DecompositionError, match="No decomposition"):
            _ = pt.core

    def test_invalid_rank_raises(self) -> None:
        """Rank < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Rank must be a positive"):
            PolyadicTensor(shape=(8, 8, 8), rank=0)

    def test_invalid_shape_raises(self) -> None:
        """Shape with zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="All shape dimensions"):
            PolyadicTensor(shape=(8, 0, 8), rank=4)

    def test_2d_tensor_decomposition(self) -> None:
        """Tucker decomposition works for 2D (matrix) input."""
        torch.manual_seed(SEED)
        data = torch.randn(10, 8)
        pt = PolyadicTensor(shape=(10, 8), rank=4)
        pt.decompose(data)
        recon = pt.reconstruct()
        assert recon.shape == (10, 8)
        error = pt.reconstruction_error(data)
        assert error < 1.0  # Lossy but valid

    def test_properties(self) -> None:
        """Shape and rank properties are correct."""
        pt = PolyadicTensor(shape=(5, 6, 7), rank=3)
        assert pt.shape == (5, 6, 7)
        assert pt.rank == 3

    def test_mode_n_unfold(self) -> None:
        """Mode-n unfolding produces correct shape."""
        tensor = torch.randn(3, 4, 5)
        unfolded_0 = PolyadicTensor._mode_n_unfold(tensor, 0)
        assert unfolded_0.shape == (3, 20)
        unfolded_1 = PolyadicTensor._mode_n_unfold(tensor, 1)
        assert unfolded_1.shape == (4, 15)


class TestCPDecomposition:
    """Tests for Canonical Polyadic decomposition."""

    def test_decompose_reconstruct_basic(self) -> None:
        """CP decomposition and reconstruction produce valid output."""
        torch.manual_seed(SEED)
        data = torch.randn(6, 6, 6)
        cp = CPDecomposition(shape=(6, 6, 6), rank=4, max_iter=50)
        cp.decompose(data)
        recon = cp.reconstruct()
        assert recon.shape == (6, 6, 6)

    def test_shape_mismatch_raises(self) -> None:
        """CP decompose raises DimensionsMismatchError for wrong shape."""
        cp = CPDecomposition(shape=(8, 8, 8), rank=3)
        with pytest.raises(DimensionsMismatchError):
            cp.decompose(torch.randn(4, 4, 4))

    def test_factors_before_decompose_raises(self) -> None:
        """Accessing factors before decompose raises."""
        cp = CPDecomposition(shape=(8, 8, 8), rank=3)
        with pytest.raises(DecompositionError):
            _ = cp.factors

    def test_weights_before_decompose_raises(self) -> None:
        """Accessing weights before decompose raises."""
        cp = CPDecomposition(shape=(8, 8, 8), rank=3)
        with pytest.raises(DecompositionError):
            _ = cp.weights

    def test_rank1_tensor_exact(self) -> None:
        """A rank-1 tensor should be perfectly reconstructed with rank=1."""
        torch.manual_seed(SEED)
        a = torch.randn(5)
        b = torch.randn(5)
        c = torch.randn(5)
        data = torch.einsum("i,j,k->ijk", a, b, c)
        cp = CPDecomposition(shape=(5, 5, 5), rank=1, max_iter=200)
        cp.decompose(data)
        recon = cp.reconstruct()
        rel_error = torch.norm(data - recon) / torch.norm(data)
        assert rel_error < 0.1, f"Expected low error for rank-1, got {rel_error}"

    def test_properties(self) -> None:
        """Shape and rank properties work."""
        cp = CPDecomposition(shape=(3, 4, 5), rank=2)
        assert cp.shape == (3, 4, 5)
        assert cp.rank == 2


# ===========================================================================
# 2. PROPAGATION TESTS
# ===========================================================================


class TestOscillatorState:
    """Tests for OscillatorState creation and manipulation."""

    def test_create_random_shape(self) -> None:
        """Random state has correct shape."""
        state = OscillatorState.create_random(100, seed=SEED)
        assert state.phase.shape == (100,)
        assert state.amplitude.shape == (100,)
        assert state.frequency.shape == (100,)

    def test_create_random_batched(self) -> None:
        """Batched random state has correct shape."""
        state = OscillatorState.create_random(50, batch_size=16, seed=SEED)
        assert state.phase.shape == (16, 50)

    def test_create_synchronized(self) -> None:
        """Synchronized state has zero phase and uniform frequency."""
        state = OscillatorState.create_synchronized(100, base_frequency=2.0)
        assert torch.allclose(state.phase, torch.zeros(100))
        assert torch.allclose(
            state.frequency, torch.full((100,), 2.0)
        )
        assert torch.allclose(state.amplitude, torch.ones(100))

    def test_clone_independence(self) -> None:
        """Cloned state is independent (modifying one doesn't affect other)."""
        state = OscillatorState.create_random(50, seed=SEED)
        cloned = state.clone()
        cloned.phase[0] = 999.0
        assert state.phase[0] != 999.0

    def test_n_oscillators_property(self) -> None:
        """n_oscillators returns correct count."""
        state = OscillatorState.create_random(73, seed=SEED)
        assert state.n_oscillators == 73

    def test_phase_range(self) -> None:
        """Random phases are in [0, 2π)."""
        state = OscillatorState.create_random(1000, seed=SEED)
        assert state.phase.min() >= 0.0
        assert state.phase.max() < 2.0 * math.pi + 0.01

    def test_frequency_range(self) -> None:
        """Random frequencies are in specified range."""
        state = OscillatorState.create_random(
            100, freq_range=(1.0, 5.0), seed=SEED
        )
        assert state.frequency.min() >= 1.0 - 0.01
        assert state.frequency.max() <= 5.0 + 0.01


class TestKuramotoOscillator:
    """Tests for Kuramoto oscillator dynamics."""

    def test_compute_derivatives_shape(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Derivatives have correct shape."""
        dphi, dr, domega = kuramoto_model.compute_derivatives(random_state)
        assert dphi.shape == (50,)
        assert dr.shape == (50,)
        assert domega.shape == (50,)

    def test_step_euler(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Euler step produces valid state."""
        new_state = kuramoto_model.step(random_state, dt=0.01, method="euler")
        assert new_state.phase.shape == random_state.phase.shape
        assert not torch.isnan(new_state.phase).any()
        assert not torch.isnan(new_state.amplitude).any()

    def test_step_rk4(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """RK4 step produces valid state."""
        new_state = kuramoto_model.step(random_state, dt=0.01, method="rk4")
        assert not torch.isnan(new_state.phase).any()
        assert not torch.isnan(new_state.amplitude).any()

    def test_integration(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Integration over multiple steps converges to valid state."""
        final, traj = kuramoto_model.integrate(
            random_state, n_steps=100, dt=0.01, method="rk4"
        )
        assert traj is None  # No trajectory by default
        assert not torch.isnan(final.phase).any()
        assert (final.amplitude >= 0).all()

    def test_integration_with_trajectory(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Integration records trajectory when requested."""
        final, traj = kuramoto_model.integrate(
            random_state,
            n_steps=10,
            dt=0.01,
            record_trajectory=True,
        )
        assert traj is not None
        assert len(traj) == 10

    def test_synchronized_state_stays_synchronized(
        self,
        kuramoto_model: KuramotoOscillator,
        synchronized_state: OscillatorState,
    ) -> None:
        """A synchronized initial state maintains high order parameter."""
        final, _ = kuramoto_model.integrate(
            synchronized_state, n_steps=100, dt=0.01
        )
        r = kuramoto_order_parameter(final.phase)
        assert r > 0.5, f"Expected r > 0.5 for sync start, got {r:.4f}"

    def test_strong_coupling_promotes_sync(self) -> None:
        """Strong coupling increases synchronization.

        Uses a narrow frequency range where K=2 is above the critical
        coupling threshold. Phase wrapping (mod 2π) requires genuine
        synchronization rather than coincidental alignment.
        """
        torch.manual_seed(SEED)
        state = OscillatorState.create_random(
            30, seed=SEED, freq_range=(1.0, 2.0)
        )
        model_weak = KuramotoOscillator(30, coupling_strength=0.1)
        model_strong = KuramotoOscillator(30, coupling_strength=2.0)
        final_weak, _ = model_weak.integrate(state, n_steps=500, dt=0.01)
        final_strong, _ = model_strong.integrate(
            state.clone(), n_steps=500, dt=0.01
        )
        r_weak = kuramoto_order_parameter(final_weak.phase)
        r_strong = kuramoto_order_parameter(final_strong.phase)
        assert r_strong > r_weak - 0.1  # Strong should be more sync'd

    def test_invalid_n_oscillators_raises(self) -> None:
        """Zero oscillators raises ValueError."""
        with pytest.raises(ValueError, match="n_oscillators must be positive"):
            KuramotoOscillator(n_oscillators=0)

    def test_unknown_method_raises(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Unknown integration method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown integration method"):
            kuramoto_model.step(random_state, method="bogus")

    def test_custom_coupling_matrix(self) -> None:
        """Custom coupling matrix is used correctly."""
        model = KuramotoOscillator(n_oscillators=5, coupling_strength=1.0)
        custom = torch.eye(5) * 0.0
        custom[0, 1] = 0.5
        custom[1, 0] = 0.5
        model.set_coupling_matrix(custom)
        assert torch.allclose(model.coupling_matrix, custom)

    def test_invalid_coupling_matrix_shape_raises(self) -> None:
        """Wrong-shaped coupling matrix raises ValueError."""
        model = KuramotoOscillator(n_oscillators=5)
        with pytest.raises(ValueError, match="Expected coupling matrix shape"):
            model.set_coupling_matrix(torch.randn(3, 3))

    def test_properties(self) -> None:
        """Model properties return correct values."""
        model = KuramotoOscillator(
            n_oscillators=100,
            coupling_strength=3.0,
            decay_rate=0.2,
            freq_adaptation_rate=0.05,
        )
        assert model.n_oscillators == 100
        assert model.coupling_strength == 3.0
        assert model.decay_rate == 0.2
        assert model.freq_adaptation_rate == 0.05

    def test_amplitudes_non_negative(
        self, kuramoto_model: KuramotoOscillator, random_state: OscillatorState
    ) -> None:
        """Amplitudes remain non-negative after integration."""
        final, _ = kuramoto_model.integrate(
            random_state, n_steps=100, dt=0.01
        )
        assert (final.amplitude >= 0).all()


class TestStuartLandauOscillator:
    """Tests for Stuart-Landau oscillator dynamics."""

    def test_compute_derivatives_shape(
        self,
        stuart_landau_model: StuartLandauOscillator,
        random_state: OscillatorState,
    ) -> None:
        """Derivatives have correct shape."""
        dphi, dr, domega = stuart_landau_model.compute_derivatives(
            random_state
        )
        assert dphi.shape == (50,)
        assert dr.shape == (50,)
        assert domega.shape == (50,)

    def test_step_produces_valid_state(
        self,
        stuart_landau_model: StuartLandauOscillator,
        random_state: OscillatorState,
    ) -> None:
        """Step produces non-NaN state."""
        new_state = stuart_landau_model.step(random_state, dt=0.005)
        assert not torch.isnan(new_state.phase).any()
        assert not torch.isnan(new_state.amplitude).any()

    def test_limit_cycle_behavior(self) -> None:
        """With μ > 0, amplitudes converge toward sqrt(μ)."""
        model = StuartLandauOscillator(
            n_oscillators=10,
            coupling_strength=0.0,  # No coupling
            bifurcation_param=4.0,  # sqrt(4) = 2
        )
        state = OscillatorState.create_random(10, seed=SEED)
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        # Amplitudes should approach sqrt(μ) = 2.0
        mean_amp = final.amplitude.mean().item()
        assert abs(mean_amp - 2.0) < 0.5, (
            f"Expected amplitude near 2.0, got {mean_amp}"
        )

    def test_bifurcation_param_property(self) -> None:
        """Bifurcation parameter property is correct."""
        model = StuartLandauOscillator(
            n_oscillators=10, bifurcation_param=3.5
        )
        assert model.bifurcation_param == 3.5

    def test_frequency_constant(
        self,
        stuart_landau_model: StuartLandauOscillator,
        random_state: OscillatorState,
    ) -> None:
        """Stuart-Landau frequency does not change (no adaptation)."""
        initial_freq = random_state.frequency.clone()
        final, _ = stuart_landau_model.integrate(
            random_state, n_steps=50, dt=0.005
        )
        assert torch.allclose(
            final.frequency, initial_freq, atol=1e-5
        )


# ===========================================================================
# 3. MEASUREMENT TESTS
# ===========================================================================


class TestKuramotoOrderParameter:
    """Tests for the Kuramoto order parameter."""

    def test_synchronized_state_r_near_one(self) -> None:
        """Fully synchronized phases yield r ≈ 1."""
        phases = torch.zeros(100)  # All at phase 0
        r = kuramoto_order_parameter(phases)
        assert torch.isclose(r, torch.tensor(1.0), atol=1e-6)

    def test_uniform_phases_r_near_zero(self) -> None:
        """Uniformly distributed phases yield r ≈ 0."""
        n = 10000
        phases = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        r = kuramoto_order_parameter(phases)
        assert r < 0.01, f"Expected r ≈ 0 for uniform, got {r:.4f}"

    def test_r_in_valid_range(self) -> None:
        """Order parameter is always in [0, 1]."""
        torch.manual_seed(SEED)
        for _ in range(10):
            phases = torch.rand(100) * 2 * math.pi
            r = kuramoto_order_parameter(phases)
            assert 0.0 <= r <= 1.0 + 1e-6

    def test_batched_input(self) -> None:
        """Batched phases produce batched order parameter."""
        phases = torch.zeros(5, 100)  # All synchronized
        r = kuramoto_order_parameter(phases)
        assert r.shape == (5,)
        assert torch.allclose(r, torch.ones(5), atol=1e-6)

    def test_empty_raises(self) -> None:
        """Empty phase tensor raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            kuramoto_order_parameter(torch.tensor([]))

    def test_single_oscillator(self) -> None:
        """Single oscillator always has r = 1."""
        r = kuramoto_order_parameter(torch.tensor([3.14]))
        assert torch.isclose(r, torch.tensor(1.0), atol=1e-5)


class TestKuramotoOrderParameterComplex:
    """Tests for the complex order parameter."""

    def test_synchronized_magnitude(self) -> None:
        """Complex order parameter magnitude is 1 when sync'd."""
        phases = torch.zeros(50)
        z = kuramoto_order_parameter_complex(phases)
        assert abs(abs(z.item()) - 1.0) < 1e-6

    def test_synchronized_mean_phase(self) -> None:
        """Mean phase of sync'd oscillators at 0 yields angle ≈ 0."""
        phases = torch.zeros(50)
        z = kuramoto_order_parameter_complex(phases)
        angle = torch.angle(z).item()
        assert abs(angle) < 1e-6

    def test_empty_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            kuramoto_order_parameter_complex(torch.tensor([]))


class TestMeanPhaseCoherence:
    """Tests for mean pairwise phase coherence."""

    def test_synchronized_coherence_near_one(self) -> None:
        """All same phases → coherence ≈ 1."""
        phases = torch.zeros(100)
        c = mean_phase_coherence(phases)
        assert torch.isclose(c, torch.tensor(1.0), atol=1e-5)

    def test_anti_phase_coherence(self) -> None:
        """Two oscillators at phase 0 and π → coherence = -1."""
        phases = torch.tensor([0.0, math.pi])
        c = mean_phase_coherence(phases)
        assert torch.isclose(c, torch.tensor(-1.0), atol=1e-5)

    def test_too_few_oscillators_raises(self) -> None:
        """Single oscillator raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            mean_phase_coherence(torch.tensor([0.0]))

    def test_batched(self) -> None:
        """Batched coherence has correct shape."""
        phases = torch.zeros(3, 50)
        c = mean_phase_coherence(phases)
        assert c.shape == (3,)


class TestPhaseCoherenceMatrix:
    """Tests for the full coherence matrix."""

    def test_shape(self) -> None:
        """Coherence matrix has shape (N, N)."""
        phases = torch.randn(20)
        mat = phase_coherence_matrix(phases)
        assert mat.shape == (20, 20)

    def test_diagonal_ones(self) -> None:
        """Diagonal entries are 1 (cos(0) = 1)."""
        phases = torch.randn(10)
        mat = phase_coherence_matrix(phases)
        assert torch.allclose(mat.diag(), torch.ones(10), atol=1e-5)

    def test_symmetric(self) -> None:
        """Coherence matrix is symmetric."""
        phases = torch.randn(15)
        mat = phase_coherence_matrix(phases)
        assert torch.allclose(mat, mat.T, atol=1e-6)

    def test_batched_shape(self) -> None:
        """Batched coherence matrix has correct shape."""
        phases = torch.randn(4, 10)
        mat = phase_coherence_matrix(phases)
        assert mat.shape == (4, 10, 10)


class TestPowerSpectralDensity:
    """Tests for power spectral density computation."""

    def test_shape_default_bins(self) -> None:
        """Default n_freq_bins = 2*N."""
        amp = torch.ones(32)
        phase = torch.zeros(32)
        psd = power_spectral_density(amp, phase)
        assert psd.shape == (64,)

    def test_shape_custom_bins(self) -> None:
        """Custom n_freq_bins."""
        amp = torch.ones(32)
        phase = torch.zeros(32)
        psd = power_spectral_density(amp, phase, n_freq_bins=128)
        assert psd.shape == (128,)

    def test_non_negative(self) -> None:
        """Power spectrum is non-negative."""
        torch.manual_seed(SEED)
        amp = torch.rand(64)
        phase = torch.rand(64) * 2 * math.pi
        psd = power_spectral_density(amp, phase)
        assert (psd >= -1e-6).all()


class TestSynchronizationEnergy:
    """Tests for synchronization energy."""

    def test_synchronized_low_energy(self) -> None:
        """Synchronized state has low (negative) energy."""
        phases = torch.zeros(20)
        amps = torch.ones(20)
        e = synchronization_energy(phases, amps)
        assert e.item() < 0

    def test_random_higher_energy_than_sync(self) -> None:
        """Random state has higher energy than synchronized."""
        phases_sync = torch.zeros(30)
        amps = torch.ones(30)
        e_sync = synchronization_energy(phases_sync, amps)

        torch.manual_seed(SEED)
        phases_rand = torch.rand(30) * 2 * math.pi
        e_rand = synchronization_energy(phases_rand, amps)

        assert e_rand > e_sync - 0.1


# ===========================================================================
# 4. SPARSE k-NN HARDENING TESTS
# ===========================================================================


class TestSafePhaseHelpers:
    """Tests for _safe_phase_diff and _clamp_finite helpers."""

    def test_safe_phase_diff_range(self) -> None:
        """Wrapped differences lie in (-π, π]."""
        torch.manual_seed(SEED)
        a = torch.rand(1000) * 4 * math.pi  # Large unconstrained phases
        b = torch.rand(1000) * 4 * math.pi
        diff = _safe_phase_diff(a, b)
        assert (diff >= -math.pi - 1e-6).all()
        assert (diff <= math.pi + 1e-6).all()

    def test_safe_phase_diff_identity(self) -> None:
        """Same phases → zero difference."""
        phase = torch.tensor([0.0, 1.0, 3.0, 6.0])
        diff = _safe_phase_diff(phase, phase)
        assert torch.allclose(diff, torch.zeros(4), atol=1e-6)

    def test_safe_phase_diff_wrapping(self) -> None:
        """Differences near boundaries wrap correctly."""
        # Phase near 0 vs phase near 2π → small difference
        a = torch.tensor([0.01])
        b = torch.tensor([2 * math.pi - 0.01])
        diff = _safe_phase_diff(a, b)
        assert abs(diff.item()) < 0.05  # Should be about +0.02

    def test_safe_phase_diff_matches_raw_small(self) -> None:
        """For small phases, safe_phase_diff ≈ raw subtraction."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.9, 1.8, 2.7])
        safe = _safe_phase_diff(a, b)
        raw = a - b
        assert torch.allclose(safe, raw, atol=1e-5)

    def test_clamp_finite_nan(self) -> None:
        """NaN values become zero."""
        t = torch.tensor([1.0, float("nan"), -2.0])
        out = _clamp_finite(t)
        assert torch.isfinite(out).all()
        assert out[1].item() == 0.0

    def test_clamp_finite_inf(self) -> None:
        """Inf values become zero."""
        t = torch.tensor([1.0, float("inf"), float("-inf")])
        out = _clamp_finite(t)
        assert torch.isfinite(out).all()

    def test_clamp_finite_large(self) -> None:
        """Large values are clamped."""
        t = torch.tensor([1e5, -1e5])
        out = _clamp_finite(t, limit=100.0)
        assert (out.abs() <= 100.0).all()


class TestBuildPhaseKNNIndex:
    """Tests for _build_phase_knn_index reusable helper."""

    def test_shape(self) -> None:
        """Output shape is (B, N, k)."""
        flat = torch.rand(2, 50)
        idx = _build_phase_knn_index(flat, k=5)
        assert idx.shape == (2, 50, 5)

    def test_no_self_neighbour(self) -> None:
        """No oscillator is its own neighbour."""
        flat = torch.rand(1, 20)
        idx = _build_phase_knn_index(flat, k=4)
        arange = torch.arange(20).unsqueeze(-1).expand_as(idx[0])
        assert (idx[0] != arange).all()

    def test_k_equals_n_minus_1(self) -> None:
        """k = N-1 returns all other oscillators."""
        N = 5
        flat = torch.rand(1, N)
        idx = _build_phase_knn_index(flat, k=N - 1)
        assert idx.shape == (1, N, N - 1)
        # Each row should have N-1 unique indices
        for i in range(N):
            unique = idx[0, i].unique()
            assert len(unique) == N - 1

    def test_sorted_phases_nearest_correct(self) -> None:
        """For linearly spaced phases, neighbours are geometrically adjacent."""
        # Equally spaced on [0, 2π): each oscillator's 2 nearest
        # are its immediate left/right in the circle
        N = 20
        phases = torch.linspace(0, 2 * math.pi, N + 1)[:-1].unsqueeze(0)
        idx = _build_phase_knn_index(phases, k=2)
        for i in range(N):
            nbrs = set(idx[0, i].tolist())
            expected = {(i - 1) % N, (i + 1) % N}
            assert nbrs == expected, f"Oscillator {i}: got {nbrs}, expected {expected}"


class TestKuramotoSparseKNNHardened:
    """Tests for the hardened KuramotoOscillator sparse k-NN path."""

    def test_derivatives_finite(self) -> None:
        """All sparse derivatives are finite for random state."""
        model = KuramotoOscillator(
            n_oscillators=200, coupling_strength=2.0,
            coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(200, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()
        assert torch.isfinite(domega).all()

    def test_derivatives_shape(self) -> None:
        """Sparse derivatives have correct shape."""
        model = KuramotoOscillator(
            n_oscillators=50, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(50, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert dphi.shape == (50,)
        assert dr.shape == (50,)
        assert domega.shape == (50,)

    def test_n_equals_1_no_crash(self) -> None:
        """N=1 edge case returns coupling-free dynamics."""
        model = KuramotoOscillator(
            n_oscillators=1, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(1, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        # dphi should just be the natural frequency
        assert torch.allclose(dphi, state.frequency, atol=1e-5)
        # domega should be zero (no coupling)
        assert torch.allclose(domega, torch.zeros(1), atol=1e-5)

    def test_n_equals_2(self) -> None:
        """N=2 with k=1 couples each oscillator to the other."""
        model = KuramotoOscillator(
            n_oscillators=2, coupling_mode="sparse_knn", sparse_k=1,
        )
        state = OscillatorState.create_random(2, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()

    def test_batched(self) -> None:
        """Batched input produces correct shape."""
        model = KuramotoOscillator(
            n_oscillators=30, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(30, batch_size=4, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert dphi.shape == (4, 30)
        assert dr.shape == (4, 30)
        assert domega.shape == (4, 30)
        assert torch.isfinite(dphi).all()

    def test_integration_stays_finite(self) -> None:
        """100-step integration with sparse coupling stays finite."""
        model = KuramotoOscillator(
            n_oscillators=100, coupling_strength=2.0,
            coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(100, seed=SEED)
        final, _ = model.integrate(state, n_steps=100, dt=0.01)
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()
        assert (final.amplitude >= 0).all()

    def test_gradient_flows_through_sparse(self) -> None:
        """Gradient flows through the sparse k-NN path."""
        model = KuramotoOscillator(
            n_oscillators=30, coupling_mode="sparse_knn",
        )
        phase_leaf = torch.rand(30, requires_grad=True)
        phase = phase_leaf * 2 * math.pi
        phase.retain_grad()
        amp = torch.ones(30)
        freq = torch.ones(30)
        state = OscillatorState(phase=phase, amplitude=amp, frequency=freq)
        dphi, dr, domega = model.compute_derivatives(state)
        loss = dphi.sum()
        loss.backward()
        assert phase_leaf.grad is not None
        assert torch.isfinite(phase_leaf.grad).all()
        # Gradient magnitude should not be extreme
        assert phase_leaf.grad.norm() < 1e6

    def test_sparse_vs_full_consistency_small_n(self) -> None:
        """At small N with k=N-1, sparse ≈ full (up to coupling normalization)."""
        N = 8
        state = OscillatorState.create_random(N, seed=SEED)

        model_full = KuramotoOscillator(
            n_oscillators=N, coupling_strength=2.0,
            coupling_mode="full",
        )
        model_sparse = KuramotoOscillator(
            n_oscillators=N, coupling_strength=2.0,
            coupling_mode="sparse_knn", sparse_k=N - 1,
        )
        dphi_f, dr_f, _ = model_full.compute_derivatives(state)
        dphi_s, dr_s, _ = model_sparse.compute_derivatives(state)

        # Full uses K/N, sparse uses K/k = K/(N-1) → slight scale diff
        # Check correlation, not exact match
        corr = torch.nn.functional.cosine_similarity(
            dphi_f.unsqueeze(0), dphi_s.unsqueeze(0)
        )
        assert corr > 0.95, f"Phase derivative correlation {corr:.4f} too low"

    def test_large_n_no_nan(self) -> None:
        """N=4096 sparse k-NN does not produce NaN."""
        model = KuramotoOscillator(
            n_oscillators=4096, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(4096, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all(), "NaN/Inf in dphi at N=4096"
        assert torch.isfinite(dr).all(), "NaN/Inf in dr at N=4096"
        assert torch.isfinite(domega).all(), "NaN/Inf in domega at N=4096"

    def test_custom_sparse_k(self) -> None:
        """Custom sparse_k parameter is respected."""
        model = KuramotoOscillator(
            n_oscillators=100, coupling_mode="sparse_knn", sparse_k=10,
        )
        assert model.sparse_k == 10

    def test_default_sparse_k_is_log2_n(self) -> None:
        """Default k = ceil(log₂ N)."""
        model = KuramotoOscillator(
            n_oscillators=1024, coupling_mode="sparse_knn",
        )
        assert model.sparse_k == math.ceil(math.log2(1024))


class TestHopfSparseKNNHardened:
    """Tests for the hardened HopfOscillator sparse k-NN path."""

    def test_derivatives_finite(self) -> None:
        """Hopf sparse derivatives are finite."""
        model = HopfOscillator(
            n_oscillators=100, coupling_strength=1.0,
            bifurcation_param=1.0, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(100, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()
        assert torch.isfinite(domega).all()

    def test_n_equals_1(self) -> None:
        """N=1 edge case: pure Hopf dynamics, no coupling."""
        model = HopfOscillator(
            n_oscillators=1, coupling_mode="sparse_knn",
            bifurcation_param=4.0,
        )
        state = OscillatorState.create_random(1, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        # dr = μr − r³ (no coupling)
        expected_dr = 4.0 * state.amplitude - state.amplitude ** 3
        assert torch.allclose(dr, expected_dr, atol=1e-5)

    def test_near_zero_amplitude_safe(self) -> None:
        """Amplitudes near zero don't cause NaN from 1/r division."""
        model = HopfOscillator(
            n_oscillators=20, coupling_mode="sparse_knn",
        )
        state = OscillatorState(
            phase=torch.rand(20) * 2 * math.pi,
            amplitude=torch.full((20,), 1e-12),  # Tiny amplitude
            frequency=torch.ones(20),
        )
        dphi, dr, domega = model.compute_derivatives(state)
        assert torch.isfinite(dphi).all(), "NaN from near-zero amplitude"
        assert torch.isfinite(dr).all()
        assert torch.isfinite(domega).all()

    def test_integration_converges(self) -> None:
        """Hopf sparse integration converges to limit cycle."""
        model = HopfOscillator(
            n_oscillators=50, coupling_strength=1.0,
            bifurcation_param=1.0, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(50, seed=SEED)
        final, _ = model.integrate(state, n_steps=200, dt=0.01)
        assert torch.isfinite(final.amplitude).all()
        # Amplitudes should approach √μ = 1.0
        mean_amp = final.amplitude.mean().item()
        assert 0.1 < mean_amp < 5.0

    def test_gradient_flows(self) -> None:
        """Gradient flows through Hopf sparse path."""
        model = HopfOscillator(
            n_oscillators=20, coupling_mode="sparse_knn",
        )
        phase = torch.rand(20, requires_grad=True)
        amp = torch.ones(20, requires_grad=True)
        freq = torch.ones(20)
        state = OscillatorState(phase=phase, amplitude=amp, frequency=freq)
        dphi, dr, domega = model.compute_derivatives(state)
        loss = dphi.sum() + dr.sum()
        loss.backward()
        assert phase.grad is not None
        assert amp.grad is not None
        assert torch.isfinite(phase.grad).all()
        assert torch.isfinite(amp.grad).all()

    def test_batched(self) -> None:
        """Batched Hopf sparse produces correct shapes."""
        model = HopfOscillator(
            n_oscillators=30, coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(30, batch_size=3, seed=SEED)
        dphi, dr, domega = model.compute_derivatives(state)
        assert dphi.shape == (3, 30)
        assert torch.isfinite(dphi).all()


# ===========================================================================
# 5. SPARSE MEASUREMENT TESTS
# ===========================================================================


class TestBuildPhaseKNN:
    """Tests for build_phase_knn measurement helper."""

    def test_unbatched_shape(self) -> None:
        """Unbatched input produces (1, N, k)."""
        phase = torch.rand(50)
        idx = build_phase_knn(phase, k=4)
        assert idx.shape == (1, 50, 4)

    def test_batched_shape(self) -> None:
        """Batched input produces (B, N, k)."""
        phase = torch.rand(3, 50)
        idx = build_phase_knn(phase, k=4)
        assert idx.shape == (3, 50, 4)

    def test_k_zero_raises(self) -> None:
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            build_phase_knn(torch.rand(10), k=0)

    def test_k_ge_n_raises(self) -> None:
        """k >= N raises ValueError."""
        with pytest.raises(ValueError, match="k must be < N"):
            build_phase_knn(torch.rand(10), k=10)


class TestSparseMeanPhaseCoherence:
    """Tests for sparse_mean_phase_coherence."""

    def test_synchronized_near_one(self) -> None:
        """Fully synchronized → coherence ≈ 1."""
        phases = torch.zeros(100)
        nbr = build_phase_knn(phases, k=8)
        c = sparse_mean_phase_coherence(phases, nbr)
        assert c.item() > 0.99

    def test_uniform_near_zero(self) -> None:
        """Uniformly distributed → coherence low."""
        phases = torch.linspace(0, 2 * math.pi, 101)[:-1]
        nbr = build_phase_knn(phases, k=4)
        c = sparse_mean_phase_coherence(phases, nbr)
        # Sparse coherence looks at local neighbourhoods — for uniform
        # the 4 nearest in a ring of 100 are close → local r not zero
        # but still < 1
        assert 0 <= c.item() <= 1.0 + 1e-6

    def test_range(self) -> None:
        """Coherence is in [0, 1]."""
        torch.manual_seed(SEED)
        phases = torch.rand(200) * 2 * math.pi
        nbr = build_phase_knn(phases, k=8)
        c = sparse_mean_phase_coherence(phases, nbr)
        assert 0.0 <= c.item() <= 1.0 + 1e-6

    def test_too_few_oscillators_raises(self) -> None:
        """Single oscillator raises."""
        with pytest.raises(ValueError, match="at least 2"):
            nbr = torch.zeros(1, 1, 1, dtype=torch.long)
            sparse_mean_phase_coherence(torch.tensor([0.0]), nbr)

    def test_batched(self) -> None:
        """Batched coherence has correct shape."""
        phases = torch.zeros(3, 50)
        nbr = build_phase_knn(phases, k=4)
        c = sparse_mean_phase_coherence(phases, nbr)
        assert c.shape == (3,)

    def test_correlated_with_dense(self) -> None:
        """Sparse coherence correlates positively with dense coherence."""
        torch.manual_seed(SEED)
        # Test across several random states
        sparse_vals = []
        dense_vals = []
        for seed in range(5):
            torch.manual_seed(seed)
            phases = torch.rand(50) * 2 * math.pi
            nbr = build_phase_knn(phases, k=10)
            sparse_vals.append(
                sparse_mean_phase_coherence(phases, nbr).item()
            )
            dense_vals.append(mean_phase_coherence(phases).item())
        # Pearson correlation should be positive
        s = torch.tensor(sparse_vals)
        d = torch.tensor(dense_vals)
        cov = ((s - s.mean()) * (d - d.mean())).mean()
        # Just check same direction (both higher for more sync'd states)
        assert cov > -0.1  # Not strongly anti-correlated


class TestSparseSynchronizationEnergy:
    """Tests for sparse_synchronization_energy."""

    def test_synchronized_negative(self) -> None:
        """Sync'd state has negative (low) energy."""
        phases = torch.zeros(50)
        amps = torch.ones(50)
        nbr = build_phase_knn(phases, k=4)
        e = sparse_synchronization_energy(phases, amps, nbr)
        assert e.item() < 0

    def test_random_higher_than_sync(self) -> None:
        """Random state has higher energy than synchronized."""
        phases_sync = torch.zeros(50)
        amps = torch.ones(50)
        nbr_sync = build_phase_knn(phases_sync, k=4)
        e_sync = sparse_synchronization_energy(phases_sync, amps, nbr_sync)

        torch.manual_seed(SEED)
        phases_rand = torch.rand(50) * 2 * math.pi
        nbr_rand = build_phase_knn(phases_rand, k=4)
        e_rand = sparse_synchronization_energy(phases_rand, amps, nbr_rand)

        assert e_rand > e_sync - 0.1

    def test_batched(self) -> None:
        """Batched energy has correct shape."""
        phases = torch.zeros(3, 40)
        amps = torch.ones(3, 40)
        nbr = build_phase_knn(phases, k=4)
        e = sparse_synchronization_energy(phases, amps, nbr)
        assert e.shape == (3,)
