"""Unit tests for prinet.utils: solvers and sparse utilities.

Covers BatchedRK45Solver, FixedStepRK4Solver, sparse_coupling_matrix,
and gradient_checkpoint_integration.
"""

from __future__ import annotations

import math

import pytest
import torch

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import (
    KuramotoOscillator,
    OscillatorState,
    StuartLandauOscillator,
)
from prinet.utils.cuda_kernels import (
    BatchedRK45Solver,
    FixedStepRK4Solver,
    SolverError,
    SolverResult,
    gradient_checkpoint_integration,
    sparse_coupling_matrix,
)

SEED = 42


@pytest.fixture
def kuramoto_100() -> KuramotoOscillator:
    """Kuramoto model with 100 oscillators."""
    return KuramotoOscillator(n_oscillators=100, coupling_strength=2.0, decay_rate=0.1)


@pytest.fixture
def state_100() -> OscillatorState:
    """Random state with 100 oscillators."""
    return OscillatorState.create_random(100, seed=SEED)


# ===========================================================================
# BATCHED RK45 SOLVER TESTS
# ===========================================================================


class TestBatchedRK45Solver:
    """Tests for the adaptive RK45 solver."""

    def test_solve_basic(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Basic solve completes without error."""
        solver = BatchedRK45Solver(atol=1e-4, rtol=1e-3)
        result = solver.solve(kuramoto_100, state_100, t_span=(0.0, 0.1), max_steps=500)
        assert isinstance(result, SolverResult)
        assert result.n_steps_taken > 0
        assert result.wall_time_seconds > 0

    def test_solve_produces_valid_state(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Solved state contains no NaN values."""
        solver = BatchedRK45Solver(atol=1e-4, rtol=1e-3)
        result = solver.solve(kuramoto_100, state_100, t_span=(0.0, 0.1), max_steps=500)
        assert not torch.isnan(result.final_state.phase).any()
        assert not torch.isnan(result.final_state.amplitude).any()
        assert (result.final_state.amplitude >= 0).all()

    def test_solve_with_trajectory(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Trajectory recording works."""
        solver = BatchedRK45Solver(atol=1e-3, rtol=1e-2)
        result = solver.solve(
            kuramoto_100,
            state_100,
            t_span=(0.0, 0.05),
            max_steps=200,
            record_trajectory=True,
        )
        assert result.trajectory is not None
        assert len(result.trajectory) == result.n_steps_taken

    def test_adaptive_step_grows(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Adaptive solver adjusts step size."""
        solver = BatchedRK45Solver(atol=1e-4, rtol=1e-3)
        result = solver.solve(
            kuramoto_100, state_100, t_span=(0.0, 0.1), max_steps=1000
        )
        # Solver should take fewer steps than a fixed dt=0.001 would
        assert result.n_steps_taken < 200

    def test_invalid_atol_raises(self) -> None:
        """Non-positive atol raises ValueError."""
        with pytest.raises(ValueError, match="atol must be positive"):
            BatchedRK45Solver(atol=0.0)

    def test_invalid_rtol_raises(self) -> None:
        """Non-positive rtol raises ValueError."""
        with pytest.raises(ValueError, match="rtol must be positive"):
            BatchedRK45Solver(rtol=-0.01)


# ===========================================================================
# FIXED STEP RK4 SOLVER TESTS
# ===========================================================================


class TestFixedStepRK4Solver:
    """Tests for the fixed-step RK4 solver."""

    def test_solve_basic(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Basic solve completes and returns result."""
        solver = FixedStepRK4Solver(dt=0.01)
        result = solver.solve(kuramoto_100, state_100, n_steps=100)
        assert isinstance(result, SolverResult)
        assert result.n_steps_taken == 100
        assert result.n_function_evals == 400  # 4 per RK4 step

    def test_solve_deterministic(self, kuramoto_100: KuramotoOscillator) -> None:
        """Same initial state produces same result."""
        solver = FixedStepRK4Solver(dt=0.01)
        state1 = OscillatorState.create_random(100, seed=SEED)
        state2 = OscillatorState.create_random(100, seed=SEED)
        r1 = solver.solve(kuramoto_100, state1, n_steps=50)
        r2 = solver.solve(kuramoto_100, state2, n_steps=50)
        assert torch.allclose(r1.final_state.phase, r2.final_state.phase, atol=1e-5)

    def test_invalid_dt_raises(self) -> None:
        """Non-positive dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            FixedStepRK4Solver(dt=0.0)

    def test_solve_with_trajectory(
        self, kuramoto_100: KuramotoOscillator, state_100: OscillatorState
    ) -> None:
        """Trajectory recording works for fixed solver."""
        solver = FixedStepRK4Solver(dt=0.01)
        result = solver.solve(
            kuramoto_100,
            state_100,
            n_steps=20,
            record_trajectory=True,
        )
        assert result.trajectory is not None
        assert len(result.trajectory) == 20


# ===========================================================================
# SPARSE COUPLING MATRIX TESTS
# ===========================================================================


class TestSparseCouplingMatrix:
    """Tests for sparse coupling matrix generation."""

    def test_shape(self) -> None:
        """Matrix has correct shape."""
        mat = sparse_coupling_matrix(100, sparsity=0.9, seed=SEED)
        assert mat.shape == (100, 100)

    def test_zero_diagonal(self) -> None:
        """Diagonal entries are zero (no self-coupling)."""
        mat = sparse_coupling_matrix(50, sparsity=0.8, seed=SEED)
        assert torch.allclose(mat.diag(), torch.zeros(50))

    def test_sparsity_level(self) -> None:
        """Approximately correct fraction of zeros."""
        mat = sparse_coupling_matrix(200, sparsity=0.9, seed=SEED)
        zero_frac = (mat == 0).float().mean().item()
        # Should be roughly 0.9 (plus diagonal zeros)
        assert zero_frac > 0.85

    def test_symmetric(self) -> None:
        """Symmetric matrix is indeed symmetric."""
        mat = sparse_coupling_matrix(50, sparsity=0.8, symmetric=True, seed=SEED)
        assert torch.allclose(mat, mat.T, atol=1e-6)

    def test_invalid_sparsity_raises(self) -> None:
        """Sparsity >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be"):
            sparse_coupling_matrix(10, sparsity=1.0)

    def test_dense_creation(self) -> None:
        """sparsity=0 creates a dense matrix."""
        mat = sparse_coupling_matrix(10, sparsity=0.0, seed=SEED)
        non_diag = mat.clone()
        non_diag.fill_diagonal_(1.0)
        # Off-diagonal should all be non-zero
        assert (non_diag != 0).all()

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same matrix."""
        m1 = sparse_coupling_matrix(50, sparsity=0.9, seed=123)
        m2 = sparse_coupling_matrix(50, sparsity=0.9, seed=123)
        assert torch.allclose(m1, m2)


# ===========================================================================
# GRADIENT CHECKPOINTING TESTS
# ===========================================================================


class TestGradientCheckpointIntegration:
    """Tests for memory-efficient checkpointed integration."""

    def test_result_matches_standard(self) -> None:
        """Checkpointed integration matches standard integration."""
        model = KuramotoOscillator(n_oscillators=20, coupling_strength=1.0)
        state = OscillatorState.create_random(20, seed=SEED)

        # Standard integration
        final_std, _ = model.integrate(state.clone(), n_steps=50, dt=0.01, method="rk4")

        # Checkpointed integration
        final_cp = gradient_checkpoint_integration(
            model, state.clone(), n_steps=50, dt=0.01, checkpoint_every=10
        )

        assert torch.allclose(final_std.phase, final_cp.phase, atol=1e-4)
        assert torch.allclose(final_std.amplitude, final_cp.amplitude, atol=1e-4)

    def test_handles_non_divisible_steps(self) -> None:
        """Works when n_steps is not divisible by checkpoint_every."""
        model = KuramotoOscillator(n_oscillators=10, coupling_strength=1.0)
        state = OscillatorState.create_random(10, seed=SEED)
        final = gradient_checkpoint_integration(
            model, state, n_steps=37, dt=0.01, checkpoint_every=10
        )
        assert not torch.isnan(final.phase).any()
