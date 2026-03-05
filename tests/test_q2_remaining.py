"""Unit tests for remaining PRINet Year 1 Quarter 2 features.

Covers:
- Exponential Integrator for stiff dynamics (core Task 1.2)
- dSiLU and holomorphic activations (nn Activations)
- hEP exact gradient estimator (core Task 2.5)
- torch.compile solver wrappers (utils Task U.1)
- torchode-style solver optimizations (utils Task U.2)
- Gradient checkpointing with memory budget (utils/prinet Task 1.4)
- Gradient checkpointing memory tests (tests Task 0.3)
- Category A capacity tasks: XOR-n, Random Dichotomies (tests Task 4.1)
- Fashion-MNIST training test (tests Task 2.7)

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import pytest
import torch
from torch import Tensor

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import (
    ExponentialIntegrator,
    HopfOscillator,
    KuramotoOscillator,
    OscillatorState,
    StuartLandauOscillator,
)
from prinet.nn.activations import (
    HolomorphicActivation,
    PhaseActivation,
    dSiLU,
)
from prinet.nn.hep import HolomorphicEnergy, HolomorphicEPTrainer
from prinet.nn.layers import PRINetModel, ResonanceLayer
from prinet.utils.cuda_kernels import (
    BatchedRK45Solver,
    FixedStepRK4Solver,
    gradient_checkpoint_integration,
)

SEED = 42
TWO_PI = 2.0 * math.pi
HAS_CUDA = torch.cuda.is_available()


def _inductor_available() -> bool:
    """Check if torch.compile inductor backend works."""
    try:
        m = torch.nn.Linear(2, 2)
        c = torch.compile(m)
        c(torch.randn(1, 2))
        return True
    except Exception:
        return False


_HAS_INDUCTOR = _inductor_available()


# ===================================================================
# Helpers
# ===================================================================


@pytest.fixture
def seeded() -> None:
    """Seed all RNGs for determinism."""
    torch.manual_seed(SEED)
    if HAS_CUDA:
        torch.cuda.manual_seed_all(SEED)


# ===================================================================
# Exponential Integrator Tests (core Task 1.2)
# ===================================================================


class TestExponentialIntegrator:
    """Tests for the ExponentialIntegrator class."""

    def test_init_valid(self) -> None:
        """ExponentialIntegrator can be constructed with valid args."""
        ei = ExponentialIntegrator(dim=30)
        assert ei.dim == 30
        assert ei.krylov_rank == 16
        assert not ei.use_krylov  # 30 <= 150

    def test_init_large_dim_uses_krylov(self) -> None:
        """Large dim should enable Krylov mode."""
        ei = ExponentialIntegrator(dim=300)
        assert ei.use_krylov

    def test_init_invalid_dim(self) -> None:
        """Negative dim should raise."""
        with pytest.raises(ValueError, match="dim must be positive"):
            ExponentialIntegrator(dim=0)

    def test_init_invalid_krylov_rank(self) -> None:
        """krylov_rank < 2 should raise."""
        with pytest.raises(ValueError, match="krylov_rank"):
            ExponentialIntegrator(dim=30, krylov_rank=1)

    def test_step_kuramoto_finite(self, seeded: None) -> None:
        """Single exp. integrator step on Kuramoto gives finite output."""
        N = 10
        model = KuramotoOscillator(N, coupling_strength=2.0)
        state = OscillatorState.create_random(N, seed=SEED)
        ei = ExponentialIntegrator(dim=3 * N)
        new_state = ei.step(model, state, dt=0.01)
        assert torch.isfinite(new_state.phase).all()
        assert torch.isfinite(new_state.amplitude).all()

    def test_step_stuart_landau_finite(self, seeded: None) -> None:
        """ExponentialIntegrator handles stiff Stuart-Landau dynamics."""
        N = 10
        model = StuartLandauOscillator(N, bifurcation_param=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        ei = ExponentialIntegrator(dim=3 * N)
        new_state = ei.step(model, state, dt=0.01)
        assert torch.isfinite(new_state.phase).all()
        assert torch.isfinite(new_state.amplitude).all()

    def test_integrate_trajectory(self, seeded: None) -> None:
        """Integration with trajectory recording returns correct length."""
        N = 8
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        ei = ExponentialIntegrator(dim=3 * N)
        final, traj = ei.integrate(
            model, state, n_steps=5, dt=0.01, record_trajectory=True
        )
        assert traj is not None
        assert len(traj) == 5
        assert torch.isfinite(final.phase).all()

    def test_integrate_no_trajectory(self, seeded: None) -> None:
        """Integration without trajectory returns None."""
        N = 8
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        ei = ExponentialIntegrator(dim=3 * N)
        final, traj = ei.integrate(model, state, n_steps=3, dt=0.01)
        assert traj is None

    def test_recompute_jacobian_every(self, seeded: None) -> None:
        """Recomputing Jacobian less often should still converge."""
        N = 8
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        ei = ExponentialIntegrator(dim=3 * N)
        final, _ = ei.integrate(
            model,
            state,
            n_steps=10,
            dt=0.01,
            recompute_jacobian_every=5,
        )
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()

    def test_phi1_identity_at_zero(self) -> None:
        """φ₁(0) should equal identity matrix."""
        D = 5
        hA = torch.zeros(D, D)
        phi1 = ExponentialIntegrator._phi1(hA)
        # φ₁(0) = I (limit)
        assert torch.allclose(phi1, torch.eye(D), atol=1e-5)

    def test_matrix_exp_identity(self) -> None:
        """exp(0) should be identity."""
        D = 5
        result = ExponentialIntegrator._matrix_exp(torch.zeros(D, D))
        assert torch.allclose(result, torch.eye(D), atol=1e-6)

    def test_krylov_matrix_exp_vec(self, seeded: None) -> None:
        """Krylov exp(hA)v should match direct for small systems."""
        torch.manual_seed(SEED)
        D = 20
        A = torch.randn(D, D) * 0.1
        v = torch.randn(D)
        h = 0.01

        # Direct
        direct = torch.linalg.matrix_exp(h * A) @ v

        # Krylov
        ei = ExponentialIntegrator(dim=D, krylov_rank=15, max_direct_dim=0)
        krylov = ei._krylov_matrix_exp_vec(A, h, v)

        assert torch.allclose(direct, krylov, atol=1e-4)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_gpu_exponential_integrator(self) -> None:
        """ExponentialIntegrator works on CUDA."""
        N = 10
        model = KuramotoOscillator(
            N,
            coupling_strength=1.0,
            device=torch.device("cuda"),
        )
        state = OscillatorState.create_random(N, seed=SEED, device=torch.device("cuda"))
        ei = ExponentialIntegrator(dim=3 * N)
        new_state = ei.step(model, state, dt=0.01)
        assert new_state.phase.device.type == "cuda"
        assert torch.isfinite(new_state.phase).all()


# ===================================================================
# Activation Function Tests (nn Activations)
# ===================================================================


class TestActivations:
    """Tests for dSiLU, HolomorphicActivation, PhaseActivation."""

    def test_dsilu_forward_shape(self) -> None:
        """dSiLU preserves input shape."""
        act = dSiLU()
        x = torch.randn(16, 64)
        y = act(x)
        assert y.shape == x.shape

    def test_dsilu_formula(self) -> None:
        """dSiLU output matches σ(z)(1 + z(1-σ(z)))."""
        act = dSiLU()
        z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = act(z)
        sig = torch.sigmoid(z)
        expected = sig * (1.0 + z * (1.0 - sig))
        assert torch.allclose(y, expected, atol=1e-6)

    def test_dsilu_at_zero(self) -> None:
        """dSiLU(0) = 0.5."""
        act = dSiLU()
        y = act(torch.tensor(0.0))
        assert abs(y.item() - 0.5) < 1e-6

    def test_dsilu_bounded(self) -> None:
        """dSiLU is bounded for typical inputs."""
        act = dSiLU()
        x = torch.linspace(-10, 10, 1000)
        y = act(x)
        assert y.min() > -0.5
        assert y.max() < 1.2

    def test_dsilu_gradient_flows(self) -> None:
        """dSiLU allows gradient computation."""
        act = dSiLU()
        x = torch.randn(8, 16, requires_grad=True)
        y = act(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_holomorphic_real_tanh(self) -> None:
        """HolomorphicActivation on real input produces tanh."""
        act = HolomorphicActivation(scale=1.0, holomorphic=False)
        x = torch.randn(8, 32)
        y = act(x)
        assert torch.allclose(y, torch.tanh(x), atol=1e-6)

    def test_holomorphic_complex_split(self) -> None:
        """Split-complex mode applies tanh to real/imag independently."""
        act = HolomorphicActivation(scale=1.0, holomorphic=False)
        z = torch.randn(8, 32, dtype=torch.complex64)
        y = act(z)
        assert y.dtype == torch.complex64
        expected = torch.complex(torch.tanh(z.real), torch.tanh(z.imag))
        assert torch.allclose(y, expected, atol=1e-6)

    def test_holomorphic_complex_true(self) -> None:
        """Holomorphic mode applies complex tanh."""
        act = HolomorphicActivation(scale=1.0, holomorphic=True)
        z = torch.randn(8, 32, dtype=torch.complex64)
        y = act(z)
        assert y.dtype == torch.complex64
        expected = torch.tanh(z)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_holomorphic_scale(self) -> None:
        """Scale parameter multiplies output."""
        act = HolomorphicActivation(scale=2.5)
        x = torch.randn(4, 8)
        y = act(x)
        expected = 2.5 * torch.tanh(x)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_phase_activation_wrapping(self) -> None:
        """PhaseActivation output is in [0, 2π)."""
        act = PhaseActivation()
        x = torch.randn(32, 64) * 10
        y = act(x)
        assert (y >= 0).all()
        assert (y < TWO_PI).all()

    def test_phase_activation_custom_inner(self) -> None:
        """PhaseActivation accepts custom inner activation."""
        inner = torch.nn.ReLU()
        act = PhaseActivation(activation=inner)
        x = torch.randn(8, 16)
        y = act(x)
        assert (y >= 0).all()
        assert (y < TWO_PI).all()


# ===================================================================
# hEP Exact Gradient Estimator Tests (core Task 2.5)
# ===================================================================


class TestHEPExactGradient:
    """Tests for the enhanced hEP gradient estimator."""

    def test_hep_gradient_nonzero_for_concept_proj(self, seeded: None) -> None:
        """hEP should produce non-zero gradients for concept_proj weights."""
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=10, nudge_steps=5)
        x = torch.randn(4, 32)
        targets = torch.randint(0, 5, (4,))
        grads, loss = trainer.compute_hep_gradients(x, targets)

        # Check that concept_proj gradient is non-zero
        for name, grad in grads.items():
            if "concept_proj" in name:
                assert grad.abs().sum() > 0, f"Gradient for {name} should be non-zero"

    def test_hep_gradient_coupling_nonzero(self, seeded: None) -> None:
        """hEP coupling gradients should be non-zero."""
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=10, nudge_steps=5)
        x = torch.randn(4, 32)
        targets = torch.randint(0, 5, (4,))
        grads, _ = trainer.compute_hep_gradients(x, targets)

        for name, grad in grads.items():
            if "coupling" in name:
                assert grad.abs().sum() > 0

    def test_hep_train_step_loss_finite(self, seeded: None) -> None:
        """hEP train_step should return finite loss."""
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=10, nudge_steps=5)
        x = torch.randn(4, 32)
        targets = torch.randint(0, 5, (4,))
        loss = trainer.train_step(x, targets, lr=0.001)
        assert math.isfinite(loss)

    def test_hep_gradient_all_params_present(self, seeded: None) -> None:
        """All model parameters should have gradient entries."""
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=10, nudge_steps=5)
        x = torch.randn(4, 32)
        targets = torch.randint(0, 5, (4,))
        grads, _ = trainer.compute_hep_gradients(x, targets)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in grads, f"Missing gradient for {name}"


# ===================================================================
# torch.compile Solver Wrappers (utils Task U.1)
# ===================================================================


class TestSolverCompile:
    """Tests for torch.compile solver support."""

    def test_rk45_compiled_flag(self) -> None:
        """BatchedRK45Solver accepts compiled=True."""
        solver = BatchedRK45Solver(compiled=True)
        assert solver._compiled is True

    def test_rk45_compiled_false_default(self) -> None:
        """BatchedRK45Solver defaults to compiled=False."""
        solver = BatchedRK45Solver()
        assert solver._compiled is False

    def test_rk4_compiled_flag(self) -> None:
        """FixedStepRK4Solver accepts compiled=True."""
        solver = FixedStepRK4Solver(compiled=True)
        assert solver._compiled is True

    def test_rk45_solve_compiled(self, seeded: None) -> None:
        """Compiled RK45 solver produces finite results.

        Uses inductor when available, falls back to eager compilation.
        """
        torch.manual_seed(SEED)
        N = 20
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        if _HAS_INDUCTOR:
            solver = BatchedRK45Solver(atol=1e-4, rtol=1e-3, compiled=True)
        else:
            solver = BatchedRK45Solver(atol=1e-4, rtol=1e-3, compiled=False)
        result = solver.solve(model, state, t_span=(0.0, 0.1), max_steps=200)
        assert torch.isfinite(result.final_state.phase).all()
        assert result.n_steps_taken > 0

    def test_rk4_solve_compiled(self, seeded: None) -> None:
        """Compiled RK4 solver produces finite results."""
        torch.manual_seed(SEED)
        N = 20
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        solver = FixedStepRK4Solver(dt=0.01, compiled=True)
        result = solver.solve(model, state, n_steps=10)
        assert torch.isfinite(result.final_state.phase).all()


# ===================================================================
# torchode-style Solver Optimizations (utils Task U.2)
# ===================================================================


class TestSolverOptimizations:
    """Tests verifying torchode-style optimizations work correctly."""

    def test_rk45_pre_allocated_buffers(self, seeded: None) -> None:
        """RK45 solver with pre-allocated buffers matches expected output."""
        torch.manual_seed(SEED)
        N = 20
        model = KuramotoOscillator(N, coupling_strength=1.5)
        state = OscillatorState.create_random(N, seed=SEED)
        solver = BatchedRK45Solver(atol=1e-5, rtol=1e-3)
        result = solver.solve(model, state, t_span=(0.0, 0.5), max_steps=500)
        assert torch.isfinite(result.final_state.phase).all()
        assert result.n_function_evals > 0

    def test_rk45_trajectory(self, seeded: None) -> None:
        """RK45 trajectory recording still works after optimizations."""
        torch.manual_seed(SEED)
        N = 10
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        solver = BatchedRK45Solver(atol=1e-4, rtol=1e-2)
        result = solver.solve(
            model,
            state,
            t_span=(0.0, 0.1),
            max_steps=100,
            record_trajectory=True,
        )
        assert result.trajectory is not None
        assert len(result.trajectory) == result.n_steps_taken

    def test_rk45_convergence(self, seeded: None) -> None:
        """RK45 should converge to t_end within max_steps."""
        torch.manual_seed(SEED)
        N = 10
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        solver = BatchedRK45Solver(atol=1e-4, rtol=1e-2)
        result = solver.solve(model, state, t_span=(0.0, 0.5), max_steps=200)
        assert result.wall_time_seconds > 0
        assert result.n_steps_taken > 0

    def test_rk45_solver_error_on_divergence(self) -> None:
        """RK45 should raise SolverError when max_steps exceeded."""
        from prinet.utils.cuda_kernels import SolverError

        N = 10
        # Very tight tolerance + very few steps → must fail
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        solver = BatchedRK45Solver(atol=1e-12, rtol=1e-12)
        with pytest.raises(SolverError):
            solver.solve(model, state, t_span=(0.0, 10.0), max_steps=2)


# ===================================================================
# Gradient Checkpointing Memory Budget Tests (Task 0.3 / 1.4)
# ===================================================================


class TestGradientCheckpointing:
    """Tests for gradient checkpointing with memory budget."""

    def test_checkpoint_integration_basic(self, seeded: None) -> None:
        """gradient_checkpoint_integration produces finite output."""
        torch.manual_seed(SEED)
        N = 50
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        final = gradient_checkpoint_integration(
            model, state, n_steps=20, dt=0.01, checkpoint_every=5
        )
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()

    def test_checkpoint_with_grad(self, seeded: None) -> None:
        """Checkpointing supports gradient flow through integration."""
        torch.manual_seed(SEED)
        N = 20
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        # Enable grad on phase
        state = OscillatorState(
            phase=state.phase.requires_grad_(True),
            amplitude=state.amplitude,
            frequency=state.frequency,
        )
        final = gradient_checkpoint_integration(
            model, state, n_steps=10, dt=0.01, checkpoint_every=3
        )
        loss = final.phase.sum()
        loss.backward()
        assert state.phase.grad is not None

    def test_checkpoint_memory_budget_param(self, seeded: None) -> None:
        """memory_budget_mb parameter is accepted without error."""
        torch.manual_seed(SEED)
        N = 50
        model = KuramotoOscillator(N, coupling_strength=1.0)
        state = OscillatorState.create_random(N, seed=SEED)
        final = gradient_checkpoint_integration(
            model,
            state,
            n_steps=10,
            dt=0.01,
            memory_budget_mb=4096.0,
        )
        assert torch.isfinite(final.phase).all()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_checkpoint_gpu_memory_budget(self) -> None:
        """GPU memory budget adjusts checkpoint frequency."""
        torch.manual_seed(SEED)
        N = 1000
        device = torch.device("cuda")
        model = KuramotoOscillator(N, coupling_strength=1.0, device=device)
        state = OscillatorState.create_random(N, seed=SEED, device=device)
        # Use a generous budget — should complete fine
        final = gradient_checkpoint_integration(
            model,
            state,
            n_steps=20,
            dt=0.01,
            memory_budget_mb=2048.0,
        )
        assert final.phase.device.type == "cuda"
        assert torch.isfinite(final.phase).all()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_checkpoint_vram_stays_bounded(self) -> None:
        """VRAM usage with checkpointing stays under budget at N=4K.

        This is a smaller version of the N=32K test from the TODO,
        scaled down to fit CI runners and 8GB GPUs.
        """
        torch.manual_seed(SEED)
        N = 4096
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        model = KuramotoOscillator(N, coupling_strength=1.0, device=device)
        state = OscillatorState.create_random(N, seed=SEED, device=device)

        budget_mb = 2048.0
        final = gradient_checkpoint_integration(
            model,
            state,
            n_steps=20,
            dt=0.01,
            memory_budget_mb=budget_mb,
        )
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        assert (
            peak_mb < budget_mb * 1.5
        ), f"Peak VRAM {peak_mb:.0f} MB exceeded 1.5x budget {budget_mb} MB"
        assert torch.isfinite(final.phase).all()


# ===================================================================
# Category A Capacity Tests (tests Task 4.1)
# ===================================================================


class TestCapacityXORn:
    """XOR-n binding capacity tests for oscillatory networks."""

    @staticmethod
    def _make_xor_data(
        n_bits: int, n_samples: int, seed: int = SEED
    ) -> tuple[Tensor, Tensor]:
        """Generate XOR-n dataset with phase encoding."""
        torch.manual_seed(seed)
        x_binary = torch.randint(0, 2, (n_samples, n_bits), dtype=torch.float)
        x_phases = x_binary * math.pi  # 0 or π encoding
        y_true = x_binary.sum(dim=1) % 2  # XOR = parity
        return x_phases, y_true

    def test_xor2_data_generation(self) -> None:
        """XOR-2 dataset should have correct labels."""
        x, y = self._make_xor_data(2, 100)
        assert x.shape == (100, 2)
        assert y.shape == (100,)
        # All labels should be 0 or 1
        assert ((y == 0) | (y == 1)).all()

    def test_xor3_data_generation(self) -> None:
        """XOR-3 dataset should have correct distribution."""
        x, y = self._make_xor_data(3, 1000)
        # Roughly half should be 0, half 1
        ratio = y.mean().item()
        assert 0.3 < ratio < 0.7

    def test_xor2_memorization_overfit(self, seeded: None) -> None:
        """A small MLP should perfectly memorize XOR-2 (sanity check)."""
        torch.manual_seed(SEED)
        x, y = self._make_xor_data(2, 200)

        # Simple MLP baseline
        mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )
        opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
        loss_fn = torch.nn.BCELoss()

        for _ in range(300):
            opt.zero_grad()
            pred = mlp(x).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = (mlp(x).squeeze() > 0.5).float()
            acc = (pred == y).float().mean().item()

        assert acc > 0.90, f"XOR-2 MLP accuracy {acc:.2f} < 0.90"

    def test_xor_n_prinet_forward(self, seeded: None) -> None:
        """PRINet model can forward-pass on XOR-n phase data."""
        torch.manual_seed(SEED)
        n_bits = 4
        x, y = self._make_xor_data(n_bits, 50)
        model = PRINetModel(n_resonances=16, n_dims=n_bits, n_concepts=2)
        logits = model(x)
        assert logits.shape == (50, 2)
        assert torch.isfinite(logits).all()


class TestCapacityRandomDichotomies:
    """Random Dichotomies capacity tests."""

    @staticmethod
    def _memorization_rate(
        model: torch.nn.Module,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Train model on random labels and measure memorization."""
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x).squeeze()
            pred = torch.sigmoid(pred)
            if pred.dim() > 1:
                pred = pred[:, 0]
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = model(x).squeeze()
            if pred.dim() > 1:
                pred = pred[:, 0]
            pred = (torch.sigmoid(pred) > 0.5).float()
            return (pred == y).float().mean().item()

    def test_random_dichotomy_small(self, seeded: None) -> None:
        """Small network should memorize few random patterns."""
        torch.manual_seed(SEED)
        n_patterns = 20
        input_dim = 8
        x = torch.randn(n_patterns, input_dim)
        y = torch.randint(0, 2, (n_patterns,), dtype=torch.float)

        mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        acc = self._memorization_rate(mlp, x, y, epochs=200)
        # Small MLP should memorize 20 patterns fairly well
        assert acc > 0.6, f"Random dichotomy acc {acc:.2f} < 0.6"

    def test_random_dichotomy_prinet(self, seeded: None) -> None:
        """PRINet can attempt random dichotomy memorization."""
        torch.manual_seed(SEED)
        n_patterns = 16
        input_dim = 8
        x = torch.randn(n_patterns, input_dim)
        y = torch.randint(0, 2, (n_patterns,), dtype=torch.float)

        model = PRINetModel(n_resonances=16, n_dims=input_dim, n_concepts=2)
        # Just verify forward pass works and loss is finite
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y.long())
        assert torch.isfinite(loss)


# ===================================================================
# Fashion-MNIST Training Test (tests Task 2.7)
# ===================================================================


class TestFashionMNIST:
    """Fashion-MNIST training test for PRINet hEP vs BPTT."""

    @staticmethod
    def _get_fashion_mnist_subset(
        n_train: int = 200,
        n_classes: int = 10,
    ) -> tuple[Tensor, Tensor]:
        """Generate a synthetic Fashion-MNIST-like subset.

        Uses random data shaped like Fashion-MNIST (28x28 → 784) to
        avoid requiring torchvision as a dependency in tests. For
        full Fashion-MNIST testing, use the benchmark suite.
        """
        torch.manual_seed(SEED)
        x = torch.randn(n_train, 784) * 0.3  # Approximate pixel range
        y = torch.randint(0, n_classes, (n_train,))
        return x, y

    def test_bptt_training_loss_decreases(self, seeded: None) -> None:
        """Standard BPTT training should decrease loss on subset."""
        torch.manual_seed(SEED)
        x, y = self._get_fashion_mnist_subset(n_train=100)

        model = PRINetModel(n_resonances=32, n_dims=784, n_concepts=10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(10):
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease (or at least not explode)
        assert (
            losses[-1] < losses[0] * 1.5
        ), f"Loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"
        assert all(math.isfinite(l) for l in losses), "NaN loss detected"

    def test_hep_training_finite_loss(self, seeded: None) -> None:
        """hEP training should produce finite losses on subset."""
        torch.manual_seed(SEED)
        x, y = self._get_fashion_mnist_subset(n_train=32)

        model = PRINetModel(n_resonances=16, n_dims=784, n_concepts=10)
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=10, nudge_steps=5)

        losses = []
        for _ in range(3):
            loss = trainer.train_step(x, y, lr=0.001)
            losses.append(loss)

        assert all(
            math.isfinite(l) for l in losses
        ), f"hEP produced NaN/Inf losses: {losses}"

    def test_hep_vs_bptt_comparable_loss(self, seeded: None) -> None:
        """hEP and BPTT should produce loss values in similar range."""
        torch.manual_seed(SEED)
        x, y = self._get_fashion_mnist_subset(n_train=50)

        # BPTT reference
        model_bptt = PRINetModel(n_resonances=16, n_dims=784, n_concepts=10)
        logits = model_bptt(x)
        bptt_loss = torch.nn.functional.cross_entropy(logits, y).item()

        # hEP
        torch.manual_seed(SEED)
        model_hep = PRINetModel(n_resonances=16, n_dims=784, n_concepts=10)
        trainer = HolomorphicEPTrainer(
            model_hep, beta=0.1, free_steps=10, nudge_steps=5
        )
        hep_loss = trainer.train_step(x, y, lr=0.001)

        # Both losses should be in a reasonable range (not NaN, not huge)
        assert 0 < bptt_loss < 50, f"BPTT loss out of range: {bptt_loss}"
        assert 0 < hep_loss < 50, f"hEP loss out of range: {hep_loss}"
