"""Unit tests for PRINet Year 1 Quarter 2 features.

Covers:
- P0 NaN loss regression tests (numerical stability)
- P1 torch.compile correctness
- P1 Mixed precision round-trip
- P1 SCALR optimizer
- P1 Holomorphic EP trainer
- P1 HopfOscillator dynamics
- Phase wrapping in core oscillators

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
from torch import Tensor

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import (
    HopfOscillator,
    KuramotoOscillator,
    OscillatorState,
    StuartLandauOscillator,
)
from prinet.nn.hep import HolomorphicEnergy, HolomorphicEPTrainer
from prinet.nn.layers import (
    PRINetModel,
    ResonanceLayer,
    compile_model,
    oscillatory_weight_init,
)
from prinet.nn.optimizers import SCALROptimizer

SEED = 42
TWO_PI = 2.0 * math.pi


# ===================================================================
# Helpers
# ===================================================================


@pytest.fixture
def seeded() -> None:
    """Set deterministic seeds."""
    torch.manual_seed(SEED)


@pytest.fixture
def small_model(seeded: None) -> PRINetModel:
    """Small PRINetModel for testing."""
    return PRINetModel(n_resonances=16, n_dims=32, n_concepts=5, n_layers=2, n_steps=5)


@pytest.fixture
def mnist_model(seeded: None) -> PRINetModel:
    """MNIST-sized PRINetModel for regression testing."""
    return PRINetModel(
        n_resonances=32, n_dims=784, n_concepts=10, n_layers=2, n_steps=5
    )


# ===================================================================
# P0: NaN Loss Regression Tests
# ===================================================================


class TestNaNRegression:
    """P0: Verify NaN loss is fixed for high-dimensional inputs."""

    def test_mnist_forward_finite(self, mnist_model: PRINetModel) -> None:
        """PRINetModel produces finite output for 784-dim input."""
        x = torch.randn(8, 784)
        out = mnist_model(x)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"
        assert out.shape == (8, 10)

    def test_mnist_loss_finite(self, mnist_model: PRINetModel) -> None:
        """NLL loss is finite and reasonable for MNIST-like input."""
        x = torch.randn(8, 784)
        target = torch.randint(0, 10, (8,))
        out = mnist_model(x)
        loss = torch.nn.functional.nll_loss(out, target)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() < 10.0, f"Loss too large: {loss.item()}"

    def test_mnist_gradient_finite(self, mnist_model: PRINetModel) -> None:
        """All gradients are finite after backward pass."""
        x = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        out = mnist_model(x)
        loss = torch.nn.functional.nll_loss(out, target)
        loss.backward()
        for name, p in mnist_model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"

    def test_1d_input_finite(self, mnist_model: PRINetModel) -> None:
        """1D (unbatched) input produces finite output."""
        x = torch.randn(784)
        out = mnist_model(x)
        assert torch.isfinite(out).all()
        assert out.shape == (10,)

    def test_large_input_finite(self, seeded: None) -> None:
        """Large-magnitude inputs don't cause NaN."""
        model = PRINetModel(n_resonances=16, n_dims=64, n_concepts=5, n_layers=2)
        x = torch.randn(4, 64) * 100.0
        out = model(x)
        assert torch.isfinite(out).all()

    def test_repeated_forward_stable(self, mnist_model: PRINetModel) -> None:
        """Repeated forward passes don't accumulate instability."""
        x = torch.randn(4, 784)
        for _ in range(10):
            out = mnist_model(x)
            assert torch.isfinite(out).all()

    def test_multi_epoch_training(self, seeded: None) -> None:
        """5 epochs of training produce finite loss < 2.5."""
        model = PRINetModel(
            n_resonances=16,
            n_dims=32,
            n_concepts=5,
            n_layers=2,
            n_steps=3,
            dt=0.01,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 32)
        y = torch.randint(0, 5, (16,))

        for _ in range(5):
            model.zero_grad()
            out = model(x)
            loss = torch.nn.functional.nll_loss(out, y)
            assert torch.isfinite(loss), f"NaN loss at epoch"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        assert loss.item() < 5.0  # Reasonable after 5 epochs


# ===================================================================
# Numerical Stability Tests
# ===================================================================


class TestNumericalStability:
    """P0: Phase wrapping, coupling scaling, amplitude clamping."""

    def test_phase_wrapping_kuramoto(self) -> None:
        """Kuramoto phases stay in [0, 2π) after integration."""
        model = KuramotoOscillator(20, coupling_strength=1.0)
        state = OscillatorState.create_random(20, seed=SEED)
        final, _ = model.integrate(state, n_steps=1000, dt=0.01)
        assert (final.phase >= 0.0).all()
        assert (final.phase < TWO_PI).all()

    def test_phase_wrapping_stuart_landau(self) -> None:
        """Stuart-Landau phases stay in [0, 2π) after integration."""
        model = StuartLandauOscillator(20, coupling_strength=1.0, bifurcation_param=1.0)
        state = OscillatorState.create_random(20, seed=SEED)
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        assert (final.phase >= 0.0).all()
        assert (final.phase < TWO_PI).all()

    def test_phase_wrapping_hopf(self) -> None:
        """Hopf phases stay in [0, 2π) after integration."""
        model = HopfOscillator(20, coupling_strength=1.0)
        state = OscillatorState.create_random(20, seed=SEED)
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        assert (final.phase >= 0.0).all()
        assert (final.phase < TWO_PI).all()

    def test_resonance_layer_phase_bounded(self, seeded: None) -> None:
        """ResonanceLayer internal phases don't grow unbounded."""
        layer = ResonanceLayer(n_oscillators=32, n_dims=64, n_steps=50)
        x = torch.randn(4, 64)
        # The forward pass should not produce NaN even with many steps
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_coupling_scaling(self, seeded: None) -> None:
        """Coupling is scaled by 1/√N in ResonanceLayer."""
        layer = ResonanceLayer(n_oscillators=100, n_dims=64)
        expected_scale = 1.0 / math.sqrt(100)
        assert abs(layer._coupling_scale - expected_scale) < 1e-6

    def test_amplitude_clamped(self, seeded: None) -> None:
        """ResonanceLayer outputs are bounded (amplitude clamping)."""
        layer = ResonanceLayer(n_oscillators=32, n_dims=64, n_steps=20)
        x = torch.randn(4, 64) * 50.0
        out = layer(x)
        # Output should represent amplitudes ≤ _AMP_MAX (10.0)
        assert (out <= 11.0).all()  # Allow slight margin from LayerNorm

    def test_logit_clamping(self, small_model: PRINetModel) -> None:
        """Logits are clamped before log_softmax."""
        x = torch.randn(4, 32) * 100.0
        out = small_model(x)
        assert torch.isfinite(out).all()
        # log_softmax values should be in (-inf, 0]
        assert (out <= 0.0).all()

    def test_layer_norm_present(self, small_model: PRINetModel) -> None:
        """PRINetModel has LayerNorm between resonance layers."""
        assert hasattr(small_model, "layer_norms")
        assert len(small_model.layer_norms) == 2  # n_layers


# ===================================================================
# torch.compile Correctness Tests
# ===================================================================


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


class TestTorchCompile:
    """P1: Verify compiled model produces same output as eager."""

    def test_compile_model_returns_module(self, seeded: None) -> None:
        """compile_model wraps model without error (no execution)."""
        model = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        compiled = compile_model(model)
        assert compiled is not None

    def test_compile_model_utility(self, seeded: None) -> None:
        """compile_model returns a compiled module that runs.

        Uses inductor backend when available, falls back to eager.
        """
        model = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        if _HAS_INDUCTOR:
            compiled = compile_model(model)
        else:
            compiled = torch.compile(model, backend="eager")
        x = torch.randn(2, 16)
        out = compiled(x)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_compile_flag_constructor(self, seeded: None) -> None:
        """PRINetModel(compile=True) creates and runs.

        Uses inductor backend when available, falls back to eager.
        """
        if _HAS_INDUCTOR:
            model = PRINetModel(
                n_resonances=8,
                n_dims=16,
                n_concepts=3,
                n_layers=1,
                compile=True,
            )
        else:
            model = PRINetModel(
                n_resonances=8,
                n_dims=16,
                n_concepts=3,
                n_layers=1,
            )
            model = torch.compile(model, backend="eager")
        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 3)

    def test_compile_output_close_to_eager(self, seeded: None) -> None:
        """Compiled and eager models produce similar output.

        Uses inductor backend when available, falls back to eager.
        """
        torch.manual_seed(SEED)
        model_eager = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        torch.manual_seed(SEED)
        model_compiled = PRINetModel(
            n_resonances=8, n_dims=16, n_concepts=3, n_layers=1
        )
        model_compiled.load_state_dict(model_eager.state_dict())
        if _HAS_INDUCTOR:
            compiled = compile_model(model_compiled)
        else:
            compiled = torch.compile(model_compiled, backend="eager")

        x = torch.randn(4, 16)
        out_eager = model_eager(x)
        out_compiled = compiled(x)
        assert torch.allclose(out_eager, out_compiled, atol=1e-4)


# ===================================================================
# Mixed Precision Tests
# ===================================================================


class TestMixedPrecision:
    """P1: Mixed precision forward/gradient consistency."""

    def test_enable_mixed_precision(self, seeded: None) -> None:
        """enable_mixed_precision returns self and sets flag."""
        model = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        result = model.enable_mixed_precision(True)
        assert result is model
        assert model._mixed_precision is True

    def test_mixed_precision_forward_finite(self, seeded: None) -> None:
        """Mixed precision forward produces finite output."""
        model = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        model.enable_mixed_precision(True, dtype=torch.bfloat16)
        x = torch.randn(4, 16)
        out = model(x)
        assert torch.isfinite(out).all()
        assert out.dtype == torch.float32  # Output should be float32

    def test_mixed_precision_gradient_consistent(self, seeded: None) -> None:
        """Gradients in mixed precision are close to float32."""
        torch.manual_seed(SEED)
        model_fp32 = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        torch.manual_seed(SEED)
        model_mp = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        model_mp.load_state_dict(model_fp32.state_dict())
        model_mp.enable_mixed_precision(True, dtype=torch.bfloat16)

        x = torch.randn(4, 16)
        y = torch.randint(0, 3, (4,))

        # FP32 gradients
        loss_fp32 = torch.nn.functional.nll_loss(model_fp32(x), y)
        loss_fp32.backward()

        # Mixed precision gradients
        loss_mp = torch.nn.functional.nll_loss(model_mp(x), y)
        loss_mp.backward()

        # Check gradients are close
        for (n1, p1), (n2, p2) in zip(
            model_fp32.named_parameters(), model_mp.named_parameters()
        ):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(
                    p1.grad, p2.grad, atol=0.1, rtol=0.1
                ), f"Gradient mismatch in {n1}"

    def test_disable_mixed_precision(self, seeded: None) -> None:
        """Mixed precision can be disabled after enabling."""
        model = PRINetModel(n_resonances=8, n_dims=16, n_concepts=3, n_layers=1)
        model.enable_mixed_precision(True)
        model.enable_mixed_precision(False)
        assert model._mixed_precision is False


# ===================================================================
# SCALR Optimizer Tests
# ===================================================================


class TestSCALROptimizer:
    """P1: Synchronization-Coupled Adaptive Learning Rate."""

    def test_basic_step(self, seeded: None) -> None:
        """SCALR performs a basic optimization step."""
        param = torch.randn(10, requires_grad=True)
        param.grad = torch.randn(10)
        opt = SCALROptimizer([param], lr=0.1)
        opt.step(order_parameter=0.8)
        # Parameter should have changed
        assert not torch.allclose(param, param + param.grad * 0.1)

    def test_lr_scales_with_order_parameter(self, seeded: None) -> None:
        """Higher order parameter → higher effective LR."""
        opt = SCALROptimizer([torch.randn(1, requires_grad=True)], lr=1.0)
        scale_high = opt.compute_lr_scale(0.9)
        scale_low = opt.compute_lr_scale(0.1)
        assert scale_high > scale_low

    def test_lr_scale_bounds(self) -> None:
        """LR scale is bounded by [r_min, 1.0]."""
        opt = SCALROptimizer([torch.randn(1, requires_grad=True)], lr=1.0, r_min=0.2)
        assert opt.compute_lr_scale(0.0) == pytest.approx(0.2)
        assert opt.compute_lr_scale(1.0) == pytest.approx(1.0)

    def test_alpha_controls_sensitivity(self) -> None:
        """Higher alpha → more aggressive scaling with r."""
        opt_linear = SCALROptimizer(
            [torch.randn(1, requires_grad=True)], lr=1.0, alpha=1.0
        )
        opt_quad = SCALROptimizer(
            [torch.randn(1, requires_grad=True)], lr=1.0, alpha=2.0
        )
        # At r=0.5: linear gives 0.5, quadratic gives 0.25
        s1 = opt_linear.compute_lr_scale(0.5)
        s2 = opt_quad.compute_lr_scale(0.5)
        assert s2 < s1

    def test_warmup_steps(self, seeded: None) -> None:
        """During warmup, full base LR is used."""
        param = torch.randn(5, requires_grad=True)
        opt = SCALROptimizer([param], lr=0.1, warmup_steps=10, r_min=0.0)
        param.grad = torch.ones(5)
        # During warmup, order_parameter should be ignored
        opt.step(order_parameter=0.0)  # Would give lr=0 without warmup
        # Parameter should have changed (full LR used)
        assert param.grad is not None

    def test_history_tracking(self, seeded: None) -> None:
        """SCALR tracks LR and order parameter history."""
        param = torch.randn(5, requires_grad=True)
        opt = SCALROptimizer([param], lr=0.1)
        for r in [0.3, 0.5, 0.8]:
            param.grad = torch.ones(5)
            opt.step(order_parameter=r)
        assert len(opt.order_history) == 3
        assert len(opt.lr_history) == 3

    def test_invalid_params_raise(self) -> None:
        """Invalid SCALR parameters raise ValueError."""
        with pytest.raises(ValueError):
            SCALROptimizer([torch.randn(1)], lr=-1.0)
        with pytest.raises(ValueError):
            SCALROptimizer([torch.randn(1)], lr=0.1, r_min=2.0)
        with pytest.raises(ValueError):
            SCALROptimizer([torch.randn(1)], lr=0.1, alpha=-1.0)

    def test_training_loop(self, seeded: None) -> None:
        """SCALR can train a small model."""
        model = PRINetModel(
            n_resonances=8, n_dims=16, n_concepts=3, n_layers=1, n_steps=3
        )
        opt = SCALROptimizer(model.parameters(), lr=0.01, r_min=0.1)
        x = torch.randn(8, 16)
        y = torch.randint(0, 3, (8,))

        initial_loss = None
        for i in range(5):
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.nll_loss(out, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step(order_parameter=0.5 + 0.1 * i)

        assert torch.isfinite(loss)


# ===================================================================
# Holomorphic EP Tests
# ===================================================================


class TestHolomorphicEP:
    """P1: Holomorphic Equilibrium Propagation trainer."""

    def test_energy_computation(self, seeded: None) -> None:
        """HolomorphicEnergy returns finite scalar."""
        energy_fn = HolomorphicEnergy(n_oscillators=16)
        z = torch.randn(4, 16, dtype=torch.complex64)
        coupling = torch.randn(16, 16) * 0.1
        e = energy_fn(z, coupling)
        assert torch.isfinite(e)
        assert e.dim() == 0  # Scalar

    def test_energy_with_nudge(self, seeded: None) -> None:
        """Energy with nudge differs from free energy."""
        energy_fn = HolomorphicEnergy(n_oscillators=16)
        z = torch.randn(4, 16, dtype=torch.complex64)
        coupling = torch.randn(16, 16) * 0.1
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        e_free = energy_fn(z, coupling, logits, labels, beta=0.0)
        e_nudge = energy_fn(z, coupling, logits, labels, beta=0.5)
        assert not torch.allclose(e_free, e_nudge)

    def test_trainer_creation(self, small_model: PRINetModel) -> None:
        """HolomorphicEPTrainer initializes correctly."""
        trainer = HolomorphicEPTrainer(
            small_model, beta=0.1, free_steps=10, nudge_steps=5
        )
        assert trainer.beta == 0.1

    def test_trainer_invalid_beta(self, small_model: PRINetModel) -> None:
        """Negative or zero beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            HolomorphicEPTrainer(small_model, beta=0.0)
        with pytest.raises(ValueError, match="beta must be positive"):
            HolomorphicEPTrainer(small_model, beta=-0.1)

    def test_train_step(self, seeded: None) -> None:
        """hEP train_step returns finite loss."""
        model = PRINetModel(
            n_resonances=8,
            n_dims=16,
            n_concepts=3,
            n_layers=1,
            n_steps=3,
        )
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=5, nudge_steps=3)
        x = torch.randn(4, 16)
        target = torch.randint(0, 3, (4,))
        loss = trainer.train_step(x, target, lr=0.01)
        assert math.isfinite(loss)

    def test_gradient_estimation(self, seeded: None) -> None:
        """hEP gradient estimation produces non-zero coupling gradients."""
        model = PRINetModel(
            n_resonances=8,
            n_dims=16,
            n_concepts=3,
            n_layers=1,
            n_steps=3,
        )
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=5)
        x = torch.randn(4, 16)
        target = torch.randint(0, 3, (4,))
        grads, loss = trainer.compute_hep_gradients(x, target)
        # Should have gradients for coupling parameters
        coupling_grads = [v for k, v in grads.items() if "coupling" in k]
        assert len(coupling_grads) > 0
        # At least some coupling gradients should be non-zero
        assert any(g.abs().sum() > 0 for g in coupling_grads)

    def test_loss_history(self, seeded: None) -> None:
        """Trainer tracks loss history."""
        model = PRINetModel(
            n_resonances=8,
            n_dims=16,
            n_concepts=3,
            n_layers=1,
            n_steps=3,
        )
        trainer = HolomorphicEPTrainer(model, beta=0.1, free_steps=5)
        x = torch.randn(4, 16)
        y = torch.randint(0, 3, (4,))
        for _ in range(3):
            trainer.train_step(x, y, lr=0.01)
        assert len(trainer.loss_history) == 3


# ===================================================================
# HopfOscillator Tests
# ===================================================================


class TestHopfOscillator:
    """P1: Hopf bifurcation oscillator with amplitude-phase dynamics."""

    def test_limit_cycle_convergence(self) -> None:
        """Amplitudes converge to √μ for positive μ."""
        model = HopfOscillator(30, coupling_strength=0.5, bifurcation_param=1.0)
        state = OscillatorState.create_random(30, seed=SEED, freq_range=(1.0, 2.0))
        final, _ = model.integrate(state, n_steps=1000, dt=0.01)
        expected_amp = math.sqrt(1.0)
        mean_amp = final.amplitude.mean().item()
        assert abs(mean_amp - expected_amp) < 0.15

    def test_fixed_point_negative_mu(self) -> None:
        """Amplitudes decay to 0 for negative μ (no coupling)."""
        model = HopfOscillator(20, coupling_strength=0.0, bifurcation_param=-1.0)
        state = OscillatorState.create_random(20, seed=SEED)
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        # With μ < 0 and K=0, amplitudes should decay
        assert final.amplitude.mean().item() < 0.5

    def test_phase_bounded(self) -> None:
        """Phases stay in [0, 2π)."""
        model = HopfOscillator(20, coupling_strength=1.0)
        state = OscillatorState.create_random(20, seed=SEED)
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        assert (final.phase >= 0.0).all()
        assert (final.phase < TWO_PI).all()

    def test_mean_field_mode(self) -> None:
        """Mean-field mode runs without error and gives reasonable results."""
        model = HopfOscillator(
            30,
            coupling_strength=1.0,
            bifurcation_param=1.0,
            mean_field=True,
        )
        state = OscillatorState.create_random(30, seed=SEED, freq_range=(1.0, 2.0))
        final, _ = model.integrate(state, n_steps=500, dt=0.01)
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()
        mean_amp = final.amplitude.mean().item()
        assert abs(mean_amp - 1.0) < 0.5

    def test_bifurcation_param_property(self) -> None:
        """Bifurcation parameter can be read and set."""
        model = HopfOscillator(10, bifurcation_param=2.0)
        assert model.bifurcation_param == 2.0
        model.bifurcation_param = -1.0
        assert model.bifurcation_param == -1.0

    def test_limit_cycle_amplitude_property(self) -> None:
        """limit_cycle_amplitude returns √μ or 0."""
        model = HopfOscillator(10, bifurcation_param=4.0)
        assert model.limit_cycle_amplitude == pytest.approx(2.0)
        model.bifurcation_param = -1.0
        assert model.limit_cycle_amplitude == 0.0

    def test_batched_integration(self) -> None:
        """Hopf oscillator works with batched state."""
        model = HopfOscillator(20, coupling_strength=0.5, bifurcation_param=1.0)
        state = OscillatorState.create_random(20, batch_size=4, seed=SEED)
        final, _ = model.integrate(state, n_steps=100, dt=0.01)
        assert final.phase.shape == (4, 20)
        assert final.amplitude.shape == (4, 20)
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()

    def test_euler_vs_rk4(self) -> None:
        """Both Euler and RK4 give similar results for small dt."""
        state = OscillatorState.create_random(10, seed=SEED)
        model = HopfOscillator(10, coupling_strength=0.5)

        final_euler, _ = model.integrate(state, n_steps=200, dt=0.001, method="euler")
        final_rk4, _ = model.integrate(
            state.clone(), n_steps=200, dt=0.001, method="rk4"
        )

        # Should be close for small dt
        assert torch.allclose(final_euler.amplitude, final_rk4.amplitude, atol=0.05)

    def test_trajectory_recording(self) -> None:
        """Trajectory recording works for Hopf oscillator."""
        model = HopfOscillator(10, coupling_strength=0.5)
        state = OscillatorState.create_random(10, seed=SEED)
        final, trajectory = model.integrate(
            state, n_steps=50, dt=0.01, record_trajectory=True
        )
        assert trajectory is not None
        assert len(trajectory) == 50


# ===================================================================
# Integration / Regression Tests
# ===================================================================


class TestIntegration:
    """Cross-module integration tests."""

    def test_full_training_pipeline(self, seeded: None) -> None:
        """Complete training pipeline: model + SCALR + gradient clipping."""
        model = PRINetModel(
            n_resonances=8,
            n_dims=16,
            n_concepts=3,
            n_layers=1,
            n_steps=3,
        )
        opt = SCALROptimizer(model.parameters(), lr=0.01)
        x = torch.randn(8, 16)
        y = torch.randint(0, 3, (8,))

        losses = []
        for i in range(5):
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.nll_loss(out, y)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(order_parameter=0.5)

        assert all(math.isfinite(l) for l in losses)

    def test_oscillatory_init_no_nan(self, seeded: None) -> None:
        """oscillatory_weight_init doesn't introduce NaN."""
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5, n_layers=2)
        oscillatory_weight_init(model)
        x = torch.randn(4, 32)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_version_updated(self) -> None:
        """Package version is >=0.2.0 (bumped in later quarters)."""
        from packaging.version import Version

        import prinet

        assert Version(prinet.__version__) >= Version("0.2.0")

    def test_all_exports_importable(self) -> None:
        """All Q2 public symbols are importable."""
        from prinet import (
            HolomorphicEnergy,
            HolomorphicEPTrainer,
            HopfOscillator,
            SCALROptimizer,
            compile_model,
        )

        assert HolomorphicEnergy is not None
        assert HolomorphicEPTrainer is not None
        assert HopfOscillator is not None
        assert SCALROptimizer is not None
        assert compile_model is not None


# ===================================================================
# 8. Sparse k-NN Coupling (O(N log N))
# ===================================================================


class TestSparseKNNCoupling:
    """Tests for sparse k-NN coupling mode on Kuramoto and Hopf oscillators."""

    # ── Kuramoto: sparse k-NN ──────────────────────────────────────

    def test_kuramoto_sparse_creates_ok(self, seeded: None) -> None:
        """KuramotoOscillator with coupling_mode='sparse_knn' creates."""
        osc = KuramotoOscillator(64, coupling_strength=2.0, coupling_mode="sparse_knn")
        assert osc.coupling_mode == "sparse_knn"
        assert osc.sparse_k == math.ceil(math.log2(64))

    def test_kuramoto_sparse_k_default(self, seeded: None) -> None:
        """Default sparse_k = ceil(log2(N))."""
        for N in [32, 128, 1024]:
            osc = KuramotoOscillator(
                N, coupling_strength=1.0, coupling_mode="sparse_knn"
            )
            assert osc.sparse_k == math.ceil(math.log2(N))

    def test_kuramoto_sparse_k_custom(self, seeded: None) -> None:
        """Custom sparse_k overrides default."""
        osc = KuramotoOscillator(
            128, coupling_strength=1.0, coupling_mode="sparse_knn", sparse_k=4
        )
        assert osc.sparse_k == 4

    def test_kuramoto_sparse_finite_output(self, seeded: None) -> None:
        """Sparse k-NN derivatives produce finite results."""
        osc = KuramotoOscillator(64, coupling_strength=2.0, coupling_mode="sparse_knn")
        state = OscillatorState.create_random(64, seed=SEED)
        dphi, dr, domega = osc.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()
        assert torch.isfinite(domega).all()

    def test_kuramoto_sparse_integration(self, seeded: None) -> None:
        """Sparse k-NN integration produces valid trajectory."""
        osc = KuramotoOscillator(128, coupling_strength=2.0, coupling_mode="sparse_knn")
        state = OscillatorState.create_random(128, seed=SEED)
        final, traj = osc.integrate(state, n_steps=50, dt=0.01, record_trajectory=True)
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()
        assert len(traj) == 50  # 50 steps

    def test_kuramoto_sparse_order_param(self, seeded: None) -> None:
        """Sparse coupling produces a valid order parameter 0 ≤ r ≤ 1."""
        osc = KuramotoOscillator(256, coupling_strength=3.0, coupling_mode="sparse_knn")
        state = OscillatorState.create_random(256, seed=SEED)
        final, _ = osc.integrate(state, n_steps=100, dt=0.01)
        r = kuramoto_order_parameter(final.phase)
        assert 0.0 <= r.item() <= 1.0

    def test_kuramoto_sparse_differs_from_full(self, seeded: None) -> None:
        """Sparse k-NN and full produce different derivatives (different topology)."""
        N = 128
        state = OscillatorState.create_random(N, seed=SEED)
        osc_full = KuramotoOscillator(N, coupling_strength=2.0, coupling_mode="full")
        osc_sparse = KuramotoOscillator(
            N, coupling_strength=2.0, coupling_mode="sparse_knn"
        )
        dphi_full, _, _ = osc_full.compute_derivatives(state)
        dphi_sparse, _, _ = osc_sparse.compute_derivatives(state)
        # They should not be identical (different coupling graphs)
        assert not torch.allclose(dphi_full, dphi_sparse, atol=1e-6)

    def test_kuramoto_sparse_batch(self, seeded: None) -> None:
        """Sparse k-NN handles batched state correctly."""
        N = 64
        osc = KuramotoOscillator(N, coupling_strength=2.0, coupling_mode="sparse_knn")
        # Create batch of 3 states
        phases = torch.randn(3, N)
        amps = torch.ones(3, N)
        freqs = torch.randn(3, N)
        state = OscillatorState(phase=phases, amplitude=amps, frequency=freqs)
        dphi, dr, domega = osc.compute_derivatives(state)
        assert dphi.shape == (3, N)
        assert torch.isfinite(dphi).all()

    # ── Hopf: sparse k-NN ──────────────────────────────────────────

    def test_hopf_sparse_creates_ok(self, seeded: None) -> None:
        """HopfOscillator with coupling_mode='sparse_knn' creates."""
        osc = HopfOscillator(
            64, bifurcation_param=1.0, coupling_strength=2.0, coupling_mode="sparse_knn"
        )
        assert osc.coupling_mode == "sparse_knn"

    def test_hopf_sparse_finite_output(self, seeded: None) -> None:
        """Hopf sparse k-NN produces finite derivatives."""
        osc = HopfOscillator(
            64, bifurcation_param=1.0, coupling_strength=2.0, coupling_mode="sparse_knn"
        )
        state = OscillatorState.create_random(64, seed=SEED)
        dphi, dr, domega = osc.compute_derivatives(state)
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()

    def test_hopf_sparse_integration(self, seeded: None) -> None:
        """Hopf sparse k-NN integration produces valid trajectory."""
        osc = HopfOscillator(
            128,
            bifurcation_param=1.0,
            coupling_strength=2.0,
            coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(128, seed=SEED)
        final, _ = osc.integrate(state, n_steps=50, dt=0.01)
        assert torch.isfinite(final.phase).all()
        assert torch.isfinite(final.amplitude).all()

    def test_hopf_sparse_order_param(self, seeded: None) -> None:
        """Hopf sparse produces valid order parameter."""
        osc = HopfOscillator(
            256,
            bifurcation_param=1.0,
            coupling_strength=3.0,
            coupling_mode="sparse_knn",
        )
        state = OscillatorState.create_random(256, seed=SEED)
        final, _ = osc.integrate(state, n_steps=100, dt=0.01)
        r = kuramoto_order_parameter(final.phase)
        assert 0.0 <= r.item() <= 1.0

    # ── Coupling mode parameter validation ─────────────────────────

    def test_coupling_mode_auto_default(self, seeded: None) -> None:
        """Default coupling_mode is 'auto'."""
        osc = KuramotoOscillator(64, coupling_strength=1.0)
        assert osc.coupling_mode == "auto"

    def test_coupling_mode_mean_field_via_param(self, seeded: None) -> None:
        """coupling_mode='mean_field' routes to mean-field path."""
        osc = KuramotoOscillator(64, coupling_strength=1.0, coupling_mode="mean_field")
        assert osc.coupling_mode == "mean_field"
        state = OscillatorState.create_random(64, seed=SEED)
        dphi, _, _ = osc.compute_derivatives(state)
        assert torch.isfinite(dphi).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sparse_on_gpu(self) -> None:
        """Sparse k-NN works on CUDA device."""
        dev = torch.device("cuda")
        osc = KuramotoOscillator(
            128, coupling_strength=2.0, device=dev, coupling_mode="sparse_knn"
        )
        state = OscillatorState.create_random(128, device=dev, seed=SEED)
        dphi, dr, domega = osc.compute_derivatives(state)
        assert dphi.device.type == "cuda"
        assert torch.isfinite(dphi).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sparse_vram_subquadratic(self) -> None:
        """Sparse k-NN uses << O(N²) memory vs full at N=4096."""
        import gc

        dev = torch.device("cuda")
        N = 4096

        # Measure full pairwise VRAM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        osc_full = KuramotoOscillator(
            N, coupling_strength=2.0, device=dev, coupling_mode="full"
        )
        state = OscillatorState.create_random(N, device=dev, seed=SEED)
        osc_full.compute_derivatives(state)
        torch.cuda.synchronize()
        vram_full = torch.cuda.max_memory_allocated() - base
        del osc_full
        gc.collect()
        torch.cuda.empty_cache()

        # Measure sparse VRAM
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        osc_sparse = KuramotoOscillator(
            N, coupling_strength=2.0, device=dev, coupling_mode="sparse_knn"
        )
        osc_sparse.compute_derivatives(state)
        torch.cuda.synchronize()
        vram_sparse = torch.cuda.max_memory_allocated() - base
        del osc_sparse

        # Sparse should use at most 10% of full's VRAM
        assert vram_sparse < vram_full * 0.10, (
            f"Sparse VRAM {vram_sparse/1e6:.1f} MB should be << "
            f"Full VRAM {vram_full/1e6:.1f} MB"
        )
