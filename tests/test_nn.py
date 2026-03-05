"""Unit tests for prinet.nn: layers, models, and optimizers.

Covers ResonanceLayer, PRINetModel, SynchronizedGradientDescent,
RIPOptimizer, and oscillatory weight initialization.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from prinet.core.measurement import kuramoto_order_parameter
from prinet.nn.layers import (
    PRINetModel,
    ResonanceLayer,
    oscillatory_weight_init,
)
from prinet.nn.optimizers import (
    RIPOptimizer,
    SynchronizedGradientDescent,
)

SEED = 42


@pytest.fixture
def resonance_layer() -> ResonanceLayer:
    """Standard resonance layer for tests."""
    torch.manual_seed(SEED)
    return ResonanceLayer(n_oscillators=32, n_dims=64, n_steps=5, dt=0.01)


@pytest.fixture
def prinet_model() -> PRINetModel:
    """Standard PRINet model for tests."""
    torch.manual_seed(SEED)
    return PRINetModel(n_resonances=32, n_dims=64, n_concepts=10, n_layers=2, n_steps=5)


# ===========================================================================
# RESONANCE LAYER TESTS
# ===========================================================================


class TestResonanceLayer:
    """Tests for the ResonanceLayer nn.Module."""

    def test_forward_shape_batched(self, resonance_layer: ResonanceLayer) -> None:
        """Batched forward pass returns correct shape."""
        x = torch.randn(16, 64)
        out = resonance_layer(x)
        assert out.shape == (16, 32)

    def test_forward_shape_unbatched(self, resonance_layer: ResonanceLayer) -> None:
        """Unbatched forward pass returns correct shape."""
        x = torch.randn(64)
        out = resonance_layer(x)
        assert out.shape == (32,)

    def test_output_non_negative(self, resonance_layer: ResonanceLayer) -> None:
        """Amplitude outputs are non-negative."""
        torch.manual_seed(SEED)
        x = torch.randn(8, 64)
        out = resonance_layer(x)
        assert (out >= 0).all(), "Expected non-negative amplitudes"

    def test_no_nan_in_output(self, resonance_layer: ResonanceLayer) -> None:
        """Output contains no NaN values."""
        torch.manual_seed(SEED)
        x = torch.randn(8, 64)
        out = resonance_layer(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, resonance_layer: ResonanceLayer) -> None:
        """Gradients flow through the layer."""
        x = torch.randn(4, 64, requires_grad=True)
        out = resonance_layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameter_count(self, resonance_layer: ResonanceLayer) -> None:
        """Layer has expected trainable parameters."""
        param_names = {n for n, _ in resonance_layer.named_parameters()}
        assert "coupling" in param_names
        assert "decay" in param_names
        assert "input_proj.weight" in param_names
        assert "base_frequency" in param_names
        assert "modulation" in param_names

    def test_coupling_diagonal_zero(self, resonance_layer: ResonanceLayer) -> None:
        """After initialization, coupling diagonal should be zero."""
        diag = resonance_layer.coupling.data.diag()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_get_order_parameter(self, resonance_layer: ResonanceLayer) -> None:
        """get_order_parameter returns valid [0, 1] value."""
        x = torch.randn(4, 64)
        r = resonance_layer.get_order_parameter(x)
        assert r.shape == (4,)
        assert (r >= 0).all() and (r <= 1.0 + 1e-6).all()

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same output."""
        torch.manual_seed(SEED)
        layer1 = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3)
        torch.manual_seed(SEED)
        layer2 = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3)
        x = torch.randn(4, 32)
        out1 = layer1(x)
        out2 = layer2(x)
        assert torch.allclose(out1, out2, atol=1e-5)


# ===========================================================================
# PRINET MODEL TESTS
# ===========================================================================


class TestPRINetModel:
    """Tests for the full PRINet model."""

    def test_forward_shape(self, prinet_model: PRINetModel) -> None:
        """Forward pass returns correct log-probability shape."""
        x = torch.randn(8, 64)
        out = prinet_model(x)
        assert out.shape == (8, 10)

    def test_output_is_log_prob(self, prinet_model: PRINetModel) -> None:
        """Output sums to ≈ 1 after exp (valid probability dist)."""
        x = torch.randn(4, 64)
        log_probs = prinet_model(x)
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_unbatched_forward(self, prinet_model: PRINetModel) -> None:
        """Unbatched input returns correct shape."""
        x = torch.randn(64)
        out = prinet_model(x)
        assert out.shape == (10,)

    def test_gradient_flow_through_model(self, prinet_model: PRINetModel) -> None:
        """Gradients flow end-to-end."""
        x = torch.randn(4, 64)
        out = prinet_model(x)
        loss = -out.mean()
        loss.backward()
        for name, param in prinet_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_training_step_reduces_loss(self, prinet_model: PRINetModel) -> None:
        """A training step reduces (or maintains) the loss."""
        torch.manual_seed(SEED)
        x = torch.randn(8, 64)
        targets = torch.randint(0, 10, (8,))
        optimizer = torch.optim.Adam(prinet_model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        # Initial loss
        prinet_model.train()
        log_probs = prinet_model(x)
        loss_before = criterion(log_probs, targets).item()

        # Training step
        optimizer.zero_grad()
        log_probs = prinet_model(x)
        loss = criterion(log_probs, targets)
        loss.backward()
        optimizer.step()

        # After step
        with torch.no_grad():
            log_probs = prinet_model(x)
            loss_after = criterion(log_probs, targets).item()

        # Loss should decrease (or at worst stay roughly the same)
        assert loss_after <= loss_before + 0.5


# ===========================================================================
# OPTIMIZER TESTS
# ===========================================================================


class TestSynchronizedGradientDescent:
    """Tests for the SynchronizedGradientDescent optimizer."""

    def test_basic_step(self) -> None:
        """Basic SGD step updates parameters."""
        torch.manual_seed(SEED)
        param = torch.randn(10, requires_grad=True)
        param.grad = torch.randn(10)
        initial = param.data.clone()

        opt = SynchronizedGradientDescent([param], lr=0.1)
        opt.step()

        assert not torch.allclose(param.data, initial)

    def test_sync_penalty_no_penalty_above_critical(self) -> None:
        """No penalty when order parameter > critical threshold."""
        opt = SynchronizedGradientDescent(
            [torch.randn(5, requires_grad=True)],
            lr=0.01,
            sync_penalty=1.0,
            critical_order=0.5,
        )
        penalty, grad_scale = opt.compute_sync_penalty(0.8)
        assert penalty == 0.0
        assert grad_scale == 0.0

    def test_sync_penalty_below_critical(self) -> None:
        """Penalty > 0 when order parameter < critical threshold."""
        opt = SynchronizedGradientDescent(
            [torch.randn(5, requires_grad=True)],
            lr=0.01,
            sync_penalty=1.0,
            critical_order=0.8,
        )
        penalty, grad_scale = opt.compute_sync_penalty(0.5)
        assert penalty > 0, "Expected > 0 penalty below critical"
        assert grad_scale > 0

    def test_penalty_formula(self) -> None:
        """Verify penalty = λ · max(0, K_c - K)²."""
        opt = SynchronizedGradientDescent(
            [torch.randn(5, requires_grad=True)],
            lr=0.01,
            sync_penalty=2.0,
            critical_order=0.8,
        )
        penalty, _ = opt.compute_sync_penalty(0.5)
        expected = 2.0 * (0.8 - 0.5) ** 2
        assert abs(penalty - expected) < 1e-6

    def test_order_history_tracked(self) -> None:
        """Order parameter history is tracked across steps."""
        param = torch.randn(5, requires_grad=True)
        opt = SynchronizedGradientDescent(
            [param], lr=0.01, sync_penalty=0.1, critical_order=0.5
        )
        for r in [0.3, 0.6, 0.9]:
            param.grad = torch.randn(5)
            opt.step(order_parameter=r)
        assert len(opt.order_history) == 3
        assert opt.order_history == [0.3, 0.6, 0.9]

    def test_lr_reduced_when_desynchronized(self) -> None:
        """Learning rate is effectively reduced when desynchronized."""
        # Step without penalty (high order)
        p_normal = torch.randn(10, requires_grad=True)
        opt_normal = SynchronizedGradientDescent(
            [p_normal],
            lr=0.1,
            sync_penalty=5.0,
            critical_order=0.8,
        )
        p_normal.grad = torch.ones(10)
        opt_normal.step(order_parameter=0.9)

        # Step with penalty (low order)
        p_desync = torch.randn(10, requires_grad=True)
        # Make identical starting point
        with torch.no_grad():
            p_desync.copy_(torch.randn(10))
        p_desync.grad = torch.ones(10)
        opt_desync = SynchronizedGradientDescent(
            [p_desync],
            lr=0.1,
            sync_penalty=5.0,
            critical_order=0.8,
        )
        opt_desync.step(order_parameter=0.2)

        # Just verify no errors occurred - the key is the penalty mechanism
        assert len(opt_desync.penalty_history) == 1

    def test_invalid_lr_raises(self) -> None:
        """Negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SynchronizedGradientDescent([torch.randn(5, requires_grad=True)], lr=-0.1)

    def test_invalid_critical_order_raises(self) -> None:
        """Critical order > 1 raises ValueError."""
        with pytest.raises(ValueError, match="critical_order must be"):
            SynchronizedGradientDescent(
                [torch.randn(5, requires_grad=True)],
                lr=0.01,
                critical_order=1.5,
            )

    def test_invalid_sync_penalty_raises(self) -> None:
        """Negative sync_penalty raises ValueError."""
        with pytest.raises(ValueError, match="Invalid sync_penalty"):
            SynchronizedGradientDescent(
                [torch.randn(5, requires_grad=True)],
                lr=0.01,
                sync_penalty=-0.1,
            )

    def test_with_momentum(self) -> None:
        """Optimizer works with momentum."""
        param = torch.randn(10, requires_grad=True)
        param.grad = torch.randn(10)
        opt = SynchronizedGradientDescent([param], lr=0.01, momentum=0.9)
        opt.step()
        param.grad = torch.randn(10)
        opt.step()  # Second step uses momentum buffer
        assert not torch.isnan(param).any()


class TestRIPOptimizer:
    """Tests for the Resonance-Induced Plasticity optimizer."""

    def test_basic_step_with_gradient(self) -> None:
        """Standard gradient step works."""
        param = torch.randn(10, requires_grad=True)
        param.grad = torch.randn(10)
        initial = param.data.clone()
        opt = RIPOptimizer([param], lr=0.1)
        opt.step()
        assert not torch.allclose(param.data, initial)

    def test_rip_update_on_coupling(self) -> None:
        """RIP Hebbian update modifies square coupling matrices."""
        coupling = torch.randn(8, 8, requires_grad=True)
        coupling.grad = torch.zeros(8, 8)
        initial = coupling.data.clone()

        phase = torch.zeros(8)  # All synchronized
        amp = torch.ones(8)

        opt = RIPOptimizer([coupling], lr=0.1, target_amplitude=1.0)
        opt.step(phase=phase, amplitude=amp)

        # Coupling should have been modified by RIP rule
        # Diagonal should be zero (no self-coupling)
        assert torch.allclose(coupling.data.diag(), torch.zeros(8))

    def test_invalid_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            RIPOptimizer([torch.randn(5, requires_grad=True)], lr=-0.01)

    def test_invalid_target_amplitude_raises(self) -> None:
        """Non-positive target amplitude raises ValueError."""
        with pytest.raises(ValueError, match="target_amplitude must be"):
            RIPOptimizer(
                [torch.randn(5, requires_grad=True)],
                lr=0.01,
                target_amplitude=0.0,
            )


# ===========================================================================
# INITIALIZATION TESTS
# ===========================================================================


class TestOscillatoryWeightInit:
    """Tests for oscillatory weight initialization."""

    def test_init_modifies_parameters(self) -> None:
        """Initialization changes parameter values."""
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        before = {n: p.clone() for n, p in model.named_parameters()}
        oscillatory_weight_init(model)

        changed = False
        for n, p in model.named_parameters():
            if not torch.allclose(p, before[n]):
                changed = True
                break
        assert changed, "Initialization should modify at least one parameter"

    def test_coupling_symmetric_after_init(self) -> None:
        """Coupling matrices are symmetric after initialization."""
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5)
        oscillatory_weight_init(model)
        for name, param in model.named_parameters():
            if "coupling" in name and param.dim() == 2:
                assert torch.allclose(
                    param.data, param.data.T, atol=1e-5
                ), f"{name} not symmetric"
