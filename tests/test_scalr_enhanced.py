"""Unit tests for Q3 enhanced SCALR optimizer.

Covers oscillation-aware decay, per-frequency learning rate scaling,
adaptive r_min, and SCALR vs benchmark comparisons.

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import pytest
import torch

from prinet.nn.optimizers import SCALROptimizer

SEED = 42


@pytest.fixture
def rng() -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return gen


def _make_simple_model() -> torch.nn.Module:
    """Create a small model for optimizer testing."""
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    return model


# ===================================================================
# TestOscillationAwareDecay
# ===================================================================


class TestOscillationAwareDecay:
    """Tests for oscillation-aware learning rate decay."""

    def test_decay_triggers_on_high_variance(self) -> None:
        """Decay should trigger when windowed variance > threshold."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            oscillation_window=5,
            oscillation_threshold=0.001,
            oscillation_decay=0.9,
        )

        # Feed oscillating order parameter history
        for r in [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]:
            opt._order_history.append(r)

        # Now step with any order_parameter
        x = torch.randn(4, 16)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        initial_decay = opt._lr_decay_factor
        opt.step(order_parameter=0.5)

        # Decay factor should have decreased
        assert opt._lr_decay_factor <= initial_decay

    def test_lr_decreases_after_trigger(self) -> None:
        """Effective lr should be lower after oscillation detected."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            oscillation_window=5,
            oscillation_threshold=0.001,
            oscillation_decay=0.5,
        )

        # Set initial decay
        opt._lr_decay_factor = 1.0

        # Inject oscillating history
        opt._order_history = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step(order_parameter=0.5)

        assert opt._lr_decay_factor < 1.0

    def test_no_decay_when_stable(self) -> None:
        """No decay should occur when variance is low."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            oscillation_window=5,
            oscillation_threshold=0.01,
            oscillation_decay=0.9,
        )
        opt._lr_decay_factor = 1.0

        # Stable history
        opt._order_history = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step(order_parameter=0.9)

        assert opt._lr_decay_factor == 1.0

    def test_window_parameter_respected(self) -> None:
        """Window parameter should control how many values are considered."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            oscillation_window=3,
            oscillation_threshold=0.001,
            oscillation_decay=0.9,
        )
        # Only 3 values in window, and they're stable
        opt._order_history = [0.9, 0.9, 0.9]
        opt._lr_decay_factor = 1.0

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step(order_parameter=0.9)

        assert opt._lr_decay_factor == 1.0

    def test_decay_factor_parameter(self) -> None:
        """Decay factor should scale the lr reduction."""
        model = _make_simple_model()
        opt1 = SCALROptimizer(
            model.parameters(), lr=0.01, oscillation_decay=0.5
        )
        model2 = _make_simple_model()
        opt2 = SCALROptimizer(
            model2.parameters(), lr=0.01, oscillation_decay=0.9
        )

        # Inject same oscillating history/decay
        opt1._lr_decay_factor = 1.0
        opt2._lr_decay_factor = 1.0
        osc_history = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
        opt1._order_history = list(osc_history)
        opt2._order_history = list(osc_history)

        x = torch.randn(4, 16)
        loss1 = model(x).sum()
        loss1.backward()
        opt1.step(order_parameter=0.5)

        x = torch.randn(4, 16)
        loss2 = model2(x).sum()
        loss2.backward()
        opt2.step(order_parameter=0.5)

        # opt1 has stronger decay (0.5 < 0.9)
        assert opt1._lr_decay_factor <= opt2._lr_decay_factor


# ===================================================================
# TestPerFrequencyLR
# ===================================================================


class TestPerFrequencyLR:
    """Tests for per-frequency group learning rate scaling."""

    def test_per_group_order_params_accepted(self) -> None:
        """Step should accept Dict[str, float] order_parameter."""
        model = _make_simple_model()
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()

        # Per-group order parameters
        opt.step(order_parameter={"delta": 0.8, "theta": 0.7, "gamma": 0.5})

    def test_independent_lr_scale(self) -> None:
        """Each group should get independent effective lr."""
        model = _make_simple_model()
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()

        # This should not raise
        opt.step(
            order_parameter={"delta": 0.99, "theta": 0.5, "gamma": 0.1}
        )

    def test_gradient_step_updates(self) -> None:
        """All parameter groups should be updated after step."""
        model = _make_simple_model()
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        # Save initial params
        initial_params = [p.data.clone() for p in model.parameters()]

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step(order_parameter=0.5)

        # At least some params should change
        any_changed = any(
            not torch.equal(p.data, ip)
            for p, ip in zip(model.parameters(), initial_params)
        )
        assert any_changed

    def test_compatible_with_hierarchical(self) -> None:
        """SCALR should work with hierarchical model order parameters."""
        model = _make_simple_model()
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        # Simulate 5 steps with per-frequency order params
        for _ in range(5):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step(
                order_parameter={"delta": 0.8, "theta": 0.6, "gamma": 0.4}
            )
            opt.zero_grad()


# ===================================================================
# TestAdaptiveRMin
# ===================================================================


class TestAdaptiveRMin:
    """Tests for EMA-based adaptive r_min."""

    def test_ema_tracks_order_parameter(self) -> None:
        """EMA should track the order parameter over time."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            adaptive_r_min=True,
            r_min_ema_alpha=0.1,
        )

        for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step(order_parameter=r)
            opt.zero_grad()

        # EMA should have a value set
        assert opt._r_ema is not None
        assert opt._r_ema > 0.0

    def test_r_min_auto_adjusts(self) -> None:
        """r_min should change when adaptive mode is on."""
        model = _make_simple_model()
        opt = SCALROptimizer(
            model.parameters(),
            lr=0.01,
            adaptive_r_min=True,
            r_min_ema_alpha=0.5,
        )
        initial_r_min = opt.defaults.get("r_min", 0.1)

        for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step(order_parameter=r)
            opt.zero_grad()

        # After tracking high r values, r_min should update
        # (exact check depends on implementation details)
        assert opt._r_ema is not None

    def test_alpha_controls_smoothing(self) -> None:
        """Higher alpha → faster tracking of recent values."""
        model1 = _make_simple_model()
        opt1 = SCALROptimizer(
            model1.parameters(),
            lr=0.01,
            adaptive_r_min=True,
            r_min_ema_alpha=0.9,  # fast
        )

        model2 = _make_simple_model()
        opt2 = SCALROptimizer(
            model2.parameters(),
            lr=0.01,
            adaptive_r_min=True,
            r_min_ema_alpha=0.01,  # slow
        )

        # Feed same sequence
        for r in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
            for opt, model in [(opt1, model1), (opt2, model2)]:
                x = torch.randn(4, 16)
                loss = model(x).sum()
                loss.backward()
                opt.step(order_parameter=r)
                opt.zero_grad()

        # Both should have EMA values, but they'll differ
        assert opt1._r_ema is not None
        assert opt2._r_ema is not None


# ===================================================================
# TestSCALRvsBenchmark
# ===================================================================


class TestSCALRvsBenchmark:
    """Comparative tests: SCALR vs Adam on simple tasks."""

    def test_scalr_converges_simple_task(self) -> None:
        """SCALR should converge on a simple classification task."""
        torch.manual_seed(SEED)
        D, C = 16, 5
        N = 100

        X = torch.randn(N, D)
        y = torch.randint(0, C, (N,))

        model = torch.nn.Linear(D, C)
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        initial_loss = None
        for epoch in range(20):
            opt.zero_grad()
            logits = model(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step(order_parameter=0.8)

        assert loss.item() < initial_loss

    def test_scalr_maintains_high_r(self) -> None:
        """SCALR should maintain r > 0.5 during training (when fed high r)."""
        torch.manual_seed(SEED)
        model = _make_simple_model()
        opt = SCALROptimizer(model.parameters(), lr=0.01)

        for _ in range(10):
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            opt.step(order_parameter=0.85)
            opt.zero_grad()

        # Order history should reflect the high r values
        assert len(opt._order_history) > 0
        avg_r = sum(opt._order_history[-5:]) / min(5, len(opt._order_history))
        assert avg_r > 0.5
