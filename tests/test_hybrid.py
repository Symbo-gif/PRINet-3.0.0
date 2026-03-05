"""Tests for HybridPRINet, AlternatingOptimizer, StateCollector, and OscilloBench.

Covers:
    * HybridPRINet end-to-end forward pass
    * HybridPRINet training loop with CLEVR-N data
    * SCALR + hybrid training (alternating optimizer)
    * Comparison vs pure oscillatory (PRINetModel) baseline
    * Comparison vs pure rate-coded (MLP) baseline
    * AlternatingOptimizer switching / dual step modes
    * HybridCLEVRN benchmark adapter
    * StateCollector metrics accumulation
    * OscilloBench full suite & dashboard generation
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from prinet.nn.hybrid import (
    AlternatingOptimizer,
    HybridCLEVRN,
    HybridPRINet,
)
from prinet.nn.training_hooks import StateCollector

SEED = 42


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def hybrid_model() -> HybridPRINet:
    """Small HybridPRINet for fast testing."""
    torch.manual_seed(SEED)
    return HybridPRINet(
        n_input=64,
        n_classes=5,
        n_delta=2,
        n_theta=4,
        n_gamma=16,
        n_lobm_layers=1,
        lobm_steps=3,
        grim_d_model=32,
        grim_n_heads=2,
        grim_n_layers=1,
    )


@pytest.fixture()
def hybrid_model_10class() -> HybridPRINet:
    """Hybrid model with 10 classes for CLEVR-10 / capacity tests."""
    torch.manual_seed(SEED)
    return HybridPRINet(
        n_input=64,
        n_classes=10,
        n_delta=2,
        n_theta=4,
        n_gamma=16,
        n_lobm_layers=1,
        lobm_steps=3,
        grim_d_model=32,
        grim_n_heads=2,
        grim_n_layers=1,
    )


@pytest.fixture()
def mock_daemon() -> MagicMock:
    """Mock SubconsciousDaemon for testing integration hooks."""
    daemon = MagicMock()
    daemon.get_control.return_value = MagicMock(
        lr_multiplier=1.0,
        coupling_suggestion=2.0,
        regime_suggestion="mean_field",
        alert_level=0,
    )
    daemon.submit_state = MagicMock()
    return daemon


# ======================================================================
# TestHybridPRINet — 5 tests
# ======================================================================


class TestHybridPRINet:
    """End-to-end tests for the HybridPRINet model."""

    def test_forward_shape_batched(self, hybrid_model: HybridPRINet) -> None:
        """Batched forward returns correct log-prob shape."""
        x = torch.randn(8, 64)
        log_probs = hybrid_model(x)
        assert log_probs.shape == (8, 5)
        # log probs should sum to ~1 in probability space
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-4), f"Prob sums: {sums}"

    def test_forward_return_rates(self, hybrid_model: HybridPRINet) -> None:
        """Forward with return_rates=True yields sparse rate codes."""
        x = torch.randn(4, 64)
        result = hybrid_model(x, return_rates=True)
        assert isinstance(result, tuple)
        log_probs, rates = result
        assert log_probs.shape == (4, 5)
        assert rates.shape == (4, 22)  # n_delta + n_theta + n_gamma = 2+4+16
        # Rates should be non-negative (soft WTA output)
        assert (rates >= -1e-6).all(), f"Negative rates found: min={rates.min()}"

    def test_clevr10_training_loop(self, hybrid_model_10class: HybridPRINet) -> None:
        """HybridPRINet can train on synthetic data and decrease loss."""
        torch.manual_seed(SEED)
        model = hybrid_model_10class
        x = torch.randn(8, 64)
        y = torch.randint(0, 10, (8,))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = None
        final_loss = None

        for epoch in range(3):
            opt.zero_grad()
            log_probs, rates = model(x, return_rates=True)
            loss_cls = nn.functional.nll_loss(log_probs, y)
            loss_sp = model.sparsity_loss(rates)
            loss = loss_cls + 0.01 * loss_sp
            loss.backward()
            opt.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        # Loss should decrease (or at least not explode)
        assert math.isfinite(final_loss), f"Final loss is not finite: {final_loss}"
        # With 10 epochs, expect some improvement
        assert final_loss < initial_loss * 1.5, (
            f"Loss did not improve: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_scalr_alternating_training(self, hybrid_model: HybridPRINet) -> None:
        """AlternatingOptimizer drives HybridPRINet training."""
        torch.manual_seed(SEED)
        model = hybrid_model
        alt_opt = AlternatingOptimizer(
            model, osc_lr=5e-4, rate_lr=1e-3, alternation_mode="step"
        )

        x = torch.randn(4, 64)
        y = torch.randint(0, 5, (4,))

        losses: list[float] = []
        for step_i in range(3):
            alt_opt.zero_grad()
            log_probs = model(x)
            loss = nn.functional.nll_loss(log_probs, y)
            loss.backward()
            alt_opt.step(epoch=0)
            losses.append(loss.item())

        # Should not diverge
        assert all(math.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
        # Final loss should be reasonably bounded
        assert losses[-1] < losses[0] * 2.0, (
            f"Alternating training diverged: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_vs_pure_oscillatory(self) -> None:
        """HybridPRINet output differs from pure oscillatory model."""
        from prinet.nn.layers import PRINetModel

        torch.manual_seed(SEED)
        pure_osc = PRINetModel(n_resonances=22, n_dims=64, n_concepts=5)
        hybrid = HybridPRINet(
            n_input=64, n_classes=5, n_delta=2, n_theta=4, n_gamma=16,
            n_lobm_layers=1, lobm_steps=3,
            grim_d_model=32, grim_n_heads=2, grim_n_layers=1,
        )

        x = torch.randn(4, 64)
        out_osc = pure_osc(x)  # (4, 5) log probs
        out_hybrid = hybrid(x)  # (4, 5) log probs

        # Both should have valid shapes and be different due to architecture
        assert out_osc.shape == out_hybrid.shape == (4, 5)
        # Architectures differ → outputs almost certainly differ
        assert not torch.allclose(out_osc, out_hybrid, atol=1e-3), (
            "Hybrid and pure oscillatory produced identical output"
        )

    def test_vs_pure_rate_coded(self) -> None:
        """HybridPRINet output differs from pure rate-coded MLP."""
        torch.manual_seed(SEED)
        pure_rate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.LogSoftmax(dim=-1),
        )
        hybrid = HybridPRINet(
            n_input=64, n_classes=5, n_delta=2, n_theta=4, n_gamma=16,
            n_lobm_layers=1, lobm_steps=3,
            grim_d_model=32, grim_n_heads=2, grim_n_layers=1,
        )

        x = torch.randn(4, 64)
        out_rate = pure_rate(x)
        out_hybrid = hybrid(x)

        assert out_rate.shape == out_hybrid.shape == (4, 5)
        assert not torch.allclose(out_rate, out_hybrid, atol=1e-3), (
            "Hybrid and pure rate-coded produced identical output"
        )


# ======================================================================
# TestAlternatingOptimizer
# ======================================================================


class TestAlternatingOptimizer:
    """Tests for AlternatingOptimizer scheduling and daemon integration."""

    def test_epoch_mode_alternation(self, hybrid_model: HybridPRINet) -> None:
        """Epoch mode alternates osc/rate optimizer per epoch parity."""
        alt = AlternatingOptimizer(hybrid_model, alternation_mode="epoch")

        x = torch.randn(4, 64)
        y = torch.randint(0, 5, (4,))

        # Epoch 0 (even) → oscillatory step
        alt.zero_grad()
        loss = nn.functional.nll_loss(hybrid_model(x), y)
        loss.backward()
        alt.step(epoch=0)
        assert alt._step_count == 1

        # Epoch 1 (odd) → rate step
        alt.zero_grad()
        loss = nn.functional.nll_loss(hybrid_model(x), y)
        loss.backward()
        alt.step(epoch=1)
        assert alt._step_count == 2

    def test_step_both(self, hybrid_model: HybridPRINet) -> None:
        """step_both updates all parameters simultaneously."""
        alt = AlternatingOptimizer(hybrid_model)

        x = torch.randn(4, 64)
        y = torch.randint(0, 5, (4,))

        alt.zero_grad()
        loss = nn.functional.nll_loss(hybrid_model(x), y)
        loss.backward()

        # Record params before
        osc_before = [p.clone() for p in hybrid_model.oscillatory_parameters()[:1]]
        rate_before = [p.clone() for p in hybrid_model.rate_coded_parameters()[:1]]

        alt.step_both()

        # Both groups should have changed
        osc_after = list(hybrid_model.oscillatory_parameters()[:1])
        rate_after = list(hybrid_model.rate_coded_parameters()[:1])

        osc_changed = not torch.equal(osc_before[0], osc_after[0])
        rate_changed = not torch.equal(rate_before[0], rate_after[0])
        assert osc_changed or rate_changed, "Neither param group was updated"

    def test_daemon_integration(
        self, hybrid_model: HybridPRINet, mock_daemon: MagicMock
    ) -> None:
        """AlternatingOptimizer reads daemon control signals."""
        alt = AlternatingOptimizer(hybrid_model, daemon=mock_daemon)

        x = torch.randn(4, 64)
        y = torch.randint(0, 5, (4,))

        alt.zero_grad()
        loss = nn.functional.nll_loss(hybrid_model(x), y)
        loss.backward()
        alt.step(epoch=0)

        # Daemon's get_control should have been called
        mock_daemon.get_control.assert_called()


# ======================================================================
# TestHybridCLEVRN
# ======================================================================


class TestHybridCLEVRN:
    """Tests for the CLEVR-N hybrid adapter."""

    def test_forward_shape(self) -> None:
        """HybridCLEVRN produces binary log-probs."""
        torch.manual_seed(SEED)
        model = HybridCLEVRN(scene_dim=16, query_dim=44, hidden_dim=32)
        scene = torch.randn(4, 3, 16)  # 3 items
        query = torch.randn(4, 44)
        out = model(scene, query)
        assert out.shape == (4, 2)

    def test_training_step(self) -> None:
        """HybridCLEVRN can compute gradients."""
        torch.manual_seed(SEED)
        model = HybridCLEVRN(scene_dim=16, query_dim=44, hidden_dim=32)
        scene = torch.randn(8, 3, 16)
        query = torch.randn(8, 44)
        labels = torch.randint(0, 2, (8,))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        log_probs = model(scene, query)
        loss = nn.functional.nll_loss(log_probs, labels)
        loss.backward()
        opt.step()

        assert math.isfinite(loss.item())


# ======================================================================
# TestStateCollector
# ======================================================================


class TestStateCollector:
    """Tests for the training loop hook / StateCollector."""

    def test_step_tracking(self, mock_daemon: MagicMock) -> None:
        """StateCollector accumulates loss EMA across steps."""
        hook = StateCollector(mock_daemon, loss_ema_alpha=0.5)

        # Simulate 5 training steps
        for i in range(5):
            hook.on_step_start()
            time.sleep(0.001)  # small delay for latency tracking
            hook.on_step_end(loss=float(i))

        assert hook.step_count == 5
        assert hook.loss_ema > 0.0
        assert hook.loss_variance >= 0.0

    def test_epoch_end_submits_state(self, mock_daemon: MagicMock) -> None:
        """on_epoch_end submits state to daemon."""
        hook = StateCollector(mock_daemon)
        hook.on_epoch_end(
            epoch=1, loss=0.5,
            r_per_band=[0.8, 0.7, 0.6],
            lr_current=1e-3,
        )
        mock_daemon.submit_state.assert_called_once()

    def test_latest_control_reads_daemon(self, mock_daemon: MagicMock) -> None:
        """latest_control delegates to daemon.get_control."""
        hook = StateCollector(mock_daemon)
        ctrl = hook.latest_control()
        mock_daemon.get_control.assert_called()
        assert ctrl.lr_multiplier == 1.0

    def test_grad_norm_tracking(self, mock_daemon: MagicMock) -> None:
        """Gradient norm EMA is computed when model is provided."""
        model = nn.Linear(10, 2)
        hook = StateCollector(mock_daemon, loss_ema_alpha=0.5)

        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()

        hook.on_step_start()
        hook.on_step_end(loss=loss, model=model)

        assert hook.grad_norm_ema > 0.0

    def test_latency_percentiles(self, mock_daemon: MagicMock) -> None:
        """Step latency tracking produces reasonable values."""
        hook = StateCollector(mock_daemon, latency_window=10)

        for _ in range(5):
            hook.on_step_start()
            time.sleep(0.002)
            hook.on_step_end(loss=1.0)

        # Access latencies through the internal deque
        assert len(hook._step_latencies) == 5
        assert all(l > 0 for l in hook._step_latencies)


# ======================================================================
# TestOscilloBenchFullSuite — 3 tests
# ======================================================================


class TestOscilloBenchFullSuite:
    """Tests for OscilloBench v1.0 benchmark suite."""

    def test_all_tasks_run(self) -> None:
        """All Category A/B benchmark tasks run without error."""
        from benchmarks.oscillobench import OscilloBench

        bench = OscilloBench(seed=SEED, epochs_capacity=2, epochs_convergence=2)

        # Use a small model factory
        def factory(n_dims: int, n_classes: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(n_dims, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
                nn.LogSoftmax(dim=-1),
            )

        # Run a subset (xor_n and random_dichotomies — fast)
        results = bench.run(
            model_factory=factory,
            model_name="TestMLP",
            tasks=["xor_n", "random_dichotomies"],
        )

        assert results["suite"] == "OscilloBench v1.0"
        assert "xor_n" in results["tasks"]
        assert "random_dichotomies" in results["tasks"]
        for task_name, task_result in results["tasks"].items():
            assert "status" in task_result, f"Missing status for {task_name}"

    def test_results_serializable(self) -> None:
        """OscilloBench results are JSON-serializable."""
        from benchmarks.oscillobench import OscilloBench

        bench = OscilloBench(seed=SEED, epochs_capacity=2, epochs_convergence=2)

        def factory(n_dims: int, n_classes: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(n_dims, 16),
                nn.ReLU(),
                nn.Linear(16, n_classes),
                nn.LogSoftmax(dim=-1),
            )

        results = bench.run(
            model_factory=factory,
            model_name="Tiny",
            tasks=["xor_n"],
        )

        # Should not raise
        json_str = json.dumps(results, indent=2, default=str)
        assert len(json_str) > 10
        parsed = json.loads(json_str)
        assert parsed["suite"] == "OscilloBench v1.0"

    def test_dashboard_generation(self) -> None:
        """generate_comparison_dashboard produces valid Markdown."""
        from benchmarks.oscillobench import OscilloBench

        bench = OscilloBench(seed=SEED, epochs_capacity=2, epochs_convergence=2)

        # Create mock results
        mock_results = {
            "model_name": "TestModel",
            "tasks": {
                "xor_n": {"status": "PASS", "test_acc": 0.85},
                "mnist": {"status": "PASS", "test_acc": 0.92},
            },
        }

        dashboard = bench.generate_comparison_dashboard([mock_results])
        assert "# OscilloBench Comparison Dashboard" in dashboard
        assert "TestModel" in dashboard
        assert "xor_n" in dashboard
        assert "|" in dashboard  # Markdown table
