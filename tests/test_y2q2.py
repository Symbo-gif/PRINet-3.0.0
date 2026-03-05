"""Year 2 Q2 Tests — Capacity, Temporal Binding & Active Control.

Tests for:
- Workstream D: CLEVR-N scaling sweep validation.
- Workstream E: Temporal CLEVR dataset, phase propagation, TemporalHybridPRINet.
- Workstream F: Active control policies, A/B test framework, controller retraining.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---- Fixtures -----------------------------------------------------------

SEED = 42


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Ensure deterministic behaviour for all tests."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# =========================================================================
# Workstream D Tests: CLEVR-N Scaling
# =========================================================================


class TestCLEVRNScaling:
    """Tests for CLEVR-N scaling sweep infrastructure."""

    def test_sweep_produces_valid_results(self) -> None:
        """CLEVR-N sweep produces valid per-N accuracy for N=2,4."""
        from benchmarks.clevr_n import (
            D_FEAT,
            D_PHASE,
            LSTMCLEVRNBaseline,
            run_clevr_n_sweep,
        )

        results = run_clevr_n_sweep(
            model_factory=lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            model_name="LSTM",
            n_items_list=[2, 4],
            n_train=100,
            n_test=50,
            n_epochs=2,
            seed=SEED,
        )

        assert len(results) == 2
        for r in results:
            assert r.model_name == "LSTM"
            assert 0.0 <= r.train_acc <= 1.0
            assert 0.0 <= r.test_acc <= 1.0
            assert r.n_items in [2, 4]

    def test_four_plus_models_in_sweep(self) -> None:
        """4+ models can be benchmarked in a single sweep."""
        from benchmarks.clevr_n import (
            D_FEAT,
            D_PHASE,
            LSTMCLEVRNBaseline,
            TransformerCLEVRN,
            run_clevr_n_sweep,
        )
        from benchmarks.y2q1_benchmarks import DiscreteDTGCLEVRN, InterleavedCLEVRN

        models = {
            "LSTM": lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            "Transformer": lambda scene_dim, query_dim: TransformerCLEVRN(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            "DiscreteDTG": lambda scene_dim, query_dim: DiscreteDTGCLEVRN(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            "Interleaved": lambda scene_dim, query_dim: InterleavedCLEVRN(
                scene_dim=scene_dim,
                query_dim=query_dim,
                n_items=4,
            ),
        }

        assert len(models) >= 4, "Need at least 4 models"

        for name, factory in models.items():
            results = run_clevr_n_sweep(
                factory,
                name,
                n_items_list=[2],
                n_train=50,
                n_test=20,
                n_epochs=1,
                seed=SEED,
            )
            assert len(results) == 1
            assert results[0].model_name == name


# =========================================================================
# Workstream E Tests: Temporal Binding
# =========================================================================


class TestTemporalCLEVR:
    """Tests for Temporal CLEVR dataset and models."""

    def test_dataset_produces_valid_sequences(self) -> None:
        """Temporal CLEVR dataset generates valid multi-frame sequences."""
        from benchmarks.y2q2_benchmarks import make_temporal_clevr

        scenes, queries, labels = make_temporal_clevr(
            n_items=4, n_frames=5, n_samples=50, seed=SEED
        )

        assert scenes.shape == (50, 5, 4, 16)  # (N, T, items, D_PHASE)
        assert queries.shape[0] == 50
        assert labels.shape == (50,)
        assert torch.all((labels == 0) | (labels == 1))
        assert not torch.isnan(scenes).any()
        assert not torch.isinf(scenes).any()

    def test_dataset_different_seeds_produce_different_data(self) -> None:
        """Different seeds produce different datasets."""
        from benchmarks.y2q2_benchmarks import make_temporal_clevr

        s1, _, l1 = make_temporal_clevr(n_items=4, n_frames=3, n_samples=20, seed=1)
        s2, _, l2 = make_temporal_clevr(n_items=4, n_frames=3, n_samples=20, seed=2)

        assert not torch.allclose(s1, s2)

    def test_phase_propagation_preserves_continuity(self) -> None:
        """Phase propagation preserves continuity across frames."""
        from prinet.core.measurement import inter_frame_phase_correlation
        from prinet.core.propagation import (
            DiscreteDeltaThetaGamma,
            TemporalPhasePropagator,
        )

        torch.manual_seed(SEED)
        prop = TemporalPhasePropagator(carry_strength=0.9)
        dynamics = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)

        B, N = 8, 28  # 4+8+16
        prev_phase = torch.rand(B, N) * 2 * math.pi
        prev_amp = torch.ones(B, N)
        input_phase = torch.rand(B, N) * 2 * math.pi
        input_amp = torch.ones(B, N)

        new_phase, new_amp = prop.propagate(
            prev_phase, prev_amp, input_phase, input_amp
        )

        # Phase should be wrapped to [0, 2π)
        assert torch.all(new_phase >= 0)
        assert torch.all(new_phase < 2 * math.pi + 0.01)

        # Amplitude should be clamped
        assert torch.all(new_amp >= 1e-6)
        assert torch.all(new_amp <= 10.0)

        # High carry_strength → phase close to previous
        corr = inter_frame_phase_correlation(new_phase, prev_phase)
        assert (
            corr.mean() > 0.5
        ), f"Expected high correlation with carry_strength=0.9, got {corr.mean():.3f}"

    def test_phase_propagation_zero_carry(self) -> None:
        """With carry_strength=0, phase equals input phase."""
        from prinet.core.propagation import TemporalPhasePropagator

        prop = TemporalPhasePropagator(carry_strength=0.0, amplitude_decay=0.0)

        prev_phase = torch.rand(4, 20) * 2 * math.pi
        prev_amp = torch.ones(4, 20) * 5.0
        input_phase = torch.rand(4, 20) * 2 * math.pi
        input_amp = torch.ones(4, 20)

        new_phase, new_amp = prop.propagate(
            prev_phase, prev_amp, input_phase, input_amp
        )

        # With zero carry, output should match input (after wrapping)
        from prinet.core.propagation import _wrap_phase

        expected_phase = _wrap_phase(input_phase)
        assert torch.allclose(new_phase, expected_phase, atol=1e-5)
        assert torch.allclose(new_amp, input_amp, atol=1e-5)

    def test_propagate_sequence(self) -> None:
        """propagate_sequence processes multiple frames correctly."""
        from prinet.core.propagation import (
            DiscreteDeltaThetaGamma,
            TemporalPhasePropagator,
        )

        torch.manual_seed(SEED)
        prop = TemporalPhasePropagator(carry_strength=0.8)
        dynamics = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        N = 28
        B, T = 4, 5

        input_phases = torch.rand(B, T, N) * 2 * math.pi
        input_amps = torch.ones(B, T, N)

        out_phases, out_amps, correlations = prop.propagate_sequence(
            dynamics, input_phases, input_amps, n_steps=3, dt=0.01
        )

        assert out_phases.shape == (B, T, N)
        assert out_amps.shape == (B, T, N)
        assert len(correlations) == T - 1  # T-1 inter-frame correlations
        for corr in correlations:
            assert corr.shape == (B,)
            assert not torch.isnan(corr).any()

    def test_temporal_hybrid_produces_finite_logits(self) -> None:
        """TemporalHybridPRINet produces finite logits on 5-frame input."""
        from prinet.nn.hybrid import TemporalHybridPRINet

        torch.manual_seed(SEED)
        n_osc = 4 + 8 + 16  # 28

        model = TemporalHybridPRINet(
            n_input=128,
            n_classes=2,
            n_tokens=n_osc,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_delta=4,
            n_theta=8,
            n_gamma=16,
            n_discrete_steps=2,
            carry_strength=0.8,
        )

        x = torch.randn(4, 5, 128)  # (B, T, D)
        log_probs = model(x)

        assert log_probs.shape == (4, 2)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()
        # Should be log probabilities
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

    def test_temporal_hybrid_single_frame_fallback(self) -> None:
        """TemporalHybridPRINet handles single-frame input (2D tensor)."""
        from prinet.nn.hybrid import TemporalHybridPRINet

        torch.manual_seed(SEED)
        model = TemporalHybridPRINet(
            n_input=128,
            n_classes=2,
            n_tokens=28,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_delta=4,
            n_theta=8,
            n_gamma=16,
        )

        x = torch.randn(4, 128)  # Single frame
        log_probs = model(x)
        assert log_probs.shape == (4, 2)
        assert not torch.isnan(log_probs).any()

    def test_temporal_hybrid_per_frame_output(self) -> None:
        """TemporalHybridPRINet per_frame=True produces per-frame log probs."""
        from prinet.nn.hybrid import TemporalHybridPRINet

        torch.manual_seed(SEED)
        model = TemporalHybridPRINet(
            n_input=128,
            n_classes=2,
            n_tokens=28,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_delta=4,
            n_theta=8,
            n_gamma=16,
        )

        x = torch.randn(4, 3, 128)
        log_probs = model(x, per_frame=True)
        assert log_probs.shape == (4, 3, 2)  # (B, T, K)
        assert not torch.isnan(log_probs).any()

    def test_temporal_hybrid_gradient_flow(self) -> None:
        """TemporalHybridPRINet: gradients flow through temporal sequence."""
        from prinet.nn.hybrid import TemporalHybridPRINet

        torch.manual_seed(SEED)
        model = TemporalHybridPRINet(
            n_input=64,
            n_classes=2,
            n_tokens=28,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_delta=4,
            n_theta=8,
            n_gamma=16,
        )

        x = torch.randn(2, 3, 64)
        labels = torch.tensor([0, 1])

        log_probs = model(x)
        loss = torch.nn.functional.nll_loss(log_probs, labels)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in model.parameters())
        assert has_grad > 0, "No parameters received gradients"


# =========================================================================
# Workstream E.measure: Inter-frame Phase Correlation
# =========================================================================


class TestInterFrameCorrelation:
    """Tests for inter-frame phase correlation measurement."""

    def test_identical_phases_give_correlation_one(self) -> None:
        """Identical phases → correlation = 1."""
        from prinet.core.measurement import inter_frame_phase_correlation

        phase = torch.rand(100) * 2 * math.pi
        rho = inter_frame_phase_correlation(phase, phase)
        assert torch.isclose(rho, torch.tensor(1.0), atol=1e-5)

    def test_small_shift_gives_high_correlation(self) -> None:
        """Small uniform phase shift → correlation near 1."""
        from prinet.core.measurement import inter_frame_phase_correlation

        prev = torch.rand(100) * 2 * math.pi
        curr = prev + 0.01  # Small shift
        rho = inter_frame_phase_correlation(curr, prev)
        assert rho > 0.99

    def test_random_phases_give_low_correlation(self) -> None:
        """Uncorrelated phases → correlation near 0."""
        from prinet.core.measurement import inter_frame_phase_correlation

        torch.manual_seed(123)
        prev = torch.rand(1000) * 2 * math.pi
        curr = torch.rand(1000) * 2 * math.pi
        rho = inter_frame_phase_correlation(curr, prev)
        assert rho < 0.2, f"Expected low correlation, got {rho:.3f}"

    def test_batched_correlation(self) -> None:
        """Batched inter-frame correlation works."""
        from prinet.core.measurement import inter_frame_phase_correlation

        prev = torch.rand(8, 50) * 2 * math.pi
        curr = prev + 0.01
        rho = inter_frame_phase_correlation(curr, prev)
        assert rho.shape == (8,)
        assert (rho > 0.99).all()

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatch raises ValueError."""
        from prinet.core.measurement import inter_frame_phase_correlation

        with pytest.raises(ValueError, match="Shape mismatch"):
            inter_frame_phase_correlation(torch.rand(10), torch.rand(20))

    def test_empty_raises(self) -> None:
        """Empty tensor raises ValueError."""
        from prinet.core.measurement import inter_frame_phase_correlation

        with pytest.raises(ValueError, match="empty"):
            inter_frame_phase_correlation(torch.tensor([]), torch.tensor([]))


# =========================================================================
# Workstream F Tests: Active Control
# =========================================================================


class TestActiveControl:
    """Tests for active subconscious control integration."""

    def test_policies_apply_max_5pct_adjustment(self) -> None:
        """Active control policies apply ≤5% adjustment per signal."""
        from prinet.nn.training_hooks import apply_lr_adjustment

        # Create a mock control object with extreme lr_multiplier
        class MockControl:
            alert_level: float = 0.9
            lr_multiplier: float = 2.0  # extreme multiplier

        control = MockControl()
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_lr = optimizer.param_groups[0]["lr"]
        mult = apply_lr_adjustment(control, optimizer, max_adjustment=0.05)

        # Must be within ±5% of 1.0
        assert 0.95 <= mult <= 1.05, f"Multiplier {mult} exceeds ±5%"

        # LR should not change by more than 5%
        new_lr = optimizer.param_groups[0]["lr"]
        ratio = new_lr / initial_lr
        assert 0.94 <= ratio <= 1.06

    def test_active_control_trainer_creation(self) -> None:
        """ActiveControlTrainer can be created and queried."""
        from prinet.nn.training_hooks import ActiveControlTrainer

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = ActiveControlTrainer(
            model=model,
            optimizer=optimizer,
            daemon=None,
            active=True,
        )

        assert trainer.active is True
        assert trainer.last_policy_applied == {}

    def test_active_control_epoch_end(self) -> None:
        """on_epoch_end returns policy dict and logs telemetry."""
        from prinet.nn.training_hooks import ActiveControlTrainer

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = ActiveControlTrainer(
            model=model,
            optimizer=optimizer,
            daemon=None,
            active=True,
        )

        policy = trainer.on_epoch_end(
            epoch=1,
            loss=0.5,
            r_per_band=[0.8, 0.6, 0.4],
        )

        assert "active" in policy
        assert "lr_mult" in policy
        assert "k_range" in policy
        assert "regime" in policy
        assert len(trainer.telemetry) == 1

    def test_passive_mode_no_adjustment(self) -> None:
        """Passive mode logs but does not apply adjustments."""
        from prinet.nn.training_hooks import ActiveControlTrainer

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_lr = optimizer.param_groups[0]["lr"]

        trainer = ActiveControlTrainer(
            model=model,
            optimizer=optimizer,
            daemon=None,
            active=False,
        )

        policy = trainer.on_epoch_end(epoch=1, loss=0.5)
        assert policy["active"] is False
        assert policy["lr_mult"] == 1.0

        # LR unchanged
        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_training_stability_with_active_control(self) -> None:
        """Training stability maintained with active control (5 epochs)."""
        from torch.utils.data import DataLoader

        from benchmarks.clevr_n import D_FEAT, D_PHASE, CLEVRNDataset, make_clevr_n
        from benchmarks.y2q1_benchmarks import DiscreteDTGCLEVRN
        from prinet.nn.training_hooks import ActiveControlTrainer

        torch.manual_seed(SEED)
        model = DiscreteDTGCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = ActiveControlTrainer(
            model=model,
            optimizer=optimizer,
            daemon=None,
            active=True,
            max_adjustment=0.05,
        )

        scenes, queries, labels = make_clevr_n(4, 100, seed=SEED)
        ds = CLEVRNDataset(scenes, queries, labels)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        losses: list[float] = []
        for epoch in range(5):
            epoch_loss = 0.0
            n_b = 0
            for s, q, lb in loader:
                optimizer.zero_grad()
                out = model(s, q)
                loss = torch.nn.functional.nll_loss(out, lb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
                # Check for NaN
                assert not torch.isnan(out).any(), f"NaN at epoch {epoch}"

            avg = epoch_loss / max(n_b, 1)
            losses.append(avg)
            trainer.on_epoch_end(epoch=epoch, loss=avg)

        # Loss should not be NaN or Inf
        assert all(math.isfinite(l) for l in losses)

    def test_ab_test_framework_produces_valid_comparison(self) -> None:
        """A/B test framework produces valid statistical comparison (2 runs)."""
        from benchmarks.y2q2_benchmarks import run_f_ab_test

        results = run_f_ab_test(
            n_runs_per_group=2,
            n_epochs=2,
            n_items=2,
            base_seed=SEED,
            device="cpu",
        )

        assert len(results["active_runs"]) == 2
        assert len(results["passive_runs"]) == 2
        assert "statistics" in results
        stats = results["statistics"]
        assert "t_statistic" in stats
        assert "p_value_approx" in stats
        assert math.isfinite(stats["t_statistic"])


# =========================================================================
# Workstream F.3: Controller Retraining
# =========================================================================


class TestControllerRetraining:
    """Tests for SubconsciousController retraining pipeline."""

    def test_retrain_from_telemetry_records(self) -> None:
        """Controller can be retrained from telemetry records."""
        from prinet.nn.subconscious_model import retrain_controller

        records = []
        for i in range(50):
            records.append(
                {
                    "epoch": i,
                    "loss": 1.0 / (i + 1),
                    "r_per_band": [0.5 + i * 0.01, 0.4, 0.3],
                    "r_global": 0.4,
                    "control": {
                        "suggested_K_min": 0.5,
                        "suggested_K_max": 5.0,
                        "lr_multiplier": 1.0,
                        "regime_mf_weight": 0.5,
                        "regime_sk_weight": 0.3,
                        "regime_full_weight": 0.2,
                        "alert_level": 0.0,
                        "coupling_mode_suggestion": 0.0,
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "controller.onnx"
            controller, metrics = retrain_controller(
                telemetry_records=records,
                n_epochs=5,
                output_onnx_path=onnx_path,
                seed=SEED,
            )

            assert metrics["n_samples"] == 50
            assert metrics["n_epochs"] == 5
            assert math.isfinite(metrics["train_loss"])
            assert onnx_path.exists()

            # Controller produces valid output
            z = torch.randn(4, 32)
            ctrl = controller(z)
            assert ctrl.shape == (4, 8)
            assert not torch.isnan(ctrl).any()

    def test_retrain_from_json_file(self) -> None:
        """Controller can be retrained from telemetry JSON file."""
        from prinet.nn.subconscious_model import retrain_controller

        records = [
            {"epoch": i, "loss": 0.5, "r_per_band": [0.5, 0.5, 0.5], "r_global": 0.5}
            for i in range(30)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "telemetry.json"
            onnx_path = Path(tmpdir) / "controller.onnx"

            with open(json_path, "w") as f:
                json.dump(records, f)

            controller, metrics = retrain_controller(
                telemetry_path=str(json_path),
                n_epochs=3,
                output_onnx_path=onnx_path,
                seed=SEED,
            )

            assert metrics["n_samples"] == 30
            assert onnx_path.exists()

    def test_retrain_empty_records_raises(self) -> None:
        """Empty telemetry raises ValueError."""
        from prinet.nn.subconscious_model import retrain_controller

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty"):
                retrain_controller(
                    telemetry_records=[],
                    output_onnx_path=Path(tmpdir) / "c.onnx",
                )

    def test_retrain_no_source_raises(self) -> None:
        """No telemetry source raises ValueError."""
        from prinet.nn.subconscious_model import retrain_controller

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="telemetry_path or telemetry_records"):
                retrain_controller(
                    output_onnx_path=Path(tmpdir) / "c.onnx",
                )


# =========================================================================
# Export / Package Tests
# =========================================================================


class TestTopLevelExports:
    """Verify all Y2 Q2 symbols are exported correctly."""

    def test_core_exports(self) -> None:
        """Core package exports temporal propagation and measurement."""
        from prinet.core import (
            TemporalPhasePropagator,
            inter_frame_phase_correlation,
        )

        assert TemporalPhasePropagator is not None
        assert callable(inter_frame_phase_correlation)

    def test_nn_exports(self) -> None:
        """NN package exports temporal and control classes."""
        from prinet.nn import (
            ActiveControlTrainer,
            TemporalHybridPRINet,
            retrain_controller,
        )

        assert TemporalHybridPRINet is not None
        assert ActiveControlTrainer is not None
        assert callable(retrain_controller)

    def test_top_level_exports(self) -> None:
        """Top-level prinet package exports all Y2 Q2 symbols."""
        from prinet import (
            ActiveControlTrainer,
            TemporalHybridPRINet,
            TemporalPhasePropagator,
            inter_frame_phase_correlation,
            retrain_controller,
        )

        assert TemporalPhasePropagator is not None
        assert callable(inter_frame_phase_correlation)
        assert TemporalHybridPRINet is not None
        assert ActiveControlTrainer is not None
        assert callable(retrain_controller)
