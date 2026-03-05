"""Unit tests for CLEVR-N benchmark task.

Covers data generation, feature encoding, baseline model forward passes,
and capacity sweep functionality.

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import json

import pytest
import torch

from benchmarks.clevr_n import (
    CLEVRNDataset,
    CLEVRNResult,
    D_FEAT,
    D_PHASE,
    HopfieldCLEVRNBaseline,
    LSTMCLEVRNBaseline,
    TransformerCLEVRN,
    encode_features_phase,
    make_clevr_n,
    run_clevr_n_sweep,
)

SEED = 42


# ===================================================================
# TestCLEVRNDataGeneration
# ===================================================================


class TestCLEVRNDataGeneration:
    """Tests for CLEVR-N data generation."""

    def test_generate_n2(self) -> None:
        scenes, queries, labels = make_clevr_n(n_items=2, n_samples=50, seed=SEED)
        assert scenes.shape == (50, 2, D_PHASE)
        assert queries.shape == (50, D_FEAT * 2)
        assert labels.shape == (50,)

    def test_generate_n4(self) -> None:
        scenes, queries, labels = make_clevr_n(n_items=4, n_samples=30, seed=SEED)
        assert scenes.shape == (30, 4, D_PHASE)

    def test_generate_n8(self) -> None:
        scenes, queries, labels = make_clevr_n(n_items=8, n_samples=20, seed=SEED)
        assert scenes.shape == (20, 8, D_PHASE)

    def test_scene_tensor_shape(self) -> None:
        for n_items in [2, 4, 6, 8]:
            scenes, _, _ = make_clevr_n(n_items=n_items, n_samples=10, seed=SEED)
            assert scenes.shape[1] == n_items
            assert scenes.shape[2] == D_PHASE

    def test_labels_valid(self) -> None:
        _, _, labels = make_clevr_n(n_items=4, n_samples=100, seed=SEED)
        assert labels.dtype == torch.long
        assert (labels >= 0).all()
        assert (labels <= 1).all()
        # Should have both classes
        assert labels.sum() > 0
        assert labels.sum() < labels.shape[0]

    def test_deterministic_with_seed(self) -> None:
        s1, q1, l1 = make_clevr_n(n_items=4, n_samples=50, seed=SEED)
        s2, q2, l2 = make_clevr_n(n_items=4, n_samples=50, seed=SEED)
        torch.testing.assert_close(s1, s2)
        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(l1, l2)


# ===================================================================
# TestFeatureEncoding
# ===================================================================


class TestFeatureEncoding:
    """Tests for phase feature encoding."""

    def test_encode_shape(self) -> None:
        colors = torch.tensor([0, 1, 2])
        shapes = torch.tensor([0, 1, 2])
        positions = torch.tensor([0, 1, 2])
        enc = encode_features_phase(colors, shapes, positions)
        assert enc.shape == (3, D_PHASE)

    def test_encode_finite(self) -> None:
        colors = torch.tensor([0, 3, 7])
        shapes = torch.tensor([0, 2, 5])
        positions = torch.tensor([0, 4, 7])
        enc = encode_features_phase(colors, shapes, positions)
        assert torch.isfinite(enc).all()

    def test_different_features_different_encodings(self) -> None:
        enc1 = encode_features_phase(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        )
        enc2 = encode_features_phase(
            torch.tensor([1]), torch.tensor([1]), torch.tensor([1])
        )
        assert not torch.allclose(enc1, enc2)


# ===================================================================
# TestCLEVRNBaselines
# ===================================================================


class TestCLEVRNBaselines:
    """Tests for baseline model forward passes."""

    def test_lstm_forward(self) -> None:
        model = LSTMCLEVRNBaseline(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
        scene = torch.randn(4, 6, D_PHASE)
        query = torch.randn(4, D_FEAT * 2)
        out = model(scene, query)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()

    def test_transformer_forward(self) -> None:
        model = TransformerCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
        scene = torch.randn(4, 6, D_PHASE)
        query = torch.randn(4, D_FEAT * 2)
        out = model(scene, query)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()

    def test_hopfield_forward(self) -> None:
        model = HopfieldCLEVRNBaseline(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
        scene = torch.randn(4, 6, D_PHASE)
        query = torch.randn(4, D_FEAT * 2)
        out = model(scene, query)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()

    def test_loss_computable(self) -> None:
        """All baselines should produce valid loss values."""
        for ModelClass in [LSTMCLEVRNBaseline, TransformerCLEVRN, HopfieldCLEVRNBaseline]:
            model = ModelClass(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
            scene = torch.randn(4, 6, D_PHASE)
            query = torch.randn(4, D_FEAT * 2)
            labels = torch.randint(0, 2, (4,))

            out = model(scene, query)
            loss = torch.nn.functional.nll_loss(out, labels)
            assert torch.isfinite(loss)
            loss.backward()


# ===================================================================
# TestCLEVRNCapacity
# ===================================================================


class TestCLEVRNCapacity:
    """Tests for capacity sweep functionality."""

    def test_sweep_runs(self) -> None:
        """Capacity sweep should complete for small N values."""
        results = run_clevr_n_sweep(
            model_factory=lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            model_name="TestLSTM",
            n_items_list=[2, 4],
            n_train=50,
            n_test=20,
            n_epochs=2,
            batch_size=16,
            seed=SEED,
        )
        assert len(results) == 2
        assert all(isinstance(r, CLEVRNResult) for r in results)

    def test_results_json_serializable(self) -> None:
        """Results should be JSON-serializable."""
        results = run_clevr_n_sweep(
            model_factory=lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
                scene_dim=scene_dim, query_dim=query_dim
            ),
            model_name="TestLSTM",
            n_items_list=[2],
            n_train=20,
            n_test=10,
            n_epochs=1,
            batch_size=8,
            seed=SEED,
        )
        serializable = [r.to_dict() for r in results]
        json_str = json.dumps(serializable)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1


# ===================================================================
# TestCLEVRNDataset
# ===================================================================


class TestCLEVRNDatasetWrapper:
    """Tests for the PyTorch Dataset wrapper."""

    def test_length(self) -> None:
        scenes, queries, labels = make_clevr_n(n_items=4, n_samples=50, seed=SEED)
        ds = CLEVRNDataset(scenes, queries, labels)
        assert len(ds) == 50

    def test_getitem(self) -> None:
        scenes, queries, labels = make_clevr_n(n_items=4, n_samples=50, seed=SEED)
        ds = CLEVRNDataset(scenes, queries, labels)
        s, q, l = ds[0]
        assert s.shape == (4, D_PHASE)
        assert q.shape == (D_FEAT * 2,)
        assert l.dim() == 0  # scalar
