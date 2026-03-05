"""Tests for Phase 1-3 experiment scripts.

Validates:
- P1: Statistical utilities (Cliff's delta, Holm-Bonferroni, bootstrap CI,
      Bayes Factor) against known values.
- P2: Scaling experiment helpers (model build, train, eval pipeline).
- P3: Profiling, gradient flow, and representation geometry primitives.

All tests run on CPU with minimal sizes for speed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = str(_PROJECT_ROOT / "benchmarks")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

SEED = 42
DEVICE = "cpu"
DET_DIM = 4


# =========================================================================
# Phase 1: Statistical Utilities
# =========================================================================


class TestCliffsDelta:
    """Validate Cliff's delta implementation."""

    def test_identical_groups(self):
        """Cliff's delta should be 0 for identical groups."""
        from phase1_statistical_hardening import cliffs_delta
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cliffs_delta(a, a) == 0.0

    def test_perfect_separation(self):
        """Cliff's delta should be +1 when A > B always."""
        from phase1_statistical_hardening import cliffs_delta
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        assert cliffs_delta(a, b) == 1.0

    def test_reverse_separation(self):
        """Cliff's delta should be -1 when B > A always."""
        from phase1_statistical_hardening import cliffs_delta
        a = [1.0, 2.0, 3.0]
        b = [10.0, 11.0, 12.0]
        assert cliffs_delta(a, b) == -1.0

    def test_interpretation_negligible(self):
        """Small effect should be 'negligible'."""
        from phase1_statistical_hardening import cliffs_delta_interpretation
        assert cliffs_delta_interpretation(0.1) == "negligible"

    def test_interpretation_large(self):
        """Large effect should be 'large'."""
        from phase1_statistical_hardening import cliffs_delta_interpretation
        assert cliffs_delta_interpretation(0.8) == "large"

    def test_empty_groups(self):
        """Empty groups should raise ValueError."""
        from phase1_statistical_hardening import cliffs_delta
        with pytest.raises(ValueError):
            cliffs_delta([], [1.0, 2.0])
        with pytest.raises(ValueError):
            cliffs_delta([1.0], [])


class TestHolmBonferroni:
    """Validate Holm-Bonferroni correction."""

    def test_all_significant(self):
        """All p-values well below alpha should remain significant."""
        from phase1_statistical_hardening import holm_bonferroni
        pvals = [0.001, 0.002, 0.003]
        result = holm_bonferroni(pvals, alpha=0.05)
        assert all(r["reject_H0"] for r in result)

    def test_none_significant(self):
        """All p-values above alpha should be non-significant."""
        from phase1_statistical_hardening import holm_bonferroni
        pvals = [0.5, 0.6, 0.7]
        result = holm_bonferroni(pvals, alpha=0.05)
        assert not any(r["reject_H0"] for r in result)

    def test_partial_rejection(self):
        """Mixed p-values: only smallest should survive correction."""
        from phase1_statistical_hardening import holm_bonferroni
        pvals = [0.01, 0.04, 0.06]
        result = holm_bonferroni(pvals, alpha=0.05)
        # Holm: sorted p-values are [0.01, 0.04, 0.06]
        # Thresholds: 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05
        # 0.01 < 0.0167 -> reject
        # 0.04 > 0.025 -> stop
        # result is indexed by original position
        assert result[0]["reject_H0"] is True   # p=0.01
        assert result[2]["reject_H0"] is False  # p=0.06

    def test_single_test(self):
        """Single test should be equivalent to uncorrected."""
        from phase1_statistical_hardening import holm_bonferroni
        result = holm_bonferroni([0.03], alpha=0.05)
        assert result[0]["reject_H0"] is True


class TestBayesFactor:
    """Validate Bayes Factor computation."""

    def test_strong_evidence(self):
        """Large t-stat should produce large BF."""
        from phase1_statistical_hardening import (
            _jeffreys_bayes_factor_t,
            bayes_factor_interpretation,
        )
        # t_stat=10 with n=20 each -> strong evidence
        bf = _jeffreys_bayes_factor_t(t_stat=10.0, n_a=20, n_b=20)
        assert bf > 10.0
        interp = bayes_factor_interpretation(bf)
        assert interp in ("strong", "very strong", "decisive",
                          "extreme_evidence_for_H1")

    def test_no_evidence(self):
        """t_stat near 0 should produce BF near or below 1."""
        from phase1_statistical_hardening import _jeffreys_bayes_factor_t
        bf = _jeffreys_bayes_factor_t(t_stat=0.01, n_a=5, n_b=5)
        assert bf < 3.0  # Not substantial


# =========================================================================
# Phase 2: Scaling Utilities
# =========================================================================


class TestPhase2Utilities:
    """Validate Phase 2 helper functions."""

    def test_bootstrap_ci(self):
        """Bootstrap CI should contain the mean."""
        from phase2_scaling_analysis import _bootstrap_ci
        values = [0.9, 0.92, 0.95, 0.91, 0.93, 0.94, 0.90]
        ci = _bootstrap_ci(values)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
        assert ci["std"] > 0

    def test_bootstrap_ci_single(self):
        """Single value should have zero-width CI."""
        from phase2_scaling_analysis import _bootstrap_ci
        ci = _bootstrap_ci([0.95])
        assert ci["mean"] == 0.95
        assert ci["std"] == 0.0

    def test_cliffs_delta_phase2(self):
        """Verify Phase 2's Cliff's delta matches expectation."""
        from phase2_scaling_analysis import _cliffs_delta
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        assert _cliffs_delta(a, b) == 1.0

    def test_welch_t_identical(self):
        """Welch's t-test on identical groups: p should be 1.0."""
        from phase2_scaling_analysis import _welch_t
        a = [1.0, 1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.0, 1.0]
        result = _welch_t(a, b)
        assert result["t_stat"] == 0.0
        assert result["p_value"] == 1.0

    def test_welch_t_separated(self):
        """Welch's t-test on well-separated groups: p should be small."""
        from phase2_scaling_analysis import _welch_t
        a = [10.0, 10.1, 10.2, 9.9, 10.3]
        b = [1.0, 1.1, 1.2, 0.9, 1.3]
        result = _welch_t(a, b)
        assert result["p_value"] < 0.001
        assert result["cohens_d"] > 1.0

    def test_build_pt(self):
        """PhaseTracker should build and forward pass on CPU."""
        from phase2_scaling_analysis import _build_pt
        pt = _build_pt(42)
        det = torch.randn(4, DET_DIM)
        with torch.no_grad():
            matches, sim = pt(det, det)
        assert sim.shape[0] == 4
        assert sim.shape[1] == 4

    def test_build_sa(self):
        """SlotAttention should build and forward pass on CPU."""
        from phase2_scaling_analysis import _build_sa
        sa = _build_sa(42)
        det = torch.randn(4, DET_DIM)
        with torch.no_grad():
            matches, sim = sa(det, det)
        assert sim.shape[0] >= 4  # May include slot count

    def test_gen_sequences(self):
        """Sequence generation should produce correct shapes."""
        from phase2_scaling_analysis import _gen
        seqs = _gen(3, n_objects=5, n_frames=10, base_seed=42)
        assert len(seqs) == 3
        assert len(seqs[0].frames) == 10
        assert seqs[0].frames[0].shape == (5, DET_DIM)


class TestPhase2Training:
    """Validate the training pipeline works end-to-end on CPU."""

    def test_train_pt_minimal(self):
        """PhaseTracker should train without error on minimal data."""
        from phase2_scaling_analysis import _build_pt, _train_model
        pt = _build_pt(42)
        pt, info = _train_model(pt, 42, n_objects=3, n_frames=5)
        assert info["epochs"] >= 1
        assert info["wall_time_s"] > 0

    def test_train_sa_minimal(self):
        """SlotAttention should train without error on minimal data."""
        from phase2_scaling_analysis import _build_sa, _train_model
        sa = _build_sa(42)
        sa, info = _train_model(sa, 42, n_objects=3, n_frames=5)
        assert info["epochs"] >= 1

    def test_eval_ip(self):
        """IP evaluation should return values in [0, 1]."""
        from phase2_scaling_analysis import _build_pt, _eval_ip, _gen
        pt = _build_pt(42)
        pt.eval()
        seqs = _gen(2, n_objects=3, n_frames=5, base_seed=42)
        ips = _eval_ip(pt, seqs, device="cpu")
        assert len(ips) == 2
        for ip in ips:
            assert 0.0 <= ip <= 1.0


# =========================================================================
# Phase 3: Scientific Experiment Primitives
# =========================================================================


class TestPhase3Profiling:
    """Validate profiling utilities."""

    def test_count_parameters_pt(self):
        """PhaseTracker should have reasonable parameter count."""
        from phase3_scientific_experiments import _build_pt
        pt = _build_pt(42)
        n = sum(p.numel() for p in pt.parameters())
        assert n > 100  # Non-trivial

    def test_count_parameters_sa(self):
        """SlotAttention should have more parameters than PT."""
        from phase3_scientific_experiments import _build_sa
        sa = _build_sa(42)
        n = sum(p.numel() for p in sa.parameters())
        assert n > 1000

    def test_parameter_ratio(self):
        """SA should have substantially more parameters than PT."""
        from phase3_scientific_experiments import _build_pt, _build_sa
        pt = _build_pt(42)
        sa = _build_sa(42)
        n_pt = sum(p.numel() for p in pt.parameters())
        n_sa = sum(p.numel() for p in sa.parameters())
        ratio = n_sa / n_pt
        assert ratio > 5.0  # SA has at least 5x more params


class TestPhase3GradientFlow:
    """Validate gradient hook registration and flow."""

    def test_hooks_collect_gradients(self):
        """Backward hooks should collect gradient norms."""
        from phase3_scientific_experiments import _build_pt, _train_model
        pt = _build_pt(42)
        hooks: dict[str, list[float]] = {}
        pt, info = _train_model(
            pt, 42, n_objects=3, n_frames=5, hooks=hooks,
        )
        # Should have collected some gradient norms
        total_norms = sum(len(v) for v in hooks.values())
        assert total_norms > 0, "No gradient norms collected"

    def test_gradient_norms_finite(self):
        """All collected gradient norms should be finite."""
        from phase3_scientific_experiments import _build_pt, _train_model
        pt = _build_pt(42)
        hooks: dict[str, list[float]] = {}
        pt, _ = _train_model(pt, 42, n_objects=3, n_frames=5, hooks=hooks)
        for layer, norms in hooks.items():
            for n in norms:
                assert math.isfinite(n), f"Non-finite gradient in {layer}"


class TestPhase3Representation:
    """Validate representation extraction."""

    def test_pt_encode(self):
        """PT encode should return phase and amplitude."""
        from phase3_scientific_experiments import _build_pt
        pt = _build_pt(42)
        pt.eval()
        det = torch.randn(5, DET_DIM)
        with torch.no_grad():
            phase, amp = pt.encode(det)
        assert phase.shape[0] == 5
        assert amp.shape[0] == 5
        # Amplitudes should be non-negative (Softplus)
        assert (amp >= 0).all()


# =========================================================================
# Integration: Save/Load Artefacts
# =========================================================================


class TestArtefactIO:
    """Validate JSON artefact save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        """JSON save and load should preserve data."""
        import phase2_scaling_analysis as p2
        original_dir = p2.RESULTS_DIR
        p2.RESULTS_DIR = tmp_path
        try:
            data = {
                "benchmark": "test",
                "values": [1.0, 2.0, 3.0],
                "nested": {"key": "value"},
            }
            p2._save("test_roundtrip", data)
            path = tmp_path / "phase2_test_roundtrip.json"
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["benchmark"] == "test"
            assert loaded["values"] == [1.0, 2.0, 3.0]
        finally:
            p2.RESULTS_DIR = original_dir
