"""Year 3 Q2 tests — Harder binding tasks.

Tests for:
- N.1: MOT17-style evaluation pipeline
- N.2: Crowded MOT (50+ objects)
- N.3: Temporal reasoning with occlusion/distractors
- N.4: Adaptive oscillator allocation
- N.5: Subconscious A/B statistical test

Requires: motmetrics, scipy.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Optional imports — skip tests if missing
try:
    import motmetrics as mm

    _HAS_MOTMETRICS = True
except ImportError:
    _HAS_MOTMETRICS = False

try:
    from scipy.stats import ttest_ind

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_skip_motmetrics = pytest.mark.skipif(
    not _HAS_MOTMETRICS, reason="motmetrics not installed"
)
_skip_scipy = pytest.mark.skipif(
    not _HAS_SCIPY, reason="scipy not installed"
)
_skip_hypothesis = pytest.mark.skipif(
    not _HAS_HYPOTHESIS, reason="hypothesis not installed"
)


def _seed(s: int = 42) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ====================================================================
# N.4 — Adaptive Oscillator Allocation
# ====================================================================


class TestN4AdaptiveAllocation:
    """Tests for AdaptiveOscillatorAllocator and DynamicPhaseTracker."""

    def test_allocator_construction_rule(self) -> None:
        """Rule-based allocator can be instantiated."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        assert alloc.strategy == "rule"
        assert alloc.min_total == 12
        assert alloc.max_total == 64

    def test_allocator_construction_learned(self) -> None:
        """Learned allocator creates an MLP."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(
            min_total=12, max_total=64, strategy="learned", complexity_dim=4,
        )
        assert alloc.strategy == "learned"
        assert alloc._mlp is not None

    def test_allocator_invalid_min(self) -> None:
        """min_total < 3 raises ValueError."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        with pytest.raises(ValueError, match="min_total must be >= 3"):
            AdaptiveOscillatorAllocator(min_total=2, max_total=64)

    def test_allocator_invalid_range(self) -> None:
        """max_total < min_total raises ValueError."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        with pytest.raises(ValueError, match="max_total"):
            AdaptiveOscillatorAllocator(min_total=64, max_total=12)

    def test_budget_dataclass(self) -> None:
        """OscillatorBudget stores and computes total correctly."""
        from prinet.nn.adaptive_allocation import OscillatorBudget

        b = OscillatorBudget(n_delta=4, n_theta=8, n_gamma=32, complexity=0.5)
        assert b.total == 44
        assert b.complexity == 0.5

    def test_rule_allocation_low_complexity(self) -> None:
        """Low complexity allocates min oscillators."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budget = alloc.allocate(complexity=0.0)
        assert budget.total == 12
        assert budget.n_delta >= 1
        assert budget.n_theta >= 1
        assert budget.n_gamma >= 1

    def test_rule_allocation_high_complexity(self) -> None:
        """High complexity allocates max oscillators."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budget = alloc.allocate(complexity=1.0)
        assert budget.total == 64

    def test_sweep_monotonicity(self) -> None:
        """Total oscillator count monotonically increases with complexity."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budgets = alloc.sweep_complexity(21)
        totals = [b.total for b in budgets]
        for i in range(1, len(totals)):
            assert totals[i] >= totals[i - 1], (
                f"Non-monotonic: totals[{i - 1}]={totals[i - 1]}, "
                f"totals[{i}]={totals[i]}"
            )

    def test_sweep_no_mid_range_dip(self) -> None:
        """Gamma oscillators don't dip in mid-range complexity."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budgets = alloc.sweep_complexity(21)
        gammas = [b.n_gamma for b in budgets]
        for i in range(1, len(gammas)):
            assert gammas[i] >= gammas[i - 1], (
                f"Mid-range dip in gamma: [{i - 1}]={gammas[i - 1]}, "
                f"[{i}]={gammas[i]}"
            )

    def test_estimate_complexity_empty(self) -> None:
        """Single detection produces low complexity."""
        from prinet.nn.adaptive_allocation import estimate_complexity

        dets = torch.randn(1, 4)
        c = estimate_complexity(dets)
        assert 0.0 <= c.item() <= 1.0

    def test_estimate_complexity_many(self) -> None:
        """50 objects saturate count term."""
        from prinet.nn.adaptive_allocation import estimate_complexity

        dets = torch.rand(50, 4)
        c = estimate_complexity(dets, max_objects=50)
        assert c.item() >= 0.5, f"Expected high complexity, got {c.item()}"

    def test_forward_equals_allocate(self) -> None:
        """forward() delegates to allocate()."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        b1 = alloc.allocate(0.5)
        b2 = alloc(0.5)
        assert b1 == b2

    def test_learned_allocation(self) -> None:
        """Learned allocator produces valid budgets."""
        _seed()
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(
            min_total=12, max_total=64, strategy="learned", complexity_dim=4,
        )
        features = torch.randn(4)
        budget = alloc.allocate(complexity=0.5, features=features)
        assert budget.n_delta >= 1
        assert budget.n_theta >= 1
        assert budget.n_gamma >= 1
        assert budget.total >= 12

    def test_dynamic_phase_tracker_forward(self) -> None:
        """DynamicPhaseTracker produces matches, sim, and budget."""
        _seed()
        from prinet.nn.adaptive_allocation import DynamicPhaseTracker

        tracker = DynamicPhaseTracker(detection_dim=4, min_total=12, max_total=28)
        dets_t = torch.randn(5, 4)
        dets_t1 = torch.randn(5, 4)
        matches, sim, budget = tracker(dets_t, dets_t1)
        assert matches.shape == (5,)
        assert sim.shape[0] == 5
        assert budget.total >= 12


# ====================================================================
# N.1 — MOT17 Evaluation Pipeline
# ====================================================================


class TestN1MOT17Pipeline:
    """Tests for MOT evaluation pipeline."""

    def test_detection_dataclass(self) -> None:
        """Detection stores fields correctly."""
        from prinet.nn.mot_evaluation import Detection

        det = Detection(frame_id=0, obj_id=1, bbox=[10, 20, 50, 80])
        assert det.frame_id == 0
        assert det.obj_id == 1
        assert det.features is None

    def test_linear_mot_generation(self) -> None:
        """Linear MOT sequence has correct structure."""
        from prinet.nn.mot_evaluation import generate_linear_mot_sequence

        seq = generate_linear_mot_sequence(n_objects=5, n_frames=10)
        assert len(seq) == 10
        # Each frame should have up to 5 detections
        for frame in seq:
            assert all(d.obj_id >= 0 for d in frame)
            assert all(len(d.bbox) == 4 for d in frame)

    def test_linear_mot_reproducibility(self) -> None:
        """Same seed produces identical sequences."""
        from prinet.nn.mot_evaluation import generate_linear_mot_sequence

        seq1 = generate_linear_mot_sequence(n_objects=3, n_frames=5, seed=123)
        seq2 = generate_linear_mot_sequence(n_objects=3, n_frames=5, seed=123)
        for f1, f2 in zip(seq1, seq2):
            assert len(f1) == len(f2)
            for d1, d2 in zip(f1, f2):
                assert d1.obj_id == d2.obj_id
                assert d1.bbox == d2.bbox

    @_skip_motmetrics
    def test_evaluate_tracking_basic(self) -> None:
        """Evaluate tracking returns valid TrackingResult."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            evaluate_tracking,
            generate_linear_mot_sequence,
        )

        tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        tracker.eval()
        seq = generate_linear_mot_sequence(n_objects=5, n_frames=10)
        result = evaluate_tracking(seq, tracker, detection_dim=4)
        assert -1.0 <= result.mota <= 1.0
        assert result.n_frames == 10
        assert result.n_objects == 5
        assert result.id_switches >= 0

    @_skip_motmetrics
    def test_tracking_result_fields(self) -> None:
        """TrackingResult contains all expected fields."""
        from prinet.nn.mot_evaluation import TrackingResult

        result = TrackingResult(
            sequence_name="test",
            n_frames=10,
            n_objects=5,
            mota=0.5,
            motp=0.3,
            idf1=0.6,
            id_switches=2,
            false_positives=1,
            false_negatives=3,
            identity_preservation=0.8,
        )
        assert result.mota == 0.5
        assert result.sequence_name == "test"

    @_skip_motmetrics
    def test_evaluate_returns_raw_metrics(self) -> None:
        """Raw metrics dict is populated."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            evaluate_tracking,
            generate_linear_mot_sequence,
        )

        tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        tracker.eval()
        seq = generate_linear_mot_sequence(n_objects=3, n_frames=5)
        result = evaluate_tracking(seq, tracker, detection_dim=4)
        assert "mota" in result.raw_metrics
        assert "idf1" in result.raw_metrics


# ====================================================================
# N.2 — Crowded MOT (50+ objects)
# ====================================================================


class TestN2CrowdedMOT:
    """Tests for crowded MOT sequences."""

    def test_crowded_generation_50_objects(self) -> None:
        """Crowded generator produces 50+ object sequences."""
        from prinet.nn.mot_evaluation import generate_crowded_mot_sequence

        seq = generate_crowded_mot_sequence(n_objects=50, n_frames=30)
        assert len(seq) == 30
        # Verify unique GT identities
        all_ids = set()
        for frame in seq:
            for det in frame:
                if det.obj_id >= 0:
                    all_ids.add(det.obj_id)
        assert len(all_ids) == 50

    def test_crowded_has_distractors(self) -> None:
        """Crowded sequence includes distractor detections (obj_id=-1)."""
        from prinet.nn.mot_evaluation import generate_crowded_mot_sequence

        seq = generate_crowded_mot_sequence(
            n_objects=50, n_frames=30, distractor_rate=0.1, seed=42,
        )
        distractor_count = sum(
            1 for frame in seq for det in frame if det.obj_id == -1
        )
        assert distractor_count > 0, "No distractors generated"

    def test_crowded_has_occlusions(self) -> None:
        """Crowded sequence has frames with fewer detections (occlusions)."""
        from prinet.nn.mot_evaluation import generate_crowded_mot_sequence

        seq = generate_crowded_mot_sequence(
            n_objects=50, n_frames=30, occlusion_rate=0.2, seed=42,
        )
        # At least some frames should have fewer than 50 real detections
        real_counts = [
            sum(1 for d in frame if d.obj_id >= 0) for frame in seq
        ]
        assert min(real_counts) < 50, "No occlusions detected"

    @_skip_motmetrics
    def test_crowded_evaluation_runs(self) -> None:
        """Evaluation completes on 50-object crowded sequence."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            evaluate_tracking,
            generate_crowded_mot_sequence,
        )

        tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        tracker.eval()
        seq = generate_crowded_mot_sequence(n_objects=50, n_frames=10)
        result = evaluate_tracking(
            seq, tracker, detection_dim=4, sequence_name="crowded",
        )
        assert result.n_objects == 50
        assert result.n_frames == 10


# ====================================================================
# N.3 — Temporal Reasoning
# ====================================================================


class TestN3TemporalReasoning:
    """Tests for multi-frame causal binding with occlusion."""

    def test_temporal_generation_structure(self) -> None:
        """Temporal sequence has correct frame count and objects."""
        from prinet.nn.mot_evaluation import generate_temporal_reasoning_sequence

        seq = generate_temporal_reasoning_sequence(n_objects=8, n_frames=15)
        assert len(seq) == 15
        all_ids = set()
        for frame in seq:
            for det in frame:
                if det.obj_id >= 0:
                    all_ids.add(det.obj_id)
        assert len(all_ids) == 8

    def test_temporal_has_occlusion_gaps(self) -> None:
        """Objects disappear during occlusion windows."""
        from prinet.nn.mot_evaluation import generate_temporal_reasoning_sequence

        seq = generate_temporal_reasoning_sequence(
            n_objects=8,
            n_frames=15,
            occlusion_frames=[(0, 5, 8), (1, 5, 8), (2, 5, 8)],
        )
        # Frames 5–7 should be missing objects 0, 1, 2
        for t in range(5, 8):
            ids_in_frame = {d.obj_id for d in seq[t]}
            for occluded_id in [0, 1, 2]:
                assert occluded_id not in ids_in_frame, (
                    f"Object {occluded_id} visible at frame {t} during occlusion"
                )

    def test_temporal_distractor_injection(self) -> None:
        """Distractors appear at specified frames."""
        from prinet.nn.mot_evaluation import generate_temporal_reasoning_sequence

        seq = generate_temporal_reasoning_sequence(
            n_objects=4,
            n_frames=10,
            distractor_injection_frames=[(4, 3), (5, 2)],
        )
        distractors_f4 = [d for d in seq[4] if d.obj_id == -1]
        distractors_f5 = [d for d in seq[5] if d.obj_id == -1]
        assert len(distractors_f4) == 3
        assert len(distractors_f5) == 2

    def test_attention_tracker_baseline(self) -> None:
        """AttentionTracker can match detections across frames."""
        _seed()
        from prinet.nn.mot_evaluation import AttentionTracker

        tracker = AttentionTracker(detection_dim=4, match_threshold=0.0)
        dets_t = torch.randn(5, 4)
        dets_t1 = torch.randn(5, 4)
        matches, sim = tracker(dets_t, dets_t1)
        assert matches.shape == (5,)
        assert sim.shape == (5, 5)

    @_skip_motmetrics
    def test_phase_vs_attention_comparison(self) -> None:
        """PhaseTracker and AttentionTracker can both be evaluated."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            AttentionTracker,
            evaluate_tracking,
            generate_temporal_reasoning_sequence,
        )

        seq = generate_temporal_reasoning_sequence(n_objects=4, n_frames=10)

        phase_tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        phase_tracker.eval()
        result_phase = evaluate_tracking(
            seq, phase_tracker, detection_dim=4, sequence_name="phase",
        )

        attn_tracker = AttentionTracker(detection_dim=4, match_threshold=0.0)
        attn_tracker.eval()
        result_attn = evaluate_tracking(
            seq, attn_tracker, detection_dim=4, sequence_name="attn",
        )

        # Both should produce valid results (not assertion on who wins —
        # that's for the benchmark)
        assert -1.0 <= result_phase.mota <= 1.0
        assert -1.0 <= result_attn.mota <= 1.0


# ====================================================================
# N.5 — Subconscious A/B Test
# ====================================================================


class TestN5SubconsciousAB:
    """Tests for subconscious A/B testing infrastructure."""

    @_skip_motmetrics
    def test_ab_test_returns_scores(self) -> None:
        """A/B test produces per-trial MOTA lists."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            generate_linear_mot_sequence,
            run_subconscious_ab_test,
        )

        tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        tracker.eval()
        seq = generate_linear_mot_sequence(n_objects=3, n_frames=5)

        results = run_subconscious_ab_test(
            tracker, seq, n_trials=3, detection_dim=4,
        )
        assert "with_daemon" in results
        assert "without_daemon" in results
        assert len(results["with_daemon"]) == 3
        assert len(results["without_daemon"]) == 3

    @_skip_motmetrics
    @_skip_scipy
    def test_ab_test_welch_t_test(self) -> None:
        """Welch's t-test can be computed from A/B results."""
        _seed()
        from prinet.nn.hybrid import PhaseTracker
        from prinet.nn.mot_evaluation import (
            generate_linear_mot_sequence,
            run_subconscious_ab_test,
        )

        tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
        tracker.eval()
        seq = generate_linear_mot_sequence(n_objects=3, n_frames=5)

        results = run_subconscious_ab_test(
            tracker, seq, n_trials=5, detection_dim=4,
        )

        t_stat, p_value = ttest_ind(
            results["with_daemon"],
            results["without_daemon"],
            equal_var=False,
        )
        # p_value can be NaN if all values identical (deterministic tracker)
        assert isinstance(float(p_value), float)
        assert math.isnan(p_value) or (0.0 <= p_value <= 1.0)


# ====================================================================
# Integration tests
# ====================================================================


class TestQ2Integration:
    """Cross-module integration tests."""

    def test_dynamic_tracker_with_crowded_scene(self) -> None:
        """DynamicPhaseTracker scales oscillators for large scenes."""
        _seed()
        from prinet.nn.adaptive_allocation import DynamicPhaseTracker

        tracker = DynamicPhaseTracker(
            detection_dim=4, min_total=12, max_total=64, max_objects=50,
        )
        # Small scene
        dets_small_t = torch.randn(3, 4)
        dets_small_t1 = torch.randn(3, 4)
        _, _, budget_small = tracker(dets_small_t, dets_small_t1)

        # Large scene
        dets_large_t = torch.randn(40, 4)
        dets_large_t1 = torch.randn(40, 4)
        _, _, budget_large = tracker(dets_large_t, dets_large_t1)

        assert budget_large.total > budget_small.total, (
            f"Large scene ({budget_large.total}) should have more oscillators "
            f"than small scene ({budget_small.total})"
        )

    @_skip_motmetrics
    def test_dynamic_tracker_evaluation(self) -> None:
        """DynamicPhaseTracker works with full evaluation pipeline."""
        _seed()
        from prinet.nn.adaptive_allocation import DynamicPhaseTracker
        from prinet.nn.mot_evaluation import (
            evaluate_tracking,
            generate_linear_mot_sequence,
        )

        tracker = DynamicPhaseTracker(
            detection_dim=4, min_total=12, max_total=28,
        )
        seq = generate_linear_mot_sequence(n_objects=5, n_frames=8)
        result = evaluate_tracking(seq, tracker, detection_dim=4)
        assert -1.0 <= result.mota <= 1.0

    def test_imports_from_nn_init(self) -> None:
        """All Q2 symbols importable from prinet.nn."""
        from prinet.nn import (
            AdaptiveOscillatorAllocator,
            AttentionTracker,
            Detection,
            DynamicPhaseTracker,
            OscillatorBudget,
            TrackingResult,
            estimate_complexity,
            evaluate_tracking,
            generate_crowded_mot_sequence,
            generate_linear_mot_sequence,
            generate_temporal_reasoning_sequence,
        )

    def test_allocator_sweep_matches_complexity_range(self) -> None:
        """Swept budgets cover full [0, 1] complexity range."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budgets = alloc.sweep_complexity(11)
        assert budgets[0].complexity == pytest.approx(0.0)
        assert budgets[-1].complexity == pytest.approx(1.0)
        assert len(budgets) == 11


# ====================================================================
# Hypothesis property-based tests
# ====================================================================


@_skip_hypothesis
class TestQ2HypothesisProperties:
    """Property-based tests for Q2 components."""

    @given(
        complexity=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_allocator_always_valid(self, complexity: float) -> None:
        """Rule allocator always returns valid budget for any complexity."""
        from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

        alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        budget = alloc.allocate(complexity)
        assert budget.n_delta >= 1
        assert budget.n_theta >= 1
        assert budget.n_gamma >= 1
        assert budget.total >= 12
        assert budget.total <= 64

    @given(
        n_objects=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_complexity_bounded(self, n_objects: int) -> None:
        """estimate_complexity always returns value in [0, 1]."""
        from prinet.nn.adaptive_allocation import estimate_complexity

        dets = torch.rand(n_objects, 4)
        c = estimate_complexity(dets, max_objects=50)
        assert 0.0 <= c.item() <= 1.0

    @given(
        n_objects=st.integers(min_value=2, max_value=20),
        n_frames=st.integers(min_value=3, max_value=15),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_linear_mot_frame_count(self, n_objects: int, n_frames: int) -> None:
        """Linear MOT generator always produces correct frame count."""
        from prinet.nn.mot_evaluation import generate_linear_mot_sequence

        seq = generate_linear_mot_sequence(
            n_objects=n_objects, n_frames=n_frames, miss_rate=0.0,
        )
        assert len(seq) == n_frames
        # With miss_rate=0, every frame has all objects
        for frame in seq:
            assert len(frame) == n_objects
