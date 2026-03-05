"""Year 4 Q1.4 tests — temporal advantage deepening.

Covers:
- PhaseTracker.track_sequence — sequence-level tracking API.
- phase_slip_rate — phase discontinuity counting.
- binding_persistence — per-object match persistence.
- coherence_decay_rate — exponential fit to coherence decay.
- rebinding_speed — recovery time after perturbation.
- cross_frequency_coupling — PAC between frequency bands.
- temporal_advantage_report — head-to-head comparison structure.
- Head-to-head PhaseTracker vs TemporalSlotAttentionMOT integration.
- Property-based tests (hypothesis) for metric contracts.
"""

import math

import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from prinet.nn.hybrid import PhaseTracker
from prinet.nn.slot_attention import TemporalSlotAttentionMOT
from prinet.utils.y4q1_tools import (
    binding_persistence,
    bootstrap_ci,
    coherence_decay_rate,
    cross_frequency_coupling,
    phase_slip_rate,
    rebinding_speed,
    temporal_advantage_report,
    welch_t_test,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def phase_tracker() -> PhaseTracker:
    """PhaseTracker with small config for fast tests."""
    torch.manual_seed(42)
    return PhaseTracker(
        detection_dim=4,
        n_delta=2,
        n_theta=4,
        n_gamma=8,
        n_discrete_steps=3,
        match_threshold=0.1,
    )


@pytest.fixture()
def slot_tracker() -> TemporalSlotAttentionMOT:
    """TemporalSlotAttentionMOT with small config for fast tests."""
    torch.manual_seed(42)
    return TemporalSlotAttentionMOT(
        detection_dim=4,
        num_slots=6,
        slot_dim=32,
        num_iterations=2,
        match_threshold=0.1,
    )


def _make_smooth_sequence(
    n_frames: int, n_objects: int, det_dim: int = 4, seed: int = 42,
) -> list[torch.Tensor]:
    """Generate a smooth detection sequence (slowly drifting features)."""
    torch.manual_seed(seed)
    base = torch.randn(n_objects, det_dim)
    frames = []
    for t in range(n_frames):
        noise = 0.02 * t * torch.randn(n_objects, det_dim)
        frames.append(base + noise)
    return frames


# =========================================================================
# A. PhaseTracker.track_sequence tests
# =========================================================================


class TestPhaseTrackerTrackSequence:
    """Tests for PhaseTracker.track_sequence method."""

    def test_basic_output_keys(self, phase_tracker: PhaseTracker) -> None:
        """track_sequence returns all expected keys."""
        frames = _make_smooth_sequence(5, 3)
        result = phase_tracker.track_sequence(frames)
        assert "phase_history" in result
        assert "identity_matches" in result
        assert "identity_preservation" in result
        assert "per_frame_similarity" in result
        assert "per_frame_phase_correlation" in result

    def test_phase_history_length(self, phase_tracker: PhaseTracker) -> None:
        """Phase history has one entry per frame."""
        frames = _make_smooth_sequence(8, 4)
        result = phase_tracker.track_sequence(frames)
        assert len(result["phase_history"]) == 8

    def test_identity_matches_length(self, phase_tracker: PhaseTracker) -> None:
        """Identity matches has T-1 entries."""
        frames = _make_smooth_sequence(10, 3)
        result = phase_tracker.track_sequence(frames)
        assert len(result["identity_matches"]) == 9

    def test_per_frame_similarity_length(self, phase_tracker: PhaseTracker) -> None:
        """Per-frame similarity has T-1 entries."""
        frames = _make_smooth_sequence(7, 3)
        result = phase_tracker.track_sequence(frames)
        assert len(result["per_frame_similarity"]) == 6

    def test_phase_correlation_length(self, phase_tracker: PhaseTracker) -> None:
        """Phase correlation has T-1 entries."""
        frames = _make_smooth_sequence(6, 3)
        result = phase_tracker.track_sequence(frames)
        assert len(result["per_frame_phase_correlation"]) == 5

    def test_identity_preservation_range(self, phase_tracker: PhaseTracker) -> None:
        """Identity preservation is in [0, 1]."""
        frames = _make_smooth_sequence(10, 4)
        result = phase_tracker.track_sequence(frames)
        ip = result["identity_preservation"]
        assert 0.0 <= ip <= 1.0

    def test_single_frame_sequence(self, phase_tracker: PhaseTracker) -> None:
        """Single frame sequence: no matches, IP = 0."""
        frames = [torch.randn(3, 4)]
        result = phase_tracker.track_sequence(frames)
        assert len(result["phase_history"]) == 1
        assert len(result["identity_matches"]) == 0
        assert result["identity_preservation"] == 0.0

    def test_two_frame_sequence(self, phase_tracker: PhaseTracker) -> None:
        """Two-frame sequence: 1 match transition."""
        frames = _make_smooth_sequence(2, 3)
        result = phase_tracker.track_sequence(frames)
        assert len(result["identity_matches"]) == 1
        assert len(result["per_frame_similarity"]) == 1

    def test_phase_history_shapes(self, phase_tracker: PhaseTracker) -> None:
        """Each phase tensor has shape (N_objects, n_osc)."""
        n_obj = 5
        frames = _make_smooth_sequence(4, n_obj)
        result = phase_tracker.track_sequence(frames)
        for ph in result["phase_history"]:
            assert ph.shape == (n_obj, phase_tracker.n_osc)

    def test_determinism(self, phase_tracker: PhaseTracker) -> None:
        """track_sequence is deterministic for same inputs."""
        frames = _make_smooth_sequence(5, 3, seed=99)
        r1 = phase_tracker.track_sequence(frames)
        r2 = phase_tracker.track_sequence(frames)
        assert r1["identity_preservation"] == r2["identity_preservation"]
        for s1, s2 in zip(r1["per_frame_similarity"], r2["per_frame_similarity"]):
            assert abs(s1 - s2) < 1e-6


# =========================================================================
# B. phase_slip_rate tests
# =========================================================================


class TestPhaseSlipRate:
    """Tests for phase_slip_rate metric."""

    def test_no_slips_smooth_trajectory(self) -> None:
        """Smooth linear phase advance → zero slips."""
        T, N = 50, 10
        t = torch.linspace(0, 1, T).unsqueeze(1).expand(T, N)
        trajectory = t * 0.1  # slow advance, never exceeds π
        result = phase_slip_rate(trajectory)
        assert result["total_slips"] == 0
        assert result["slip_fraction"] == 0.0

    def test_all_slips_alternating_antipodal(self) -> None:
        """Alternating 0 / π phases → every step is a near-π slip."""
        T, N = 100, 20
        trajectory = torch.zeros(T, N)
        trajectory[1::2] = math.pi  # odd frames at π → circular dist = π
        # Use default threshold (0.8π) — circular dist π > 0.8π ✓
        result = phase_slip_rate(trajectory)
        assert result["total_slips"] > 0
        assert 0.0 < result["slip_fraction"] <= 1.0

    def test_single_step(self) -> None:
        """T=2: single transition."""
        trajectory = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        result = phase_slip_rate(trajectory)
        assert result["total_slips"] == 0
        assert result["slips_per_step"] == 0.0

    def test_shape_validation(self) -> None:
        """Rejects non-2D tensors."""
        with pytest.raises(ValueError, match="Expected 2-D"):
            phase_slip_rate(torch.rand(10))

    def test_per_oscillator_length(self) -> None:
        """per_oscillator_slips has correct length."""
        trajectory = torch.rand(20, 5) * 2 * math.pi
        result = phase_slip_rate(trajectory)
        assert len(result["per_oscillator_slips"]) == 5

    def test_custom_threshold(self) -> None:
        """Lower threshold → more slips detected."""
        torch.manual_seed(42)
        trajectory = torch.rand(50, 10) * 2 * math.pi
        result_high = phase_slip_rate(trajectory, threshold=math.pi * 0.9)
        result_low = phase_slip_rate(trajectory, threshold=0.1)
        assert result_low["total_slips"] >= result_high["total_slips"]

    def test_short_trajectory_edge(self) -> None:
        """T=1: degenerate case."""
        trajectory = torch.zeros(1, 5)
        result = phase_slip_rate(trajectory)
        assert result["total_slips"] == 0
        assert result["slip_fraction"] == 0.0


# =========================================================================
# C. binding_persistence tests
# =========================================================================


class TestBindingPersistence:
    """Tests for binding_persistence metric."""

    def test_perfect_persistence(self) -> None:
        """All matches valid → persistence = 1.0."""
        matches = [torch.tensor([0, 1, 2])] * 10
        result = binding_persistence(matches, n_objects=3)
        assert result["mean_persistence"] == 1.0
        assert result["min_persistence"] == 1.0

    def test_zero_persistence(self) -> None:
        """All unmatched → persistence = 0.0."""
        matches = [torch.tensor([-1, -1, -1])] * 5
        result = binding_persistence(matches, n_objects=3)
        assert result["mean_persistence"] == 0.0

    def test_partial_persistence(self) -> None:
        """Mix of matched/unmatched → intermediate persistence."""
        matches = [
            torch.tensor([0, -1, 2]),
            torch.tensor([0, 1, -1]),
        ]
        result = binding_persistence(matches, n_objects=3)
        # obj 0: matched 2/2 = 1.0
        # obj 1: matched 1/2 = 0.5
        # obj 2: matched 1/2 = 0.5
        assert abs(result["mean_persistence"] - 2.0 / 3.0) < 1e-6
        assert result["min_persistence"] == 0.5

    def test_empty_history(self) -> None:
        """Empty history → all zeros."""
        result = binding_persistence([], n_objects=3)
        assert result["mean_persistence"] == 0.0
        assert result["n_frames"] == 0

    def test_per_object_length(self) -> None:
        """per_object_persistence has correct length."""
        matches = [torch.tensor([0, 1])] * 5
        result = binding_persistence(matches, n_objects=2)
        assert len(result["per_object_persistence"]) == 2


# =========================================================================
# D. coherence_decay_rate tests
# =========================================================================


class TestCoherenceDecayRate:
    """Tests for coherence_decay_rate metric."""

    def test_constant_coherence(self) -> None:
        """Constant coherence → ~zero decay."""
        series = [0.9] * 20
        result = coherence_decay_rate(series)
        assert abs(result["decay_rate"]) < 0.01
        assert result["half_life"] > 100  # essentially infinite

    def test_exponential_decay(self) -> None:
        """Known exponential → correct lambda recovery."""
        import numpy as np
        lam = 0.1
        series = [math.exp(-lam * t) for t in range(50)]
        result = coherence_decay_rate(series)
        assert abs(result["decay_rate"] - lam) < 0.02
        assert result["r_squared"] > 0.95

    def test_single_value(self) -> None:
        """Single value → default outputs."""
        result = coherence_decay_rate([0.5])
        assert result["decay_rate"] == 0.0
        assert result["half_life"] == float("inf")

    def test_positive_decay_rate(self) -> None:
        """Decaying series → positive decay rate."""
        series = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        result = coherence_decay_rate(series)
        assert result["decay_rate"] > 0

    def test_near_zero_values(self) -> None:
        """Near-zero coherence is clamped, no crash."""
        series = [0.001, 0.0001, 0.00001, 0.0]
        result = coherence_decay_rate(series)
        assert math.isfinite(result["decay_rate"])


# =========================================================================
# E. rebinding_speed tests
# =========================================================================


class TestRebindingSpeed:
    """Tests for rebinding_speed metric."""

    def test_instant_recovery(self) -> None:
        """Immediate recovery → 0 frames."""
        before = [torch.tensor([0, 1, 2])] * 5
        after = [torch.tensor([0, 1, 2])] * 5
        result = rebinding_speed(before, after, n_objects=3)
        assert result["recovery_frames"] == 0

    def test_no_recovery(self) -> None:
        """No valid matches → recovery = -1."""
        before = [torch.tensor([0, 1])] * 5
        after = [torch.tensor([-1, -1])] * 5
        result = rebinding_speed(before, after, n_objects=2)
        assert result["recovery_frames"] == -1

    def test_delayed_recovery(self) -> None:
        """Recovery after 3 frames."""
        before = [torch.tensor([0, 1])] * 5
        after = [
            torch.tensor([-1, -1]),
            torch.tensor([-1, -1]),
            torch.tensor([-1, -1]),
            torch.tensor([0, 1]),  # recovery at frame 3
            torch.tensor([0, 1]),
        ]
        result = rebinding_speed(before, after, n_objects=2)
        assert result["recovery_frames"] == 3

    def test_post_match_rates_length(self) -> None:
        """post_match_rates has one entry per after-frame."""
        before = [torch.tensor([0, 1])] * 3
        after = [torch.tensor([0, -1])] * 4
        result = rebinding_speed(before, after, n_objects=2)
        assert len(result["post_match_rates"]) == 4

    def test_empty_before(self) -> None:
        """Empty before segment → pre_rate defaults to 1.0."""
        after = [torch.tensor([0, 1])] * 3
        result = rebinding_speed([], after, n_objects=2)
        assert result["pre_match_rate"] == 1.0


# =========================================================================
# F. cross_frequency_coupling tests
# =========================================================================


class TestCrossFrequencyCoupling:
    """Tests for cross_frequency_coupling metric."""

    def test_identical_phases_zero_pac(self) -> None:
        """Identical phases → zero PAC (sin(0) = 0)."""
        phases = torch.ones(100) * 1.5
        result = cross_frequency_coupling(phases, phases)
        assert result["pac"] < 0.01

    def test_orthogonal_phases(self) -> None:
        """π/2 offset → maximal PAC."""
        low = torch.zeros(100)
        high = torch.ones(100) * math.pi / 2
        result = cross_frequency_coupling(low, high)
        # sin(-π/2) = -1, |mean| = 1.0
        assert result["pac"] > 0.9

    def test_2d_input(self) -> None:
        """2-D input (T, N) works and returns per_step values."""
        phases_low = torch.rand(10, 20) * 2 * math.pi
        phases_high = torch.rand(10, 20) * 2 * math.pi
        result = cross_frequency_coupling(phases_low, phases_high)
        assert len(result["pac_per_step"]) == 10
        assert 0.0 <= result["pac"] <= 1.0

    def test_shape_mismatch_error(self) -> None:
        """Mismatched shapes → ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            cross_frequency_coupling(torch.rand(5), torch.rand(10))

    def test_pac_bounded(self) -> None:
        """PAC metric is always in [0, 1]."""
        torch.manual_seed(123)
        for _ in range(10):
            low = torch.rand(50) * 2 * math.pi
            high = torch.rand(50) * 2 * math.pi
            result = cross_frequency_coupling(low, high)
            assert 0.0 <= result["pac"] <= 1.0


# =========================================================================
# G. temporal_advantage_report tests
# =========================================================================


class TestTemporalAdvantageReport:
    """Tests for temporal_advantage_report utility."""

    def test_basic_keys(self) -> None:
        """Report contains all expected keys."""
        pt_result = {
            "identity_preservation": 0.85,
            "per_frame_similarity": [0.7, 0.8],
            "per_frame_phase_correlation": [0.9, 0.95],
        }
        sa_result = {
            "identity_preservation": 0.75,
            "per_frame_similarity": [0.6, 0.7],
        }
        report = temporal_advantage_report(pt_result, sa_result)
        assert "ip_phase" in report
        assert "ip_slot" in report
        assert "ip_advantage" in report
        assert "mean_sim_phase" in report
        assert "mean_sim_slot" in report
        assert "mean_rho_phase" in report

    def test_advantage_positive_when_phase_wins(self) -> None:
        """ip_advantage is positive when PhaseTracker leads."""
        pt = {"identity_preservation": 0.9, "per_frame_similarity": [],
              "per_frame_phase_correlation": []}
        sa = {"identity_preservation": 0.7, "per_frame_similarity": []}
        report = temporal_advantage_report(pt, sa)
        assert report["ip_advantage"] == pytest.approx(0.2, abs=1e-6)

    def test_advantage_negative_when_slot_wins(self) -> None:
        """ip_advantage is negative when SlotAttention leads."""
        pt = {"identity_preservation": 0.5, "per_frame_similarity": [],
              "per_frame_phase_correlation": []}
        sa = {"identity_preservation": 0.8, "per_frame_similarity": []}
        report = temporal_advantage_report(pt, sa)
        assert report["ip_advantage"] < 0


# =========================================================================
# H. Integration: Head-to-head comparison
# =========================================================================


class TestHeadToHeadTracking:
    """Integration tests running both trackers on the same sequences."""

    def test_both_trackers_produce_valid_results(
        self, phase_tracker: PhaseTracker, slot_tracker: TemporalSlotAttentionMOT,
    ) -> None:
        """Both trackers run without error on the same data."""
        frames = _make_smooth_sequence(10, 4)
        pt_result = phase_tracker.track_sequence(frames)
        sa_result = slot_tracker.track_sequence(frames)

        assert 0.0 <= pt_result["identity_preservation"] <= 1.0
        assert 0.0 <= sa_result["identity_preservation"] <= 1.0

    def test_report_from_both_trackers(
        self, phase_tracker: PhaseTracker, slot_tracker: TemporalSlotAttentionMOT,
    ) -> None:
        """temporal_advantage_report works with real tracker outputs."""
        frames = _make_smooth_sequence(8, 3)
        pt_result = phase_tracker.track_sequence(frames)
        sa_result = slot_tracker.track_sequence(frames)

        report = temporal_advantage_report(pt_result, sa_result)
        assert math.isfinite(report["ip_advantage"])
        assert math.isfinite(report["mean_rho_phase"])

    def test_phase_slip_from_tracker_output(
        self, phase_tracker: PhaseTracker,
    ) -> None:
        """phase_slip_rate works on PhaseTracker phase_history."""
        frames = _make_smooth_sequence(15, 4)
        result = phase_tracker.track_sequence(frames)
        trajectory = torch.stack(result["phase_history"])  # (T, N, n_osc)
        # Average over oscillators for a per-object trajectory
        mean_phase = trajectory.mean(dim=-1)  # (T, N)
        psr = phase_slip_rate(mean_phase)
        assert psr["slip_fraction"] >= 0.0

    def test_binding_persistence_from_tracker_output(
        self, phase_tracker: PhaseTracker,
    ) -> None:
        """binding_persistence works on PhaseTracker matches."""
        frames = _make_smooth_sequence(10, 4)
        result = phase_tracker.track_sequence(frames)
        bp = binding_persistence(result["identity_matches"], n_objects=4)
        assert 0.0 <= bp["mean_persistence"] <= 1.0

    def test_coherence_decay_from_tracker_output(
        self, phase_tracker: PhaseTracker,
    ) -> None:
        """coherence_decay_rate works on per_frame_phase_correlation."""
        frames = _make_smooth_sequence(20, 4)
        result = phase_tracker.track_sequence(frames)
        rho = result["per_frame_phase_correlation"]
        if len(rho) >= 2:
            decay = coherence_decay_rate(rho)
            assert math.isfinite(decay["decay_rate"])

    def test_sequence_length_scaling(
        self, phase_tracker: PhaseTracker, slot_tracker: TemporalSlotAttentionMOT,
    ) -> None:
        """Both trackers handle increasing sequence lengths."""
        for n_frames in [5, 10, 20]:
            frames = _make_smooth_sequence(n_frames, 3)
            pt = phase_tracker.track_sequence(frames)
            sa = slot_tracker.track_sequence(frames)
            assert 0.0 <= pt["identity_preservation"] <= 1.0
            assert 0.0 <= sa["identity_preservation"] <= 1.0


# =========================================================================
# I. Property-based tests (hypothesis)
# =========================================================================


class TestPropertyBased:
    """Hypothesis property-based tests for metric contracts."""

    @given(
        n_steps=st.integers(min_value=2, max_value=50),
        n_osc=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_psr_slip_fraction_bounded(self, n_steps: int, n_osc: int) -> None:
        """slip_fraction always in [0, 1]."""
        trajectory = torch.rand(n_steps, n_osc) * 2 * math.pi
        result = phase_slip_rate(trajectory)
        assert 0.0 <= result["slip_fraction"] <= 1.0

    @given(
        n_frames=st.integers(min_value=1, max_value=30),
        n_obj=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_binding_persistence_bounded(self, n_frames: int, n_obj: int) -> None:
        """Mean persistence always in [0, 1]."""
        matches = [
            torch.randint(-1, 3, (n_obj,)) for _ in range(n_frames)
        ]
        result = binding_persistence(matches, n_objects=n_obj)
        assert 0.0 <= result["mean_persistence"] <= 1.0

    @given(
        n_values=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_coherence_decay_finite(self, n_values: int) -> None:
        """coherence_decay_rate returns finite values for random input."""
        import random
        random.seed(42)
        series = [random.uniform(0.01, 1.0) for _ in range(n_values)]
        result = coherence_decay_rate(series)
        assert math.isfinite(result["decay_rate"])
        assert math.isfinite(result["r_squared"])

    @given(
        n_osc=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_pac_bounded_property(self, n_osc: int) -> None:
        """PAC is in [0, 1] for any phase pair."""
        low = torch.rand(n_osc) * 2 * math.pi
        high = torch.rand(n_osc) * 2 * math.pi
        result = cross_frequency_coupling(low, high)
        assert 0.0 <= result["pac"] <= 1.0


# =========================================================================
# J. Version consistency
# =========================================================================


class TestVersionConsistency:
    """Ensure version consistency across the codebase."""

    def test_version_string(self) -> None:
        import prinet
        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    def test_pyproject_version(self) -> None:
        import prinet, tomllib
        from pathlib import Path
        pp = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pp.read_text())
        assert data["project"]["version"] == prinet.__version__
