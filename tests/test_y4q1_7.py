"""Year 4 Q1.7 tests — definitive temporal advantage benchmarks.

Covers:
- Temporal metrics: TS, IDSW, TFR, IOC, MT/ML, track duration, RS, BRS.
- Dataset generation: generate_temporal_clevr_n, generate_dataset.
- Loss functions: hungarian_similarity_loss, temporal_smoothness_loss.
- Parameter counting: count_parameters (complex-aware).
- TemporalTrainer: training loop, evaluation, snapshots.
- Ablation variants: PT-frozen, PT-static, SA-no-GRU, SA-frozen.
- Integration: end-to-end training + evaluation pipeline.
- Version: 2.5.0.
"""

import math

import pytest
import torch

from prinet.nn.hybrid import PhaseTracker
from prinet.nn.slot_attention import TemporalSlotAttentionMOT
from prinet.nn.ablation_variants import (
    PhaseTrackerFrozen,
    PhaseTrackerStatic,
    SlotAttentionNoGRU,
    SlotAttentionFrozen,
    create_ablation_tracker,
)
from prinet.utils.temporal_metrics import (
    TemporalMetrics,
    binding_robustness_score,
    compute_full_temporal_metrics,
    identity_overcount,
    identity_switches,
    mostly_tracked_lost,
    recovery_speed,
    temporal_smoothness,
    track_duration_stats,
    track_fragmentation_rate,
)
from prinet.utils.temporal_training import (
    MultiSeedResult,
    SequenceData,
    TemporalTrainer,
    TrainingResult,
    TrainingSnapshot,
    count_parameters,
    generate_dataset,
    generate_temporal_clevr_n,
    hungarian_similarity_loss,
    temporal_smoothness_loss,
    train_multi_seed,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def phase_tracker() -> PhaseTracker:
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
    torch.manual_seed(42)
    return TemporalSlotAttentionMOT(
        detection_dim=4,
        num_slots=6,
        slot_dim=32,
        num_iterations=2,
        match_threshold=0.1,
    )


@pytest.fixture()
def sample_sequence() -> SequenceData:
    return generate_temporal_clevr_n(
        n_objects=3,
        n_frames=10,
        det_dim=4,
        seed=42,
    )


@pytest.fixture()
def short_dataset() -> list[SequenceData]:
    return generate_dataset(
        n_sequences=5,
        n_objects=3,
        n_frames=8,
        det_dim=4,
        base_seed=42,
    )


@pytest.fixture()
def val_dataset() -> list[SequenceData]:
    return generate_dataset(
        n_sequences=3,
        n_objects=3,
        n_frames=8,
        det_dim=4,
        base_seed=9999,
    )


def _make_perfect_matches(n_frames: int, n_objects: int) -> list[torch.Tensor]:
    """Matches where each object consistently maps to itself."""
    return [torch.arange(n_objects) for _ in range(n_frames - 1)]


def _make_switch_matches(n_frames: int, n_objects: int) -> list[torch.Tensor]:
    """Matches with identity switches at every other frame."""
    matches = []
    for t in range(n_frames - 1):
        if t % 2 == 0:
            matches.append(torch.arange(n_objects))
        else:
            m = torch.arange(n_objects)
            if n_objects >= 2:
                m[0], m[1] = 1, 0  # swap first two
            matches.append(m)
    return matches


# =========================================================================
# A. Temporal Metrics Tests
# =========================================================================


class TestTemporalSmoothness:
    """Tests for temporal_smoothness metric."""

    def test_constant_position_is_smooth(self) -> None:
        pos = torch.ones(10, 3, 2) * 5.0
        ts = temporal_smoothness(pos)
        assert ts == pytest.approx(0.0, abs=1e-6)

    def test_linear_motion_is_smooth(self) -> None:
        pos = torch.zeros(10, 2, 2)
        for t in range(10):
            pos[t, :, 0] = float(t)
            pos[t, :, 1] = float(t) * 0.5
        ts = temporal_smoothness(pos)
        assert ts == pytest.approx(0.0, abs=1e-6)

    def test_jittery_motion_is_not_smooth(self) -> None:
        torch.manual_seed(42)
        pos = torch.randn(20, 4, 2) * 10.0
        ts = temporal_smoothness(pos)
        assert ts > 0.1

    def test_short_sequence_returns_zero(self) -> None:
        pos = torch.randn(2, 3, 2)
        ts = temporal_smoothness(pos)
        assert ts == pytest.approx(0.0)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected positions shape"):
            temporal_smoothness(torch.randn(10, 3))


class TestIdentitySwitches:
    """Tests for identity_switches metric."""

    def test_no_switches_with_perfect_matches(self) -> None:
        matches = _make_perfect_matches(10, 4)
        assert identity_switches(matches, 4) == 0

    def test_switches_counted_correctly(self) -> None:
        matches = _make_switch_matches(10, 4)
        sw = identity_switches(matches, 4)
        assert sw > 0

    def test_single_frame_returns_zero(self) -> None:
        assert identity_switches([torch.tensor([0, 1])], 2) == 0

    def test_empty_returns_zero(self) -> None:
        assert identity_switches([], 3) == 0


class TestTrackFragmentationRate:
    """Tests for track_fragmentation_rate metric."""

    def test_perfect_tracking_gives_one(self) -> None:
        matches = _make_perfect_matches(10, 3)
        tfr = track_fragmentation_rate(matches, 3)
        assert tfr == pytest.approx(1.0)

    def test_empty_returns_one(self) -> None:
        assert track_fragmentation_rate([], 3) == pytest.approx(1.0)

    def test_fragmented_tracking_gives_higher(self) -> None:
        # Object 0 is matched, then lost, then matched again
        matches = [
            torch.tensor([0, 1, 2]),
            torch.tensor([-1, 1, 2]),
            torch.tensor([0, 1, 2]),
        ]
        tfr = track_fragmentation_rate(matches, 3)
        assert tfr >= 1.0


class TestIdentityOvercount:
    """Tests for identity_overcount metric."""

    def test_perfect_tracking_gives_one(self) -> None:
        matches = _make_perfect_matches(10, 3)
        ioc = identity_overcount(matches, 3)
        assert ioc == pytest.approx(1.0)

    def test_empty_returns_one(self) -> None:
        assert identity_overcount([], 3) == pytest.approx(1.0)

    def test_overcounting(self) -> None:
        # More unique IDs than objects
        matches = [
            torch.tensor([0, 1, 2]),
            torch.tensor([3, 4, 5]),
        ]
        ioc = identity_overcount(matches, 3)
        assert ioc == pytest.approx(2.0)


class TestMostlyTrackedLost:
    """Tests for mostly_tracked_lost metric."""

    def test_perfect_tracking(self) -> None:
        matches = _make_perfect_matches(10, 3)
        mt, ml = mostly_tracked_lost(matches, 3)
        assert mt == pytest.approx(1.0)
        assert ml == pytest.approx(0.0)

    def test_never_tracked(self) -> None:
        matches = [torch.tensor([-1, -1]) for _ in range(10)]
        mt, ml = mostly_tracked_lost(matches, 2)
        assert mt == pytest.approx(0.0)
        assert ml == pytest.approx(1.0)

    def test_empty_returns_zero_one(self) -> None:
        mt, ml = mostly_tracked_lost([], 3)
        assert mt == pytest.approx(0.0)
        assert ml == pytest.approx(1.0)


class TestTrackDurationStats:
    """Tests for track_duration_stats metric."""

    def test_perfect_tracking_duration(self) -> None:
        matches = _make_perfect_matches(10, 3)
        mean_d, median_d = track_duration_stats(matches, 3)
        assert mean_d == pytest.approx(9.0)
        assert median_d == pytest.approx(9.0)

    def test_empty_returns_zeros(self) -> None:
        mean_d, median_d = track_duration_stats([], 3)
        assert mean_d == pytest.approx(0.0)
        assert median_d == pytest.approx(0.0)


class TestRecoverySpeed:
    """Tests for recovery_speed metric."""

    def test_no_occlusion_returns_nan(self) -> None:
        occ = torch.ones(10, 3)  # All visible
        matches = _make_perfect_matches(10, 3)
        rs = recovery_speed(matches, occ, 3)
        assert math.isnan(rs)

    def test_with_occlusion(self) -> None:
        occ = torch.ones(10, 3)
        occ[3, 0] = 0  # Object 0 occluded at frame 3
        occ[4, 0] = 0  # still occluded
        # frame 5: reappears
        matches = _make_perfect_matches(10, 3)
        rs = recovery_speed(matches, occ, 3)
        assert isinstance(rs, float)
        assert not math.isnan(rs)


class TestBindingRobustness:
    """Tests for binding_robustness_score metric."""

    def test_no_degradation(self) -> None:
        assert binding_robustness_score(0.9, 0.9) == pytest.approx(1.0)

    def test_degradation(self) -> None:
        assert binding_robustness_score(0.5, 1.0) == pytest.approx(0.5)

    def test_zero_baseline(self) -> None:
        assert binding_robustness_score(0.5, 0.0) == pytest.approx(0.0)


class TestComputeFullMetrics:
    """Tests for compute_full_temporal_metrics."""

    def test_returns_dataclass(self) -> None:
        matches = _make_perfect_matches(10, 3)
        m = compute_full_temporal_metrics(matches, 3)
        assert isinstance(m, TemporalMetrics)
        assert 0.0 <= m.ip <= 1.0
        assert m.idsw >= 0
        assert m.track_fragmentation_rate >= 1.0

    def test_with_positions(self) -> None:
        matches = _make_perfect_matches(10, 3)
        pos = torch.randn(10, 3, 2)
        m = compute_full_temporal_metrics(matches, 3, positions=pos)
        assert isinstance(m.temporal_smoothness, float)

    def test_with_occlusion(self) -> None:
        matches = _make_perfect_matches(10, 3)
        occ = torch.ones(10, 3)
        occ[3, 0] = 0
        m = compute_full_temporal_metrics(matches, 3, occlusion_mask=occ)
        assert isinstance(m.recovery_speed, float)


# =========================================================================
# B. Dataset Generation Tests
# =========================================================================


class TestGenerateTemporalClevrN:
    """Tests for generate_temporal_clevr_n."""

    def test_basic_output(self) -> None:
        seq = generate_temporal_clevr_n(n_objects=4, n_frames=20, seed=42)
        assert isinstance(seq, SequenceData)
        assert seq.n_objects == 4
        assert seq.n_frames == 20
        assert len(seq.frames) == 20

    def test_frame_shape(self) -> None:
        seq = generate_temporal_clevr_n(n_objects=3, n_frames=10, det_dim=4)
        assert seq.frames[0].shape == (3, 4)

    def test_positions_shape(self) -> None:
        seq = generate_temporal_clevr_n(n_objects=4, n_frames=15)
        assert seq.positions.shape == (15, 4, 2)
        assert seq.velocities.shape == (15, 4, 2)

    def test_identities_shape(self) -> None:
        seq = generate_temporal_clevr_n(n_objects=3, n_frames=10)
        assert seq.identities.shape == (10, 3)

    def test_occlusion_mask_shape(self) -> None:
        seq = generate_temporal_clevr_n(n_objects=3, n_frames=10)
        assert seq.occlusion_mask.shape == (10, 3)

    def test_deterministic(self) -> None:
        s1 = generate_temporal_clevr_n(seed=42)
        s2 = generate_temporal_clevr_n(seed=42)
        assert torch.allclose(s1.positions, s2.positions)

    def test_different_seeds_differ(self) -> None:
        s1 = generate_temporal_clevr_n(seed=42)
        s2 = generate_temporal_clevr_n(seed=99)
        assert not torch.allclose(s1.positions, s2.positions)

    def test_occlusion_rate(self) -> None:
        seq = generate_temporal_clevr_n(
            n_objects=4, n_frames=50, occlusion_rate=0.5, seed=42
        )
        # Should have some occluded frames
        assert (seq.occlusion_mask < 0.5).any()
        # First frame always visible
        assert seq.occlusion_mask[0].min() >= 0.5

    def test_noise_sigma(self) -> None:
        s_clean = generate_temporal_clevr_n(noise_sigma=0.0, seed=42)
        s_noisy = generate_temporal_clevr_n(noise_sigma=1.0, seed=42)
        # Noisy should differ in detection features
        assert not torch.allclose(s_clean.frames[1], s_noisy.frames[1])

    def test_reversal_count(self) -> None:
        seq = generate_temporal_clevr_n(
            n_objects=2, n_frames=20, reversal_count=3, seed=42
        )
        assert seq.n_frames == 20

    def test_swap_rate(self) -> None:
        seq = generate_temporal_clevr_n(
            n_objects=3, n_frames=20, swap_rate=0.3, seed=42
        )
        assert seq.n_frames == 20


class TestGenerateDataset:
    """Tests for generate_dataset."""

    def test_correct_length(self) -> None:
        ds = generate_dataset(10, n_objects=3, n_frames=5)
        assert len(ds) == 10

    def test_each_element_is_sequence_data(self) -> None:
        ds = generate_dataset(3, n_objects=2, n_frames=5)
        for seq in ds:
            assert isinstance(seq, SequenceData)


# =========================================================================
# C. Loss Function Tests
# =========================================================================


class TestHungarianSimilarityLoss:
    """Tests for hungarian_similarity_loss."""

    def test_identity_sim_low_loss(self) -> None:
        # Perfect similarity → low loss
        sim = torch.eye(4) * 10.0
        loss = hungarian_similarity_loss(sim, 4)
        assert loss.item() < 0.1

    def test_uniform_sim_higher_loss(self) -> None:
        sim = torch.ones(4, 4)
        loss = hungarian_similarity_loss(sim, 4)
        assert loss.item() > 1.0

    def test_empty_returns_zero(self) -> None:
        sim = torch.randn(0, 0)
        loss = hungarian_similarity_loss(sim, 0)
        assert loss.item() == pytest.approx(0.0)

    def test_gradients_flow(self) -> None:
        sim = torch.randn(3, 3, requires_grad=True)
        loss = hungarian_similarity_loss(sim, 3)
        loss.backward()
        assert sim.grad is not None


class TestTemporalSmoothnessLoss:
    """Tests for temporal_smoothness_loss."""

    def test_identical_sims_zero_loss(self) -> None:
        sim = torch.eye(3)
        loss = temporal_smoothness_loss([sim, sim, sim])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_sims_nonzero_loss(self) -> None:
        loss = temporal_smoothness_loss([torch.eye(3), torch.zeros(3, 3)])
        assert loss.item() > 0.0

    def test_single_sim_returns_zero(self) -> None:
        loss = temporal_smoothness_loss([torch.eye(3)])
        assert loss.item() == pytest.approx(0.0)


# =========================================================================
# D. Parameter Counting Tests
# =========================================================================


class TestCountParameters:
    """Tests for count_parameters."""

    def test_simple_model(self) -> None:
        m = torch.nn.Linear(10, 5)
        counts = count_parameters(m)
        assert counts["total"] == 55  # 10*5 + 5
        assert counts["trainable"] == 55
        assert counts["frozen"] == 0

    def test_frozen_model(self) -> None:
        m = torch.nn.Linear(10, 5)
        for p in m.parameters():
            p.requires_grad = False
        counts = count_parameters(m)
        assert counts["trainable"] == 0
        assert counts["frozen"] == 55

    def test_phase_tracker(self, phase_tracker: PhaseTracker) -> None:
        counts = count_parameters(phase_tracker)
        assert counts["total"] > 0
        assert counts["trainable"] > 0

    def test_slot_tracker(self, slot_tracker: TemporalSlotAttentionMOT) -> None:
        counts = count_parameters(slot_tracker)
        assert counts["total"] > 0
        assert counts["trainable"] > 0


# =========================================================================
# E. Ablation Variant Tests
# =========================================================================


class TestPhaseTrackerFrozen:
    """Tests for PhaseTrackerFrozen ablation variant."""

    def test_dynamics_frozen(self) -> None:
        torch.manual_seed(42)
        m = PhaseTrackerFrozen(detection_dim=4)
        for p in m._inner.dynamics.parameters():
            assert not p.requires_grad

    def test_encoder_trainable(self) -> None:
        m = PhaseTrackerFrozen(detection_dim=4)
        for p in m._inner.det_to_phase.parameters():
            assert p.requires_grad

    def test_forward_runs(self) -> None:
        torch.manual_seed(42)
        m = PhaseTrackerFrozen(detection_dim=4)
        d0 = torch.randn(3, 4)
        d1 = torch.randn(3, 4)
        matches, sim = m(d0, d1)
        assert matches.shape == (3,)
        assert sim.shape == (3, 3)

    def test_track_sequence_runs(self) -> None:
        torch.manual_seed(42)
        m = PhaseTrackerFrozen(detection_dim=4, match_threshold=0.1)
        frames = [torch.randn(3, 4) for _ in range(5)]
        result = m.track_sequence(frames)
        assert "identity_preservation" in result


class TestPhaseTrackerStatic:
    """Tests for PhaseTrackerStatic (no coupling)."""

    def test_no_coupling_weights(self) -> None:
        m = PhaseTrackerStatic(detection_dim=4)
        # Should not have W_delta, W_theta, W_gamma
        has_coupling = any("W_delta" in n for n, _ in m.named_parameters())
        assert not has_coupling

    def test_forward_runs(self) -> None:
        torch.manual_seed(42)
        m = PhaseTrackerStatic(detection_dim=4)
        d0 = torch.randn(3, 4)
        d1 = torch.randn(3, 4)
        matches, sim = m(d0, d1)
        assert matches.shape == (3,)
        assert sim.shape == (3, 3)

    def test_track_sequence_runs(self) -> None:
        torch.manual_seed(42)
        m = PhaseTrackerStatic(detection_dim=4, match_threshold=0.1)
        frames = [torch.randn(3, 4) for _ in range(5)]
        result = m.track_sequence(frames)
        assert "identity_preservation" in result
        assert "phase_history" in result


class TestSlotAttentionNoGRU:
    """Tests for SlotAttentionNoGRU (no temporal carry-over)."""

    def test_no_gru(self) -> None:
        m = SlotAttentionNoGRU(detection_dim=4, num_slots=4, slot_dim=32)
        # Should not have temporal_gru attribute
        assert not hasattr(m, "temporal_gru")

    def test_forward_runs(self) -> None:
        torch.manual_seed(42)
        m = SlotAttentionNoGRU(detection_dim=4, num_slots=4, slot_dim=32)
        d0 = torch.randn(3, 4)
        d1 = torch.randn(3, 4)
        matches, sim = m(d0, d1)
        assert sim.shape == (4, 4)

    def test_track_sequence_runs(self) -> None:
        torch.manual_seed(42)
        m = SlotAttentionNoGRU(
            detection_dim=4, num_slots=4, slot_dim=32, match_threshold=0.1
        )
        frames = [torch.randn(3, 4) for _ in range(5)]
        result = m.track_sequence(frames)
        assert "identity_preservation" in result

    def test_process_frame_ignores_prev(self) -> None:
        torch.manual_seed(42)
        m = SlotAttentionNoGRU(detection_dim=4, num_slots=4, slot_dim=32)
        dets = torch.randn(3, 4)
        prev_slots = torch.randn(1, 4, 32)
        # NoGRU should have no GRU or temporal_norm attributes
        assert not hasattr(m, "temporal_gru")
        assert not hasattr(m, "temporal_norm")
        # process_frame still runs even with prev_slots argument
        slots = m.process_frame(dets, prev_slots)
        assert slots.shape == (1, 4, 32)


class TestSlotAttentionFrozen:
    """Tests for SlotAttentionFrozen (all weights frozen)."""

    def test_all_frozen(self) -> None:
        m = SlotAttentionFrozen(detection_dim=4)
        for p in m.parameters():
            assert not p.requires_grad

    def test_track_sequence_runs(self) -> None:
        torch.manual_seed(42)
        m = SlotAttentionFrozen(
            detection_dim=4, num_slots=4, slot_dim=32, match_threshold=0.1
        )
        frames = [torch.randn(3, 4) for _ in range(5)]
        result = m.track_sequence(frames)
        assert "identity_preservation" in result


class TestCreateAblationTracker:
    """Tests for create_ablation_tracker factory."""

    @pytest.mark.parametrize(
        "variant",
        [
            "pt_full",
            "pt_frozen",
            "pt_static",
            "sa_full",
            "sa_no_gru",
            "sa_frozen",
        ],
    )
    def test_all_variants_constructable(self, variant: str) -> None:
        torch.manual_seed(42)
        m = create_ablation_tracker(variant, detection_dim=4)
        assert isinstance(m, torch.nn.Module)

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            create_ablation_tracker("unknown")


# =========================================================================
# F. Temporal Trainer Tests
# =========================================================================


class TestTemporalTrainer:
    """Tests for TemporalTrainer class."""

    def test_pt_train_epoch(
        self, phase_tracker: PhaseTracker, short_dataset: list[SequenceData]
    ) -> None:
        trainer = TemporalTrainer(phase_tracker, lr=1e-3, max_epochs=2)
        loss = trainer.train_epoch(short_dataset)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_sa_train_epoch(
        self, slot_tracker: TemporalSlotAttentionMOT, short_dataset: list[SequenceData]
    ) -> None:
        trainer = TemporalTrainer(slot_tracker, lr=1e-3, max_epochs=2)
        loss = trainer.train_epoch(short_dataset)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_evaluate(
        self, phase_tracker: PhaseTracker, val_dataset: list[SequenceData]
    ) -> None:
        trainer = TemporalTrainer(phase_tracker, lr=1e-3)
        metrics = trainer.evaluate(val_dataset)
        assert "loss" in metrics
        assert "ip" in metrics
        assert "idsw" in metrics
        assert "tfr" in metrics

    def test_full_training_pt(
        self,
        phase_tracker: PhaseTracker,
        short_dataset: list[SequenceData],
        val_dataset: list[SequenceData],
    ) -> None:
        trainer = TemporalTrainer(
            phase_tracker,
            lr=1e-3,
            max_epochs=3,
            patience=2,
            warmup_epochs=1,
        )
        result = trainer.train(short_dataset, val_dataset)
        assert isinstance(result, TrainingResult)
        assert result.total_epochs >= 1
        assert result.wall_time_s > 0
        assert len(result.train_losses) > 0

    def test_full_training_sa(
        self,
        slot_tracker: TemporalSlotAttentionMOT,
        short_dataset: list[SequenceData],
        val_dataset: list[SequenceData],
    ) -> None:
        trainer = TemporalTrainer(
            slot_tracker,
            lr=1e-3,
            max_epochs=3,
            patience=2,
            warmup_epochs=1,
        )
        result = trainer.train(short_dataset, val_dataset)
        assert isinstance(result, TrainingResult)
        assert result.total_epochs >= 1

    def test_snapshots_captured(
        self,
        phase_tracker: PhaseTracker,
        short_dataset: list[SequenceData],
        val_dataset: list[SequenceData],
    ) -> None:
        trainer = TemporalTrainer(
            phase_tracker,
            lr=1e-3,
            max_epochs=3,
            snapshot_epochs=(0, 1, 2),
        )
        result = trainer.train(short_dataset, val_dataset)
        assert len(result.snapshots) > 0
        for snap in result.snapshots:
            assert isinstance(snap, TrainingSnapshot)
            assert snap.epoch >= 0

    def test_gradient_clipping(
        self, phase_tracker: PhaseTracker, short_dataset: list[SequenceData]
    ) -> None:
        trainer = TemporalTrainer(phase_tracker, lr=1e-3, grad_clip=0.5)
        trainer.train_epoch(short_dataset)
        # Verify gradients were clipped
        for p in phase_tracker.parameters():
            if p.grad is not None:
                assert p.grad.norm() <= 0.5 + 0.01  # small tolerance


# =========================================================================
# G. Integration Tests
# =========================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_pt_train_and_evaluate(self) -> None:
        torch.manual_seed(42)
        pt = PhaseTracker(
            detection_dim=4,
            n_delta=2,
            n_theta=4,
            n_gamma=8,
            n_discrete_steps=2,
            match_threshold=0.1,
        )
        train_data = generate_dataset(3, n_objects=3, n_frames=6, base_seed=42)
        val_data = generate_dataset(2, n_objects=3, n_frames=6, base_seed=999)
        trainer = TemporalTrainer(pt, lr=1e-3, max_epochs=2, patience=2)
        result = trainer.train(train_data, val_data)
        assert result.total_epochs >= 1

        # Evaluate with temporal metrics
        frames = [f.detach() for f in val_data[0].frames]
        tracking = pt.track_sequence(frames)
        metrics = compute_full_temporal_metrics(
            tracking["identity_matches"],
            val_data[0].n_objects,
            positions=val_data[0].positions,
        )
        assert isinstance(metrics, TemporalMetrics)

    def test_sa_train_and_evaluate(self) -> None:
        torch.manual_seed(42)
        sa = TemporalSlotAttentionMOT(
            detection_dim=4,
            num_slots=4,
            slot_dim=32,
            num_iterations=2,
            match_threshold=0.1,
        )
        train_data = generate_dataset(3, n_objects=3, n_frames=6, base_seed=42)
        val_data = generate_dataset(2, n_objects=3, n_frames=6, base_seed=999)
        trainer = TemporalTrainer(sa, lr=1e-3, max_epochs=2, patience=2)
        result = trainer.train(train_data, val_data)
        assert result.total_epochs >= 1

    def test_ablation_variants_all_trainable(self) -> None:
        """All ablation variants produce valid tracking output."""
        torch.manual_seed(42)
        frames = [torch.randn(3, 4) for _ in range(5)]
        for variant in [
            "pt_full",
            "pt_frozen",
            "pt_static",
            "sa_full",
            "sa_no_gru",
            "sa_frozen",
        ]:
            torch.manual_seed(42)
            m = create_ablation_tracker(variant, detection_dim=4, match_threshold=0.1)
            result = m.track_sequence(frames)
            assert "identity_preservation" in result

    def test_dataset_stress_variants(self) -> None:
        """Datasets with anomalies generate valid SequenceData."""
        for occ in [0.0, 0.3, 0.6]:
            seq = generate_temporal_clevr_n(
                n_objects=3, n_frames=10, occlusion_rate=occ, seed=42
            )
            assert seq.frames[0].shape == (3, 4)

        for swap in [0.0, 0.1, 0.3]:
            seq = generate_temporal_clevr_n(
                n_objects=3, n_frames=10, swap_rate=swap, seed=42
            )
            assert seq.n_frames == 10

        for rev in [0, 1, 5]:
            seq = generate_temporal_clevr_n(
                n_objects=3, n_frames=20, reversal_count=rev, seed=42
            )
            assert seq.n_frames == 20

        for noise in [0.0, 0.1, 0.5]:
            seq = generate_temporal_clevr_n(
                n_objects=3, n_frames=10, noise_sigma=noise, seed=42
            )
            assert seq.frames[0].shape == (3, 4)


# =========================================================================
# H. Version Check
# =========================================================================


class TestVersion:
    """Verify version is a valid semver."""

    def test_version(self) -> None:
        import prinet

        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)
