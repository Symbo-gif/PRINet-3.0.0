"""Year 4 Q1.9 tests -- Reviewer Gap Analysis benchmark validation.

Tests validate:
- G1: 7-seed statistical utilities (Welch's t, bootstrap CI, _train_model)
- G2: Embedding analysis (cosine similarity, effective rank)
- G3: Object/sequence scaling pipeline
- G4: Chimera metrics extraction from trained PT
- G5: Fine occlusion sweep
- G6: pt_static vs pt_full stress pipeline
- G7: Trained coherence measurement
- G8: PAC significance with null distribution

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

# Add benchmarks/ to path for importing y4q1_9_benchmarks
_BENCH_DIR = str(Path(__file__).resolve().parent.parent / "benchmarks")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from prinet.nn.ablation_variants import create_ablation_tracker
from prinet.nn.hybrid import PhaseTracker
from prinet.nn.slot_attention import TemporalSlotAttentionMOT
from prinet.utils.temporal_training import (
    SequenceData,
    generate_dataset,
    generate_temporal_clevr_n,
    count_parameters,
)
from prinet.utils.y4q1_tools import (
    PhaseTrackerLarge,
    phase_slip_rate,
    coherence_decay_rate,
    cross_frequency_coupling,
    binding_persistence,
)
from prinet.utils.oscillosim import bimodality_index

SEED = 42
DEVICE = "cpu"
DET_DIM = 4
RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def pt_model() -> PhaseTracker:
    torch.manual_seed(SEED)
    return PhaseTracker(
        detection_dim=DET_DIM,
        n_delta=2,
        n_theta=4,
        n_gamma=8,
        n_discrete_steps=3,
        match_threshold=0.1,
    )


@pytest.fixture()
def sa_model() -> TemporalSlotAttentionMOT:
    torch.manual_seed(SEED)
    return TemporalSlotAttentionMOT(
        detection_dim=DET_DIM,
        num_slots=6,
        slot_dim=32,
        num_iterations=2,
        match_threshold=0.1,
    )


@pytest.fixture()
def pt_large_model() -> PhaseTrackerLarge:
    torch.manual_seed(SEED)
    return PhaseTrackerLarge(detection_dim=DET_DIM)


@pytest.fixture()
def sample_dataset() -> list[SequenceData]:
    return generate_dataset(
        n_sequences=3,
        n_objects=3,
        n_frames=8,
        det_dim=DET_DIM,
        base_seed=SEED,
    )


@pytest.fixture()
def long_dataset() -> list[SequenceData]:
    return generate_dataset(
        n_sequences=2,
        n_objects=3,
        n_frames=30,
        det_dim=DET_DIM,
        base_seed=SEED,
    )


@pytest.fixture()
def sample_dets() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    return torch.randn(3, DET_DIM), torch.randn(3, DET_DIM)


# =========================================================================
# A. Statistical Utilities (supports G1)
# =========================================================================


class TestBootstrapCI:
    """Tests for _bootstrap_ci utility."""

    def test_ci_contains_mean(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        values = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87]
        ci = bench._bootstrap_ci(values)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
        assert ci["std"] > 0

    def test_single_value(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        ci = bench._bootstrap_ci([0.5])
        assert ci["mean"] == pytest.approx(0.5)
        assert ci["std"] == 0.0

    def test_identical_values(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        ci = bench._bootstrap_ci([0.9, 0.9, 0.9, 0.9, 0.9])
        assert ci["ci_low"] == pytest.approx(0.9)
        assert ci["ci_high"] == pytest.approx(0.9)


class TestWelchT:
    """Tests for _welch_t utility."""

    def test_identical_distributions(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        a = [0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5]
        result = bench._welch_t(a, b)
        assert result["p_value"] >= 0.99
        assert result["cohens_d"] == pytest.approx(0.0)

    def test_different_distributions(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        a = [0.9, 0.91, 0.92, 0.93, 0.94]
        b = [0.1, 0.11, 0.12, 0.13, 0.14]
        result = bench._welch_t(a, b)
        assert result["p_value"] < 0.01
        assert abs(result["cohens_d"]) > 1.0
        assert result["n_a"] == 5
        assert result["n_b"] == 5


# =========================================================================
# B. Embedding Analysis (supports G2)
# =========================================================================


class TestEmbeddingAnalysis:
    """Tests for embedding discriminability and effective rank computations."""

    def test_pt_encode_produces_phases(
        self, pt_model: PhaseTracker, sample_dets: tuple
    ) -> None:
        dets, _ = sample_dets
        phase, amp = pt_model.encode(dets)
        n_osc = 2 + 4 + 8  # n_delta + n_theta + n_gamma
        assert phase.shape == (3, n_osc)
        assert amp.shape == (3, n_osc)

    def test_pt_large_encode_produces_phases(
        self, pt_large_model: PhaseTrackerLarge, sample_dets: tuple
    ) -> None:
        dets, _ = sample_dets
        phase, amp = pt_large_model.encode(dets)
        # PT-Large: 16 + 32 + 64 = 112 oscillators
        assert phase.shape[1] == 112

    def test_cosine_similarity_computation(self) -> None:
        """Verify cosine similarity is correct for known vectors."""
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0]])
        v3 = torch.tensor([[1.0, 0.0, 0.0]])
        sim_ortho = torch.nn.functional.cosine_similarity(v1, v2).item()
        sim_same = torch.nn.functional.cosine_similarity(v1, v3).item()
        assert abs(sim_ortho) < 1e-5
        assert abs(sim_same - 1.0) < 1e-5

    def test_effective_rank_identity(self) -> None:
        """Identity matrix should have effective rank equal to its size."""
        w = torch.eye(4)
        svs = torch.linalg.svdvals(w)
        svs_norm = svs / svs.sum()
        entropy = -float((svs_norm * svs_norm.log()).sum().item())
        erank = math.exp(entropy)
        assert abs(erank - 4.0) < 0.01

    def test_effective_rank_rank1(self) -> None:
        """Rank-1 matrix should have effective rank ~1."""
        w = torch.ones(4, 4)  # rank 1
        svs = torch.linalg.svdvals(w)
        svs_norm = svs / svs.sum()
        svs_norm = svs_norm[svs_norm > 1e-10]
        entropy = -float((svs_norm * svs_norm.log()).sum().item())
        erank = math.exp(entropy)
        assert erank < 1.5


# =========================================================================
# C. Object Scaling (supports G3)
# =========================================================================


class TestObjectScaling:
    """Tests for scaling object counts in datasets and models."""

    @pytest.mark.parametrize("n_objects", [4, 8, 12])
    def test_generate_dataset_varying_objects(self, n_objects: int) -> None:
        data = generate_dataset(
            n_sequences=2,
            n_objects=n_objects,
            n_frames=8,
            det_dim=DET_DIM,
            base_seed=SEED,
        )
        assert len(data) == 2
        for seq in data:
            assert seq.n_objects == n_objects
            assert len(seq.frames) == 8
            for f in seq.frames:
                # May have fewer due to occlusion
                assert f.shape[1] == DET_DIM

    def test_pt_handles_variable_objects(self, pt_model: PhaseTracker) -> None:
        """PT should handle different numbers of detections per frame."""
        for n in [3, 6, 10]:
            frames = [torch.randn(n, DET_DIM) for _ in range(5)]
            res = pt_model.track_sequence(frames)
            assert "identity_preservation" in res

    def test_sa_handles_high_object_count(self) -> None:
        """SA with enough slots should handle high object counts."""
        torch.manual_seed(SEED)
        sa = TemporalSlotAttentionMOT(
            detection_dim=DET_DIM,
            num_slots=14,
            slot_dim=32,
            num_iterations=2,
            match_threshold=0.1,
        )
        frames = [torch.randn(12, DET_DIM) for _ in range(5)]
        res = sa.track_sequence(frames)
        assert "identity_preservation" in res


# =========================================================================
# D. Chimera Metrics in Tracking (supports G4)
# =========================================================================


class TestChimeraInTracking:
    """Tests for extracting chimera-like metrics from tracking results."""

    def test_phase_history_available(
        self, pt_model: PhaseTracker, sample_dataset: list
    ) -> None:
        seq = sample_dataset[0]
        res = pt_model.track_sequence(seq.frames)
        assert "phase_history" in res
        assert len(res["phase_history"]) > 0

    def test_phase_history_shape(
        self, pt_model: PhaseTracker, sample_dataset: list
    ) -> None:
        seq = sample_dataset[0]
        res = pt_model.track_sequence(seq.frames)
        phases = res["phase_history"]
        # Each entry: (N_obj, N_osc)
        for ph in phases:
            assert ph.dim() == 2
            n_osc = 2 + 4 + 8  # delta + theta + gamma
            assert ph.shape[1] == n_osc

    def test_bimodality_index_format(self) -> None:
        values = torch.randn(100)
        bc = bimodality_index(values)
        assert isinstance(bc, float)
        assert 0.0 <= bc  # BC is non-negative

    def test_phase_slip_rate_format(self) -> None:
        phases = torch.randn(20, 14)  # (T, N_osc)
        psr = phase_slip_rate(phases)
        assert "slip_fraction" in psr
        assert 0.0 <= psr["slip_fraction"] <= 1.0

    def test_cross_frequency_coupling_format(self) -> None:
        low = torch.randn(20)
        high = torch.randn(20)
        pac = cross_frequency_coupling(low, high)
        assert "pac" in pac
        assert 0.0 <= pac["pac"] <= 1.0

    def test_intra_inter_coherence_computation(
        self, pt_model: PhaseTracker, sample_dataset: list
    ) -> None:
        """Verify intra/inter object coherence can be computed."""
        seq = sample_dataset[0]
        res = pt_model.track_sequence(seq.frames)
        phases = torch.stack(res["phase_history"])  # (T, N_obj, N_osc)
        T, N_obj, N_osc = phases.shape

        # Intra-object coherence (complex mean resultant length)
        for obj_i in range(N_obj):
            z = torch.exp(1j * phases[:, obj_i, :].to(torch.complex64))
            r = z.mean(dim=-1).abs()  # (T,)
            assert r.shape == (T,)
            assert (r >= 0).all()
            assert (r <= 1.01).all()  # Allow slight floating point overshoot


# =========================================================================
# E. Fine Occlusion (supports G5)
# =========================================================================


class TestFineOcclusion:
    """Tests for fine-grained occlusion dataset generation."""

    @pytest.mark.parametrize("occ_rate", [0.0, 0.05, 0.10, 0.15, 0.80])
    def test_dataset_generation_with_occlusion(self, occ_rate: float) -> None:
        data = generate_dataset(
            n_sequences=2,
            n_objects=3,
            n_frames=8,
            det_dim=DET_DIM,
            occlusion_rate=occ_rate,
            base_seed=SEED,
        )
        assert len(data) == 2
        for seq in data:
            assert len(seq.frames) == 8

    def test_higher_occlusion_fewer_detections(self) -> None:
        """On average, higher occlusion should yield fewer detections."""
        rng = np.random.default_rng(SEED)
        low_data = generate_dataset(
            n_sequences=10,
            n_objects=4,
            n_frames=20,
            det_dim=DET_DIM,
            occlusion_rate=0.0,
            base_seed=100,
        )
        high_data = generate_dataset(
            n_sequences=10,
            n_objects=4,
            n_frames=20,
            det_dim=DET_DIM,
            occlusion_rate=0.8,
            base_seed=200,
        )
        low_dets = sum(f.shape[0] for seq in low_data for f in seq.frames)
        high_dets = sum(f.shape[0] for seq in high_data for f in seq.frames)
        assert high_dets <= low_dets


# =========================================================================
# F. pt_static Stress (supports G6)
# =========================================================================


class TestStaticStress:
    """Tests for pt_static ablation variant under stress conditions."""

    def test_pt_static_constructable(self) -> None:
        m = create_ablation_tracker("pt_static", detection_dim=DET_DIM)
        assert isinstance(m, torch.nn.Module)

    def test_pt_static_track_sequence(self) -> None:
        torch.manual_seed(SEED)
        m = create_ablation_tracker(
            "pt_static",
            detection_dim=DET_DIM,
            n_delta=2,
            n_theta=4,
            n_gamma=8,
            n_discrete_steps=3,
            match_threshold=0.1,
        )
        frames = [torch.randn(3, DET_DIM) for _ in range(10)]
        res = m.track_sequence(frames)
        assert "identity_preservation" in res
        assert 0.0 <= res["identity_preservation"] <= 1.0

    def test_pt_full_vs_static_both_run(self) -> None:
        """Both pt_full and pt_static should complete on stress conditions."""
        torch.manual_seed(SEED)
        pt_full = PhaseTracker(
            detection_dim=DET_DIM,
            n_delta=2,
            n_theta=4,
            n_gamma=8,
            n_discrete_steps=3,
            match_threshold=0.1,
        )
        torch.manual_seed(SEED)
        pt_static = create_ablation_tracker(
            "pt_static",
            detection_dim=DET_DIM,
            n_delta=2,
            n_theta=4,
            n_gamma=8,
            n_discrete_steps=3,
            match_threshold=0.1,
        )
        # Stress: 8 objects, 15 frames
        frames = [torch.randn(8, DET_DIM) for _ in range(15)]
        res_full = pt_full.track_sequence(frames)
        res_static = pt_static.track_sequence(frames)
        assert "identity_preservation" in res_full
        assert "identity_preservation" in res_static


# =========================================================================
# G. Trained Coherence (supports G7)
# =========================================================================


class TestTrainedCoherence:
    """Tests for coherence measurements on trained models."""

    def test_coherence_decay_rate_output_format(self) -> None:
        """coherence_decay_rate should return dict with expected keys."""
        series = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        cdr = coherence_decay_rate(series)
        assert "decay_rate" in cdr
        assert "half_life" in cdr
        assert "initial_coherence" in cdr

    def test_coherence_decay_rate_decreasing(self) -> None:
        """Monotonically decreasing series should have positive decay rate."""
        series = [1.0 * (0.95**t) for t in range(20)]
        cdr = coherence_decay_rate(series)
        assert cdr["decay_rate"] > 0
        assert cdr["half_life"] > 0

    def test_binding_persistence_perfect(self) -> None:
        """Perfect matches should have persistence = 1.0."""
        matches = [torch.arange(4) for _ in range(10)]
        bp = binding_persistence(matches, 4)
        assert bp["mean_persistence"] >= 0.99

    def test_binding_persistence_output_format(self) -> None:
        matches = [torch.arange(3) for _ in range(5)]
        bp = binding_persistence(matches, 3)
        assert "mean_persistence" in bp
        assert 0.0 <= bp["mean_persistence"] <= 1.0

    def test_phase_slip_on_constant_phases(self) -> None:
        """Constant phases should produce zero slips."""
        phases = torch.ones(20, 10)  # Constant
        psr = phase_slip_rate(phases)
        assert psr["total_slips"] == 0
        assert psr["slip_fraction"] == 0.0


# =========================================================================
# H. PAC Significance (supports G8)
# =========================================================================


class TestPACSignificance:
    """Tests for phase-amplitude coupling significance testing."""

    def test_pac_value_range(self) -> None:
        """PAC should be between 0 and 1."""
        low = torch.randn(50)
        high = torch.randn(50)
        pac = cross_frequency_coupling(low, high)
        assert 0.0 <= pac["pac"] <= 1.0

    def test_pac_identical_signals(self) -> None:
        """Identical low and high should still produce valid PAC."""
        signal = torch.sin(torch.linspace(0, 6.28, 100))
        pac = cross_frequency_coupling(signal, signal)
        assert 0.0 <= pac["pac"] <= 1.0

    def test_null_distribution_construction(self) -> None:
        """Phase-shuffled surrogates should produce a null distribution."""
        low = torch.sin(torch.linspace(0, 6.28, 50))
        high = torch.sin(torch.linspace(0, 12.56, 50))
        rng = np.random.default_rng(42)
        null_pacs = []
        for _ in range(50):  # Small n for speed
            shift = rng.integers(1, 49)
            high_shuffled = torch.roll(high, int(shift))
            pac = cross_frequency_coupling(low, high_shuffled)
            null_pacs.append(pac["pac"])
        assert len(null_pacs) == 50
        assert all(0.0 <= p <= 1.0 for p in null_pacs)

    def test_significance_computation(self) -> None:
        """p-value should be fraction of null >= observed."""
        null = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        observed = 0.08
        p_val = sum(1 for n in null if n >= observed) / len(null)
        assert p_val == pytest.approx(0.3)  # 3 out of 10

    def test_fisher_combined_p_value(self) -> None:
        """Fisher's method on independent p-values."""
        from scipy import stats

        p_values = [0.01, 0.03, 0.05]
        chi2_stat = -2 * sum(math.log(p) for p in p_values)
        combined_p = float(stats.chi2.sf(chi2_stat, df=2 * len(p_values)))
        assert combined_p < 0.01  # Very significant when combined


# =========================================================================
# I. Integration -- Benchmark Functions Import
# =========================================================================


class TestBenchmarkImports:
    """Verify all benchmark functions are importable and listed."""

    def test_all_benchmarks_list(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        names = [n for n, _ in bench.ALL_BENCHMARKS]
        expected = [
            "preregistration",
            "g1_1_7seed_comparison",
            "g1_2_7seed_noise",
            "g2_1_embedding_analysis",
            "g2_2_regularization_ablation",
            "g3_1_object_scaling",
            "g3_2_sequence_scaling",
            "g4_1_chimera_in_tracking",
            "g5_1_fine_occlusion",
            "g6_1_static_stress",
            "g7_1_trained_coherence",
            "g8_1_pac_significance",
        ]
        assert names == expected

    def test_preregistration_runs(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        result = bench.bench_preregistration()
        assert "sha256" in result
        assert len(result["sha256"]) == 64
        assert result["session"] == "Y4_Q1.9"

    def test_seeds_7_count(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        assert len(bench.SEEDS_7) == 7

    def test_seeds_3_subset(self) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        for s in bench.SEEDS_3:
            assert s in bench.SEEDS_7


# =========================================================================
# J. Training Pipeline (supports G1, G2, G3, G6)
# =========================================================================


class TestTrainingPipeline:
    """Tests for _train_model utility."""

    def test_train_model_returns_model_and_info(self, pt_model: PhaseTracker) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        # Override training config for fast test
        original_seqs = bench.TRAIN_SEQS
        original_val = bench.VAL_SEQS
        original_epochs = bench.MAX_EPOCHS
        bench.TRAIN_SEQS = 3
        bench.VAL_SEQS = 2
        bench.MAX_EPOCHS = 2
        bench.WARMUP = 0
        try:
            model, info = bench._train_model(pt_model, seed=42)
            assert isinstance(model, torch.nn.Module)
            assert "epochs" in info
            assert "wall_time_s" in info
            assert "final_val_ip" in info
            assert info["epochs"] > 0
        finally:
            bench.TRAIN_SEQS = original_seqs
            bench.VAL_SEQS = original_val
            bench.MAX_EPOCHS = original_epochs
            bench.WARMUP = 2

    def test_eval_ip_returns_list(
        self, pt_model: PhaseTracker, sample_dataset: list
    ) -> None:
        import importlib

        bench = importlib.import_module("y4q1_9_benchmarks")
        ips = bench._eval_ip(pt_model, sample_dataset, device="cpu")
        assert isinstance(ips, list)
        assert len(ips) == 3
        for ip in ips:
            assert isinstance(ip, float)
            assert 0.0 <= ip <= 1.0


# =========================================================================
# K. JSON Artefact Validation
# =========================================================================


class TestJSONArtefacts:
    """Validate structure of existing Q1.9 JSON artefacts (if present)."""

    def _load_artefact(self, name: str) -> dict | None:
        path = RESULTS_DIR / f"y4q1_9_{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def test_preregistration_hash_schema(self) -> None:
        data = self._load_artefact("preregistration_hash")
        if data is None:
            pytest.skip("y4q1_9_preregistration_hash.json not yet generated")
        assert "sha256" in data
        assert "session" in data
        assert data["session"] == "Y4_Q1.9"

    def test_7seed_comparison_schema(self) -> None:
        data = self._load_artefact("7seed_comparison")
        if data is None:
            pytest.skip("y4q1_9_7seed_comparison.json not yet generated")
        assert data["n_seeds"] == 7
        assert len(data["pt_ips"]) == 7
        assert "welch_t" in data
        assert "p_value" in data["welch_t"]

    def test_pac_significance_schema(self) -> None:
        data = self._load_artefact("pac_significance")
        if data is None:
            pytest.skip("y4q1_9_pac_significance.json not yet generated")
        assert data["n_surrogates"] == 1000
        assert "aggregate" in data
        assert "combined_p_value" in data["aggregate"]

    def test_chimera_in_tracking_schema(self) -> None:
        data = self._load_artefact("chimera_in_tracking")
        if data is None:
            pytest.skip("y4q1_9_chimera_in_tracking.json not yet generated")
        assert "chimera_like_dynamics_detected" in data
        assert "aggregate" in data

    def test_fine_occlusion_schema(self) -> None:
        data = self._load_artefact("fine_occlusion")
        if data is None:
            pytest.skip("y4q1_9_fine_occlusion.json not yet generated")
        assert "pt_degradation" in data
        assert "linear_r_squared" in data["pt_degradation"]

    def test_trained_coherence_schema(self) -> None:
        data = self._load_artefact("trained_coherence")
        if data is None:
            pytest.skip("y4q1_9_trained_coherence.json not yet generated")
        assert data["n_seeds"] == 7
        assert "comparison_to_untrained" in data
