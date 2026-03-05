"""Year 4 Q1 tests — paper-critical experiment infrastructure.

Covers:
- T.1 Ring/small-world topology (ring_topology, small_world_topology).
- T.2 Chimera detection utilities (local_order_parameter, bimodality_index).
- T.3 Temporal Slot Attention MOT (TemporalSlotAttentionMOT).
- T.5 Ablation framework (AblationHybridPRINetV2, create_ablation_model).
- T.6 FLOPs counting and wall-time measurement (count_flops, measure_wall_time).
- OscilloSim ring/small_world coupling integration.
"""

import math

import pytest
import torch

from prinet.utils.oscillosim import (
    OscilloSim,
    bimodality_index,
    local_order_parameter,
    ring_topology,
    small_world_topology,
)
from prinet.utils.y4q1_tools import (
    AblationConfig,
    AblationHybridPRINetV2,
    count_flops,
    create_ablation_model,
    measure_wall_time,
)
from prinet.nn.slot_attention import TemporalSlotAttentionMOT


# =====================================================================
# T.1: Ring / Small-World Topology
# =====================================================================


class TestRingTopology:
    """Tests for :func:`ring_topology`."""

    def test_shape(self) -> None:
        N, k = 64, 4
        idx = ring_topology(N, k)
        assert idx.shape == (N, k)

    def test_dtype(self) -> None:
        idx = ring_topology(16, 4)
        assert idx.dtype == torch.int64

    def test_values_in_range(self) -> None:
        N, k = 32, 6
        idx = ring_topology(N, k)
        assert idx.min().item() >= 0
        assert idx.max().item() < N

    def test_no_self_loop(self) -> None:
        N, k = 32, 6
        idx = ring_topology(N, k)
        for i in range(N):
            assert i not in idx[i].tolist()

    def test_symmetry(self) -> None:
        """Neighbors should be symmetric around each node."""
        N, k = 16, 4
        idx = ring_topology(N, k)
        for i in range(N):
            nbrs = idx[i].tolist()
            assert len(nbrs) == k
            # Should include k/2 neighbors on each side
            for offset in range(1, k // 2 + 1):
                right = (i + offset) % N
                left = (i - offset) % N
                assert right in nbrs
                assert left in nbrs

    def test_k_equals_2(self) -> None:
        idx = ring_topology(10, 2)
        # Node 0 should have neighbors 1 and 9
        assert set(idx[0].tolist()) == {1, 9}

    def test_device_cpu(self) -> None:
        idx = ring_topology(8, 2, device="cpu")
        assert idx.device.type == "cpu"


class TestSmallWorldTopology:
    """Tests for :func:`small_world_topology`."""

    def test_shape(self) -> None:
        N, k = 64, 4
        idx = small_world_topology(N, k, p_rewire=0.3)
        assert idx.shape == (N, k)

    def test_dtype(self) -> None:
        idx = small_world_topology(16, 4, p_rewire=0.1)
        assert idx.dtype == torch.int64

    def test_values_in_range(self) -> None:
        N, k = 32, 6
        idx = small_world_topology(N, k, p_rewire=0.5)
        assert idx.min().item() >= 0
        assert idx.max().item() < N

    def test_no_self_loop(self) -> None:
        N, k = 32, 6
        idx = small_world_topology(N, k, p_rewire=0.5, seed=42)
        for i in range(N):
            assert i not in idx[i].tolist()

    def test_zero_rewire_equals_ring(self) -> None:
        """p_rewire=0 should produce identical ring topology."""
        N, k = 16, 4
        ring = ring_topology(N, k)
        sw = small_world_topology(N, k, p_rewire=0.0, seed=0)
        assert torch.equal(ring, sw)

    def test_high_rewire_differs_from_ring(self) -> None:
        """p_rewire=1.0 should differ substantially from ring."""
        N, k = 32, 6
        ring = ring_topology(N, k)
        sw = small_world_topology(N, k, p_rewire=1.0, seed=42)
        # At least some entries should differ
        assert not torch.equal(ring, sw)

    def test_reproducible_with_seed(self) -> None:
        a = small_world_topology(16, 4, p_rewire=0.5, seed=99)
        b = small_world_topology(16, 4, p_rewire=0.5, seed=99)
        assert torch.equal(a, b)


# =====================================================================
# T.2: Chimera Detection Utilities
# =====================================================================


class TestLocalOrderParameter:
    """Tests for :func:`local_order_parameter`."""

    def test_output_shape(self) -> None:
        N, k = 64, 4
        phase = torch.randn(N)
        nbr_idx = ring_topology(N, k)
        r = local_order_parameter(phase, nbr_idx)
        assert r.shape == (N,)

    def test_all_synchronized_gives_high_r(self) -> None:
        """All phases equal → R ≈ 1 everywhere."""
        N, k = 32, 4
        phase = torch.zeros(N)
        nbr_idx = ring_topology(N, k)
        r = local_order_parameter(phase, nbr_idx)
        assert torch.allclose(r, torch.ones(N), atol=1e-5)

    def test_uniform_random_gives_low_r(self) -> None:
        """Uniformly random phases → low R on average."""
        torch.manual_seed(0)
        N, k = 256, 4
        phase = torch.rand(N) * 2 * math.pi
        nbr_idx = ring_topology(N, k)
        r = local_order_parameter(phase, nbr_idx)
        assert r.mean().item() < 0.8

    def test_values_in_01(self) -> None:
        """R should be in [0, 1]."""
        phase = torch.randn(64)
        nbr_idx = ring_topology(64, 6)
        r = local_order_parameter(phase, nbr_idx)
        assert r.min().item() >= -1e-6
        assert r.max().item() <= 1.0 + 1e-6


class TestBimodalityIndex:
    """Tests for :func:`bimodality_index`."""

    def test_unimodal_below_threshold(self) -> None:
        """Normal distribution → BC < 5/9."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        bc = bimodality_index(x)
        assert bc < 0.555  # 5/9 threshold

    def test_bimodal_above_threshold(self) -> None:
        """Two well-separated peaks → BC > 5/9."""
        torch.manual_seed(42)
        cluster_a = torch.randn(500) - 5.0
        cluster_b = torch.randn(500) + 5.0
        x = torch.cat([cluster_a, cluster_b])
        bc = bimodality_index(x)
        assert bc > 0.555

    def test_returns_float(self) -> None:
        bc = bimodality_index(torch.randn(100))
        assert isinstance(bc, float)


# =====================================================================
# T.1 + T.2 Integration: OscilloSim ring/small_world coupling
# =====================================================================


class TestOscilloSimRingMode:
    """OscilloSim with ring coupling mode."""

    def test_ring_mode_runs(self) -> None:
        sim = OscilloSim(
            n_oscillators=32, coupling_mode="ring",
            k_neighbors=4, coupling_strength=2.0,
        )
        result = sim.run(n_steps=50, record_trajectory=True, record_interval=1)
        assert result.trajectory_phase is not None
        assert result.trajectory_phase.shape[1] == 32
        assert result.final_phase.shape == (32,)

    def test_ring_mode_with_phase_lag(self) -> None:
        sim = OscilloSim(
            n_oscillators=32, coupling_mode="ring",
            k_neighbors=4, coupling_strength=2.0,
            phase_lag=1.457,
        )
        result = sim.run(n_steps=50)
        assert result.final_phase.shape == (32,)
        assert result.coupling_mode == "ring"

    def test_small_world_mode_runs(self) -> None:
        sim = OscilloSim(
            n_oscillators=32, coupling_mode="small_world",
            k_neighbors=4, coupling_strength=2.0,
            p_rewire=0.3,
        )
        result = sim.run(n_steps=50)
        assert result.final_phase.shape == (32,)
        assert result.coupling_mode == "small_world"

    def test_state_summary_includes_mode(self) -> None:
        sim = OscilloSim(
            n_oscillators=16, coupling_mode="ring",
            k_neighbors=4, coupling_strength=1.0,
        )
        summary = sim.state_summary()
        assert summary["coupling_mode"] == "ring"


# =====================================================================
# T.3: Temporal Slot Attention MOT
# =====================================================================


class TestTemporalSlotAttentionMOT:
    """Tests for :class:`TemporalSlotAttentionMOT`."""

    def test_construction(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=6, slot_dim=32,
        )
        assert model.num_slots == 6
        assert model.slot_dim == 32

    def test_process_single_frame(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=4, slot_dim=16, num_iterations=2,
        )
        dets = torch.randn(5, 4)  # 5 detections, dim=4
        slots = model.process_frame(dets, prev_slots=None)
        # Output is (1, num_slots, slot_dim) with batch dim
        assert slots.shape[-2:] == (4, 16)

    def test_process_frame_with_carry(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=4, slot_dim=16, num_iterations=2,
        )
        dets = torch.randn(5, 4)
        prev = torch.randn(1, 4, 16)
        slots = model.process_frame(dets, prev_slots=prev)
        assert slots.shape[-2:] == (4, 16)

    def test_slot_similarity(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=4, slot_dim=16,
        )
        a = torch.randn(4, 16)
        b = torch.randn(4, 16)
        sim = model.slot_similarity(a, b)
        assert sim.shape == (4, 4)
        # Self-similarity diagonal should be close to 1
        self_sim = model.slot_similarity(a, a)
        diag = torch.diag(self_sim)
        assert diag.min().item() > 0.9

    def test_track_sequence(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=4, slot_dim=16, num_iterations=2,
        )
        frames = [torch.randn(5, 4) for _ in range(3)]
        result = model.track_sequence(frames)
        assert "slot_history" in result
        assert len(result["slot_history"]) == 3
        assert "identity_preservation" in result
        assert isinstance(result["identity_preservation"], float)

    def test_track_empty_sequence(self) -> None:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=4, slot_dim=16,
        )
        result = model.track_sequence([])
        assert result["identity_preservation"] == 0.0


# =====================================================================
# T.5: Ablation Framework
# =====================================================================


class TestAblationConfig:
    """Tests for :class:`AblationConfig`."""

    def test_default_variant(self) -> None:
        cfg = AblationConfig()
        assert cfg.variant == "full"

    def test_custom_fields(self) -> None:
        cfg = AblationConfig(variant="oscillator_only", d_model=128)
        assert cfg.d_model == 128
        assert cfg.variant == "oscillator_only"


class TestAblationHybridPRINetV2:
    """Tests for :class:`AblationHybridPRINetV2`."""

    @pytest.fixture(params=["full", "attention_only", "oscillator_only", "shared_phase"])
    def variant(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_forward_shape(self, variant: str) -> None:
        model = create_ablation_model(variant, n_input=64, n_classes=5)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 5)

    def test_forward_1d(self, variant: str) -> None:
        model = create_ablation_model(variant, n_input=64, n_classes=5)
        x = torch.randn(64)
        out = model(x)
        assert out.shape == (5,)

    def test_log_probabilities(self, variant: str) -> None:
        """Output should be valid log-probabilities (sum to ~1 in exp)."""
        model = create_ablation_model(variant, n_input=64, n_classes=5)
        x = torch.randn(4, 64)
        out = model(x)
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones(4), atol=1e-4)

    def test_variant_stored(self, variant: str) -> None:
        model = create_ablation_model(variant)
        assert model.variant == variant

    def test_attention_only_has_no_dynamics(self) -> None:
        model = create_ablation_model("attention_only")
        assert model.dynamics is None

    def test_oscillator_only_has_no_attention(self) -> None:
        model = create_ablation_model("oscillator_only")
        assert model.attn_layers is None

    def test_full_has_all_components(self) -> None:
        model = create_ablation_model("full")
        assert model.dynamics is not None
        assert model.attn_layers is not None

    def test_shared_phase_has_dynamics(self) -> None:
        model = create_ablation_model("shared_phase")
        assert model.dynamics is not None
        assert model._shared_phase is True

    def test_parameter_counts_differ(self) -> None:
        """Different variants should have different parameter counts
        (except full and shared_phase which may be identical)."""
        full = create_ablation_model("full", n_input=64, n_classes=5)
        attn = create_ablation_model("attention_only", n_input=64, n_classes=5)
        osc = create_ablation_model("oscillator_only", n_input=64, n_classes=5)
        p_full = sum(p.numel() for p in full.parameters())
        p_attn = sum(p.numel() for p in attn.parameters())
        p_osc = sum(p.numel() for p in osc.parameters())
        # attention_only and oscillator_only should differ from each other
        assert p_attn != p_osc


# =====================================================================
# T.6: FLOPs and Wall-Time
# =====================================================================


class TestCountFlops:
    """Tests for :func:`count_flops`."""

    def test_returns_dict_keys(self) -> None:
        model = create_ablation_model("attention_only", n_input=64, n_classes=5)
        result = count_flops(model, (4, 64))
        assert "total_flops" in result
        assert "total_params" in result
        assert "layer_flops" in result

    def test_total_flops_positive(self) -> None:
        model = create_ablation_model("full", n_input=64, n_classes=5)
        result = count_flops(model, (4, 64))
        assert result["total_flops"] > 0

    def test_total_params_positive(self) -> None:
        model = create_ablation_model("full", n_input=64, n_classes=5)
        result = count_flops(model, (4, 64))
        assert result["total_params"] > 0

    def test_layer_details_non_empty(self) -> None:
        model = create_ablation_model("full", n_input=64, n_classes=5)
        result = count_flops(model, (4, 64))
        assert len(result["layer_flops"]) > 0


class TestMeasureWallTime:
    """Tests for :func:`measure_wall_time`."""

    def test_returns_dict_keys(self) -> None:
        model = create_ablation_model("attention_only", n_input=64, n_classes=5)
        x = torch.randn(4, 64)
        result = measure_wall_time(model, x, n_warmup=1, n_runs=3)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result

    def test_mean_positive(self) -> None:
        model = create_ablation_model("attention_only", n_input=64, n_classes=5)
        x = torch.randn(4, 64)
        result = measure_wall_time(model, x, n_warmup=1, n_runs=3)
        assert result["mean_ms"] > 0.0

    def test_min_le_max(self) -> None:
        model = create_ablation_model("attention_only", n_input=64, n_classes=5)
        x = torch.randn(4, 64)
        result = measure_wall_time(model, x, n_warmup=1, n_runs=3)
        assert result["min_ms"] <= result["max_ms"]
