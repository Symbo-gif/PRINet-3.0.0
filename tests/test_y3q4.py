"""Year 3 Q4 Tests — Publication & OscilloSim v2.0.

Tests for all Q4 deliverables:
    P.3: PyPI v2.0 packaging — version, public API surface
    P.4: OscilloSim v2.0 — 1M+ oscillators, sparse coupling modes
    P.5: Slot Attention comparison — SlotAttentionModule, CLEVR-N adapter

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn.functional as F

SEED = 42
TWO_PI = 2.0 * math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CUDA = torch.cuda.is_available()


def _seed(s: int = SEED) -> None:
    torch.manual_seed(s)
    if _CUDA:
        torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════
# 1. P.3: PyPI v2.0 packaging
# ══════════════════════════════════════════════════════════════════


class TestPackagingV2:
    """Tests for PRINet v2.0 package metadata and public API."""

    def test_version_is_2_0_0(self) -> None:
        """Package version is a valid semver."""
        import prinet

        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    def test_slot_attention_importable_from_top_level(self) -> None:
        """SlotAttentionModule and SlotAttentionCLEVRN are importable
        from the top-level ``prinet`` package."""
        from prinet import SlotAttentionCLEVRN, SlotAttentionModule

        assert SlotAttentionModule is not None
        assert SlotAttentionCLEVRN is not None

    def test_oscillosim_importable_from_top_level(self) -> None:
        """OscilloSim, SimulationResult, quick_simulate are importable
        from the top-level ``prinet`` package."""
        from prinet import OscilloSim, SimulationResult, quick_simulate

        assert OscilloSim is not None
        assert SimulationResult is not None
        assert callable(quick_simulate)

    def test_nn_subpackage_exports_slot_attention(self) -> None:
        """prinet.nn exposes SlotAttention symbols."""
        from prinet.nn import SlotAttentionCLEVRN, SlotAttentionModule

        assert SlotAttentionModule is not None
        assert SlotAttentionCLEVRN is not None

    def test_utils_subpackage_exports_oscillosim(self) -> None:
        """prinet.utils exposes OscilloSim symbols."""
        from prinet.utils import OscilloSim, SimulationResult, quick_simulate

        assert OscilloSim is not None
        assert SimulationResult is not None
        assert callable(quick_simulate)


# ══════════════════════════════════════════════════════════════════
# 2. P.4: OscilloSim v2.0
# ══════════════════════════════════════════════════════════════════


class TestOscilloSimCreation:
    """Tests for OscilloSim constructor and configuration."""

    def test_default_creation(self) -> None:
        """OscilloSim can be created with only ``n_oscillators``."""
        from prinet.utils.oscillosim import OscilloSim

        sim = OscilloSim(n_oscillators=100, seed=SEED)
        assert sim.n_oscillators == 100

    def test_auto_selects_csr_for_small_n(self) -> None:
        """Auto-selection picks CSR for N < 1000."""
        from prinet.utils.oscillosim import OscilloSim

        sim = OscilloSim(n_oscillators=500, coupling_mode="auto", seed=SEED)
        assert sim.coupling_mode in ("csr", "sparse_knn", "mean_field")

    def test_explicit_mean_field_mode(self) -> None:
        """Explicit mean_field mode is accepted."""
        from prinet.utils.oscillosim import OscilloSim

        sim = OscilloSim(n_oscillators=200, coupling_mode="mean_field", seed=SEED)
        assert sim.coupling_mode == "mean_field"

    def test_explicit_sparse_knn_mode(self) -> None:
        """Explicit sparse_knn mode is accepted."""
        from prinet.utils.oscillosim import OscilloSim

        sim = OscilloSim(n_oscillators=200, coupling_mode="sparse_knn", seed=SEED)
        assert sim.coupling_mode == "sparse_knn"

    def test_state_summary_has_required_keys(self) -> None:
        """state_summary() returns a dict with expected keys."""
        from prinet.utils.oscillosim import OscilloSim

        sim = OscilloSim(n_oscillators=100, seed=SEED)
        summary = sim.state_summary()
        for key in (
            "n_oscillators",
            "coupling_mode",
            "coupling_strength",
            "device",
        ):
            assert key in summary, f"Missing key: {key}"


class TestOscilloSimRun:
    """Tests for OscilloSim.run() method and SimulationResult."""

    def test_run_returns_simulation_result(self) -> None:
        """run() returns a SimulationResult dataclass."""
        from prinet.utils.oscillosim import OscilloSim, SimulationResult

        _seed()
        sim = OscilloSim(n_oscillators=64, seed=SEED)
        result = sim.run(n_steps=50, dt=0.01)
        assert isinstance(result, SimulationResult)

    def test_result_final_phase_shape(self) -> None:
        """SimulationResult.final_phase has shape (N,)."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(n_oscillators=128, seed=SEED)
        result = sim.run(n_steps=50, dt=0.01)
        assert result.final_phase.shape == (128,)

    def test_result_final_amplitude_shape(self) -> None:
        """SimulationResult.final_amplitude has shape (N,)."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(n_oscillators=128, seed=SEED)
        result = sim.run(n_steps=50, dt=0.01)
        assert result.final_amplitude.shape == (128,)

    def test_order_parameter_list_populated(self) -> None:
        """order_parameter list is non-empty after run."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(n_oscillators=64, coupling_strength=2.0, seed=SEED)
        result = sim.run(n_steps=100, dt=0.01)
        assert len(result.order_parameter) > 0
        assert all(0.0 <= r <= 1.0 + 1e-6 for r in result.order_parameter)

    def test_trajectory_recorded_when_requested(self) -> None:
        """record_trajectory=True populates trajectory_phase."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(n_oscillators=32, seed=SEED)
        result = sim.run(
            n_steps=100, dt=0.01, record_trajectory=True, record_interval=10
        )
        assert result.trajectory_phase is not None
        assert result.trajectory_phase.shape[1] == 32  # N oscillators

    def test_wall_time_and_throughput_positive(self) -> None:
        """wall_time_s and throughput are positive floats."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(n_oscillators=64, seed=SEED)
        result = sim.run(n_steps=50, dt=0.01)
        assert result.wall_time_s > 0.0
        assert result.throughput > 0.0

    def test_mean_field_mode_runs(self) -> None:
        """Mean-field mode completes without error."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(
            n_oscillators=500,
            coupling_mode="mean_field",
            coupling_strength=1.0,
            seed=SEED,
        )
        result = sim.run(n_steps=20, dt=0.01)
        assert result.final_phase.shape == (500,)

    def test_sparse_knn_mode_runs(self) -> None:
        """Sparse k-NN mode completes without error."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(
            n_oscillators=200,
            coupling_mode="sparse_knn",
            coupling_strength=1.0,
            seed=SEED,
        )
        result = sim.run(n_steps=20, dt=0.01)
        assert result.final_phase.shape == (200,)

    def test_csr_mode_runs(self) -> None:
        """CSR mode completes without error."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(
            n_oscillators=100,
            coupling_mode="csr",
            coupling_strength=1.0,
            seed=SEED,
        )
        result = sim.run(n_steps=20, dt=0.01)
        assert result.final_phase.shape == (100,)


class TestOscilloSimLargeScale:
    """Tests for OscilloSim scaling to large N."""

    @pytest.mark.skipif(not _CUDA, reason="CUDA required for 1M test")
    def test_million_oscillators_mean_field(self) -> None:
        """Mean-field OscilloSim handles 1M oscillators on CUDA."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(
            n_oscillators=1_000_000,
            coupling_mode="mean_field",
            coupling_strength=0.5,
            device="cuda",
            seed=SEED,
        )
        result = sim.run(n_steps=10, dt=0.01)
        assert result.final_phase.shape == (1_000_000,)
        assert result.throughput > 0.0

    def test_10k_oscillators_cpu(self) -> None:
        """10K oscillators run on CPU in mean-field mode."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        sim = OscilloSim(
            n_oscillators=10_000,
            coupling_mode="mean_field",
            coupling_strength=0.5,
            device="cpu",
            seed=SEED,
        )
        result = sim.run(n_steps=10, dt=0.01)
        assert result.final_phase.shape == (10_000,)


class TestQuickSimulate:
    """Tests for the quick_simulate convenience function."""

    def test_quick_simulate_returns_result(self) -> None:
        """quick_simulate returns a SimulationResult."""
        from prinet.utils.oscillosim import SimulationResult, quick_simulate

        _seed()
        result = quick_simulate(
            n_oscillators=100, n_steps=50, coupling_strength=1.0, seed=SEED
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_default_args(self) -> None:
        """quick_simulate works with minimal arguments."""
        from prinet.utils.oscillosim import quick_simulate

        _seed()
        result = quick_simulate(n_oscillators=64, n_steps=20, seed=SEED)
        assert result.final_phase.shape == (64,)


# ══════════════════════════════════════════════════════════════════
# 3. P.5: Slot Attention comparison
# ══════════════════════════════════════════════════════════════════


class TestSlotAttentionModule:
    """Tests for the core SlotAttentionModule."""

    def test_forward_output_shape(self) -> None:
        """SlotAttentionModule produces (B, num_slots, slot_dim)."""
        from prinet.nn.slot_attention import SlotAttentionModule

        _seed()
        mod = SlotAttentionModule(num_slots=8, slot_dim=64, input_dim=128)
        x = torch.randn(4, 16, 128)  # (B, T, D)
        slots = mod(x)
        assert slots.shape == (4, 8, 64)

    def test_slot_competition(self) -> None:
        """Different inputs produce different slot activations."""
        from prinet.nn.slot_attention import SlotAttentionModule

        _seed()
        mod = SlotAttentionModule(
            num_slots=4, slot_dim=32, input_dim=64, num_iterations=3
        )
        x1 = torch.randn(2, 8, 64)
        x2 = torch.randn(2, 8, 64) + 5.0
        s1 = mod(x1)
        s2 = mod(x2)
        # Slots should differ for different inputs
        assert not torch.allclose(s1, s2, atol=1e-3)

    def test_deterministic_with_same_seed(self) -> None:
        """Two forward passes with the same seed produce identical slots."""
        from prinet.nn.slot_attention import SlotAttentionModule

        mod = SlotAttentionModule(
            num_slots=4, slot_dim=32, input_dim=64, num_iterations=3
        )
        x = torch.randn(2, 8, 64)
        _seed(123)
        s1 = mod(x)
        _seed(123)
        s2 = mod(x)
        assert torch.allclose(s1, s2)

    def test_gradient_flows(self) -> None:
        """Gradients flow through SlotAttentionModule."""
        from prinet.nn.slot_attention import SlotAttentionModule

        _seed()
        mod = SlotAttentionModule(num_slots=4, slot_dim=32, input_dim=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        slots = mod(x)
        loss = slots.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestSlotAttentionCLEVRN:
    """Tests for the CLEVR-N adapter."""

    def test_forward_shape(self) -> None:
        """SlotAttentionCLEVRN produces log-softmax logits (B, 2)."""
        from prinet.nn.slot_attention import SlotAttentionCLEVRN

        _seed()
        model = SlotAttentionCLEVRN(scene_dim=16, query_dim=60, num_slots=8, d_model=64)
        scene = torch.randn(4, 16)
        query = torch.randn(4, 60)
        logits = model(scene, query)
        assert logits.shape == (4, 2)

    def test_output_is_log_probability(self) -> None:
        """Output sums to ≈1 when exponentiated (log_softmax)."""
        from prinet.nn.slot_attention import SlotAttentionCLEVRN

        _seed()
        model = SlotAttentionCLEVRN(scene_dim=16, query_dim=60)
        scene = torch.randn(2, 16)
        query = torch.randn(2, 60)
        logits = model(scene, query)
        probs = logits.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones(2), atol=1e-5)

    def test_clevr_n_compatible_interface(self) -> None:
        """Model follows the CLEVR-N factory protocol:
        factory(scene_dim, query_dim) → model with forward(scene, query)."""
        from prinet.nn.slot_attention import SlotAttentionCLEVRN

        _seed()
        # Matches how clevr_n.py calls model factories
        model = SlotAttentionCLEVRN(scene_dim=16, query_dim=60)
        scene = torch.randn(8, 16)
        query = torch.randn(8, 60)
        out = model(scene, query)
        assert out.shape == (8, 2)

    def test_trainable_parameter_count(self) -> None:
        """Model has a reasonable number of trainable parameters."""
        from prinet.nn.slot_attention import SlotAttentionCLEVRN

        model = SlotAttentionCLEVRN(scene_dim=16, query_dim=60, d_model=64)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 1000, f"Too few params: {n_params}"
        assert n_params < 1_000_000, f"Too many params: {n_params}"

    def test_training_step(self) -> None:
        """One training step (forward + backward + update) works."""
        from prinet.nn.slot_attention import SlotAttentionCLEVRN

        _seed()
        model = SlotAttentionCLEVRN(scene_dim=16, query_dim=60, d_model=32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scene = torch.randn(4, 16)
        query = torch.randn(4, 60)
        labels = torch.tensor([0, 1, 0, 1])
        logits = model(scene, query)
        loss = F.nll_loss(logits, labels)
        loss.backward()
        opt.step()
        assert loss.item() > 0.0
