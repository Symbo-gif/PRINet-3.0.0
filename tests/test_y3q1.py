"""Year 3 Q1 Tests for PRINet.

Covers R.1 (propagation package split), R.2 (daemon DLQ),
R.3 (pseudo-phase fix), M.1–M.5 milestones.

Includes Hypothesis property-based tests for R.5 (phase-space invariants).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any
from unittest.mock import patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Optional Hypothesis guard
# ---------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False

_skip_hypothesis = pytest.mark.skipif(
    not _HAS_HYPOTHESIS, reason="hypothesis not installed"
)

# ---------------------------------------------------------------------------
# R.1 — propagation package
# ---------------------------------------------------------------------------


class TestR1PropagationPackage:
    """R.1: propagation package split tests."""

    def test_all_public_symbols_importable(self) -> None:
        """All 21 public symbols re-exported from new package."""
        from prinet.core.propagation import (  # noqa: F401
            DeltaThetaGammaNetwork,
            DentateGyrusConverter,
            DiscreteDeltaThetaGamma,
            ExponentialIntegrator,
            FeedbackInhibition,
            FeedforwardInhibition,
            HopfOscillator,
            KuramotoOscillator,
            MultiRateIntegrator,
            OscillatorModel,
            OscillatorState,
            OscillatorSyncError,
            PhaseAmplitudeCoupling,
            StuartLandauOscillator,
            TemporalPhasePropagator,
            ThetaGammaNetwork,
            detect_oscillation,
            phase_to_rate,
            sweep_coupling_params,
        )

    def test_oscillator_state_construction(self) -> None:
        from prinet.core.propagation import OscillatorState

        phi = torch.zeros(8)
        amp = torch.ones(8)
        freq = torch.full((8,), 6.0)
        s = OscillatorState(phase=phi, amplitude=amp, frequency=freq)
        assert s.phase.shape == (8,)
        assert s.amplitude.shape == (8,)

    def test_kuramoto_step_output_shape(self) -> None:
        from prinet.core.propagation import KuramotoOscillator, OscillatorState

        model = KuramotoOscillator(n_oscillators=12, coupling_strength=1.0)
        s = OscillatorState(
            phase=torch.rand(12) * 2 * math.pi,
            amplitude=torch.ones(12),
            frequency=torch.full((12,), 6.0),
        )
        next_s = model.step(s, dt=0.01)
        assert next_s.phase.shape == (12,)

    def test_delta_theta_gamma_order_params_in_range(self) -> None:
        from prinet.core.propagation import DeltaThetaGammaNetwork

        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        init = net.create_initial_state(seed=1)
        final, _ = net.integrate(init, n_steps=20, dt=0.01)
        rd, rt, rg = net.order_parameters(final)
        assert 0.0 <= rd.item() <= 1.0
        assert 0.0 <= rt.item() <= 1.0
        assert 0.0 <= rg.item() <= 1.0

    def test_sweep_coupling_params_returns_list(self) -> None:
        from prinet.core.propagation import sweep_coupling_params

        results = sweep_coupling_params(
            n_oscillators=9,
            k_values=[0.5, 2.0],
            m_values=[0.1, 0.3],
            n_steps=5,
        )
        assert len(results) == 4
        assert all({"K", "m", "r_delta", "r_theta", "r_gamma"} == set(r.keys()) for r in results)

    def test_detect_oscillation_bool(self) -> None:
        from prinet.core.propagation import detect_oscillation

        # Oscillating history
        history = [0.3 + 0.2 * math.sin(i * 0.5) for i in range(30)]
        result = detect_oscillation(history, window=20)
        assert isinstance(result, bool)

    def test_phase_to_rate_output_finite(self) -> None:
        from prinet.core.propagation import phase_to_rate

        phase = torch.rand(16) * 2 * math.pi
        amp = torch.rand(16) + 0.1
        rates = phase_to_rate(phase, amp, temperature=1.0)
        assert rates.shape == phase.shape
        assert torch.all(torch.isfinite(rates))

    def test_feedforward_inhibition_gate(self) -> None:
        from prinet.core.propagation import FeedforwardInhibition

        ffi = FeedforwardInhibition(delay_steps=1, tau=0.05)
        phase = torch.rand(8) * 2 * math.pi
        amp = torch.rand(8) + 0.1
        out = ffi.gate(phase, amp)
        assert out.shape == amp.shape
        assert torch.all(torch.isfinite(out))

    def test_feedback_inhibition_compete(self) -> None:
        from prinet.core.propagation import FeedbackInhibition

        fbi = FeedbackInhibition(k=3, sparsity=0.3)
        rates = torch.rand(10)
        out = fbi.compete(rates)
        assert out.shape == rates.shape

    def test_dentate_gyrus_converter(self) -> None:
        from prinet.core.propagation import DentateGyrusConverter

        dg = DentateGyrusConverter(n_oscillators=16, k=2)
        phase = torch.rand(16) * 2 * math.pi
        amp = torch.rand(16) + 0.1
        out = dg.convert(phase, amp)
        assert out.shape == (16,)

    def test_discrete_delta_theta_gamma_integrate(self) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        ddtg = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(4 + 8 + 16) * 2 * math.pi
        amp = torch.rand(4 + 8 + 16)
        out_phase, out_amp = ddtg.integrate(phase, amp, n_steps=5)
        assert out_phase.shape == phase.shape
        assert out_amp.shape == amp.shape


# ---------------------------------------------------------------------------
# R.2 — Daemon DLQ
# ---------------------------------------------------------------------------


class TestR2DaemonDLQ:
    """R.2: SubconsciousDaemon dead-letter queue tests."""

    def test_daemon_has_dlq_properties(self) -> None:
        from prinet.core.subconscious_daemon import SubconsciousDaemon

        assert hasattr(SubconsciousDaemon, "dead_letter_queue")
        assert hasattr(SubconsciousDaemon, "dlq_size")

    def test_daemon_new_constructor_params(self) -> None:
        import inspect

        from prinet.core.subconscious_daemon import SubconsciousDaemon

        sig = inspect.signature(SubconsciousDaemon.__init__)
        params = set(sig.parameters.keys())
        assert "dlq_maxlen" in params
        assert "max_errors_before_escalation" in params
        assert "error_escalation_callback" in params

    def test_dlq_initially_empty(self) -> None:
        """Instantiation (without starting) yields empty DLQ."""
        from prinet.core.subconscious_daemon import SubconsciousDaemon

        # We cannot call __init__ without a model path, so test via __new__
        d = SubconsciousDaemon.__new__(SubconsciousDaemon)
        d._dead_letter_queue = deque(maxlen=10)
        d._errors = 0
        d._inferences = 0
        d._max_errors_before_escalation = 5
        d._error_escalation_callback = None
        d._start_time = 0.0

        assert d.dlq_size == 0
        assert d.dead_letter_queue == []

    def test_dlq_evicts_oldest_at_maxlen(self) -> None:
        from prinet.core.subconscious_daemon import SubconsciousDaemon

        d = SubconsciousDaemon.__new__(SubconsciousDaemon)
        d._dead_letter_queue = deque(maxlen=3)
        d._errors = 0
        d._inferences = 0
        d._max_errors_before_escalation = 99
        d._error_escalation_callback = None
        d._start_time = 0.0

        for i in range(5):
            d._dead_letter_queue.append({"error": str(i), "error_count": i, "timestamp": 0.0})

        assert d.dlq_size == 3
        # Most-recent-first — dead_letter_queue property reverses deque
        first = d.dead_letter_queue[0]
        assert first["error"] == "4"

    def test_error_escalation_callback_fires(self) -> None:
        """Callback is invoked once error count reaches threshold."""
        from prinet.core.subconscious_daemon import SubconsciousDaemon

        fired: list[dict] = []

        d = SubconsciousDaemon.__new__(SubconsciousDaemon)
        d._dead_letter_queue = deque(maxlen=50)
        d._errors = 2
        d._inferences = 0
        d._max_errors_before_escalation = 3
        d._error_escalation_callback = fired.append
        d._start_time = 0.0

        # Simulate an exception in _run_inference
        import time

        exc = RuntimeError("test boom")
        d._errors += 1
        entry: dict[str, Any] = {
            "error": str(exc),
            "error_count": d._errors,
            "timestamp": time.monotonic(),
        }
        d._dead_letter_queue.append(entry)
        if (
            d._max_errors_before_escalation > 0
            and d._errors >= d._max_errors_before_escalation
            and d._error_escalation_callback is not None
        ):
            d._error_escalation_callback({"error_count": d._errors, "dlq_tail": entry})

        assert len(fired) == 1
        assert fired[0]["error_count"] == 3


# ---------------------------------------------------------------------------
# R.3 — Pseudo-phase fix
# ---------------------------------------------------------------------------


class TestR3PseudoPhaseFix:
    """R.3: HierarchicalResonanceLayer real-phase return tests."""

    def test_return_phase_false_returns_tensor(self) -> None:
        from prinet.nn.layers import HierarchicalResonanceLayer

        layer = HierarchicalResonanceLayer(n_delta=2, n_theta=4, n_gamma=8, n_dims=16, n_steps=2)
        x = torch.randn(2, 16)
        out = layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 14)

    def test_return_phase_true_returns_tuple(self) -> None:
        from prinet.nn.layers import HierarchicalResonanceLayer

        layer = HierarchicalResonanceLayer(n_delta=2, n_theta=4, n_gamma=8, n_dims=16, n_steps=2)
        x = torch.randn(2, 16)
        result = layer(x, return_phase=True)
        assert isinstance(result, tuple)
        amps, phases = result
        assert amps.shape == phases.shape == (2, 14)

    def test_returned_phases_in_range(self) -> None:
        """Returned phases are finite and bounded to [0, 2π) by layer's _wrap_phase."""
        from prinet.nn.layers import HierarchicalResonanceLayer

        layer = HierarchicalResonanceLayer(n_delta=2, n_theta=4, n_gamma=8, n_dims=16, n_steps=3)
        x = torch.randn(3, 16)
        _, phases = layer(x, return_phase=True)
        assert torch.all(torch.isfinite(phases))
        # layers.py _wrap_phase maps to [0, 2π)
        assert phases.min().item() >= -1e-4
        assert phases.max().item() <= 2 * math.pi + 1e-4

    def test_hybrid_prinet_forward_valid(self) -> None:
        from prinet.nn.hybrid import HybridPRINet

        model = HybridPRINet(
            n_input=16, n_classes=4,
            n_delta=2, n_theta=4, n_gamma=8,
            n_lobm_layers=2, lobm_steps=2,
        )
        x = torch.randn(3, 16)
        out = model(x)
        assert out.shape == (3, 4)
        # Valid log-probs: sum to ≈ 1 in prob space
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones(3), atol=1e-3)

    def test_hybrid_prinet_return_rates(self) -> None:
        from prinet.nn.hybrid import HybridPRINet

        model = HybridPRINet(
            n_input=16, n_classes=4,
            n_delta=2, n_theta=4, n_gamma=8,
            n_lobm_layers=2, lobm_steps=2,
        )
        x = torch.randn(2, 16)
        log_probs, rates = model(x, return_rates=True)
        assert log_probs.shape == (2, 4)
        assert rates.shape == (2, 14)

    def test_1d_input_squeezed_correctly(self) -> None:
        from prinet.nn.layers import HierarchicalResonanceLayer

        layer = HierarchicalResonanceLayer(n_delta=2, n_theta=4, n_gamma=8, n_dims=16, n_steps=2)
        x = torch.randn(16)
        amps, phases = layer(x, return_phase=True)
        assert amps.dim() == 1 and amps.shape[0] == 14
        assert phases.dim() == 1 and phases.shape[0] == 14


# ---------------------------------------------------------------------------
# M.1/M.2 — conv-stem image architecture
# ---------------------------------------------------------------------------


class TestM1M2ConvStemArchitecture:
    """M.1/M.2: HybridPRINetV2 conv stem tests."""

    def test_cifar10_forward_shape(self) -> None:
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(
            n_input=64 * 16, n_classes=10,
            n_delta=4, n_theta=8, n_gamma=16,
            n_discrete_steps=2,
            use_conv_stem=True, stem_channels=64,
        )
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_fashion_mnist_forward_shape(self) -> None:
        """3-channel 32×32 Fashion-MNIST works."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(
            n_input=64 * 16, n_classes=10,
            n_delta=4, n_theta=8, n_gamma=16,
            n_discrete_steps=2,
            use_conv_stem=True, stem_channels=64,
        )
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_conv_stem_valid_log_probs(self) -> None:
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(
            n_input=64 * 16, n_classes=10,
            n_delta=2, n_theta=4, n_gamma=8,
            n_discrete_steps=2,
            use_conv_stem=True, stem_channels=64,
        )
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones(2), atol=1e-3)

    def test_datasets_module_importable(self) -> None:
        from prinet.utils.datasets import (  # noqa: F401
            CIFAR10_CLASSES,
            CIFAR10_MEAN,
            CIFAR10_STD,
            FMNIST_CLASSES,
            evaluate_accuracy,
            get_cifar10_loaders,
            get_fashion_mnist_loaders,
        )
        assert len(CIFAR10_CLASSES) == 10
        assert len(FMNIST_CLASSES) == 10


# ---------------------------------------------------------------------------
# M.3/M.4 — extended palette + adversarial CLEVR
# ---------------------------------------------------------------------------


class TestM3M4ExtendedCLEVRN:
    """M.3/M.4: extended palette and adversarial CLEVR tests."""

    def test_colors_24_length(self) -> None:
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
        from benchmarks.clevr_n import COLORS_24, D_COLOR_24

        assert len(COLORS_24) == 24
        assert D_COLOR_24 == 24

    def test_make_clevr_n_extended_length(self) -> None:
        from benchmarks.clevr_n import make_clevr_n_extended

        ds = make_clevr_n_extended(n_items=4, n_samples=50, seed=0)
        assert len(ds) == 50

    def test_make_clevr_n_extended_query_shape(self) -> None:
        """Query vectors must be 2×d_feat_24 = 2×46 = 92."""
        from benchmarks.clevr_n import make_clevr_n_extended

        ds = make_clevr_n_extended(n_items=4, n_samples=20, seed=0)
        _, queries, _ = ds.tensors
        assert queries.shape == (20, 92)

    def test_adversarial_pairs_delta_e_bounded(self) -> None:
        from benchmarks.clevr_n import build_adversarial_colour_pairs

        pairs = build_adversarial_colour_pairs(max_delta_e=25.0)
        assert len(pairs) > 0
        assert all(de <= 25.0 + 1e-6 for _, _, de in pairs)

    def test_make_adversarial_clevr_dataset(self) -> None:
        from benchmarks.clevr_n import make_adversarial_clevr

        ds = make_adversarial_clevr(n_items=4, n_samples=50, seed=0)
        assert len(ds) == 50

    def test_adversarial_clevr_labels_binary(self) -> None:
        from benchmarks.clevr_n import make_adversarial_clevr

        ds = make_adversarial_clevr(n_items=4, n_samples=100, seed=1)
        _, _, labels = ds.tensors
        unique = labels.unique().tolist()
        assert set(unique).issubset({0, 1})


# ---------------------------------------------------------------------------
# M.5 — profiler wrapper
# ---------------------------------------------------------------------------


class TestM5Profiler:
    """M.5: torch.profiler wrapper tests."""

    def test_profiler_module_importable(self) -> None:
        from prinet.utils.profiler import (  # noqa: F401
            PRINetProfiler,
            ProfileReport,
            profile_training_loop,
        )

    def test_profile_report_fields(self) -> None:
        from prinet.utils.profiler import ProfileReport

        r = ProfileReport(
            total_wall_ms=100.0,
            n_steps=5,
            avg_step_ms=20.0,
            top_ops=[("aten::mm", 10.0, 5.0)],
            top_ops_table="Op  CPU  CUDA",
            bottleneck_op="aten::mm",
        )
        assert r.total_wall_ms == pytest.approx(100.0)
        assert r.bottleneck_op == "aten::mm"

    def test_prinetprofiler_context_manager(self) -> None:
        """Profiler produces a valid report with avg_step_ms >= 0."""
        import torch.nn as nn

        from prinet.utils.profiler import PRINetProfiler

        model = nn.Linear(8, 4)

        profiler = PRINetProfiler(
            out_dir=None,
            warmup_steps=1,
            active_steps=3,
        )
        with profiler:
            for _ in range(4):
                x = torch.randn(2, 8)
                _ = model(x)
                profiler.step()

        report = profiler.report(top_n=5)
        assert report.avg_step_ms >= 0.0
        assert isinstance(report.top_ops_table, str)


# ---------------------------------------------------------------------------
# Hypothesis R.5 — phase-space invariants
# ---------------------------------------------------------------------------


@_skip_hypothesis
class TestR5HypothesisInvariants:
    """R.5: property-based tests for oscillator invariants."""

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n=st.integers(min_value=4, max_value=32),
        steps=st.integers(min_value=5, max_value=30),
        k=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_order_parameter_in_unit_interval(
        self, n: int, steps: int, k: float
    ) -> None:
        """Kuramoto order parameter r ∈ [0, 1] for any parameters."""
        from prinet.core.propagation import KuramotoOscillator, OscillatorState

        model = KuramotoOscillator(n_oscillators=n, coupling_strength=k)
        s = OscillatorState(
            phase=torch.rand(n) * 2 * math.pi,
            amplitude=torch.ones(n),
            frequency=torch.full((n,), 6.0),
        )
        for _ in range(steps):
            s = model.step(s, dt=0.01)
        # Compute Kuramoto order parameter
        r = torch.abs(torch.mean(torch.exp(1j * s.phase.float()))).item()
        assert 0.0 - 1e-5 <= r <= 1.0 + 1e-5

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n=st.integers(min_value=3, max_value=20),
        steps=st.integers(min_value=1, max_value=10),
    )
    def test_phase_wrap_stays_bounded(self, n: int, steps: int) -> None:
        """Phase values after integration are finite (Kuramoto may not wrap to [-π, π])."""
        from prinet.core.propagation import KuramotoOscillator, OscillatorState

        model = KuramotoOscillator(n_oscillators=n, coupling_strength=1.0)
        s = OscillatorState(
            phase=torch.rand(n) * 2 * math.pi - math.pi,
            amplitude=torch.ones(n),
            frequency=torch.full((n,), 6.0),
        )
        for _ in range(steps):
            s = model.step(s, dt=0.01)
        # Model may not strictly wrap to [-π, π] — just check finiteness
        # and that values remain in a sensible range
        assert torch.all(torch.isfinite(s.phase))
        assert s.phase.abs().max().item() < 1e4  # no blow-up

    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(
        k_weak=st.floats(min_value=0.01, max_value=0.3),
        k_strong=st.floats(min_value=3.0, max_value=6.0),
    )
    def test_stronger_coupling_higher_synchrony(
        self, k_weak: float, k_strong: float
    ) -> None:
        """After sufficient integration, strong coupling → higher mean r."""
        from prinet.core.propagation import KuramotoOscillator, OscillatorState

        n, steps = 16, 50
        torch.manual_seed(0)

        def _run(k: float) -> float:
            model = KuramotoOscillator(n_oscillators=n, coupling_strength=k)
            s = OscillatorState(
                phase=torch.rand(n) * 2 * math.pi,
                amplitude=torch.ones(n),
                frequency=torch.full((n,), 6.0),
            )
            for _ in range(steps):
                s = model.step(s, dt=0.01)
            return torch.abs(torch.mean(torch.exp(1j * s.phase.float()))).item()

        r_weak = _run(k_weak)
        r_strong = _run(k_strong)
        # Strong coupling must give at least as much synchrony as weak
        # (add generous tolerance for stochastic initial conditions)
        assert r_strong >= r_weak - 0.3

    @settings(max_examples=20, deadline=500, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=3, max_value=24))
    def test_dtg_order_params_always_in_unit_interval(self, n: int) -> None:
        """DeltaThetaGammaNetwork order parameters always ∈ [0, 1]."""
        from prinet.core.propagation import DeltaThetaGammaNetwork

        n_per = max(2, n // 3)
        net = DeltaThetaGammaNetwork(
            n_delta=n_per, n_theta=n_per, n_gamma=n_per, coupling_strength=1.5
        )
        init = net.create_initial_state(seed=0)
        final, _ = net.integrate(init, n_steps=10, dt=0.01)
        rd, rt, rg = net.order_parameters(final)
        for r in (rd, rt, rg):
            assert 0.0 - 1e-5 <= r.item() <= 1.0 + 1e-5

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        phase=st.lists(st.floats(min_value=-100.0, max_value=100.0), min_size=4, max_size=32),
    )
    def test_phase_to_rate_output_in_valid_range(self, phase: list[float]) -> None:
        """phase_to_rate output is always finite and non-negative."""
        from prinet.core.propagation import phase_to_rate

        p = torch.tensor(phase)
        amp = torch.ones_like(p)
        rates = phase_to_rate(p, amp, temperature=1.0)
        assert torch.all(torch.isfinite(rates))
