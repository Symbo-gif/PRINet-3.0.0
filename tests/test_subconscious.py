"""Test Suite for the Subconscious Layer (Q4 Workstream 5).

Covers:
    * Backend detection and session creation
    * SubconsciousState ↔ tensor round-trip
    * ControlSignals ↔ tensor round-trip
    * ControlSignalBuffer thread-safety
    * SubconsciousController forward pass and parameter count
    * ONNX export + inference round-trip
    * SubconsciousDaemon lifecycle (start → submit → read → stop)
    * collect_system_state helper
    * Edge cases (NaN handling, queue overflow, empty buffers)
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from prinet.core.subconscious import (
    CONTROL_DIM,
    STATE_DIM,
    ControlSignalBuffer,
    ControlSignals,
    SubconsciousState,
)
from prinet.core.subconscious_daemon import (
    SubconsciousDaemon,
    collect_system_state,
)
from prinet.nn.subconscious_model import SubconsciousController
from prinet.utils.npu_backend import (
    BackendType,
    backend_info,
    detect_best_backend,
    directml_available,
    npu_available,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def tmp_onnx(tmp_path: Path) -> Path:
    """Export a fresh ONNX model to a temporary directory."""
    model = SubconsciousController()
    return model.export_to_onnx(tmp_path / "test_controller.onnx")


# ======================================================================
# Backend detection tests
# ======================================================================


class TestNpuBackend:
    """Tests for ``prinet.utils.npu_backend``."""

    def test_detect_best_backend_returns_valid_type(self) -> None:
        result = detect_best_backend()
        assert result in {"npu", "directml", "cpu"}

    def test_npu_available_returns_bool(self) -> None:
        assert isinstance(npu_available(), bool)

    def test_directml_available_returns_bool(self) -> None:
        assert isinstance(directml_available(), bool)

    def test_backend_info_keys(self) -> None:
        info = backend_info()
        assert "ort_available" in info
        assert "available_eps" in info
        assert "best_backend" in info
        assert "npu_firmware_found" in info
        assert isinstance(info["available_eps"], list)

    def test_backend_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRINET_SUBCONSCIOUS_BACKEND", "cpu")
        assert detect_best_backend() == "cpu"

    def test_backend_override_invalid_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRINET_SUBCONSCIOUS_BACKEND", "quantum")
        # Should fall through to auto-detection (not crash)
        result = detect_best_backend()
        assert result in {"npu", "directml", "cpu"}


# ======================================================================
# SubconsciousState tests
# ======================================================================


class TestSubconsciousState:
    """Tests for ``prinet.core.subconscious.SubconsciousState``."""

    def test_default_factory(self) -> None:
        state = SubconsciousState.default()
        assert state.epoch == 0
        assert state.regime == "mean_field"
        assert state.timestamp > 0.0

    def test_to_tensor_shape_and_dtype(self) -> None:
        state = SubconsciousState.default()
        z = state.to_tensor()
        assert z.shape == (STATE_DIM,)
        assert z.dtype == np.float32

    def test_to_tensor_values_are_finite(self) -> None:
        state = SubconsciousState(
            r_per_band=[0.8, 0.5, 0.3],
            r_global=0.6,
            loss_ema=1.5,
            loss_variance=0.1,
            grad_norm_ema=2.0,
            lr_current=1e-4,
            scalr_alpha=0.7,
            gpu_temp=72.0,
            gpu_util=0.85,
            vram_pct=0.60,
            cpu_util=0.40,
            step_latency_p50=0.012,
            step_latency_p95=0.025,
            throughput=5000.0,
            epoch=42,
            regime="sparse_knn",
            timestamp=time.time(),
        )
        z = state.to_tensor()
        assert np.all(np.isfinite(z))

    def test_to_tensor_padding_is_zero(self) -> None:
        state = SubconsciousState.default()
        z = state.to_tensor()
        # Last 13 elements should be zero (padding)
        assert np.allclose(z[19:], 0.0)

    def test_clone_produces_equal_copy(self) -> None:
        state = SubconsciousState(
            r_per_band=[0.1, 0.2, 0.3],
            r_global=0.5,
            loss_ema=2.0,
            epoch=10,
            regime="full",
            timestamp=1234.0,
        )
        cloned = state.clone()
        assert cloned is not state
        np.testing.assert_array_equal(state.to_tensor(), cloned.to_tensor())

    def test_regime_encoding(self) -> None:
        for regime, expected_idx in [("mean_field", 0), ("sparse_knn", 1), ("full", 2)]:
            state = SubconsciousState(regime=regime)
            z = state.to_tensor()
            # regime is at index 17, normalized by /2.0
            assert z[17] == pytest.approx(expected_idx / 2.0)


# ======================================================================
# ControlSignals tests
# ======================================================================


class TestControlSignals:
    """Tests for ``prinet.core.subconscious.ControlSignals``."""

    def test_default_signals(self) -> None:
        ctrl = ControlSignals.default()
        assert ctrl.lr_multiplier == 1.0
        assert ctrl.alert_level == 0.0

    def test_to_tensor_shape(self) -> None:
        ctrl = ControlSignals.default()
        arr = ctrl.to_tensor()
        assert arr.shape == (CONTROL_DIM,)
        assert arr.dtype == np.float32

    def test_round_trip(self) -> None:
        original = ControlSignals(
            suggested_K_min=1.0,
            suggested_K_max=4.0,
            lr_multiplier=0.8,
            regime_mf_weight=0.5,
            regime_sk_weight=0.3,
            regime_full_weight=0.2,
            alert_level=0.6,
            coupling_mode_suggestion=1.0,
        )
        arr = original.to_tensor()
        restored = ControlSignals.from_tensor(arr)
        assert restored.suggested_K_min == pytest.approx(1.0)
        assert restored.suggested_K_max == pytest.approx(4.0)
        assert restored.lr_multiplier == pytest.approx(0.8)
        assert restored.alert_level == pytest.approx(0.6)

    def test_from_tensor_clamps_alert(self) -> None:
        arr = np.array([1, 2, 3, 0.3, 0.3, 0.4, 5.0, 0], dtype=np.float32)
        ctrl = ControlSignals.from_tensor(arr)
        assert ctrl.alert_level == pytest.approx(1.0)

    def test_from_tensor_clamps_lr(self) -> None:
        arr = np.array([1, 2, -999, 0.3, 0.3, 0.4, 0.5, 0], dtype=np.float32)
        ctrl = ControlSignals.from_tensor(arr)
        assert ctrl.lr_multiplier == pytest.approx(0.1)

    def test_from_tensor_rejects_short(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            ControlSignals.from_tensor(np.zeros(3, dtype=np.float32))

    def test_from_tensor_accepts_2d(self) -> None:
        arr = np.zeros((1, 8), dtype=np.float32)
        ctrl = ControlSignals.from_tensor(arr)
        assert ctrl.is_finite()

    def test_is_finite_detects_nan(self) -> None:
        ctrl = ControlSignals(alert_level=float("nan"))
        assert not ctrl.is_finite()

    def test_preferred_regime(self) -> None:
        ctrl = ControlSignals(
            regime_mf_weight=0.1,
            regime_sk_weight=0.8,
            regime_full_weight=0.1,
        )
        assert ctrl.preferred_regime == "sparse_knn"


# ======================================================================
# ControlSignalBuffer tests
# ======================================================================


class TestControlSignalBuffer:
    """Tests for ``prinet.core.subconscious.ControlSignalBuffer``."""

    def test_initial_value_is_default(self) -> None:
        buf = ControlSignalBuffer()
        ctrl = buf.latest()
        assert ctrl.lr_multiplier == 1.0

    def test_update_and_read(self) -> None:
        buf = ControlSignalBuffer()
        buf.update(ControlSignals(alert_level=0.9))
        assert buf.latest().alert_level == pytest.approx(0.9)

    def test_thread_safety(self) -> None:
        """Concurrent writes and reads should not corrupt data."""
        buf = ControlSignalBuffer()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    buf.update(ControlSignals(alert_level=float(i) / 200.0))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(200):
                    ctrl = buf.latest()
                    assert ctrl.is_finite()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors, f"Thread-safety errors: {errors}"


# ======================================================================
# SubconsciousController (PyTorch) tests
# ======================================================================


class TestSubconsciousController:
    """Tests for ``prinet.nn.subconscious_model.SubconsciousController``."""

    def test_forward_shape(self) -> None:
        model = SubconsciousController()
        z = torch.randn(4, STATE_DIM)
        out = model(z)
        assert out.shape == (4, CONTROL_DIM)

    def test_forward_finite(self) -> None:
        model = SubconsciousController()
        z = torch.randn(8, STATE_DIM)
        out = model(z)
        assert torch.all(torch.isfinite(out))

    def test_k_range_positive(self) -> None:
        """K bounds should be positive (softplus output)."""
        model = SubconsciousController()
        z = torch.randn(16, STATE_DIM)
        out = model(z)
        assert torch.all(out[:, 0:2] > 0)

    def test_lr_multiplier_positive(self) -> None:
        model = SubconsciousController()
        z = torch.randn(16, STATE_DIM)
        out = model(z)
        assert torch.all(out[:, 2] > 0)

    def test_regime_weights_sum_to_one(self) -> None:
        model = SubconsciousController()
        z = torch.randn(16, STATE_DIM)
        out = model(z)
        sums = out[:, 3:6].sum(dim=-1)
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-5)

    def test_alert_level_bounded(self) -> None:
        model = SubconsciousController()
        z = torch.randn(16, STATE_DIM)
        out = model(z)
        assert torch.all(out[:, 6] >= 0.0)
        assert torch.all(out[:, 6] <= 1.0)

    def test_parameter_count(self) -> None:
        model = SubconsciousController()
        # 32*128 + 128 + 128*128 + 128 + 128*8 + 8 = 21,768
        assert model.num_parameters == 21_768

    def test_custom_dims(self) -> None:
        model = SubconsciousController(state_dim=16, hidden=64, control_dim=4)
        z = torch.randn(2, 16)
        out = model(z)
        assert out.shape == (2, 4)

    def test_eval_mode_no_dropout(self) -> None:
        model = SubconsciousController()
        model.eval()
        z = torch.randn(1, STATE_DIM)
        out1 = model(z)
        out2 = model(z)
        torch.testing.assert_close(out1, out2)


# ======================================================================
# ONNX export and inference tests
# ======================================================================


class TestOnnxExport:
    """Tests for ONNX export and inference round-trip."""

    def test_export_creates_file(self, tmp_onnx: Path) -> None:
        assert tmp_onnx.exists()
        assert tmp_onnx.stat().st_size > 0

    def test_onnx_inference_shape(self, tmp_onnx: Path) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(tmp_onnx), providers=["CPUExecutionProvider"]
        )
        z = np.random.randn(1, STATE_DIM).astype(np.float32)
        outputs = session.run(None, {"state_vector": z})
        assert outputs[0].shape == (1, CONTROL_DIM)

    def test_onnx_inference_finite(self, tmp_onnx: Path) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(tmp_onnx), providers=["CPUExecutionProvider"]
        )
        z = np.random.randn(4, STATE_DIM).astype(np.float32)
        outputs = session.run(None, {"state_vector": z})
        assert np.all(np.isfinite(outputs[0]))

    def test_onnx_pytorch_agreement(self, tmp_onnx: Path) -> None:
        """ONNX and PyTorch should produce approximately the same output."""
        import onnxruntime as ort

        model = SubconsciousController()
        model.eval()

        z_np = np.random.randn(1, STATE_DIM).astype(np.float32)
        z_pt = torch.from_numpy(z_np)

        # PyTorch
        with torch.no_grad():
            pt_out = model(z_pt).numpy()

        # Re-export (since weights are new, we need a fresh export)
        tmp_path = tmp_onnx.parent / "agreement_test.onnx"
        model.export_to_onnx(tmp_path)

        session = ort.InferenceSession(
            str(tmp_path), providers=["CPUExecutionProvider"]
        )
        onnx_out = session.run(None, {"state_vector": z_np})[0]

        np.testing.assert_allclose(pt_out, onnx_out, atol=1e-5, rtol=1e-4)

    def test_onnx_dynamic_batch(self, tmp_onnx: Path) -> None:
        """ONNX model should accept variable batch sizes."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(tmp_onnx), providers=["CPUExecutionProvider"]
        )
        for batch in [1, 4, 16]:
            z = np.random.randn(batch, STATE_DIM).astype(np.float32)
            out = session.run(None, {"state_vector": z})
            assert out[0].shape == (batch, CONTROL_DIM)


# ======================================================================
# SubconsciousDaemon tests
# ======================================================================


class TestSubconsciousDaemon:
    """Tests for ``prinet.core.subconscious_daemon.SubconsciousDaemon``."""

    def test_daemon_lifecycle(self, tmp_onnx: Path) -> None:
        """Start → submit → read → stop without errors."""
        daemon = SubconsciousDaemon(
            tmp_onnx, backend="cpu", interval=0.5, warmup=True
        )
        daemon.start()
        time.sleep(0.5)  # let it warm up

        state = SubconsciousState.default()
        daemon.submit_state(state)
        time.sleep(1.0)  # wait for inference

        ctrl = daemon.get_control()
        assert ctrl.is_finite()
        assert daemon.inference_count >= 1

        daemon.stop(timeout=5.0)
        assert not daemon.is_alive()

    def test_daemon_is_daemon_thread(self, tmp_onnx: Path) -> None:
        daemon = SubconsciousDaemon(tmp_onnx, backend="cpu")
        assert daemon.daemon is True

    def test_daemon_submit_overflow(self, tmp_onnx: Path) -> None:
        """Queue overflow should not raise."""
        daemon = SubconsciousDaemon(
            tmp_onnx, backend="cpu", interval=60.0, queue_size=5
        )
        daemon.start()
        for _ in range(20):
            daemon.submit_state(SubconsciousState.default())
        daemon.stop(timeout=3.0)

    def test_daemon_default_control_before_inference(self, tmp_onnx: Path) -> None:
        daemon = SubconsciousDaemon(tmp_onnx, backend="cpu", interval=60.0)
        daemon.start()
        ctrl = daemon.get_control()
        # Should return safe defaults since no inference has run
        assert ctrl.lr_multiplier == 1.0
        daemon.stop(timeout=3.0)

    def test_daemon_multiple_inferences(self, tmp_onnx: Path) -> None:
        daemon = SubconsciousDaemon(
            tmp_onnx, backend="cpu", interval=0.1, warmup=False
        )
        daemon.start()
        for i in range(5):
            state = SubconsciousState(epoch=i, timestamp=time.time())
            daemon.submit_state(state)
            time.sleep(0.3)

        daemon.stop(timeout=5.0)
        assert daemon.inference_count >= 3
        assert daemon.error_count == 0

    def test_daemon_uptime(self, tmp_onnx: Path) -> None:
        daemon = SubconsciousDaemon(tmp_onnx, backend="cpu", interval=0.5)
        assert daemon.uptime == 0.0
        daemon.start()
        time.sleep(1.0)
        assert daemon.uptime >= 0.5
        daemon.stop(timeout=3.0)


# ======================================================================
# collect_system_state tests
# ======================================================================


class TestCollectSystemState:
    """Tests for ``prinet.core.subconscious_daemon.collect_system_state``."""

    def test_basic_collection(self) -> None:
        state = collect_system_state(
            r_per_band=[0.5, 0.4, 0.3],
            r_global=0.4,
            loss_ema=1.0,
            epoch=5,
        )
        assert state.r_per_band == [0.5, 0.4, 0.3]
        assert state.epoch == 5
        assert state.timestamp > 0.0

    def test_tensor_from_collected_state(self) -> None:
        state = collect_system_state()
        z = state.to_tensor()
        assert z.shape == (STATE_DIM,)
        assert np.all(np.isfinite(z))

    def test_cpu_util_populated(self) -> None:
        """cpu_util should be populated if psutil is available."""
        try:
            import psutil  # noqa: F401

            state = collect_system_state()
            # cpu_util can be 0.0 on first call (no interval), that's OK
            assert 0.0 <= state.cpu_util <= 1.0
        except ImportError:
            pytest.skip("psutil not installed")


# ======================================================================
# Integration: end-to-end state → daemon → control signals
# ======================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_state_through_daemon(self, tmp_onnx: Path) -> None:
        """Full pipeline: state → tensor → daemon → control signals."""
        daemon = SubconsciousDaemon(
            tmp_onnx, backend="cpu", interval=0.2, warmup=True
        )
        daemon.start()
        time.sleep(0.5)

        state = SubconsciousState(
            r_per_band=[0.9, 0.7, 0.5],
            r_global=0.7,
            loss_ema=0.5,
            loss_variance=0.01,
            grad_norm_ema=1.0,
            lr_current=1e-3,
            scalr_alpha=1.0,
            gpu_temp=65.0,
            gpu_util=0.75,
            vram_pct=0.50,
            cpu_util=0.30,
            step_latency_p50=0.010,
            step_latency_p95=0.020,
            throughput=3000.0,
            epoch=10,
            regime="sparse_knn",
            timestamp=time.time(),
        )
        daemon.submit_state(state)
        time.sleep(1.0)

        ctrl = daemon.get_control()
        assert ctrl.is_finite()
        assert ctrl.suggested_K_min > 0
        assert ctrl.suggested_K_max > 0
        assert 0.0 <= ctrl.alert_level <= 1.0

        daemon.stop(timeout=5.0)

    def test_no_gpu_throughput_regression(self, tmp_onnx: Path) -> None:
        """Daemon should not significantly affect CPU-bound work."""
        daemon = SubconsciousDaemon(
            tmp_onnx, backend="cpu", interval=0.5, warmup=True
        )

        # Baseline: compute without daemon
        t0 = time.perf_counter()
        _dummy_work(n=100_000)
        baseline = time.perf_counter() - t0

        # With daemon running
        daemon.start()
        time.sleep(0.5)
        for _ in range(3):
            daemon.submit_state(SubconsciousState.default())

        t0 = time.perf_counter()
        _dummy_work(n=100_000)
        with_daemon = time.perf_counter() - t0

        daemon.stop(timeout=5.0)

        # Daemon should add < 30% overhead (generous margin for CI / background load)
        ratio = with_daemon / max(baseline, 1e-9)
        assert ratio < 1.30, f"Throughput ratio {ratio:.2f} > 1.30"


def _dummy_work(n: int) -> float:
    """Simulate CPU-bound work for regression testing."""
    total = 0.0
    for i in range(n):
        total += math.sin(float(i))
    return total
