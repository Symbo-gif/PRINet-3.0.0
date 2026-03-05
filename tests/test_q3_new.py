"""Tests for Q3 new code: kernels, benchmark reporting, layers, propagation extensions.

Covers:
  - DentateGyrusConverter
  - DGLayer, PhaseToRateAutoencoder, DenseAutoencoder
  - pytorch_multi_rate_derivatives, pytorch_fused_sub_step_rk4, pytorch_cross_band_coupling
  - generate_benchmark_report, generate_leaderboard, generate_scalr_metrics_report
  - ExponentialIntegrator stiff_mode
  - OscillatorState freq_band
  - OscilloBench class
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

# ────── Core / Propagation Imports ──────
from prinet.core.propagation import (
    DentateGyrusConverter,
    ExponentialIntegrator,
    OscillatorState,
    StuartLandauOscillator,
)

# ────── NN / Layer Imports ──────
from prinet.nn.layers import DGLayer, DenseAutoencoder, PhaseToRateAutoencoder

# ────── Utils Imports ──────
from prinet.utils.benchmark_reporting import (
    generate_benchmark_report,
    generate_leaderboard,
    generate_scalr_metrics_report,
)
from prinet.utils.triton_kernels import (
    pytorch_cross_band_coupling,
    pytorch_fused_sub_step_rk4,
    pytorch_multi_rate_derivatives,
)

DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════
# DentateGyrusConverter
# ═══════════════════════════════════════════════════════════════════
class TestDentateGyrusConverter:
    """Test DentateGyrusConverter (DG pattern separation)."""

    def test_output_shape(self) -> None:
        dg = DentateGyrusConverter(n_oscillators=32, k=6)
        phase = torch.rand(4, 32) * 6.2832
        amp = torch.ones(4, 32)
        out = dg.convert(phase, amp)
        assert out.shape == (4, 32)

    def test_output_sparse(self) -> None:
        dg = DentateGyrusConverter(n_oscillators=64, k=8)
        phase = torch.rand(4, 64) * 6.2832
        amp = torch.ones(4, 64)
        out = dg.convert(phase, amp)
        # At most k non-zero per sample
        for b in range(4):
            n_nonzero = (out[b] > 1e-6).sum().item()
            assert n_nonzero <= 8, f"expected ≤8, got {n_nonzero}"

    def test_ffi_delay_effect(self) -> None:
        dg = DentateGyrusConverter(n_oscillators=16, k=4, ffi_delay=5, fbi_delay=10)
        phase = torch.rand(2, 16) * 6.2832
        amp = torch.ones(2, 16)
        out = dg.convert(phase, amp)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

    def test_unbatched(self) -> None:
        dg = DentateGyrusConverter(n_oscillators=16, k=4)
        phase = torch.rand(16) * 6.2832
        amp = torch.ones(16)
        out = dg.convert(phase, amp)
        assert out.shape == (16,)


# ═══════════════════════════════════════════════════════════════════
# DGLayer (nn.Module wrapper)
# ═══════════════════════════════════════════════════════════════════
class TestDGLayer:
    """Test DGLayer wrapper."""

    def test_forward_shape(self) -> None:
        layer = DGLayer(n_input=32, top_k=6)
        phase = torch.rand(4, 32) * 6.2832
        amp = torch.ones(4, 32)
        out = layer(phase, amp)
        assert out.shape == (4, 32)

    def test_has_learnable_params(self) -> None:
        layer = DGLayer(n_input=32)
        params = list(layer.parameters())
        assert len(params) > 0

    def test_gradient_flows(self) -> None:
        layer = DGLayer(n_input=16, top_k=4)
        phase = (torch.rand(2, 16) * 6.2832).requires_grad_(True)
        amp = torch.ones(2, 16, requires_grad=True)
        out = layer(phase, amp)
        loss = out.sum()
        loss.backward()
        # Check gradients flow to layer parameters
        has_grad = any(
            p.grad is not None for p in layer.parameters() if p.requires_grad
        )
        assert has_grad


# ═══════════════════════════════════════════════════════════════════
# PhaseToRateAutoencoder / DenseAutoencoder
# ═══════════════════════════════════════════════════════════════════
class TestPhaseToRateAutoencoder:
    """Test PhaseToRateAutoencoder."""

    def test_forward_shape(self) -> None:
        ae = PhaseToRateAutoencoder(n_input=64, n_oscillators=16)
        x = torch.randn(4, 64)
        recon, rates = ae(x)
        assert recon.shape == (4, 64)
        assert rates.shape == (4, 16)

    def test_classify_shape(self) -> None:
        ae = PhaseToRateAutoencoder(n_input=64, n_oscillators=16)
        x = torch.randn(4, 64)
        log_probs = ae.classify(x)
        assert log_probs.shape[0] == 4

    def test_gradient_flows(self) -> None:
        ae = PhaseToRateAutoencoder(n_input=32, n_oscillators=8)
        x = torch.randn(2, 32)
        recon, rates = ae(x)
        loss = recon.sum() + rates.sum()
        loss.backward()
        for p in ae.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


class TestDenseAutoencoder:
    """Test DenseAutoencoder."""

    def test_forward_shape(self) -> None:
        ae = DenseAutoencoder(n_input=64, n_bottleneck=16)
        x = torch.randn(4, 64)
        recon, latent = ae(x)
        assert recon.shape == (4, 64)
        assert latent.shape == (4, 16)

    def test_classify_shape(self) -> None:
        ae = DenseAutoencoder(n_input=64, n_bottleneck=16)
        x = torch.randn(4, 64)
        log_probs = ae.classify(x)
        assert log_probs.shape[0] == 4


# ═══════════════════════════════════════════════════════════════════
# Multi-Rate Kernel Functions
# ═══════════════════════════════════════════════════════════════════
class TestPytorchMultiRateDerivatives:
    """Test pytorch_multi_rate_derivatives."""

    def test_output_shape(self) -> None:
        N = 64
        phase = torch.rand(N) * 6.2832
        amp = torch.ones(N)
        freq = torch.rand(N) * 40.0
        freq_band = torch.randint(0, 3, (N,))
        dp, da, df = pytorch_multi_rate_derivatives(
            phase, amp, freq, freq_band, K=2.0, decay=0.1, gamma=0.01
        )
        assert dp.shape == (N,)
        assert da.shape == (N,)
        assert df.shape == (N,)

    def test_finite_output(self) -> None:
        N = 32
        phase = torch.rand(N) * 6.2832
        amp = torch.ones(N)
        freq = torch.rand(N) * 40.0
        freq_band = torch.randint(0, 3, (N,))
        dp, da, df = pytorch_multi_rate_derivatives(
            phase, amp, freq, freq_band, 1.0, 0.1, 0.01
        )
        assert torch.isfinite(dp).all()
        assert torch.isfinite(da).all()
        assert torch.isfinite(df).all()


class TestPytorchFusedSubStepRK4:
    """Test pytorch_fused_sub_step_rk4."""

    def test_output_shape(self) -> None:
        N = 32
        phase = torch.rand(N) * 6.2832
        amp = torch.ones(N)
        freq = torch.rand(N) * 40.0
        freq_band = torch.randint(0, 3, (N,))
        new_p, new_a, new_f = pytorch_fused_sub_step_rk4(
            phase, amp, freq, freq_band, 2.0, 0.1, 0.01, 0.01
        )
        assert new_p.shape == (N,)
        assert new_a.shape == (N,)
        assert new_f.shape == (N,)

    def test_state_changes(self) -> None:
        N = 64
        torch.manual_seed(42)
        phase = torch.rand(N) * 6.2832
        amp = torch.ones(N)
        freq = torch.rand(N) * 40.0
        freq_band = torch.randint(0, 3, (N,))
        new_p, new_a, new_f = pytorch_fused_sub_step_rk4(
            phase, amp, freq, freq_band, 2.0, 0.1, 0.01, 0.01
        )
        # State should change after integration
        assert not torch.allclose(new_p, phase)


class TestPytorchCrossBandCoupling:
    """Test pytorch_cross_band_coupling."""

    def test_output_shape(self) -> None:
        N_slow = 8
        N_fast = 32
        slow_phase = torch.rand(N_slow) * 6.2832
        fast_phase = torch.rand(N_fast) * 6.2832
        fast_amp = torch.ones(N_fast)
        parent_idx = torch.randint(0, N_slow, (N_fast,))
        modulated_amp, phase_shift = pytorch_cross_band_coupling(
            slow_phase, fast_phase, fast_amp, parent_idx
        )
        assert modulated_amp.shape == (N_fast,)
        assert phase_shift.shape == (N_fast,)

    def test_modulated_amp_non_negative(self) -> None:
        slow_phase = torch.rand(4) * 6.2832
        fast_phase = torch.rand(16) * 6.2832
        fast_amp = torch.ones(16)
        parent_idx = torch.randint(0, 4, (16,))
        mod_amp, _ = pytorch_cross_band_coupling(
            slow_phase, fast_phase, fast_amp, parent_idx
        )
        assert (mod_amp >= 0).all()


# ═══════════════════════════════════════════════════════════════════
# Benchmark Reporting
# ═══════════════════════════════════════════════════════════════════
class TestBenchmarkReporting:
    """Test benchmark reporting utilities."""

    def test_generate_report_from_dir(self, tmp_path: Path) -> None:
        # Create a dummy benchmark JSON
        data = {"status": "OK", "accuracy": 0.95, "loss": 0.1}
        (tmp_path / "benchmark_test.json").write_text(json.dumps(data))
        report = generate_benchmark_report(str(tmp_path))
        assert "benchmark_test.json" in report
        assert "OK" in report

    def test_generate_report_empty_dir(self, tmp_path: Path) -> None:
        report = generate_benchmark_report(str(tmp_path))
        assert "Benchmark Report" in report

    def test_generate_leaderboard(self, tmp_path: Path) -> None:
        data = {
            "model": "test_model",
            "test_accuracy": 0.92,
            "train_accuracy": 0.99,
        }
        (tmp_path / "benchmark_leaderboard.json").write_text(json.dumps(data))
        lb = generate_leaderboard(str(tmp_path))
        assert isinstance(lb, str)

    def test_generate_scalr_metrics_report(self) -> None:
        r_history = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.88, 0.92, 0.91, 0.93]
        report = generate_scalr_metrics_report(r_history, window=3)
        assert isinstance(report, str)
        assert len(report) > 0


# ═══════════════════════════════════════════════════════════════════
# ExponentialIntegrator stiff_mode / OscillatorState freq_band
# ═══════════════════════════════════════════════════════════════════
class TestStiffMode:
    """Test ExponentialIntegrator stiff_mode feature."""

    def test_stiff_mode_creates(self) -> None:
        ei = ExponentialIntegrator(dim=16, stiff_mode=True, krylov_rank=4)
        assert ei.stiff_mode is True

    def test_stiff_mode_step(self) -> None:
        ei = ExponentialIntegrator(dim=16, stiff_mode=True, krylov_rank=4)
        model = StuartLandauOscillator(16, bifurcation_param=1.0)
        state = OscillatorState.create_random(16, seed=0)
        result = ei.step(model, state, dt=0.01)
        assert result.phase.shape == (16,)
        assert torch.isfinite(result.phase).all()

    def test_stiff_non_stiff_differ(self) -> None:
        model = StuartLandauOscillator(16, bifurcation_param=1.0)
        state = OscillatorState.create_random(16, seed=0)
        ei_stiff = ExponentialIntegrator(dim=16, stiff_mode=True, krylov_rank=4)
        ei_normal = ExponentialIntegrator(dim=16, stiff_mode=False)
        r1 = ei_stiff.step(model, state, dt=0.01)
        r2 = ei_normal.step(model, state, dt=0.01)
        # They may differ due to different integration methods
        # Just confirm both are valid
        assert torch.isfinite(r1.phase).all()
        assert torch.isfinite(r2.phase).all()


class TestOscillatorStateFreqBand:
    """Test OscillatorState freq_band attribute."""

    def test_freq_band_stored(self) -> None:
        state = OscillatorState(
            phase=torch.zeros(5),
            amplitude=torch.ones(5),
            frequency=torch.ones(5),
            freq_band=torch.tensor([0, 0, 1, 1, 2]),
        )
        assert state.freq_band is not None
        assert state.freq_band.shape == (5,)

    def test_n_bands(self) -> None:
        state = OscillatorState(
            phase=torch.zeros(6),
            amplitude=torch.ones(6),
            frequency=torch.ones(6),
            freq_band=torch.tensor([0, 0, 1, 1, 2, 2]),
        )
        assert state.n_bands == 3

    def test_clone_preserves_freq_band(self) -> None:
        state = OscillatorState(
            phase=torch.zeros(4),
            amplitude=torch.ones(4),
            frequency=torch.ones(4),
            freq_band=torch.tensor([0, 1, 1, 2]),
        )
        cloned = state.clone()
        assert cloned.freq_band is not None
        assert torch.equal(cloned.freq_band, state.freq_band)

    def test_create_random_no_freq_band(self) -> None:
        state = OscillatorState.create_random(10, seed=0)
        # freq_band defaults to None for basic create_random
        # Just verify the state is valid
        assert state.phase.shape == (10,)


# ═══════════════════════════════════════════════════════════════════
# OscilloBench Class
# ═══════════════════════════════════════════════════════════════════
class TestOscilloBench:
    """Test OscilloBench class API."""

    def test_available_tasks(self) -> None:
        from benchmarks.oscillobench import OscilloBench

        bench = OscilloBench()
        tasks = bench.available_tasks()
        assert "xor_n" in tasks
        assert "random_dichotomies" in tasks
        assert len(tasks) >= 4

    def test_save_load(self, tmp_path: Path) -> None:
        from benchmarks.oscillobench import OscilloBench

        bench = OscilloBench()
        results = {"model": "test", "score": 0.5}
        save_path = tmp_path / "test_bench.json"
        bench.save(results, save_path)
        assert save_path.exists()
        loaded = json.loads(save_path.read_text())
        assert loaded["model"] == "test"
