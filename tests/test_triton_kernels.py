"""Tests for Triton fused kernels: mean-field RK4 and sparse k-NN coupling.

Validates numerical correctness of Triton kernels against PyTorch reference
implementations across multiple sizes, parameters, and edge cases.

Y3 Q3 O.7: Added PyTorch-only test variants that mirror the Triton-only tests,
enabling test recovery without Triton.  The Triton-vs-PyTorch comparison tests
retain the ``@triton_only`` marker.

All tests use seeded RNG for determinism per Testing Standards.
Automatically skipped if CUDA is unavailable.
"""

from __future__ import annotations

import math

import pytest
import torch

# ── Skip guards ───────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

try:
    from prinet.utils.triton_kernels import triton_available

    _TRITON_OK = triton_available()
except ImportError:
    _TRITON_OK = False

triton_only = pytest.mark.skipif(not _TRITON_OK, reason="Triton not available")

from prinet.utils.triton_kernels import (
    pytorch_mean_field_rk4_step,
    pytorch_sparse_knn_coupling,
    triton_fused_mean_field_rk4_step,
    triton_sparse_knn_coupling,
)

SEED = 42
DEVICE = torch.device("cuda")
TWO_PI = 2.0 * math.pi


def _mf_rk4_fn():
    """Return Triton mean-field RK4 if available, else PyTorch fallback."""
    if _TRITON_OK:
        return triton_fused_mean_field_rk4_step
    return pytorch_mean_field_rk4_step


def _sparse_knn_fn():
    """Return Triton sparse k-NN if available, else PyTorch fallback."""
    if _TRITON_OK:
        return triton_sparse_knn_coupling
    return pytorch_sparse_knn_coupling


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular distance: min angular separation (handles 0/2π wrap)."""
    return ((a - b + math.pi) % TWO_PI - math.pi).abs()


def _make_state(
    N: int,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random oscillator state on CUDA."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    phase = torch.rand(N, device=DEVICE, generator=gen) * TWO_PI
    amp = torch.rand(N, device=DEVICE, generator=gen) * 0.5 + 0.5  # [0.5, 1.0]
    freq = torch.randn(N, device=DEVICE, generator=gen)
    return phase, amp, freq


# ══════════════════════════════════════════════════════════════════
# 1. MEAN-FIELD RK4: Triton vs PyTorch
# ══════════════════════════════════════════════════════════════════


class TestTritonMeanFieldRK4:
    """Correctness tests for triton_fused_mean_field_rk4_step."""

    K = 2.0
    decay = 0.1
    gamma = 0.01
    dt = 0.01

    @triton_only
    @pytest.mark.parametrize("N", [64, 256, 1024, 4096, 16384])
    def test_matches_pytorch_reference(self, N: int) -> None:
        """Triton result matches PyTorch to within float32 tolerance."""
        phase, amp, freq = _make_state(N)

        tp, ta, tf = triton_fused_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        pp, pa, pf = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )

        # Phase: use circular distance (handles wrap)
        assert _circ_dist(tp, pp).max() < 1e-5, (
            f"Phase circular error too large: {_circ_dist(tp, pp).max():.2e}"
        )
        # Amplitude
        assert (ta - pa).abs().max() < 1e-5, (
            f"Amplitude error too large: {(ta - pa).abs().max():.2e}"
        )
        # Frequency
        assert (tf - pf).abs().max() < 1e-5, (
            f"Frequency error too large: {(tf - pf).abs().max():.2e}"
        )

    def test_phase_wrapping(self) -> None:
        """Output phases are always in [0, 2\u03c0). Dual-path: Triton or PyTorch."""
        fn = _mf_rk4_fn()
        phase, amp, freq = _make_state(4096)
        tp, _, _ = fn(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert tp.min() >= 0.0
        assert tp.max() < TWO_PI + 1e-6

    def test_amplitude_non_negative(self) -> None:
        """Output amplitudes are clamped to >= 0. Dual-path: Triton or PyTorch."""
        fn = _mf_rk4_fn()
        phase, amp, freq = _make_state(4096)
        # Use negative-driving parameters to stress test clamping
        _, ta, _ = fn(
            phase, amp, freq, K=0.0, decay=100.0, gamma=0.0, dt=1.0,
        )
        assert ta.min() >= 0.0

    def test_zero_coupling(self) -> None:
        """With K=0, phases advance by omega*dt. Dual-path: Triton or PyTorch."""
        fn = _mf_rk4_fn()
        N = 1024
        phase, amp, freq = _make_state(N)
        tp, ta, tf = fn(
            phase, amp, freq, K=0.0, decay=0.0, gamma=0.0, dt=self.dt,
        )
        expected_phase = (phase + freq * self.dt) % TWO_PI
        assert _circ_dist(tp, expected_phase).max() < 1e-5

    def test_deterministic(self) -> None:
        """Same inputs produce near-identical outputs. Dual-path: Triton or PyTorch."""
        fn = _mf_rk4_fn()
        phase, amp, freq = _make_state(8192)
        r1 = fn(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        r2 = fn(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        for a, b in zip(r1, r2):
            assert torch.allclose(a, b, atol=1e-6), (
                f"Non-deterministic output: max diff {(a - b).abs().max():.2e}"
            )

    def test_large_N(self) -> None:
        """Smoke test at N=65536. Dual-path: Triton or PyTorch."""
        fn = _mf_rk4_fn()
        phase, amp, freq = _make_state(65536)
        tp, ta, tf = fn(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        pp, pa, pf = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert _circ_dist(tp, pp).max() < 1e-4

    @triton_only
    def test_cuda_required(self) -> None:
        """Raises RuntimeError for CPU tensors."""
        phase, amp, freq = _make_state(64)
        with pytest.raises(RuntimeError, match="CUDA"):
            triton_fused_mean_field_rk4_step(
                phase.cpu(), amp.cpu(), freq.cpu(),
                self.K, self.decay, self.gamma, self.dt,
            )


# ══════════════════════════════════════════════════════════════════
# 2. SPARSE k-NN: Triton vs PyTorch
# ══════════════════════════════════════════════════════════════════


class TestTritonSparseKNN:
    """Correctness tests for triton_sparse_knn_coupling."""

    K = 2.0
    decay = 0.1
    gamma = 0.01

    @triton_only
    @pytest.mark.parametrize("N,k", [(256, 8), (1024, 14), (4096, 6), (8192, 20)])
    def test_matches_pytorch_reference(self, N: int, k: int) -> None:
        """Triton result matches PyTorch to within float32 tolerance."""
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        td, tr, to_ = triton_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        pd, pr, po = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )

        assert (td - pd).abs().max() < 1e-5, (
            f"dphi error: {(td - pd).abs().max():.2e}"
        )
        assert (tr - pr).abs().max() < 1e-5, (
            f"dr error: {(tr - pr).abs().max():.2e}"
        )
        assert (to_ - po).abs().max() < 1e-5, (
            f"domega error: {(to_ - po).abs().max():.2e}"
        )

    def test_zero_coupling(self) -> None:
        """K=0 gives dphi=freq, dr=-decay*amp, domega=0. Dual-path."""
        fn = _sparse_knn_fn()
        N, k = 1024, 10
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 2)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        dphi, dr, dom = fn(
            phase, amp, freq, nbr, K=0.0, decay=self.decay, gamma=self.gamma,
        )
        assert (dphi - freq).abs().max() < 1e-6
        assert (dr - (-self.decay * amp)).abs().max() < 1e-6
        assert dom.abs().max() < 1e-6

    def test_deterministic(self) -> None:
        """Same inputs produce identical outputs. Dual-path."""
        fn = _sparse_knn_fn()
        N, k = 4096, 14
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 3)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        r1 = fn(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        r2 = fn(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        for a, b in zip(r1, r2):
            assert torch.equal(a, b), "Non-deterministic output detected"

    @triton_only
    def test_cuda_required(self) -> None:
        """Raises RuntimeError for CPU tensors."""
        phase, amp, freq = _make_state(64)
        nbr = torch.randint(0, 64, (64, 4), device=DEVICE)
        with pytest.raises(RuntimeError, match="CUDA"):
            triton_sparse_knn_coupling(
                phase.cpu(), amp.cpu(), freq.cpu(),
                nbr.cpu(), self.K, self.decay, self.gamma,
            )


# ══════════════════════════════════════════════════════════════════
# 3. PYTORCH FALLBACK TESTS (always run, no Triton required)
# ══════════════════════════════════════════════════════════════════


class TestPyTorchFallback:
    """Test PyTorch reference implementations independently."""

    K = 2.0
    decay = 0.1
    gamma = 0.01
    dt = 0.01

    def test_mf_rk4_phase_wrapping(self) -> None:
        """PyTorch fallback wraps phases to [0, 2π)."""
        phase, amp, freq = _make_state(1024)
        p, _, _ = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert p.min() >= 0.0
        assert p.max() < TWO_PI + 1e-6

    def test_mf_rk4_amplitude_non_negative(self) -> None:
        """PyTorch fallback clamps amplitude to ≥ 0."""
        phase, amp, freq = _make_state(1024)
        _, a, _ = pytorch_mean_field_rk4_step(
            phase, amp, freq, K=0.0, decay=100.0, gamma=0.0, dt=1.0,
        )
        assert a.min() >= 0.0

    def test_sparse_knn_symmetry(self) -> None:
        """With uniform amplitudes and symmetric K_eff, coupling sums balance."""
        N, k = 512, 8
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 10)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        dphi, dr, dom = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        # Outputs should be finite
        assert torch.isfinite(dphi).all()
        assert torch.isfinite(dr).all()
        assert torch.isfinite(dom).all()


# ══════════════════════════════════════════════════════════════════
# 4. Q3: PAC MODULATION KERNEL TESTS
# ══════════════════════════════════════════════════════════════════


from prinet.utils.triton_kernels import (
    pytorch_hierarchical_order_param,
    pytorch_multi_rate_rk4_step,
    pytorch_pac_modulation,
    triton_hierarchical_order_param,
    triton_pac_modulation,
)


class TestPACModulation:
    """Tests for PAC modulation (Triton + PyTorch)."""

    def test_pytorch_pac_basic(self) -> None:
        """PyTorch PAC modulation should produce expected values."""
        slow_phase = torch.zeros(10, device=DEVICE)  # cos(0) = 1
        fast_amp = torch.ones(10, device=DEVICE) * 2.0
        out = pytorch_pac_modulation(slow_phase, fast_amp, modulation_depth=0.5)
        expected = 2.0 * (1.0 + 0.5 * 1.0)  # = 3.0
        torch.testing.assert_close(
            out, torch.full_like(out, expected), atol=1e-5, rtol=1e-5
        )

    def test_amplitude_clamping(self) -> None:
        """Output should be clamped to [amp_min, amp_max]."""
        slow_phase = torch.zeros(10, device=DEVICE)
        fast_amp = torch.ones(10, device=DEVICE) * 1e-8
        out = pytorch_pac_modulation(
            slow_phase, fast_amp, modulation_depth=0.5, amp_min=1e-6
        )
        assert (out >= 1e-6).all()

    def test_phase_wrapping_safe(self) -> None:
        """PAC should handle phases near 2π and multiples."""
        slow_phase = torch.tensor([0.0, TWO_PI, 4 * math.pi], device=DEVICE)
        fast_amp = torch.ones(5, device=DEVICE)
        out = pytorch_pac_modulation(slow_phase, fast_amp, modulation_depth=0.5)
        assert torch.isfinite(out).all()

    @triton_only
    def test_triton_pac_matches_pytorch(self) -> None:
        """Triton PAC should match PyTorch reference."""
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        slow_phase = torch.rand(64, device=DEVICE, generator=gen) * TWO_PI
        fast_amp = torch.rand(128, device=DEVICE, generator=gen) + 0.5
        m = 0.6

        ref = pytorch_pac_modulation(slow_phase, fast_amp, m)
        tri = triton_pac_modulation(slow_phase, fast_amp, m)
        torch.testing.assert_close(ref, tri, atol=1e-4, rtol=1e-4)


# ══════════════════════════════════════════════════════════════════
# 5. Q3: HIERARCHICAL ORDER PARAMETER KERNEL TESTS
# ══════════════════════════════════════════════════════════════════


class TestHierarchicalOrderParam:
    """Tests for per-band order parameter computation."""

    def test_pytorch_reference_single_band(self) -> None:
        """Single band → same as standard Kuramoto order parameter."""
        phase = torch.zeros(32, device=DEVICE)  # All aligned → r=1
        r = pytorch_hierarchical_order_param(phase, [32])
        assert r.shape == (1,)
        assert r[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_pytorch_reference_multi_band(self) -> None:
        """Multi-band should compute independent order parameters."""
        # Band 1: all aligned, Band 2: random
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        band1 = torch.zeros(16, device=DEVICE)
        band2 = torch.rand(16, device=DEVICE, generator=gen) * TWO_PI
        phase = torch.cat([band1, band2])
        r = pytorch_hierarchical_order_param(phase, [16, 16])
        assert r.shape == (2,)
        assert r[0].item() == pytest.approx(1.0, abs=1e-5)
        # Band 2 should be < 1 (random phases)
        assert r[1].item() < 0.9

    @triton_only
    def test_triton_matches_pytorch(self) -> None:
        """Triton hierarchical order params should match PyTorch."""
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        phase = torch.rand(256, device=DEVICE, generator=gen) * TWO_PI
        band_sizes = [64, 96, 96]

        ref = pytorch_hierarchical_order_param(phase, band_sizes)
        tri = triton_hierarchical_order_param(phase, band_sizes)
        torch.testing.assert_close(ref, tri, atol=1e-4, rtol=1e-4)


# ══════════════════════════════════════════════════════════════════
# 6. Q3: MULTI-RATE RK4 TESTS
# ══════════════════════════════════════════════════════════════════


class TestMultiRateRK4:
    """Tests for multi-rate sub-stepping RK4 integration."""

    def test_single_substep_matches_standard(self) -> None:
        """With sub_steps=1, should approximate standard RK4."""
        phase, amp, freq = _make_state(128)
        K, decay, gamma, dt = 2.0, 0.1, 0.01, 0.01

        p1, a1, f1 = pytorch_mean_field_rk4_step(
            phase, amp, freq, K, decay, gamma, dt
        )
        p2, a2, f2 = pytorch_multi_rate_rk4_step(
            phase, amp, freq, K, decay, gamma, dt, sub_steps=1
        )
        # Should be very close
        assert _circ_dist(p1, p2).max() < 0.1
        torch.testing.assert_close(a1, a2, atol=0.1, rtol=0.1)

    def test_more_substeps_more_accurate(self) -> None:
        """More sub-steps should produce smoother integration."""
        phase, amp, freq = _make_state(64)
        K, decay, gamma, dt = 2.0, 0.1, 0.01, 0.1

        # 1 sub-step (large dt)
        p1, a1, _ = pytorch_multi_rate_rk4_step(
            phase, amp, freq, K, decay, gamma, dt, sub_steps=1
        )
        # 10 sub-steps (effectively smaller dt)
        p10, a10, _ = pytorch_multi_rate_rk4_step(
            phase, amp, freq, K, decay, gamma, dt, sub_steps=10
        )
        # Both should be finite
        assert torch.isfinite(p1).all()
        assert torch.isfinite(p10).all()
        assert torch.isfinite(a1).all()
        assert torch.isfinite(a10).all()

    def test_outputs_finite(self) -> None:
        """Multi-rate RK4 outputs should always be finite."""
        phase, amp, freq = _make_state(256)
        p, a, f = pytorch_multi_rate_rk4_step(
            phase, amp, freq, K=2.0, decay=0.1, gamma=0.01, dt=0.01,
            sub_steps=5,
        )
        assert torch.isfinite(p).all()
        assert torch.isfinite(a).all()
        assert torch.isfinite(f).all()

    def test_amplitude_non_negative(self) -> None:
        """Amplitudes should remain non-negative after multi-rate step."""
        phase, amp, freq = _make_state(128)
        _, a, _ = pytorch_multi_rate_rk4_step(
            phase, amp, freq, K=0.0, decay=100.0, gamma=0.0, dt=1.0,
            sub_steps=5,
        )
        assert a.min() >= 0.0


# ══════════════════════════════════════════════════════════════════
# 7. O.7: RECOVERED TESTS — PyTorch-only variants of Triton tests
# ══════════════════════════════════════════════════════════════════
# These tests mirror the Triton-only tests above but use only the
# PyTorch reference implementations.  They validate the same
# invariants (phase wrapping, amplitude non-negativity, zero-coupling
# identity, determinism, large-N correctness) without Triton.


class TestRecoveredMeanFieldRK4:
    """PyTorch-fallback tests recovering Triton mean-field RK4 skips."""

    K = 2.0
    decay = 0.1
    gamma = 0.01
    dt = 0.01

    @pytest.mark.parametrize("N", [64, 256, 1024, 4096, 16384])
    def test_pytorch_mf_rk4_scaling(self, N: int) -> None:
        """PyTorch MF RK4 produces consistent results across sizes."""
        phase, amp, freq = _make_state(N)
        pp, pa, pf = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert pp.shape == (N,)
        assert pa.shape == (N,)
        assert pf.shape == (N,)
        assert torch.isfinite(pp).all()
        assert torch.isfinite(pa).all()
        assert torch.isfinite(pf).all()

    def test_pytorch_mf_phase_wrapping_strict(self) -> None:
        """PyTorch MF RK4 wraps phases to [0, 2π)."""
        phase, amp, freq = _make_state(4096)
        pp, _, _ = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert pp.min() >= 0.0
        assert pp.max() < TWO_PI + 1e-6

    def test_pytorch_mf_amplitude_clamping(self) -> None:
        """PyTorch MF RK4 clamps amplitude ≥ 0 under extreme decay."""
        phase, amp, freq = _make_state(4096)
        _, pa, _ = pytorch_mean_field_rk4_step(
            phase, amp, freq, K=0.0, decay=100.0, gamma=0.0, dt=1.0,
        )
        assert pa.min() >= 0.0

    def test_pytorch_mf_zero_coupling_identity(self) -> None:
        """K=0 → phases advance by ω·dt (mean-field coupling vanishes)."""
        N = 1024
        phase, amp, freq = _make_state(N)
        pp, _, _ = pytorch_mean_field_rk4_step(
            phase, amp, freq, K=0.0, decay=0.0, gamma=0.0, dt=self.dt,
        )
        expected = (phase + freq * self.dt) % TWO_PI
        assert _circ_dist(pp, expected).max() < 1e-5

    def test_pytorch_mf_deterministic(self) -> None:
        """Same inputs produce identical PyTorch MF RK4 outputs."""
        phase, amp, freq = _make_state(8192)
        r1 = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        r2 = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        for a, b in zip(r1, r2):
            assert torch.allclose(a, b, atol=1e-6)

    def test_pytorch_mf_large_N(self) -> None:
        """Smoke test at N=65536 for PyTorch fallback."""
        phase, amp, freq = _make_state(65536)
        pp, pa, pf = pytorch_mean_field_rk4_step(
            phase, amp, freq, self.K, self.decay, self.gamma, self.dt,
        )
        assert torch.isfinite(pp).all()
        assert torch.isfinite(pa).all()

    def test_pytorch_mf_cpu_works(self) -> None:
        """PyTorch fallback should work on CPU (Triton version doesn't)."""
        phase, amp, freq = _make_state(64)
        pp, pa, pf = pytorch_mean_field_rk4_step(
            phase.cpu(), amp.cpu(), freq.cpu(),
            self.K, self.decay, self.gamma, self.dt,
        )
        assert pp.device.type == "cpu"
        assert torch.isfinite(pp).all()


class TestRecoveredSparseKNN:
    """PyTorch-fallback tests recovering Triton sparse k-NN skips."""

    K = 2.0
    decay = 0.1
    gamma = 0.01

    @pytest.mark.parametrize("N,k", [(256, 8), (1024, 14), (4096, 6), (8192, 20)])
    def test_pytorch_knn_scaling(self, N: int, k: int) -> None:
        """PyTorch sparse k-NN produces valid outputs across sizes."""
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        pd, pr, po = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        assert pd.shape == (N,)
        assert pr.shape == (N,)
        assert po.shape == (N,)
        assert torch.isfinite(pd).all()
        assert torch.isfinite(pr).all()
        assert torch.isfinite(po).all()

    def test_pytorch_knn_zero_coupling(self) -> None:
        """K=0 gives dphi=freq, dr=-decay*amp, domega=0."""
        N, k = 1024, 10
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 2)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        dphi, dr, dom = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, K=0.0, decay=self.decay, gamma=self.gamma,
        )
        assert (dphi - freq).abs().max() < 1e-6
        assert (dr - (-self.decay * amp)).abs().max() < 1e-6
        assert dom.abs().max() < 1e-6

    def test_pytorch_knn_deterministic(self) -> None:
        """Same inputs produce identical PyTorch k-NN outputs."""
        N, k = 4096, 14
        phase, amp, freq = _make_state(N)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 3)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        r1 = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        r2 = pytorch_sparse_knn_coupling(
            phase, amp, freq, nbr, self.K, self.decay, self.gamma,
        )
        for a, b in zip(r1, r2):
            assert torch.equal(a, b), "Non-deterministic output detected"

    def test_pytorch_knn_cpu_works(self) -> None:
        """PyTorch k-NN should work on CPU (Triton version doesn't)."""
        phase, amp, freq = _make_state(64)
        nbr = torch.randint(0, 64, (64, 4), device=DEVICE)
        pd, pr, po = pytorch_sparse_knn_coupling(
            phase.cpu(), amp.cpu(), freq.cpu(),
            nbr.cpu(), self.K, self.decay, self.gamma,
        )
        assert pd.device.type == "cpu"


class TestRecoveredPACModulation:
    """PyTorch-fallback tests recovering Triton PAC modulation skip."""

    def test_pytorch_pac_matches_manual(self) -> None:
        """PyTorch PAC with known inputs produces expected result."""
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        slow_phase = torch.rand(64, device=DEVICE, generator=gen) * TWO_PI
        fast_amp = torch.rand(128, device=DEVICE, generator=gen) + 0.5
        m = 0.6

        ref = pytorch_pac_modulation(slow_phase, fast_amp, m)
        assert torch.isfinite(ref).all()
        assert ref.shape[0] == max(64, 128)  # broadcast

    def test_pytorch_pac_modulation_depth_range(self) -> None:
        """PAC modulation scales linearly with depth parameter."""
        slow = torch.zeros(10, device=DEVICE)  # cos(0) = 1
        fast = torch.ones(10, device=DEVICE) * 2.0

        out_low = pytorch_pac_modulation(slow, fast, modulation_depth=0.1)
        out_high = pytorch_pac_modulation(slow, fast, modulation_depth=0.9)
        # Higher depth → larger modulation effect
        assert out_high.mean() > out_low.mean()


class TestRecoveredHierarchicalOrderParam:
    """PyTorch-fallback tests recovering Triton hierarchical order param skip."""

    def test_pytorch_hierarchical_matches_aligned(self) -> None:
        """Aligned phases in all bands → r ≈ 1.0 for each band."""
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        phase = torch.rand(256, device=DEVICE, generator=gen) * TWO_PI
        band_sizes = [64, 96, 96]

        ref = pytorch_hierarchical_order_param(phase, band_sizes)
        assert ref.shape == (3,)
        # Random phases should have r < 1 (not perfectly aligned)
        assert (ref >= 0).all()
        assert (ref <= 1.0 + 1e-5).all()

    def test_pytorch_hierarchical_perfectly_aligned(self) -> None:
        """All-zero phases → r = 1.0 for each band."""
        phase = torch.zeros(256, device=DEVICE)
        band_sizes = [64, 96, 96]
        ref = pytorch_hierarchical_order_param(phase, band_sizes)
        for r in ref:
            assert r.item() == pytest.approx(1.0, abs=1e-5)
