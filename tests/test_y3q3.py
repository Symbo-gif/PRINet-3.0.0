"""Year 3 Q3 Tests — Efficiency & Scaling.

Tests for all Q3 deliverables:
    O.1: torch.compile integration on HybridPRINetV2
    O.2: CUDA C++ fused discrete step (via PyTorch fallback)
    O.3: Mixed-precision training
    R.4: CSR sparse coupling matrix
    O.4: 100+ oscillator systems with sparse k-NN coupling
    O.5: Async CPU+GPU pipeline
    O.6: Model pruning

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
# 1. O.1: torch.compile integration
# ══════════════════════════════════════════════════════════════════


class TestTorchCompile:
    """Tests for torch.compile integration on HybridPRINetV2."""

    def test_compile_method_exists(self) -> None:
        """HybridPRINetV2 has a compile() method."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=64, n_classes=5)
        assert hasattr(model, "compile")
        assert callable(model.compile)

    def test_is_compiled_property_default_false(self) -> None:
        """is_compiled is False before compile() is called."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=64, n_classes=5)
        assert not model.is_compiled

    def test_compile_returns_self(self) -> None:
        """compile() returns the model itself for chaining."""
        from prinet.nn.hybrid import HybridPRINetV2

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5)
        result = model.compile(backend="eager")  # eager = no-op compile
        assert result is model
        assert model.is_compiled

    def test_compiled_forward_matches_uncompiled(self) -> None:
        """compiled_forward() matches forward() output (eval mode, no stochastic ops)."""
        from prinet.nn.hybrid import HybridPRINetV2

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        model.eval()  # disable dropout / stochastic layers
        x = torch.randn(4, 64, device=DEVICE)

        with torch.no_grad():
            # Uncompiled
            out1 = model(x)

            # compiled_forward without compile() → should fall back
            out2 = model.compiled_forward(x)
        torch.testing.assert_close(out1, out2)

    def test_compile_eager_backend_works(self) -> None:
        """torch.compile with eager backend runs without error."""
        from prinet.nn.hybrid import HybridPRINetV2

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        model.compile(backend="eager")

        x = torch.randn(4, 64, device=DEVICE)
        out = model.compiled_forward(x)
        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════
# 2. O.2: CUDA C++ fused discrete step
# ══════════════════════════════════════════════════════════════════


class TestFusedDiscreteStep:
    """Tests for the fused discrete step (PyTorch reference)."""

    def test_pytorch_fused_step_basic(self) -> None:
        """PyTorch fused step produces valid outputs."""
        from prinet.utils.fused_kernels import pytorch_fused_discrete_step_full

        _seed()
        B, nd, nt, ng = 4, 4, 8, 16
        N = nd + nt + ng
        phase = torch.rand(B, N, device=DEVICE) * TWO_PI
        amp = torch.ones(B, N, device=DEVICE)

        fd = torch.full((nd,), 2.0, device=DEVICE)
        ft = torch.full((nt,), 6.0, device=DEVICE)
        fg = torch.full((ng,), 40.0, device=DEVICE)
        Wd = torch.randn(nd, nd, device=DEVICE) * 0.5
        Wt = torch.randn(nt, nt, device=DEVICE) * 0.25
        Wg = torch.randn(ng, ng, device=DEVICE) * 0.125
        W_dt_w = torch.randn(nt, 2 * nd, device=DEVICE) * 0.1
        W_dt_b = torch.zeros(nt, device=DEVICE)
        W_tg_w = torch.randn(ng, 2 * nt, device=DEVICE) * 0.1
        W_tg_b = torch.zeros(ng, device=DEVICE)

        new_p, new_a = pytorch_fused_discrete_step_full(
            phase,
            amp,
            fd,
            ft,
            fg,
            Wd,
            Wt,
            Wg,
            W_dt_w,
            W_dt_b,
            W_tg_w,
            W_tg_b,
            1.0,
            1.0,
            1.0,
            0.01,
            nd,
            nt,
            ng,
        )

        assert new_p.shape == (B, N)
        assert new_a.shape == (B, N)
        assert torch.isfinite(new_p).all()
        assert torch.isfinite(new_a).all()
        # Phase in [0, 2π)
        assert new_p.min() >= 0.0
        assert new_p.max() < TWO_PI + 1e-5

    def test_fused_step_amplitude_clamped(self) -> None:
        """Amplitudes are clamped to [1e-6, 10.0]."""
        from prinet.utils.fused_kernels import pytorch_fused_discrete_step_full

        _seed()
        B, nd, nt, ng = 2, 2, 4, 8
        N = nd + nt + ng
        phase = torch.rand(B, N, device=DEVICE) * TWO_PI
        amp = torch.ones(B, N, device=DEVICE) * 0.001  # Very small amplitudes

        fd = torch.full((nd,), 2.0, device=DEVICE)
        ft = torch.full((nt,), 6.0, device=DEVICE)
        fg = torch.full((ng,), 40.0, device=DEVICE)
        Wd = torch.zeros(nd, nd, device=DEVICE)
        Wt = torch.zeros(nt, nt, device=DEVICE)
        Wg = torch.zeros(ng, ng, device=DEVICE)
        W_dt_w = torch.zeros(nt, 2 * nd, device=DEVICE)
        W_dt_b = torch.zeros(nt, device=DEVICE)
        W_tg_w = torch.zeros(ng, 2 * nt, device=DEVICE)
        W_tg_b = torch.zeros(ng, device=DEVICE)

        _, new_a = pytorch_fused_discrete_step_full(
            phase,
            amp,
            fd,
            ft,
            fg,
            Wd,
            Wt,
            Wg,
            W_dt_w,
            W_dt_b,
            W_tg_w,
            W_tg_b,
            1.0,
            1.0,
            1.0,
            0.01,
            nd,
            nt,
            ng,
        )
        assert (new_a >= 1e-6).all()
        assert (new_a <= 10.0).all()

    def test_fused_step_deterministic(self) -> None:
        """Fused step is deterministic with same inputs."""
        from prinet.utils.fused_kernels import pytorch_fused_discrete_step_full

        _seed()
        B, nd, nt, ng = 4, 4, 8, 16
        N = nd + nt + ng
        phase = torch.rand(B, N, device=DEVICE) * TWO_PI
        amp = torch.ones(B, N, device=DEVICE)

        fd = torch.full((nd,), 2.0, device=DEVICE)
        ft = torch.full((nt,), 6.0, device=DEVICE)
        fg = torch.full((ng,), 40.0, device=DEVICE)
        Wd = torch.randn(nd, nd, device=DEVICE) * 0.5
        Wt = torch.randn(nt, nt, device=DEVICE) * 0.25
        Wg = torch.randn(ng, ng, device=DEVICE) * 0.125
        W_dt_w = torch.randn(nt, 2 * nd, device=DEVICE) * 0.1
        W_dt_b = torch.zeros(nt, device=DEVICE)
        W_tg_w = torch.randn(ng, 2 * nt, device=DEVICE) * 0.1
        W_tg_b = torch.zeros(ng, device=DEVICE)

        args = (
            phase,
            amp,
            fd,
            ft,
            fg,
            Wd,
            Wt,
            Wg,
            W_dt_w,
            W_dt_b,
            W_tg_w,
            W_tg_b,
            1.0,
            1.0,
            1.0,
            0.01,
            nd,
            nt,
            ng,
        )
        p1, a1 = pytorch_fused_discrete_step_full(*args)
        p2, a2 = pytorch_fused_discrete_step_full(*args)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(a1, a2)

    def test_cuda_fused_kernel_available_check(self) -> None:
        """cuda_fused_kernel_available() returns a boolean."""
        from prinet.utils.fused_kernels import cuda_fused_kernel_available

        result = cuda_fused_kernel_available()
        assert isinstance(result, bool)


# ══════════════════════════════════════════════════════════════════
# 3. O.3: Mixed-Precision Training
# ══════════════════════════════════════════════════════════════════


class TestMixedPrecision:
    """Tests for mixed-precision training wrapper."""

    def test_mixed_precision_trainer_creation(self) -> None:
        """MixedPrecisionTrainer can be instantiated."""
        from prinet.nn.hybrid import HybridPRINetV2
        from prinet.utils.fused_kernels import MixedPrecisionTrainer

        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(
            model, opt, device_type="cuda" if _CUDA else "cpu"
        )
        assert trainer.step_count == 0

    @pytest.mark.skipif(not _CUDA, reason="CUDA required for AMP")
    def test_mixed_precision_train_step(self) -> None:
        """One train step completes with mixed precision."""
        from prinet.nn.hybrid import HybridPRINetV2
        from prinet.utils.fused_kernels import MixedPrecisionTrainer

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(model, opt, enabled=True)

        x = torch.randn(8, 64, device="cuda")
        y = torch.randint(0, 5, (8,), device="cuda")

        loss = trainer.train_step(x, y, F.nll_loss)
        assert isinstance(loss, float)
        assert trainer.step_count == 1

    def test_mixed_precision_disabled_mode(self) -> None:
        """Disabled mode runs in FP32."""
        from prinet.nn.hybrid import HybridPRINetV2
        from prinet.utils.fused_kernels import MixedPrecisionTrainer

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(model, opt, enabled=False)

        x = torch.randn(4, 64, device=DEVICE)
        y = torch.randint(0, 5, (4,), device=DEVICE)

        loss = trainer.train_step(x, y, F.nll_loss)
        assert isinstance(loss, float)

    def test_mixed_precision_state_dict(self) -> None:
        """State dict roundtrip for checkpointing."""
        from prinet.nn.hybrid import HybridPRINetV2
        from prinet.utils.fused_kernels import MixedPrecisionTrainer

        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MixedPrecisionTrainer(model, opt, enabled=False)

        state = trainer.state_dict()
        assert "scaler" in state
        assert "step_count" in state

        trainer.load_state_dict(state)
        assert trainer.step_count == 0


# ══════════════════════════════════════════════════════════════════
# 4. R.4: CSR Sparse Coupling Matrix
# ══════════════════════════════════════════════════════════════════


class TestCSRSparseCoupling:
    """Tests for CSR sparse coupling matrix."""

    def test_csr_matrix_creation(self) -> None:
        """sparse_coupling_matrix_csr produces CSR tensor."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        C = sparse_coupling_matrix_csr(100, sparsity=0.9, seed=42)
        assert C.layout == torch.sparse_csr

    def test_csr_sparsity_correct(self) -> None:
        """CSR matrix has approximately correct sparsity."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        N = 500
        sparsity = 0.95
        C = sparse_coupling_matrix_csr(N, sparsity=sparsity, seed=42)
        nnz = C.values().numel()
        total = N * N - N  # Exclude diagonal
        actual_density = nnz / total
        # Allow some tolerance for random sampling
        assert actual_density < 0.15  # Should be around 5%

    def test_csr_symmetric(self) -> None:
        """CSR matrix is symmetric when symmetric=True."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        C = sparse_coupling_matrix_csr(50, sparsity=0.8, symmetric=True, seed=42)
        dense = C.to_dense()
        torch.testing.assert_close(dense, dense.T, atol=1e-6, rtol=1e-6)

    def test_csr_zero_diagonal(self) -> None:
        """CSR matrix has zero diagonal (no self-coupling)."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        C = sparse_coupling_matrix_csr(50, sparsity=0.8, seed=42)
        dense = C.to_dense()
        assert (dense.diag() == 0).all()

    def test_csr_vram_savings(self) -> None:
        """CSR uses less memory than dense for high sparsity."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        N = 1000
        sparsity = 0.95
        csr = sparse_coupling_matrix_csr(N, sparsity=sparsity, seed=42)

        # Dense would use N*N*4 bytes
        dense_bytes = N * N * 4
        # CSR uses: nnz*4 (values) + nnz*4 (col_indices) + (N+1)*4 (crow)
        nnz = csr.values().numel()
        csr_bytes = nnz * 4 + nnz * 4 + (N + 1) * 4

        assert csr_bytes < dense_bytes * 0.5  # < 50% of dense

    def test_csr_coupling_step_basic(self) -> None:
        """CSR coupling step produces finite results."""
        from prinet.utils.fused_kernels import (
            csr_coupling_step,
            sparse_coupling_matrix_csr,
        )

        N = 100
        csr = sparse_coupling_matrix_csr(N, sparsity=0.9, seed=42)
        phase = torch.rand(4, N) * TWO_PI

        coupling = csr_coupling_step(phase, csr)
        assert coupling.shape == (4, N)
        assert torch.isfinite(coupling).all()

    def test_csr_coupling_matches_dense(self) -> None:
        """CSR coupling produces same result as dense coupling."""
        from prinet.utils.fused_kernels import (
            csr_coupling_step,
            sparse_coupling_matrix_csr,
        )

        _seed()
        N = 50
        csr = sparse_coupling_matrix_csr(N, sparsity=0.8, seed=42)
        dense = csr.to_dense()
        phase = torch.rand(2, N) * TWO_PI

        csr_result = csr_coupling_step(phase, csr)

        # Dense reference
        for b in range(2):
            p = phase[b]
            diff = p.unsqueeze(0) - p.unsqueeze(1)  # (N, N)
            sin_diff = torch.sin(diff)
            dense_coupling = (dense * sin_diff).sum(dim=1)
            torch.testing.assert_close(
                csr_result[b], dense_coupling, atol=1e-4, rtol=1e-4
            )

    def test_csr_invalid_sparsity_raises(self) -> None:
        """Invalid sparsity raises ValueError."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        with pytest.raises(ValueError, match="sparsity"):
            sparse_coupling_matrix_csr(100, sparsity=1.0)
        with pytest.raises(ValueError, match="sparsity"):
            sparse_coupling_matrix_csr(100, sparsity=-0.1)


# ══════════════════════════════════════════════════════════════════
# 5. O.4: 100+ Oscillator Systems
# ══════════════════════════════════════════════════════════════════


class TestLargeScaleOscillators:
    """Tests for large-scale oscillator system with sparse k-NN."""

    def test_knn_neighbors_shape(self) -> None:
        """build_knn_neighbors produces correct shape."""
        from prinet.utils.fused_kernels import build_knn_neighbors

        nbr = build_knn_neighbors(200, k=8, seed=42)
        assert nbr.shape == (200, 8)
        assert nbr.dtype == torch.long

    def test_knn_neighbors_no_self(self) -> None:
        """No oscillator has itself as a neighbor."""
        from prinet.utils.fused_kernels import build_knn_neighbors

        nbr = build_knn_neighbors(100, k=10, seed=42)
        for i in range(100):
            assert i not in nbr[i].tolist()

    def test_sparse_knn_coupling_basic(self) -> None:
        """Sparse k-NN coupling produces finite results."""
        from prinet.utils.fused_kernels import (
            build_knn_neighbors,
            sparse_knn_coupling_step,
        )

        _seed()
        N = 200
        nbr = build_knn_neighbors(N, k=8, seed=42, device=DEVICE)
        phase = torch.rand(4, N, device=DEVICE) * TWO_PI
        amp = torch.ones(4, N, device=DEVICE)

        coupling = sparse_knn_coupling_step(phase, amp, nbr, coupling_strength=2.0)
        assert coupling.shape == (4, N)
        assert torch.isfinite(coupling).all()

    def test_large_scale_system_step(self) -> None:
        """LargeScaleOscillatorSystem.step() runs with N=200."""
        from prinet.utils.fused_kernels import LargeScaleOscillatorSystem

        _seed()
        sys = LargeScaleOscillatorSystem(n_oscillators=200, k_neighbors=8, seed=42).to(
            DEVICE
        )
        phase = torch.rand(4, 200, device=DEVICE) * TWO_PI
        amp = torch.ones(4, 200, device=DEVICE)

        new_p, new_a = sys.step(phase, amp, dt=0.01)
        assert new_p.shape == (4, 200)
        assert new_a.shape == (4, 200)
        assert torch.isfinite(new_p).all()
        assert torch.isfinite(new_a).all()

    def test_large_scale_system_integrate(self) -> None:
        """LargeScaleOscillatorSystem.integrate() runs 10+ steps."""
        from prinet.utils.fused_kernels import LargeScaleOscillatorSystem

        _seed()
        sys = LargeScaleOscillatorSystem(n_oscillators=100, k_neighbors=6, seed=42).to(
            DEVICE
        )
        phase = torch.rand(2, 100, device=DEVICE) * TWO_PI
        amp = torch.ones(2, 100, device=DEVICE)

        final_p, final_a = sys.integrate(phase, amp, n_steps=10, dt=0.01)
        assert torch.isfinite(final_p).all()
        assert torch.isfinite(final_a).all()
        # Amplitudes should still be in valid range
        assert (final_a >= 1e-6).all()
        assert (final_a <= 10.0).all()

    def test_large_scale_500_oscillators(self) -> None:
        """System scales to N=500 without error."""
        from prinet.utils.fused_kernels import LargeScaleOscillatorSystem

        _seed()
        sys = LargeScaleOscillatorSystem(n_oscillators=500, k_neighbors=10, seed=42).to(
            DEVICE
        )
        phase = torch.rand(2, 500, device=DEVICE) * TWO_PI
        amp = torch.ones(2, 500, device=DEVICE)

        new_p, new_a = sys.step(phase, amp, dt=0.01)
        assert new_p.shape == (2, 500)


# ══════════════════════════════════════════════════════════════════
# 6. O.5: Async CPU+GPU Pipeline
# ══════════════════════════════════════════════════════════════════


class TestAsyncPipeline:
    """Tests for async CPU+GPU pipeline."""

    def test_pipeline_creation(self) -> None:
        """AsyncCPUGPUPipeline can be instantiated."""
        from prinet.utils.fused_kernels import AsyncCPUGPUPipeline
        from prinet.nn.hybrid import HybridPRINetV2

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Use a mock daemon (just an object with start/stop/get_control)
        class MockDaemon:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def get_control(self):
                return None

        pipeline = AsyncCPUGPUPipeline(MockDaemon(), model, opt)
        assert pipeline.step_count == 0
        assert not pipeline.is_running

    def test_pipeline_train_step(self) -> None:
        """Pipeline.train_step completes one iteration."""
        from prinet.utils.fused_kernels import AsyncCPUGPUPipeline
        from prinet.nn.hybrid import HybridPRINetV2

        _seed()
        model = HybridPRINetV2(n_input=64, n_classes=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        class MockDaemon:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def get_control(self):
                return None

        pipeline = AsyncCPUGPUPipeline(MockDaemon(), model, opt)
        pipeline.start()
        assert pipeline.is_running

        x = torch.randn(4, 64, device=DEVICE)
        y = torch.randint(0, 5, (4,), device=DEVICE)
        loss = pipeline.train_step(x, y, F.nll_loss)
        assert isinstance(loss, float)
        assert pipeline.step_count == 1

        pipeline.stop()
        assert not pipeline.is_running


# ══════════════════════════════════════════════════════════════════
# 7. O.6: Model Pruning
# ══════════════════════════════════════════════════════════════════


class TestOscillatorPruning:
    """Tests for oscillator pruning."""

    def test_pruner_analyze(self) -> None:
        """OscillatorPruner.analyze() produces valid stats."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma
        from prinet.utils.fused_kernels import OscillatorPruner

        _seed()
        nd, nt, ng = 4, 8, 16
        N = nd + nt + ng
        dynamics = DiscreteDeltaThetaGamma(n_delta=nd, n_theta=nt, n_gamma=ng).to(
            DEVICE
        )

        phase = torch.rand(4, N, device=DEVICE) * TWO_PI
        amp = torch.ones(4, N, device=DEVICE)

        pruner = OscillatorPruner(threshold=0.1, n_eval_steps=20)
        stats = pruner.analyze(dynamics, phase, amp)

        assert "n_total" in stats
        assert "n_active" in stats
        assert "n_inactive" in stats
        assert "active_mask" in stats
        assert "mean_amplitudes" in stats
        assert "reduction_pct" in stats
        assert stats["n_total"] == N
        assert stats["n_active"] + stats["n_inactive"] == N

    def test_pruner_indices(self) -> None:
        """prune_indices provides per-band information."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma
        from prinet.utils.fused_kernels import OscillatorPruner

        _seed()
        nd, nt, ng = 4, 8, 16
        N = nd + nt + ng
        dynamics = DiscreteDeltaThetaGamma(n_delta=nd, n_theta=nt, n_gamma=ng).to(
            DEVICE
        )

        phase = torch.rand(4, N, device=DEVICE) * TWO_PI
        amp = torch.ones(4, N, device=DEVICE)

        pruner = OscillatorPruner(threshold=0.1, n_eval_steps=20)
        result = pruner.prune_indices(dynamics, phase, amp, nd, nt, ng)

        assert "delta_active" in result
        assert "theta_active" in result
        assert "gamma_active" in result
        assert "total_pruned" in result
        assert "reduction_pct" in result

    def test_high_threshold_prunes_more(self) -> None:
        """Higher threshold prunes more oscillators."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma
        from prinet.utils.fused_kernels import OscillatorPruner

        _seed()
        nd, nt, ng = 4, 8, 16
        N = nd + nt + ng
        dynamics = DiscreteDeltaThetaGamma(n_delta=nd, n_theta=nt, n_gamma=ng).to(
            DEVICE
        )

        phase = torch.rand(4, N, device=DEVICE) * TWO_PI
        amp = torch.ones(4, N, device=DEVICE)

        low = OscillatorPruner(threshold=0.01, n_eval_steps=20)
        high = OscillatorPruner(threshold=5.0, n_eval_steps=20)

        stats_low = low.analyze(dynamics, phase, amp)
        stats_high = high.analyze(dynamics, phase, amp)

        assert stats_high["n_inactive"] >= stats_low["n_inactive"]
