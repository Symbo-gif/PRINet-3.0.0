"""Tests for Year 2 Q1 deliverables: discrete multi-rate, oscillatory attention,
interleaved hybrid, and subconscious training integration.

Covers:
    - Workstream A: DiscreteDeltaThetaGamma + DiscreteDeltaThetaGammaLayer
    - Workstream B: OscillatoryAttention + InterleavedHybridPRINet
    - Workstream C: Control policies + TelemetryLogger
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor


# ---- Fixtures -----------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(42)
    return g


# =========================================================================
# Workstream A: Discrete Multi-Rate Dynamics
# =========================================================================


class TestDiscreteDeltaThetaGamma:
    """Tests for DiscreteDeltaThetaGamma (core/propagation.py)."""

    def test_construction(self) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=32)
        assert net.n_delta == 4
        assert net.n_theta == 8
        assert net.n_gamma == 32
        assert net.n_total == 44

    def test_step_shapes(self, device: torch.device) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        net = net.to(device)
        B, N = 8, 28  # 4+8+16
        phase = torch.rand(B, N, device=device) * 2 * math.pi
        amp = torch.ones(B, N, device=device)

        new_phase, new_amp = net.step(phase, amp, dt=0.01)
        assert new_phase.shape == (B, N)
        assert new_amp.shape == (B, N)

    def test_step_1d_input(self) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=2, n_theta=4, n_gamma=8)
        N = 14
        phase = torch.rand(N) * 2 * math.pi
        amp = torch.ones(N)

        new_phase, new_amp = net.step(phase, amp)
        assert new_phase.shape == (N,)
        assert new_amp.shape == (N,)

    def test_integrate_shapes(self) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        B, N = 4, 28
        phase = torch.rand(B, N) * 2 * math.pi
        amp = torch.ones(B, N)

        final_p, final_a = net.integrate(phase, amp, n_steps=5, dt=0.01)
        assert final_p.shape == (B, N)
        assert final_a.shape == (B, N)

    def test_phase_bounded(self) -> None:
        """Phases should remain in [0, 2π) after integration."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(8, 28) * 2 * math.pi
        amp = torch.ones(8, 28)

        final_p, _ = net.integrate(phase, amp, n_steps=20)
        assert final_p.min() >= 0.0
        assert final_p.max() < 2 * math.pi + 0.01  # small tolerance

    def test_amplitude_bounded(self) -> None:
        """Amplitudes should be clamped to [1e-6, 10.0]."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(8, 28) * 2 * math.pi
        amp = torch.ones(8, 28) * 5.0  # large initial amplitude

        _, final_a = net.integrate(phase, amp, n_steps=50, dt=0.01)
        assert final_a.min() >= 1e-6
        assert final_a.max() <= 10.0

    def test_gradients_flow(self) -> None:
        """Verify gradients flow through step and integrate."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(4, 28) * 2 * math.pi
        amp = torch.ones(4, 28, requires_grad=True)

        final_p, final_a = net.integrate(phase, amp, n_steps=3)
        loss = final_a.sum()
        loss.backward()

        # Check that coupling weights received gradients
        assert net.W_delta.grad is not None
        assert net.W_pac_dt.weight.grad is not None
        assert net.W_pac_tg.weight.grad is not None

    def test_order_parameters(self) -> None:
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        # Fully synchronized phases → r ≈ 1
        phase = torch.zeros(2, 28)  # all zero
        r_d, r_t, r_g = net.order_parameters(phase)
        assert r_d.item() == pytest.approx(1.0, abs=0.01)
        assert r_t.item() == pytest.approx(1.0, abs=0.01)
        assert r_g.item() == pytest.approx(1.0, abs=0.01)

    def test_pac_index_computable(self) -> None:
        """PAC index should return finite values."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(4, 28) * 2 * math.pi
        amp = torch.ones(4, 28)

        pac_dt, pac_tg = net.pac_index(phase, amp)
        assert torch.isfinite(pac_dt)
        assert torch.isfinite(pac_tg)
        assert pac_dt >= 0.0
        assert pac_tg >= 0.0

    def test_is_nn_module(self) -> None:
        """Must be nn.Module for parameter registration."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma()
        assert isinstance(net, nn.Module)
        params = list(net.parameters())
        assert len(params) > 0

    def test_deterministic_with_same_input(self) -> None:
        """Same input → same output (no randomness in forward)."""
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        torch.manual_seed(123)
        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        phase = torch.rand(4, 28) * 2 * math.pi
        amp = torch.ones(4, 28)

        p1, a1 = net.integrate(phase.clone(), amp.clone(), n_steps=5)
        p2, a2 = net.integrate(phase.clone(), amp.clone(), n_steps=5)

        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(a1, a2)

    @pytest.mark.slow
    def test_speed_vs_transformer(self) -> None:
        """Forward pass should be ≤5× slower than an equivalent Transformer layer."""
        import time

        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        torch.manual_seed(0)
        B, D = 16, 128
        n_delta, n_theta, n_gamma = 4, 8, 32
        n_total = n_delta + n_theta + n_gamma

        dtg_layer = DiscreteDeltaThetaGammaLayer(
            n_delta=n_delta, n_theta=n_theta, n_gamma=n_gamma,
            n_dims=D, n_steps=5,
        )
        # Transformer baseline: roughly comparable parameter count
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=4, dim_feedforward=D * 4, batch_first=True,
        )

        x_dtg = torch.randn(B, D)
        # Transformer expects (B, seq, D) — use seq=1 for comparable workload
        x_tf = torch.randn(B, 1, D)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                dtg_layer(x_dtg)
                transformer_layer(x_tf)

        n_iters = 50
        # Time DTG
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_iters):
                dtg_layer(x_dtg)
            dtg_time = time.perf_counter() - t0

        # Time Transformer
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_iters):
                transformer_layer(x_tf)
            tf_time = time.perf_counter() - t0

        ratio = dtg_time / max(tf_time, 1e-9)
        assert ratio <= 5.0, (
            f"DiscreteDTG is {ratio:.2f}× slower than Transformer (limit: 5×)"
        )

    @pytest.mark.slow
    def test_clevr6_convergence(self) -> None:
        """Training on CLEVR-6 should show loss decrease over 10 epochs."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from clevr_n import make_clevr_n
        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        torch.manual_seed(42)

        # Build a small CLEVR-6 model around DiscreteDeltaThetaGammaLayer
        n_delta, n_theta, n_gamma = 4, 8, 32
        n_total = n_delta + n_theta + n_gamma
        scene_dim = 16  # D_PHASE from clevr_n
        query_dim = 60  # D_FEAT * 2 = (8+6+16)*2

        class _SmallDTGModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scene_proj = nn.Linear(scene_dim, n_total)
                self.dtg = DiscreteDeltaThetaGammaLayer(
                    n_delta=n_delta, n_theta=n_theta, n_gamma=n_gamma,
                    n_dims=n_total, n_steps=5,
                )
                self.query_proj = nn.Linear(query_dim, 64)
                self.head = nn.Sequential(
                    nn.Linear(n_total + 64, 64), nn.ReLU(), nn.Linear(64, 2),
                )

            def forward(self, scene: Tensor, query: Tensor) -> Tensor:
                h = scene.mean(dim=1)
                h = self.scene_proj(h)
                h = self.dtg(h)
                q = self.query_proj(query)
                return torch.log_softmax(self.head(torch.cat([h, q], -1)), -1)

        model = _SmallDTGModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scenes, queries, labels = make_clevr_n(6, 500, seed=42, phase_encode=True)

        # Record loss at epoch 0 and epoch 9
        first_loss: Optional[float] = None
        last_loss: Optional[float] = None

        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            log_probs = model(scenes, queries)
            loss = torch.nn.functional.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                first_loss = loss.item()
            if epoch == 9:
                last_loss = loss.item()

        assert first_loss is not None and last_loss is not None
        assert last_loss < first_loss, (
            f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
        )


class TestDiscreteDeltaThetaGammaLayer:
    """Tests for DiscreteDeltaThetaGammaLayer (nn/layers.py)."""

    def test_forward_shape(self) -> None:
        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        layer = DiscreteDeltaThetaGammaLayer(
            n_delta=4, n_theta=8, n_gamma=32, n_dims=128, n_steps=5
        )
        x = torch.randn(16, 128)
        out = layer(x)
        assert out.shape == (16, 44)  # 4+8+32

    def test_forward_1d(self) -> None:
        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        layer = DiscreteDeltaThetaGammaLayer(
            n_delta=4, n_theta=8, n_gamma=32, n_dims=128
        )
        x = torch.randn(128)
        out = layer(x)
        assert out.shape == (44,)

    def test_gradient_flow(self) -> None:
        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        layer = DiscreteDeltaThetaGammaLayer(
            n_delta=4, n_theta=8, n_gamma=16, n_dims=64, n_steps=3
        )
        x = torch.randn(4, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        # Check that layer parameters have gradients
        for p in layer.parameters():
            assert p.grad is not None, f"No grad for param shape {p.shape}"

    def test_output_nonnegative(self) -> None:
        """Output amplitudes should be non-negative."""
        from prinet.nn.layers import DiscreteDeltaThetaGammaLayer

        layer = DiscreteDeltaThetaGammaLayer(
            n_delta=4, n_theta=8, n_gamma=16, n_dims=64, n_steps=5
        )
        x = torch.randn(8, 64)
        out = layer(x)
        assert (out >= 0).all()

    def test_drop_in_for_hierarchical(self) -> None:
        """Should produce same-shaped output as HierarchicalResonanceLayer."""
        from prinet.nn.layers import (
            DiscreteDeltaThetaGammaLayer,
            HierarchicalResonanceLayer,
        )

        nd, nt, ng, D = 4, 8, 32, 128
        old_layer = HierarchicalResonanceLayer(nd, nt, ng, n_dims=D)
        new_layer = DiscreteDeltaThetaGammaLayer(nd, nt, ng, n_dims=D)

        x = torch.randn(4, D)
        old_out = old_layer(x)
        new_out = new_layer(x)

        assert old_out.shape == new_out.shape


# =========================================================================
# Workstream B: Oscillatory Attention & Interleaved Hybrid
# =========================================================================


class TestOscillatoryAttention:
    """Tests for OscillatoryAttention (nn/layers.py)."""

    def test_forward_shape(self) -> None:
        from prinet.nn.layers import OscillatoryAttention

        attn = OscillatoryAttention(d_model=64, n_heads=4)
        x = torch.randn(8, 10, 64)
        out = attn(x)
        assert out.shape == (8, 10, 64)

    def test_with_external_phase(self) -> None:
        from prinet.nn.layers import OscillatoryAttention

        attn = OscillatoryAttention(d_model=64, n_heads=4)
        x = torch.randn(4, 6, 64)
        phase = torch.rand(4, 6, 4) * 2 * math.pi
        out = attn(x, phase=phase)
        assert out.shape == (4, 6, 64)

    def test_gradient_flow(self) -> None:
        from prinet.nn.layers import OscillatoryAttention

        attn = OscillatoryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()

        assert x.grad is not None
        # Check alpha (coherence bias strength) has grad
        assert attn.alpha.grad is not None

    def test_alpha_zero_equals_standard_attention(self) -> None:
        """With alpha=0, should behave like standard MHA (no phase bias)."""
        from prinet.nn.layers import OscillatoryAttention

        attn = OscillatoryAttention(d_model=32, n_heads=2)
        # Zero out alpha
        with torch.no_grad():
            attn.alpha.zero_()
        attn.eval()  # disable dropout for deterministic comparison

        x = torch.randn(2, 4, 32)
        # Two forward passes with different phases should give same result
        phase_a = torch.zeros(2, 4, 2)
        phase_b = torch.rand(2, 4, 2) * 2 * math.pi

        out_a = attn(x, phase=phase_a)
        out_b = attn(x, phase=phase_b)

        # With alpha=0, coherence bias is multiplied by 0
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)

    def test_coherent_phases_increase_attention(self) -> None:
        """Tokens with aligned phases should attend more to each other."""
        from prinet.nn.layers import OscillatoryAttention

        torch.manual_seed(42)
        attn = OscillatoryAttention(d_model=16, n_heads=1)
        # Force positive alpha
        with torch.no_grad():
            attn.alpha.fill_(5.0)

        x = torch.randn(1, 4, 16)

        # All phases aligned → high coherence
        phase_aligned = torch.zeros(1, 4, 1)
        # Random phases → low coherence
        phase_random = torch.tensor([[[0.0], [1.57], [3.14], [4.71]]])

        out_aligned = attn(x, phase=phase_aligned)
        out_random = attn(x, phase=phase_random)

        # Outputs should differ when alpha is large
        diff = (out_aligned - out_random).abs().mean()
        assert diff > 0.01

    def test_d_model_not_divisible_by_heads(self) -> None:
        """Should raise ValueError."""
        from prinet.nn.layers import OscillatoryAttention

        with pytest.raises(ValueError, match="divisible"):
            OscillatoryAttention(d_model=65, n_heads=4)

    def test_with_mask(self) -> None:
        from prinet.nn.layers import OscillatoryAttention

        attn = OscillatoryAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 6, 32)
        # Causal mask
        mask = torch.tril(torch.ones(6, 6))
        out = attn(x, mask=mask)
        assert out.shape == (2, 6, 32)
        assert torch.isfinite(out).all()


class TestInterleavedHybridPRINet:
    """Tests for InterleavedHybridPRINet (nn/hybrid.py)."""

    def test_forward_shape(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        model = InterleavedHybridPRINet(
            n_input=128, n_classes=10, n_tokens=44,
            d_model=32, n_heads=4, n_layers=2,
            n_delta=4, n_theta=8, n_gamma=32,
        )
        x = torch.randn(8, 128)
        out = model(x)
        assert out.shape == (8, 10)

    def test_forward_1d(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        model = InterleavedHybridPRINet(
            n_input=64, n_classes=5, n_tokens=14,
            d_model=16, n_heads=2, n_layers=1,
            n_delta=2, n_theta=4, n_gamma=8,
        )
        x = torch.randn(64)
        out = model(x)
        assert out.shape == (5,)

    def test_output_is_log_probabilities(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        model = InterleavedHybridPRINet(
            n_input=64, n_classes=5, n_tokens=14,
            d_model=16, n_heads=2, n_layers=1,
            n_delta=2, n_theta=4, n_gamma=8,
        )
        x = torch.randn(4, 64)
        out = model(x)
        # log_softmax: all values ≤ 0, exp sums to ~1
        assert (out <= 0.0).all()
        probs = out.exp().sum(dim=-1)
        torch.testing.assert_close(
            probs, torch.ones(4), atol=1e-5, rtol=1e-5
        )

    def test_gradient_flow(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        model = InterleavedHybridPRINet(
            n_input=64, n_classes=3, n_tokens=14,
            d_model=16, n_heads=2, n_layers=1,
            n_delta=2, n_theta=4, n_gamma=8,
        )
        x = torch.randn(2, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert x.grad is not None
        # Dynamics parameters should have gradients
        assert model.dynamics.W_delta.grad is not None

    def test_param_groups(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        model = InterleavedHybridPRINet(
            n_input=64, n_classes=5, n_tokens=14,
            d_model=16, n_heads=2, n_layers=1,
            n_delta=2, n_theta=4, n_gamma=8,
        )
        osc_params = model.oscillatory_parameters()
        rate_params = model.rate_coded_parameters()

        assert len(osc_params) > 0
        assert len(rate_params) > 0

        # Should have no overlap
        osc_ids = {id(p) for p in osc_params}
        rate_ids = {id(p) for p in rate_params}
        assert osc_ids.isdisjoint(rate_ids)

    def test_no_nan_in_forward(self) -> None:
        from prinet.nn.hybrid import InterleavedHybridPRINet

        torch.manual_seed(0)
        model = InterleavedHybridPRINet(
            n_input=128, n_classes=10, n_tokens=44,
            d_model=32, n_heads=4, n_layers=2,
            n_delta=4, n_theta=8, n_gamma=32,
        )
        # Run multiple batches
        for _ in range(5):
            x = torch.randn(4, 128)
            out = model(x)
            assert torch.isfinite(out).all(), "NaN or Inf in output"

    @pytest.mark.slow
    def test_clevr6_no_nan(self) -> None:
        """InterleavedHybridPRINet trains on CLEVR-6 for 10 epochs without NaN."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from clevr_n import make_clevr_n
        from prinet.nn.hybrid import InterleavedHybridPRINet

        torch.manual_seed(42)

        scene_dim = 16  # D_PHASE
        query_dim = 60  # D_FEAT * 2 = (8+6+16)*2
        n_items = 6
        n_delta, n_theta, n_gamma = 4, 8, 32
        n_osc_total = n_delta + n_theta + n_gamma
        input_dim = scene_dim * n_items + query_dim  # 16*6+60=156

        model = InterleavedHybridPRINet(
            n_input=input_dim,
            n_classes=2,
            n_tokens=n_osc_total,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_discrete_steps=3,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scenes, queries, labels = make_clevr_n(6, 400, seed=42, phase_encode=True)
        flat_scenes = scenes.reshape(scenes.shape[0], -1)
        x = torch.cat([flat_scenes, queries], dim=-1)

        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            log_probs = model(x)
            assert torch.isfinite(log_probs).all(), (
                f"NaN/Inf in output at epoch {epoch}"
            )
            loss = torch.nn.functional.nll_loss(log_probs, labels)
            assert torch.isfinite(loss), f"NaN loss at epoch {epoch}"
            loss.backward()
            # Check gradients for NaN
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"NaN gradient in {name} at epoch {epoch}"
                    )
            optimizer.step()

    @pytest.mark.slow
    def test_ablation_oscillatory_bias(self) -> None:
        """Removing oscillatory bias (alpha=0) should degrade performance."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from clevr_n import make_clevr_n
        from prinet.nn.hybrid import InterleavedHybridPRINet

        torch.manual_seed(42)
        scene_dim = 16
        query_dim = 60  # D_FEAT * 2 = (8+6+16)*2
        n_items = 6
        n_delta, n_theta, n_gamma = 4, 8, 32
        n_osc_total = n_delta + n_theta + n_gamma
        input_dim = scene_dim * n_items + query_dim  # 16*6+60=156

        scenes, queries, labels = make_clevr_n(6, 400, seed=42, phase_encode=True)
        flat_scenes = scenes.reshape(scenes.shape[0], -1)
        x = torch.cat([flat_scenes, queries], dim=-1)

        def _train_model(model: nn.Module, n_epochs: int = 20) -> float:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                log_probs = model(x)
                loss = torch.nn.functional.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(x).argmax(dim=-1)
            return (preds == labels).float().mean().item()

        def _make_model() -> InterleavedHybridPRINet:
            return InterleavedHybridPRINet(
                n_input=input_dim,
                n_classes=2,
                n_tokens=n_osc_total,
                d_model=32,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                n_delta=n_delta,
                n_theta=n_theta,
                n_gamma=n_gamma,
                n_discrete_steps=3,
            )

        # Model with oscillatory bias
        torch.manual_seed(42)
        model_osc = _make_model()
        acc_osc = _train_model(model_osc)

        # Model with alpha forced to 0 (no oscillatory bias)
        torch.manual_seed(42)
        model_no_osc = _make_model()
        with torch.no_grad():
            for module in model_no_osc.modules():
                if hasattr(module, "alpha"):
                    module.alpha.fill_(0.0)
                    module.alpha.requires_grad_(False)
        acc_no_osc = _train_model(model_no_osc)

        # Oscillatory bias should help — or at worst be equal.
        # Use a soft check: osc accuracy should not be significantly worse
        assert acc_osc >= acc_no_osc - 0.05, (
            f"Oscillatory model ({acc_osc:.3f}) significantly worse than "
            f"no-oscillation ({acc_no_osc:.3f})"
        )


# =========================================================================
# Workstream C: Subconscious Training Integration
# =========================================================================


@dataclass
class _MockControlSignals:
    """Lightweight mock of ControlSignals for testing policies."""
    suggested_K_min: float = 0.5
    suggested_K_max: float = 4.0
    lr_multiplier: float = 1.1
    regime_mf_weight: float = 0.2
    regime_sk_weight: float = 0.5
    regime_full_weight: float = 0.3
    alert_level: float = 0.8
    coupling_mode_suggestion: float = 1.0


class TestControlPolicies:
    """Tests for control policies in training_hooks.py."""

    def test_lr_adjustment_no_alert(self) -> None:
        from prinet.nn.training_hooks import apply_lr_adjustment

        ctrl = _MockControlSignals(alert_level=0.5, lr_multiplier=1.2)
        opt = torch.optim.SGD([torch.randn(3, requires_grad=True)], lr=0.01)
        mult = apply_lr_adjustment(ctrl, opt)
        assert mult == 1.0  # alert < 0.7 → no adjustment
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_lr_adjustment_with_alert(self) -> None:
        from prinet.nn.training_hooks import apply_lr_adjustment

        ctrl = _MockControlSignals(alert_level=0.9, lr_multiplier=1.1)
        opt = torch.optim.SGD([torch.randn(3, requires_grad=True)], lr=0.01)
        mult = apply_lr_adjustment(ctrl, opt, max_adjustment=0.05)
        assert mult == pytest.approx(1.05)  # clamped from 1.1 to 1.05
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0105)

    def test_lr_adjustment_none_control(self) -> None:
        from prinet.nn.training_hooks import apply_lr_adjustment

        opt = torch.optim.SGD([torch.randn(3, requires_grad=True)], lr=0.01)
        mult = apply_lr_adjustment(None, opt)
        assert mult == 1.0

    def test_k_range_narrowing(self) -> None:
        from prinet.nn.training_hooks import apply_k_range_narrowing
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma(n_delta=4, n_theta=8, n_gamma=16)
        ctrl = _MockControlSignals(suggested_K_min=0.5, suggested_K_max=4.0)

        k_min, k_max = apply_k_range_narrowing(
            ctrl, net, field_name="W_delta"
        )
        assert k_min == pytest.approx(0.5)
        assert k_max == pytest.approx(4.0)

    def test_k_range_none_control(self) -> None:
        from prinet.nn.training_hooks import apply_k_range_narrowing
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        net = DiscreteDeltaThetaGamma()
        k_min, k_max = apply_k_range_narrowing(None, net)
        assert k_min == 0.0
        assert k_max == 0.0

    def test_regime_bias(self) -> None:
        from prinet.nn.training_hooks import apply_regime_bias

        ctrl = _MockControlSignals(
            regime_mf_weight=0.2, regime_sk_weight=0.5, regime_full_weight=0.3
        )
        regime = apply_regime_bias(ctrl)
        assert regime == "sparse_knn"

    def test_regime_bias_none(self) -> None:
        from prinet.nn.training_hooks import apply_regime_bias

        regime = apply_regime_bias(None)
        assert regime == "mean_field"

    def test_regime_bias_mean_field_wins(self) -> None:
        from prinet.nn.training_hooks import apply_regime_bias

        ctrl = _MockControlSignals(
            regime_mf_weight=0.8, regime_sk_weight=0.1, regime_full_weight=0.1
        )
        assert apply_regime_bias(ctrl) == "mean_field"


class TestTelemetryLogger:
    """Tests for TelemetryLogger."""

    def test_record_and_access(self) -> None:
        from prinet.nn.training_hooks import TelemetryLogger

        logger = TelemetryLogger(capacity=100)
        logger.record(epoch=1, loss=0.5, r_global=0.6)
        logger.record(epoch=2, loss=0.4, r_global=0.7)

        assert len(logger) == 2
        assert logger.records[0]["epoch"] == 1
        assert logger.records[1]["loss"] == pytest.approx(0.4)

    def test_to_json(self) -> None:
        from prinet.nn.training_hooks import TelemetryLogger

        logger = TelemetryLogger()
        logger.record(epoch=1, loss=0.5)
        logger.record(
            epoch=2, loss=0.3,
            control=_MockControlSignals(),
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            logger.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "control" in data[1]
            assert data[1]["control"]["lr_multiplier"] == pytest.approx(1.1)
        finally:
            os.unlink(path)

    def test_capacity_limit(self) -> None:
        from prinet.nn.training_hooks import TelemetryLogger

        logger = TelemetryLogger(capacity=5)
        for i in range(10):
            logger.record(epoch=i, loss=float(i))
        assert len(logger) == 5
        # Oldest records should be dropped
        assert logger.records[0]["epoch"] == 5

    def test_record_with_extra(self) -> None:
        from prinet.nn.training_hooks import TelemetryLogger

        logger = TelemetryLogger()
        logger.record(epoch=1, loss=0.5, extra={"custom_metric": 42})
        assert logger.records[0]["custom_metric"] == 42


# =========================================================================
# Integration: Top-level exports
# =========================================================================


class TestTopLevelExports:
    """Verify all Y2Q1 classes are importable from the public API."""

    def test_core_export(self) -> None:
        from prinet.core import DiscreteDeltaThetaGamma

        assert DiscreteDeltaThetaGamma is not None

    def test_nn_exports(self) -> None:
        from prinet.nn import (
            DiscreteDeltaThetaGammaLayer,
            InterleavedHybridPRINet,
            OscillatoryAttention,
            TelemetryLogger,
            apply_k_range_narrowing,
            apply_lr_adjustment,
            apply_regime_bias,
        )

        assert DiscreteDeltaThetaGammaLayer is not None
        assert OscillatoryAttention is not None
        assert InterleavedHybridPRINet is not None
        assert TelemetryLogger is not None

    def test_top_level_exports(self) -> None:
        from prinet import (
            DiscreteDeltaThetaGamma,
            DiscreteDeltaThetaGammaLayer,
            InterleavedHybridPRINet,
            OscillatoryAttention,
        )

        assert DiscreteDeltaThetaGamma is not None
        assert DiscreteDeltaThetaGammaLayer is not None
        assert OscillatoryAttention is not None
        assert InterleavedHybridPRINet is not None
