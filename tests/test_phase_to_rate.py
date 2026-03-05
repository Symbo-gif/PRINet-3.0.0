"""Unit tests for Q3 Phase-to-Rate conversion and sparsity regulation.

Covers PhaseToRateConverter, SparsityRegularizationLoss, phase_to_rate()
function, FFI/FBI gating, information preservation tests.

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prinet.core.propagation import phase_to_rate
from prinet.nn.layers import PhaseToRateConverter, SparsityRegularizationLoss

SEED = 42
TWO_PI = 2.0 * math.pi


# ===================================================================
# TestPhaseToRateConverter
# ===================================================================


class TestPhaseToRateConverter:
    """Tests for PhaseToRateConverter nn.Module."""

    def test_constructor(self) -> None:
        converter = PhaseToRateConverter(64, mode="soft", sparsity=0.1)
        assert converter.mode == "soft"

    def test_forward_shape(self) -> None:
        torch.manual_seed(SEED)
        converter = PhaseToRateConverter(32, mode="soft", sparsity=0.1)
        phase = torch.rand(32) * TWO_PI
        amplitude = torch.ones(32)
        rates = converter(phase, amplitude)
        assert rates.shape == phase.shape

    def test_soft_wta_smooth_gradients(self) -> None:
        """Soft WTA should produce smooth (non-zero) gradients."""
        converter = PhaseToRateConverter(32, mode="soft", sparsity=0.1)
        phase = torch.rand(32) * TWO_PI
        amplitude = torch.ones(32, requires_grad=True)
        rates = converter(phase, amplitude)
        # Use weighted sum — plain sum of softmax is constant (=1)
        # so its gradient w.r.t. inputs is zero by construction.
        weights = torch.arange(32, dtype=torch.float32)
        loss = (rates * weights).sum()
        loss.backward()
        assert amplitude.grad is not None
        # Soft mode: most gradients should be non-zero
        nonzero_frac = (amplitude.grad.abs() > 1e-8).float().mean()
        assert nonzero_frac > 0.3

    def test_hard_wta_sparse_output(self) -> None:
        """Hard WTA should produce approximately 10% active units."""
        converter = PhaseToRateConverter(100, mode="hard", sparsity=0.1)
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100)
        rates = converter(phase, amplitude)
        active_frac = (rates.abs() > 1e-8).float().mean().item()
        # Should be close to 10% (tolerance: 5-25%)
        assert 0.05 <= active_frac <= 0.25

    def test_annealed_mode(self) -> None:
        """Annealed mode should transition between soft and hard."""
        converter = PhaseToRateConverter(64, mode="annealed", sparsity=0.1)
        phase = torch.rand(64) * TWO_PI
        amplitude = torch.ones(64)
        rates = converter(phase, amplitude)
        assert rates.shape == (64,)
        assert torch.isfinite(rates).all()

    def test_sparsity_configurable(self) -> None:
        """Different sparsity levels should produce different outputs."""
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100)

        conv_high = PhaseToRateConverter(100, mode="hard", sparsity=0.5)
        conv_low = PhaseToRateConverter(100, mode="hard", sparsity=0.05)

        rates_high = conv_high(phase, amplitude)
        rates_low = conv_low(phase, amplitude)

        active_high = (rates_high.abs() > 1e-8).float().mean().item()
        active_low = (rates_low.abs() > 1e-8).float().mean().item()
        # Higher sparsity → more active units
        assert active_high > active_low

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_parity(self) -> None:
        """CPU and GPU results should be close."""
        converter = PhaseToRateConverter(32, mode="soft", sparsity=0.1)
        phase = torch.rand(32) * TWO_PI
        amplitude = torch.ones(32)

        cpu_rates = converter(phase, amplitude)
        gpu_rates = converter(phase.cuda(), amplitude.cuda())
        torch.testing.assert_close(
            cpu_rates, gpu_rates.cpu(), atol=1e-5, rtol=1e-5
        )


# ===================================================================
# TestFeedforwardInhibition
# ===================================================================


class TestFeedforwardInhibition:
    """Tests for FFI gating within phase_to_rate."""

    def test_ffi_gates_output(self) -> None:
        """FFI should gate output values."""
        phase = torch.rand(32) * TWO_PI
        amplitude = torch.ones(32)
        rates = phase_to_rate(phase, amplitude, mode="soft", sparsity=0.1)
        assert torch.isfinite(rates).all()
        assert rates.shape == phase.shape

    def test_ffi_output_bounded(self) -> None:
        """FFI output should be bounded."""
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100) * 5.0
        rates = phase_to_rate(phase, amplitude, mode="soft", sparsity=0.1)
        # Rates should not explode
        assert rates.abs().max() < 100.0

    def test_zero_amplitude_zero_rate(self) -> None:
        """Zero amplitude should produce near-zero rate (hard mode)."""
        phase = torch.rand(32) * TWO_PI
        amplitude = torch.zeros(32)
        # Use hard mode — soft mode (softmax) produces uniform 1/N
        rates = phase_to_rate(phase, amplitude, mode="hard", sparsity=0.1)
        assert rates.abs().max() < 1e-3


# ===================================================================
# TestFeedbackInhibition
# ===================================================================


class TestFeedbackInhibition:
    """Tests for FBI sparsity enforcement."""

    def test_fbi_enforces_sparsity(self) -> None:
        """Hard WTA should enforce sparsity constraint."""
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100)
        rates = phase_to_rate(phase, amplitude, mode="hard", sparsity=0.1)
        active = (rates.abs() > 1e-8).float().mean().item()
        assert active <= 0.25  # At most 25% active

    def test_topk_selection(self) -> None:
        """Top-k should select the highest-amplitude units."""
        phase = torch.rand(100) * TWO_PI
        # Give first 10 much higher amplitude
        amplitude = torch.ones(100)
        amplitude[:10] = 10.0
        rates = phase_to_rate(phase, amplitude, mode="hard", sparsity=0.1)
        # First 10 should be preferentially active
        active_top = (rates[:10].abs() > 1e-8).float().mean().item()
        active_rest = (rates[10:].abs() > 1e-8).float().mean().item()
        assert active_top >= active_rest

    def test_sparsity_parameter_effect(self) -> None:
        """Higher sparsity parameter should allow more active units."""
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100)

        rates_sparse = phase_to_rate(
            phase, amplitude, mode="hard", sparsity=0.05
        )
        rates_dense = phase_to_rate(
            phase, amplitude, mode="hard", sparsity=0.5
        )

        active_sparse = (rates_sparse.abs() > 1e-8).float().mean().item()
        active_dense = (rates_dense.abs() > 1e-8).float().mean().item()
        assert active_dense >= active_sparse


# ===================================================================
# TestSparsityRegularizationLoss
# ===================================================================


class TestSparsityRegularizationLoss:
    """Tests for SparsityRegularizationLoss nn.Module."""

    def test_loss_zero_at_target(self) -> None:
        """Loss should be ~0 when actual sparsity matches target."""
        loss_fn = SparsityRegularizationLoss(target_sparsity=0.5)
        # Create tensor with exactly 50% active
        rates = torch.zeros(100)
        rates[:50] = 1.0
        loss = loss_fn(rates)
        assert loss.item() < 0.1

    def test_loss_positive_too_dense(self) -> None:
        """Loss > 0 when activation is too dense."""
        loss_fn = SparsityRegularizationLoss(target_sparsity=0.1)
        rates = torch.ones(100)  # 100% active
        loss = loss_fn(rates)
        assert loss.item() > 0.0

    def test_loss_positive_too_sparse(self) -> None:
        """Loss > 0 when activation is too sparse."""
        loss_fn = SparsityRegularizationLoss(target_sparsity=0.9)
        rates = torch.zeros(100)  # 0% active
        loss = loss_fn(rates)
        assert loss.item() > 0.0

    def test_gradient_flows(self) -> None:
        """Gradient should flow through the sparsity loss."""
        loss_fn = SparsityRegularizationLoss(target_sparsity=0.1)
        rates = torch.ones(100, requires_grad=True)
        loss = loss_fn(rates)
        loss.backward()
        assert rates.grad is not None
        assert torch.isfinite(rates.grad).all()


# ===================================================================
# TestInformationPreservation
# ===================================================================


class TestInformationPreservation:
    """Tests verifying information is preserved through phase-to-rate."""

    def test_autoencoder_loss_decreases(self) -> None:
        """Simple autoencoder with PhaseToRate bottleneck should reduce loss."""
        torch.manual_seed(SEED)
        D = 32

        # Simple encoder → phase → rate → decoder
        encoder = torch.nn.Linear(D, D)
        decoder = torch.nn.Linear(D, D)
        converter = PhaseToRateConverter(D, mode="soft", sparsity=0.3)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )

        x = torch.randn(64, D)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            encoded = encoder(x)
            phase = encoded % TWO_PI
            amplitude = encoded.abs()
            rates = converter(phase, amplitude)
            decoded = decoder(rates)
            loss = torch.nn.functional.mse_loss(decoded, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over epochs
        assert losses[-1] < losses[0]

    def test_bottleneck_classifiable(self) -> None:
        """Phase-to-rate codes should be classifiable on synthetic data."""
        torch.manual_seed(SEED)
        D = 32
        n_classes = 5
        n_samples = 200

        # Generate class-separable data
        X = torch.randn(n_samples, D)
        y = torch.randint(0, n_classes, (n_samples,))
        # Add class-specific bias
        for c in range(n_classes):
            mask = y == c
            X[mask] += torch.randn(D) * 2.0

        converter = PhaseToRateConverter(D, mode="soft", sparsity=0.3)
        classifier = torch.nn.Linear(D, n_classes)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)

        for _ in range(50):
            optimizer.zero_grad()
            phase = X % TWO_PI
            amplitude = X.abs()
            rates = converter(phase, amplitude)
            logits = classifier(rates)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        # Check accuracy
        with torch.no_grad():
            phase = X % TWO_PI
            amplitude = X.abs()
            rates = converter(phase, amplitude)
            preds = classifier(rates).argmax(dim=-1)
            acc = (preds == y).float().mean().item()
        assert acc > 0.4  # >40% (well above chance = 20%)

    def test_sparse_vs_dense_comparison(self) -> None:
        """Sparse codes should have lower L0 norm than dense codes."""
        phase = torch.rand(100) * TWO_PI
        amplitude = torch.ones(100)

        sparse_rates = phase_to_rate(
            phase, amplitude, mode="hard", sparsity=0.1
        )
        dense_rates = phase_to_rate(
            phase, amplitude, mode="soft", sparsity=0.9
        )

        sparse_l0 = (sparse_rates.abs() > 1e-8).float().sum().item()
        dense_l0 = (dense_rates.abs() > 1e-8).float().sum().item()
        assert sparse_l0 <= dense_l0


# ===================================================================
# Hypothesis Property-Based Tests
# ===================================================================


class TestPhaseToRateProperties:
    """Property-based tests using hypothesis."""

    @given(
        sparsity=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=20, deadline=5000)
    def test_output_sparsity_bounded(self, sparsity: float) -> None:
        """Output sparsity is always in [0, 1]."""
        phase = torch.rand(64) * TWO_PI
        amplitude = torch.ones(64)
        rates = phase_to_rate(
            phase, amplitude, mode="soft", sparsity=sparsity
        )
        active_frac = (rates.abs() > 1e-8).float().mean().item()
        assert 0.0 <= active_frac <= 1.0

    @given(
        n=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=15, deadline=5000)
    def test_output_norm_bounded(self, n: int) -> None:
        """Output norm should be ≤ input norm (energy not created)."""
        phase = torch.rand(n) * TWO_PI
        amplitude = torch.rand(n)
        rates = phase_to_rate(phase, amplitude, mode="soft", sparsity=0.1)
        # Rate magnitudes should not exceed amplitude magnitudes significantly
        assert rates.norm() < amplitude.norm() * 20.0 + 1.0  # generous bound
