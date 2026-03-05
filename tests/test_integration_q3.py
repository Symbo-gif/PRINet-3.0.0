"""Q3 Integration Tests — End-to-end pipeline validation.

Tests:
    1. DeltaThetaGammaNetwork → HierarchicalResonanceLayer forward pass
       through a PRINetModel-like pipeline with shape validation.
    2. DeltaThetaGammaNetwork → PhaseToRateConverter → Linear classifier
       producing finite loss and non-zero gradients.

Reference:
    src/prinet/TODO_Year1_Q3_Q4.md — Hierarchical Model Support.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from prinet import (
    DeltaThetaGammaNetwork,
    HierarchicalResonanceLayer,
    OscillatorState,
    PhaseAmplitudeCoupling,
    PhaseToRateConverter,
    SparsityRegularizationLoss,
    phase_to_rate,
)

SEED: int = 42
DEVICE: torch.device = torch.device("cpu")


# ── Helpers ────────────────────────────────────────────────────────


class HierarchicalClassifier(nn.Module):
    """Minimal end-to-end model: HierarchicalResonanceLayer → Linear.

    Mimics a PRINetModel-like pipeline:
        input (B, D) → hierarchical dynamics → concat amps (B, N_total)
        → linear → logits (B, C).
    """

    def __init__(
        self,
        n_dims: int = 128,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_classes: int = 5,
        n_steps: int = 5,
    ) -> None:
        super().__init__()
        n_total = n_delta + n_theta + n_gamma
        self.hier = HierarchicalResonanceLayer(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_dims=n_dims,
            n_steps=n_steps,
            dt=0.01,
            coupling_strength=2.0,
            pac_depth=0.3,
        )
        self.classifier = nn.Linear(n_total, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        amps = self.hier(x)
        return self.classifier(amps)


class PhaseToRatePipeline(nn.Module):
    """End-to-end pipeline: DeltaThetaGamma → PhaseToRate → Linear.

    For each sample:
        1. Create DeltaThetaGammaNetwork, integrate to get final state.
        2. Concatenate phases and amplitudes from all bands.
        3. PhaseToRateConverter produces sparse rate codes.
        4. Linear classifier produces logits.
    """

    def __init__(
        self,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_classes: int = 5,
        n_steps: int = 5,
    ) -> None:
        super().__init__()
        self.n_delta = n_delta
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        n_total = n_delta + n_theta + n_gamma
        self.n_total = n_total
        self.n_steps = n_steps

        self.converter = PhaseToRateConverter(
            n_oscillators=n_total, mode="soft", sparsity=0.1
        )
        self.classifier = nn.Linear(n_total, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, n_total) — used to seed initial amplitudes."""
        B = x.shape[0]
        all_rates = []

        for b in range(B):
            # Create a network per sample
            net = DeltaThetaGammaNetwork(
                n_delta=self.n_delta,
                n_theta=self.n_theta,
                n_gamma=self.n_gamma,
                coupling_strength=2.0,
                pac_depth_dt=0.3,
                pac_depth_tg=0.3,
                device=x.device,
                dtype=x.dtype,
            )
            state = net.create_initial_state(seed=SEED + b)
            # Perturb amplitudes with input features for gradient flow
            ds, ts, gs = state
            feature = x[b]
            ds = OscillatorState(
                phase=ds.phase,
                amplitude=ds.amplitude + feature[: self.n_delta].abs() * 0.01,
                frequency=ds.frequency,
            )
            ts = OscillatorState(
                phase=ts.phase,
                amplitude=ts.amplitude
                + feature[self.n_delta : self.n_delta + self.n_theta].abs() * 0.01,
                frequency=ts.frequency,
            )
            gs = OscillatorState(
                phase=gs.phase,
                amplitude=gs.amplitude
                + feature[self.n_delta + self.n_theta :].abs() * 0.01,
                frequency=gs.frequency,
            )

            final, _ = net.integrate((ds, ts, gs), n_steps=self.n_steps, dt=0.01)
            # Concatenate phases and amplitudes
            phase = torch.cat([final[0].phase, final[1].phase, final[2].phase])
            amplitude = torch.cat(
                [final[0].amplitude, final[1].amplitude, final[2].amplitude]
            )
            rate = self.converter(phase.unsqueeze(0), amplitude.unsqueeze(0))
            all_rates.append(rate.squeeze(0))

        rates = torch.stack(all_rates, dim=0)  # (B, n_total)
        return self.classifier(rates)


# ══════════════════════════════════════════════════════════════════
# Test Suite 1: Delta-Theta-Gamma Forward through Pipeline
# ══════════════════════════════════════════════════════════════════


class TestHierarchicalPipeline:
    """Integration tests for hierarchical dynamics in a model pipeline."""

    def test_hierarchical_layer_output_shape(self) -> None:
        """HierarchicalResonanceLayer produces correct output shape."""
        torch.manual_seed(SEED)
        layer = HierarchicalResonanceLayer(
            n_delta=4, n_theta=8, n_gamma=32, n_dims=128, n_steps=3
        )
        x = torch.randn(8, 128)
        out = layer(x)
        assert out.shape == (8, 44), f"Expected (8, 44), got {out.shape}"

    def test_hierarchical_layer_1d_input(self) -> None:
        """HierarchicalResonanceLayer handles 1-D (unbatched) input."""
        torch.manual_seed(SEED)
        layer = HierarchicalResonanceLayer(
            n_delta=4, n_theta=8, n_gamma=32, n_dims=64, n_steps=3
        )
        x = torch.randn(64)
        out = layer(x)
        assert out.shape == (44,), f"Expected (44,), got {out.shape}"

    def test_end_to_end_classifier_shape(self) -> None:
        """Full pipeline: input → hierarchical layer → classifier → logits."""
        torch.manual_seed(SEED)
        model = HierarchicalClassifier(
            n_dims=128,
            n_delta=4,
            n_theta=8,
            n_gamma=32,
            n_classes=5,
            n_steps=3,
        )
        x = torch.randn(4, 128)
        logits = model(x)
        assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    def test_end_to_end_finite_output(self) -> None:
        """Pipeline outputs are finite (no NaN/Inf)."""
        torch.manual_seed(SEED)
        model = HierarchicalClassifier(
            n_dims=64,
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        x = torch.randn(4, 64)
        logits = model(x)
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf"

    def test_end_to_end_loss_finite(self) -> None:
        """Cross-entropy loss on pipeline output is finite."""
        torch.manual_seed(SEED)
        model = HierarchicalClassifier(
            n_dims=64,
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        x = torch.randn(4, 64)
        targets = torch.randint(0, 3, (4,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_end_to_end_gradients_flow(self) -> None:
        """Gradients flow from loss through hierarchical layer to input projection."""
        torch.manual_seed(SEED)
        model = HierarchicalClassifier(
            n_dims=64,
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        x = torch.randn(4, 64)
        targets = torch.randint(0, 3, (4,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Check classifier linear has gradients
        assert model.classifier.weight.grad is not None
        assert torch.isfinite(model.classifier.weight.grad).all()
        # Check at least one hierarchical projection has gradients
        has_hier_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.hier.parameters()
        )
        assert has_hier_grad, "No gradients in HierarchicalResonanceLayer"

    def test_pac_depth_registered_as_parameter(self) -> None:
        """PAC depth parameters are registered nn.Parameters.

        Note: Gradients don't flow through .item() in current impl.
        The parameters are registered for future differentiability.
        """
        torch.manual_seed(SEED)
        model = HierarchicalClassifier(
            n_dims=64,
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        # Verify they are registered parameters
        param_names = {n for n, _ in model.hier.named_parameters()}
        assert "pac_depth_dt" in param_names
        assert "pac_depth_tg" in param_names
        assert model.hier.pac_depth_dt.requires_grad
        assert model.hier.pac_depth_tg.requires_grad

    def test_hierarchical_layer_different_configs(self) -> None:
        """Layer works with various n_delta/n_theta/n_gamma configs."""
        torch.manual_seed(SEED)
        configs = [
            (2, 4, 8),
            (4, 16, 64),
            (1, 2, 4),
        ]
        for nd, nt, ng in configs:
            layer = HierarchicalResonanceLayer(
                n_delta=nd,
                n_theta=nt,
                n_gamma=ng,
                n_dims=32,
                n_steps=2,
            )
            x = torch.randn(2, 32)
            out = layer(x)
            expected = nd + nt + ng
            assert out.shape == (2, expected), (
                f"Config ({nd},{nt},{ng}): expected (2, {expected}), "
                f"got {out.shape}"
            )


# ══════════════════════════════════════════════════════════════════
# Test Suite 2: PhaseToRate Pipeline
# ══════════════════════════════════════════════════════════════════


class TestPhaseToRatePipeline:
    """Integration tests for DeltaThetaGamma → PhaseToRate → classifier."""

    def test_pipeline_output_shape(self) -> None:
        """Pipeline produces correct logit shape."""
        torch.manual_seed(SEED)
        n_total = 4 + 8 + 32
        model = PhaseToRatePipeline(
            n_delta=4,
            n_theta=8,
            n_gamma=32,
            n_classes=5,
            n_steps=3,
        )
        x = torch.randn(4, n_total)
        logits = model(x)
        assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    def test_pipeline_finite_loss(self) -> None:
        """Pipeline produces finite cross-entropy loss."""
        torch.manual_seed(SEED)
        n_total = 4 + 8 + 32
        model = PhaseToRatePipeline(
            n_delta=4,
            n_theta=8,
            n_gamma=32,
            n_classes=5,
            n_steps=3,
        )
        x = torch.randn(4, n_total)
        targets = torch.randint(0, 5, (4,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_pipeline_gradients_exist(self) -> None:
        """Loss.backward() produces non-zero gradients in classifier."""
        torch.manual_seed(SEED)
        n_total = 2 + 4 + 16
        model = PhaseToRatePipeline(
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        x = torch.randn(2, n_total)
        targets = torch.randint(0, 3, (2,))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        assert model.classifier.weight.grad is not None
        assert torch.isfinite(model.classifier.weight.grad).all()
        assert model.classifier.weight.grad.abs().sum() > 0

    def test_converter_temperature_registered(self) -> None:
        """PhaseToRateConverter temperature is a learnable parameter.

        Note: Gradients don't flow through .item() in current impl.
        The parameter is registered for future differentiability.
        """
        torch.manual_seed(SEED)
        n_total = 2 + 4 + 16
        model = PhaseToRatePipeline(
            n_delta=2,
            n_theta=4,
            n_gamma=16,
            n_classes=3,
            n_steps=3,
        )
        temp_param = model.converter.temperature
        assert isinstance(temp_param, nn.Parameter)
        assert temp_param.requires_grad
        assert temp_param.item() > 0.0

    def test_sparsity_loss_integration(self) -> None:
        """SparsityRegularizationLoss can be combined with task loss."""
        torch.manual_seed(SEED)
        n_total = 2 + 4 + 16
        converter = PhaseToRateConverter(
            n_oscillators=n_total, mode="soft", sparsity=0.1
        )
        sparsity_loss_fn = SparsityRegularizationLoss(
            target_sparsity=0.9, temperature=0.1
        )

        phase = torch.randn(4, n_total)
        amplitude = torch.ones(4, n_total)
        rate = converter(phase, amplitude)

        sparsity_loss = sparsity_loss_fn(rate)
        assert torch.isfinite(
            sparsity_loss
        ), f"Sparsity loss not finite: {sparsity_loss.item()}"
        assert sparsity_loss.item() >= 0.0, "Sparsity loss should be >= 0"

    def test_phase_to_rate_deterministic(self) -> None:
        """Same inputs produce same outputs (deterministic)."""
        torch.manual_seed(SEED)
        n = 22
        converter = PhaseToRateConverter(n_oscillators=n, mode="soft", sparsity=0.1)
        phase = torch.randn(4, n)
        amplitude = torch.ones(4, n)

        r1 = converter(phase, amplitude)
        r2 = converter(phase, amplitude)
        assert torch.allclose(r1, r2), "PhaseToRateConverter not deterministic"

    def test_hard_mode_sparsity(self) -> None:
        """Hard mode produces truly sparse outputs."""
        torch.manual_seed(SEED)
        n = 100
        converter = PhaseToRateConverter(n_oscillators=n, mode="hard", sparsity=0.1)
        phase = torch.randn(8, n)
        amplitude = torch.ones(8, n)
        rate = converter(phase, amplitude)

        # In hard mode, ~10% should be active (non-zero)
        active_frac = (rate > 0).float().mean().item()
        assert (
            active_frac <= 0.15
        ), f"Hard mode too dense: {active_frac:.2%} active, expected ≤15%"


# ══════════════════════════════════════════════════════════════════
# Test Suite 3: Core-to-NN Boundary Checks
# ══════════════════════════════════════════════════════════════════


class TestCoreToNNBoundary:
    """Verify data types and shapes across core↔nn boundaries."""

    def test_dtg_state_to_layer_compatibility(self) -> None:
        """DeltaThetaGammaNetwork state is compatible with nn layer input."""
        net = DeltaThetaGammaNetwork(
            n_delta=4,
            n_theta=8,
            n_gamma=32,
            coupling_strength=2.0,
        )
        state = net.create_initial_state(seed=SEED)
        ds, ts, gs = state
        # All states should be float32 CPU tensors
        for s, name in [(ds, "delta"), (ts, "theta"), (gs, "gamma")]:
            assert s.phase.dtype == torch.float32, f"{name} phase dtype wrong"
            assert s.amplitude.dtype == torch.float32, f"{name} amp dtype wrong"
            assert s.phase.device.type == "cpu", f"{name} not on CPU"

    def test_dtg_integrate_shape_consistency(self) -> None:
        """Integrate preserves oscillator count in final state."""
        net = DeltaThetaGammaNetwork(
            n_delta=4,
            n_theta=8,
            n_gamma=32,
        )
        state = net.create_initial_state(seed=SEED)
        final, _ = net.integrate(state, n_steps=10, dt=0.01)
        ds, ts, gs = final
        assert ds.phase.shape == (4,)
        assert ts.phase.shape == (8,)
        assert gs.phase.shape == (32,)

    def test_phase_to_rate_preserves_batch_dim(self) -> None:
        """phase_to_rate function preserves batch dimension."""
        phase = torch.randn(16, 64)
        amplitude = torch.ones(16, 64)
        rate = phase_to_rate(phase, amplitude, mode="soft")
        assert rate.shape == (16, 64)

    def test_order_parameters_finite_after_integration(self) -> None:
        """Order parameters are finite and in [0, 1] after integration."""
        net = DeltaThetaGammaNetwork(
            n_delta=4,
            n_theta=8,
            n_gamma=32,
            coupling_strength=2.0,
        )
        state = net.create_initial_state(seed=SEED)
        final, _ = net.integrate(state, n_steps=50, dt=0.01)
        r_d, r_t, r_g = net.order_parameters(final)

        for r, name in [(r_d, "delta"), (r_t, "theta"), (r_g, "gamma")]:
            assert torch.isfinite(r), f"r_{name} not finite: {r}"
            assert 0.0 <= r.item() <= 1.0 + 1e-6, f"r_{name} out of range: {r.item()}"
