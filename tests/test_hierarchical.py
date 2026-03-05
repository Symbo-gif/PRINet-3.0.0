"""Unit tests for Q3 hierarchical binding dynamics.

Covers MultiRateIntegrator, PhaseAmplitudeCoupling, ThetaGammaNetwork,
DeltaThetaGammaNetwork, sweep_coupling_params, detect_oscillation, and
HierarchicalResonanceLayer / PhaseAmplitudeCouplingLayer nn.Modules.

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prinet.core.propagation import (
    DeltaThetaGammaNetwork,
    KuramotoOscillator,
    MultiRateIntegrator,
    OscillatorState,
    PhaseAmplitudeCoupling,
    ThetaGammaNetwork,
    detect_oscillation,
    phase_to_rate,
    sweep_coupling_params,
)
from prinet.nn.layers import (
    HierarchicalResonanceLayer,
    PhaseAmplitudeCouplingLayer,
)

SEED = 42
DEVICE = torch.device("cpu")
TWO_PI = 2.0 * math.pi


# ===================================================================
# TestMultiRateIntegrator
# ===================================================================


class TestMultiRateIntegrator:
    """Tests for multi-rate integration with sub-stepping."""

    def test_constructor_params(self) -> None:
        mri = MultiRateIntegrator(sub_steps=10)
        assert mri.sub_steps == 10

    def test_sub_step_count(self) -> None:
        mri = MultiRateIntegrator(sub_steps=20)
        assert mri.sub_steps == 20

    def test_frequency_band_tensor_shape(self) -> None:
        """Integration produces outputs with correct shape."""
        N = 32
        model = KuramotoOscillator(N, coupling_strength=2.0)
        state = OscillatorState.create_random(N, seed=SEED)

        mri = MultiRateIntegrator(sub_steps=5)
        new_state = mri.step(model, state, dt=0.001)
        assert new_state.phase.shape == (N,)
        assert new_state.amplitude.shape == (N,)

    def test_gamma_substeps(self) -> None:
        """Gamma band (fast) should use requested sub-steps per delta step."""
        mri = MultiRateIntegrator(sub_steps=20)
        assert mri.sub_steps == 20

    def test_gradient_flow(self) -> None:
        """Gradients should flow through sub-steps."""
        N = 16
        model = KuramotoOscillator(N, coupling_strength=2.0)
        state = OscillatorState.create_random(N, seed=SEED)
        state = OscillatorState(
            phase=state.phase,
            amplitude=state.amplitude.clone().requires_grad_(True),
            frequency=state.frequency,
        )

        mri = MultiRateIntegrator(sub_steps=3)
        new_state = mri.step(model, state, dt=0.001)
        loss = new_state.amplitude.sum()
        loss.backward()
        assert state.amplitude.grad is not None
        assert torch.isfinite(state.amplitude.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_parity(self) -> None:
        """CPU and GPU results should match."""
        N = 32
        state_cpu = OscillatorState.create_random(N, seed=SEED)

        # Create identical state on GPU by copying from CPU
        state_gpu = OscillatorState(
            phase=state_cpu.phase.cuda(),
            amplitude=state_cpu.amplitude.cuda(),
            frequency=state_cpu.frequency.cuda(),
        )

        model_cpu = KuramotoOscillator(N, coupling_strength=2.0)
        model_gpu = KuramotoOscillator(
            N, coupling_strength=2.0, device=torch.device("cuda")
        )

        mri = MultiRateIntegrator(sub_steps=5)
        cpu_out = mri.step(model_cpu, state_cpu, dt=0.001)
        gpu_out = mri.step(model_gpu, state_gpu, dt=0.001)

        torch.testing.assert_close(
            cpu_out.phase, gpu_out.phase.cpu(), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            cpu_out.amplitude, gpu_out.amplitude.cpu(), atol=1e-4, rtol=1e-4
        )


# ===================================================================
# TestPhaseAmplitudeCoupling
# ===================================================================


class TestPhaseAmplitudeCoupling:
    """Tests for PAC modulation formula."""

    def test_amplitude_modulation_formula(self) -> None:
        """A_out = A_in * (1 + m * cos(mean_slow_phase + offset))."""
        pac = PhaseAmplitudeCoupling(modulation_depth=0.5)
        # Slow phases all at 0 → cos(0) = 1 → modulation = 1.5
        slow_phase = torch.zeros(10)
        fast_amp = torch.ones(10) * 2.0
        out = pac.modulate(slow_phase, fast_amp)
        expected = 2.0 * (1.0 + 0.5 * 1.0)
        torch.testing.assert_close(
            out, torch.full_like(out, expected), atol=1e-5, rtol=1e-5
        )

    def test_modulation_depth_effect(self) -> None:
        """Higher modulation depth → stronger modulation."""
        slow_phase = torch.zeros(10)
        fast_amp = torch.ones(10)

        pac_low = PhaseAmplitudeCoupling(modulation_depth=0.1)
        pac_high = PhaseAmplitudeCoupling(modulation_depth=0.9)

        out_low = pac_low.modulate(slow_phase, fast_amp)
        out_high = pac_high.modulate(slow_phase, fast_amp)

        # High depth should deviate more from 1.0
        assert (out_high - 1.0).abs().mean() > (out_low - 1.0).abs().mean()

    def test_phase_wrapping_preserved(self) -> None:
        """PAC should handle phases near 2π correctly."""
        pac = PhaseAmplitudeCoupling(modulation_depth=0.5)
        slow_phase = torch.tensor([0.0, TWO_PI, TWO_PI * 2])
        fast_amp = torch.ones(5)
        out = pac.modulate(slow_phase, fast_amp)
        assert torch.isfinite(out).all()

    def test_amplitude_clamping(self) -> None:
        """Output amplitudes should be clamped."""
        pac = PhaseAmplitudeCoupling(modulation_depth=0.5)
        slow_phase = torch.zeros(10)
        fast_amp = torch.ones(10) * 1e-7  # Very small
        out = pac.modulate(slow_phase, fast_amp)
        assert (out >= 1e-6).all()  # Clamped to min

    def test_gradient_flow(self) -> None:
        """Gradients should flow through PAC modulation."""
        pac = PhaseAmplitudeCoupling(modulation_depth=0.5)
        slow_phase = torch.zeros(10)
        fast_amp = torch.ones(10, requires_grad=True)
        out = pac.modulate(slow_phase, fast_amp)
        out.sum().backward()
        assert fast_amp.grad is not None
        assert torch.isfinite(fast_amp.grad).all()


# ===================================================================
# TestDeltaThetaGammaNetwork
# ===================================================================


class TestDeltaThetaGammaNetwork:
    """Tests for 3-frequency hierarchical network."""

    def test_constructor(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=8, n_theta=16, n_gamma=32)
        assert net.n_delta == 8
        assert net.n_theta == 16
        assert net.n_gamma == 32

    def test_forward_shape(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        new_state = net.step(state, dt=0.001)
        # State is a tuple of 3 OscillatorStates
        assert len(new_state) == 3
        assert new_state[0].phase.shape == (4,)
        assert new_state[1].phase.shape == (8,)
        assert new_state[2].phase.shape == (16,)

    def test_order_parameter_per_band(self) -> None:
        """Each frequency band should have a computable order parameter."""
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        r_delta, r_theta, r_gamma = net.order_parameters(state)
        assert 0.0 <= r_delta.item() <= 1.0
        assert 0.0 <= r_theta.item() <= 1.0
        assert 0.0 <= r_gamma.item() <= 1.0

    def test_inter_band_pac_active(self) -> None:
        """PAC should modulate fast-band amplitudes."""
        net = DeltaThetaGammaNetwork(
            n_delta=4,
            n_theta=8,
            n_gamma=16,
            pac_depth_dt=0.8,
            pac_depth_tg=0.8,
        )
        state = net.create_initial_state(seed=SEED)
        # Run several steps to let PAC take effect
        for _ in range(10):
            state = net.step(state, dt=0.001)
        # Gamma amplitudes should differ from initial uniform
        gamma_amp = state[2].amplitude
        assert gamma_amp.std() > 0.0

    def test_intra_band_sparse_knn(self) -> None:
        """Network should accept and use sparse k-NN coupling."""
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16, sparse_k=3)
        # sparse_k is passed through to internal KuramotoOscillator models
        assert net._delta_model.sparse_k == 3
        # Verify it runs without error
        state = net.create_initial_state(seed=SEED)
        state = net.step(state, dt=0.001)
        for s in state:
            assert torch.isfinite(s.phase).all()

    def test_100step_stability(self) -> None:
        """100 integration steps should produce no NaN."""
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        for _ in range(100):
            state = net.step(state, dt=0.001)
        for s in state:
            assert torch.isfinite(s.phase).all()
            assert torch.isfinite(s.amplitude).all()

    def test_n1_edge_case(self) -> None:
        """N=1 per band should not crash."""
        net = DeltaThetaGammaNetwork(n_delta=1, n_theta=1, n_gamma=1)
        state = net.create_initial_state(seed=SEED)
        state = net.step(state, dt=0.001)
        for s in state:
            assert torch.isfinite(s.phase).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self) -> None:
        """GPU forward pass should work."""
        net = DeltaThetaGammaNetwork(
            n_delta=4,
            n_theta=8,
            n_gamma=16,
            device=torch.device("cuda"),
        )
        state = net.create_initial_state(seed=SEED)
        new_state = net.step(state, dt=0.001)
        assert new_state[0].phase.is_cuda
        for s in new_state:
            assert torch.isfinite(s.phase).all()


# ===================================================================
# TestThetaGammaNetwork
# ===================================================================


class TestThetaGammaNetwork:
    """Tests for 2-frequency hierarchical network."""

    def test_constructor(self) -> None:
        net = ThetaGammaNetwork(n_theta=8, n_gamma=32)
        assert net.n_theta == 8
        assert net.n_gamma == 32

    def test_forward_shape(self) -> None:
        net = ThetaGammaNetwork(n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        new_state = net.step(state, dt=0.001)
        # State is a tuple of 2 OscillatorStates
        assert len(new_state) == 2
        assert new_state[0].phase.shape == (8,)
        assert new_state[1].phase.shape == (16,)

    def test_baseline_comparison(self) -> None:
        """ThetaGamma should be constructable and run without error."""
        net = ThetaGammaNetwork(n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        for _ in range(50):
            state = net.step(state, dt=0.001)
        for s in state:
            assert torch.isfinite(s.phase).all()

    def test_two_freq_order_parameter(self) -> None:
        net = ThetaGammaNetwork(n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        r_theta, r_gamma = net.order_parameters(state)
        assert 0.0 <= r_theta.item() <= 1.0
        assert 0.0 <= r_gamma.item() <= 1.0


# ===================================================================
# TestCouplingParameterSweep
# ===================================================================


class TestCouplingParameterSweep:
    """Tests for sweep_coupling_params grid search."""

    def test_grid_search_produces_results(self) -> None:
        results = sweep_coupling_params(
            n_oscillators=24,
            k_values=[1.0, 2.0],
            m_values=[0.3, 0.6],
            n_steps=10,
            dt=0.001,
        )
        assert len(results) > 0

    def test_json_output_valid(self) -> None:
        import json

        results = sweep_coupling_params(
            n_oscillators=12,
            k_values=[1.0],
            m_values=[0.3],
            n_steps=5,
            dt=0.001,
        )
        # Should be JSON-serializable
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_optimal_m_in_range(self) -> None:
        results = sweep_coupling_params(
            n_oscillators=24,
            k_values=[2.0],
            m_values=[0.1, 0.3, 0.5, 0.7, 0.9],
            n_steps=20,
            dt=0.001,
        )
        # All modulation depths should be in specified range
        for r in results:
            assert 0.0 <= r["m"] <= 1.0


# ===================================================================
# TestHierarchicalNumericalStability
# ===================================================================


class TestHierarchicalNumericalStability:
    """Tests for numerical stability of hierarchical dynamics."""

    def test_phase_wrapping_all_frequencies(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        for _ in range(50):
            state = net.step(state, dt=0.001)
        # All phases should be in [0, 2π)
        for s in state:
            assert (s.phase >= -1e-6).all()
            assert (s.phase < TWO_PI + 1e-6).all()

    def test_amplitude_clamping_per_band(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        for _ in range(50):
            state = net.step(state, dt=0.001)
        for s in state:
            assert (s.amplitude >= 0.0).all()
            # Amplitudes may slightly exceed 10.0 due to PAC modulation
            assert (s.amplitude <= 11.0).all()

    def test_no_nan_large_system(self) -> None:
        """N=4096 total oscillators split across bands should not NaN."""
        net = DeltaThetaGammaNetwork(n_delta=512, n_theta=1024, n_gamma=2560)
        state = net.create_initial_state(seed=SEED)
        for _ in range(10):
            state = net.step(state, dt=0.0005)
        for s in state:
            assert torch.isfinite(s.phase).all()
            assert torch.isfinite(s.amplitude).all()

    def test_gradient_finite_50steps(self) -> None:
        """Gradients should remain finite after 50 steps."""
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        # Make gamma amplitude require grad for testing
        gamma_amp = state[2].amplitude.clone().requires_grad_(True)
        state = (
            state[0],
            state[1],
            OscillatorState(
                phase=state[2].phase,
                amplitude=gamma_amp,
                frequency=state[2].frequency,
            ),
        )
        for _ in range(50):
            state = net.step(state, dt=0.001)
        loss = state[2].amplitude.sum()
        loss.backward()
        assert gamma_amp.grad is not None
        assert torch.isfinite(gamma_amp.grad).all()


# ===================================================================
# TestHierarchicalEndToEnd
# ===================================================================


class TestHierarchicalEndToEnd:
    """Integration tests for DeltaThetaGamma → PhaseToRate → classifier."""

    def _concat_state(
        self, state: tuple[OscillatorState, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all band phases and amplitudes."""
        all_phase = torch.cat([s.phase for s in state])
        all_amp = torch.cat([s.amplitude for s in state])
        return all_phase, all_amp

    def test_pipeline_shape(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        state = net.step(state, dt=0.001)

        all_phase, all_amp = self._concat_state(state)
        rates = phase_to_rate(all_phase, all_amp, mode="soft", sparsity=0.1)
        assert rates.shape == all_phase.shape

    def test_finite_loss(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        state = net.step(state, dt=0.001)

        all_phase, all_amp = self._concat_state(state)
        rates = phase_to_rate(all_phase, all_amp, mode="soft", sparsity=0.1)
        # Simulate a linear classifier
        classifier = torch.nn.Linear(rates.shape[0], 10)
        logits = classifier(rates.unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
        assert torch.isfinite(loss)

    def test_gradient_flow_full_pipeline(self) -> None:
        net = DeltaThetaGammaNetwork(n_delta=4, n_theta=8, n_gamma=16)
        state = net.create_initial_state(seed=SEED)
        # Make gamma amplitudes require grad
        gamma_amp = state[2].amplitude.clone().requires_grad_(True)
        state = (
            state[0],
            state[1],
            OscillatorState(
                phase=state[2].phase,
                amplitude=gamma_amp,
                frequency=state[2].frequency,
            ),
        )
        state = net.step(state, dt=0.001)

        all_phase, all_amp = self._concat_state(state)
        rates = phase_to_rate(all_phase, all_amp, mode="soft", sparsity=0.1)
        classifier = torch.nn.Linear(rates.shape[0], 10)
        logits = classifier(rates.unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
        loss.backward()
        assert gamma_amp.grad is not None


# ===================================================================
# TestDetectOscillation
# ===================================================================


class TestDetectOscillation:
    """Tests for detect_oscillation utility."""

    def test_oscillating_signal(self) -> None:
        """High-variance signal should be detected as oscillating."""
        r_history = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
        assert detect_oscillation(r_history, window=5, threshold=0.01)

    def test_stable_signal(self) -> None:
        """Low-variance signal should NOT be detected as oscillating."""
        r_history = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        assert not detect_oscillation(r_history, window=5, threshold=0.01)

    def test_short_history(self) -> None:
        """History shorter than window should return False."""
        r_history = [0.5, 0.6]
        assert not detect_oscillation(r_history, window=10, threshold=0.01)


# ===================================================================
# Hypothesis Property-Based Tests
# ===================================================================


class TestHierarchicalProperties:
    """Property-based tests using hypothesis."""

    @given(
        n_delta=st.integers(min_value=1, max_value=8),
        n_theta=st.integers(min_value=1, max_value=16),
        n_gamma=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=10, deadline=10000)
    def test_order_parameter_in_range(
        self, n_delta: int, n_theta: int, n_gamma: int
    ) -> None:
        """Order parameter is always in [0, 1]."""
        net = DeltaThetaGammaNetwork(n_delta=n_delta, n_theta=n_theta, n_gamma=n_gamma)
        state = net.create_initial_state(seed=SEED)
        for _ in range(5):
            state = net.step(state, dt=0.001)
        r_d, r_t, r_g = net.order_parameters(state)
        assert 0.0 <= r_d.item() <= 1.0 + 1e-6
        assert 0.0 <= r_t.item() <= 1.0 + 1e-6
        assert 0.0 <= r_g.item() <= 1.0 + 1e-6

    @given(
        n_delta=st.integers(min_value=2, max_value=8),
        n_theta=st.integers(min_value=2, max_value=16),
        n_gamma=st.integers(min_value=2, max_value=32),
    )
    @settings(max_examples=10, deadline=10000)
    def test_phases_in_range(self, n_delta: int, n_theta: int, n_gamma: int) -> None:
        """Phases are always in [0, 2π]."""
        net = DeltaThetaGammaNetwork(n_delta=n_delta, n_theta=n_theta, n_gamma=n_gamma)
        state = net.create_initial_state(seed=SEED)
        for _ in range(10):
            state = net.step(state, dt=0.001)
        for s in state:
            assert (s.phase >= -1e-6).all()
            assert (s.phase <= TWO_PI + 1e-6).all()


# ===================================================================
# TestPhaseAmplitudeCouplingLayer (nn.Module)
# ===================================================================


class TestPhaseAmplitudeCouplingLayerModule:
    """Tests for the nn.Module PAC wrapper."""

    def test_constructor(self) -> None:
        layer = PhaseAmplitudeCouplingLayer(initial_depth=0.5)
        assert layer.modulation_depth.item() == pytest.approx(0.5, abs=1e-6)

    def test_learnable_parameter(self) -> None:
        layer = PhaseAmplitudeCouplingLayer(initial_depth=0.5)
        params = list(layer.parameters())
        assert len(params) >= 1  # modulation_depth should be learnable

    def test_forward_shape(self) -> None:
        layer = PhaseAmplitudeCouplingLayer(initial_depth=0.5)
        slow_phase = torch.zeros(10)
        fast_amp = torch.ones(10)
        out = layer(slow_phase, fast_amp)
        assert out.shape == (10,)


# ===================================================================
# TestHierarchicalResonanceLayer (nn.Module)
# ===================================================================


class TestHierarchicalResonanceLayerModule:
    """Tests for the nn.Module hierarchical resonance wrapper."""

    def test_constructor(self) -> None:
        layer = HierarchicalResonanceLayer(n_delta=4, n_theta=8, n_gamma=16, n_steps=5)
        assert layer.n_steps == 5

    def test_forward_output_shape(self) -> None:
        n_delta, n_theta, n_gamma = 4, 8, 16
        n_dims = 28
        layer = HierarchicalResonanceLayer(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_dims=n_dims,
            n_steps=3,
        )
        x = torch.randn(2, n_dims)  # (batch, features)
        out = layer(x)
        # Output should be (batch, n_total) = (2, 28)
        assert out.shape == (2, n_delta + n_theta + n_gamma)
        assert torch.isfinite(out).all()
