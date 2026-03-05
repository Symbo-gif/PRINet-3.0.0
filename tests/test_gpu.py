"""GPU integration tests for all PRINet components.

Validates that every core module (decomposition, propagation, measurement),
nn module (layers, optimizers), and utility (solvers, sparse) operates
correctly on CUDA devices. Skipped automatically when no GPU is available.

All tests use seeded RNG for determinism per Testing Standards.
"""

from __future__ import annotations

import math

import pytest
import torch

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

from prinet.core.decomposition import CPDecomposition, PolyadicTensor
from prinet.core.measurement import (
    extract_concept_probabilities,
    kuramoto_order_parameter,
    kuramoto_order_parameter_complex,
    mean_phase_coherence,
    phase_coherence_matrix,
    power_spectral_density,
    synchronization_energy,
)
from prinet.core.propagation import (
    KuramotoOscillator,
    OscillatorState,
    StuartLandauOscillator,
)
from prinet.nn.layers import PRINetModel, ResonanceLayer, oscillatory_weight_init
from prinet.nn.optimizers import RIPOptimizer, SynchronizedGradientDescent
from prinet.utils.cuda_kernels import (
    BatchedRK45Solver,
    FixedStepRK4Solver,
    gradient_checkpoint_integration,
    sparse_coupling_matrix,
)

SEED = 42
DEVICE = torch.device("cuda")
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> torch.Generator:
    """Seeded random generator on CUDA."""
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED)
    return gen


@pytest.fixture
def gpu_phases() -> torch.Tensor:
    """Random phase tensor on GPU."""
    torch.manual_seed(SEED)
    return torch.rand(64, device=DEVICE, dtype=DTYPE) * 2 * math.pi


@pytest.fixture
def gpu_amplitudes() -> torch.Tensor:
    """Random amplitude tensor on GPU."""
    torch.manual_seed(SEED)
    return torch.rand(64, device=DEVICE, dtype=DTYPE) + 0.1


# ===========================================================================
# DECOMPOSITION ON GPU
# ===========================================================================


class TestPolyadicTensorGPU:
    """PolyadicTensor (Tucker/HOSVD) on CUDA."""

    def test_decompose_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        tensor = torch.randn(8, 8, 8, device=DEVICE)
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4, device=DEVICE)
        pt.decompose(tensor)
        assert pt.core is not None
        assert pt.core.device.type == "cuda"
        for f in pt.factors:
            assert f.device.type == "cuda"

    def test_reconstruct_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        tensor = torch.randn(8, 8, 8, device=DEVICE)
        pt = PolyadicTensor(shape=(8, 8, 8), rank=8, device=DEVICE)
        pt.decompose(tensor)
        recon = pt.reconstruct()
        assert recon.device.type == "cuda"
        assert torch.allclose(recon, tensor, atol=1e-4)

    def test_reconstruction_error_gpu(self) -> None:
        torch.manual_seed(SEED)
        tensor = torch.randn(8, 8, 8, device=DEVICE)
        pt = PolyadicTensor(shape=(8, 8, 8), rank=4, device=DEVICE)
        pt.decompose(tensor)
        err = pt.reconstruction_error(tensor)
        assert isinstance(err, float)
        assert 0.0 <= err <= 1.0


class TestCPDecompositionGPU:
    """CP (ALS) decomposition on CUDA."""

    def test_decompose_and_reconstruct(self) -> None:
        torch.manual_seed(SEED)
        tensor = torch.randn(6, 6, 6, device=DEVICE)
        cp = CPDecomposition(shape=(6, 6, 6), rank=4, max_iter=50, device=DEVICE)
        cp.decompose(tensor)
        recon = cp.reconstruct()
        assert recon.device.type == "cuda"
        err = torch.norm(tensor - recon) / torch.norm(tensor)
        assert err.item() < 0.8  # CP with low rank on noisy data


# ===========================================================================
# PROPAGATION ON GPU
# ===========================================================================


class TestOscillatorStateGPU:
    """OscillatorState creation and device placement."""

    def test_random_state_on_gpu(self) -> None:
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert state.phase.device.type == "cuda"
        assert state.amplitude.device.type == "cuda"
        assert state.frequency.device.type == "cuda"
        assert state.device.type == "cuda"

    def test_synchronized_state_on_gpu(self) -> None:
        state = OscillatorState.create_synchronized(
            n_oscillators=16, device=DEVICE, dtype=DTYPE
        )
        assert state.phase.device.type == "cuda"
        assert state.amplitude.device.type == "cuda"


class TestKuramotoOscillatorGPU:
    """Kuramoto oscillator dynamics on CUDA."""

    def test_step_on_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=32, coupling_strength=2.0, device=DEVICE)
        state = OscillatorState.create_random(
            n_oscillators=32, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        new_state = osc.step(state, dt=0.01)
        assert new_state.phase.device.type == "cuda"
        assert new_state.amplitude.device.type == "cuda"
        assert new_state.frequency.device.type == "cuda"

    def test_step_with_coupling_matrix_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=16, coupling_strength=2.0, device=DEVICE)
        K = sparse_coupling_matrix(16, sparsity=0.5, device=DEVICE)
        osc.set_coupling_matrix(K)
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        new_state = osc.step(state, dt=0.01)
        assert new_state.phase.device.type == "cuda"

    def test_mean_field_mode_gpu(self) -> None:
        osc = KuramotoOscillator(
            n_oscillators=128,
            coupling_strength=2.0,
            device=DEVICE,
            mean_field=True,
        )
        state = OscillatorState.create_random(
            n_oscillators=128, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        new_state = osc.step(state, dt=0.01)
        assert new_state.phase.device.type == "cuda"

    def test_derivatives_on_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=16, coupling_strength=2.0, device=DEVICE)
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        dphase, damp, dfreq = osc.compute_derivatives(state)
        assert dphase.device.type == "cuda"
        assert damp.device.type == "cuda"
        assert dfreq.device.type == "cuda"


class TestStuartLandauOscillatorGPU:
    """Stuart-Landau oscillator dynamics on CUDA."""

    def test_step_on_gpu(self) -> None:
        osc = StuartLandauOscillator(
            n_oscillators=16, coupling_strength=1.0, device=DEVICE
        )
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        new_state = osc.step(state, dt=0.01)
        assert new_state.phase.device.type == "cuda"
        assert new_state.amplitude.device.type == "cuda"

    def test_derivatives_on_gpu(self) -> None:
        osc = StuartLandauOscillator(
            n_oscillators=16, coupling_strength=1.0, device=DEVICE
        )
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        dphase, damp, dfreq = osc.compute_derivatives(state)
        assert dphase.device.type == "cuda"
        assert damp.device.type == "cuda"


# ===========================================================================
# MEASUREMENT ON GPU
# ===========================================================================


class TestMeasurementGPU:
    """All measurement functions on CUDA tensors."""

    def test_kuramoto_order_parameter_gpu(self, gpu_phases: torch.Tensor) -> None:
        r = kuramoto_order_parameter(gpu_phases)
        assert 0.0 <= r.item() <= 1.0

    def test_kuramoto_order_parameter_complex_gpu(
        self, gpu_phases: torch.Tensor
    ) -> None:
        z = kuramoto_order_parameter_complex(gpu_phases)
        assert z.is_complex()
        assert torch.abs(z).item() <= 1.0 + 1e-6

    def test_mean_phase_coherence_gpu(self, gpu_phases: torch.Tensor) -> None:
        c = mean_phase_coherence(gpu_phases)
        assert 0.0 <= c.item() <= 1.0

    def test_phase_coherence_matrix_gpu(self, gpu_phases: torch.Tensor) -> None:
        mat = phase_coherence_matrix(gpu_phases)
        assert mat.device.type == "cuda"
        assert mat.shape == (64, 64)
        diag = torch.diag(mat)
        assert torch.allclose(diag, torch.ones(64, device=DEVICE))

    def test_power_spectral_density_gpu(
        self, gpu_phases: torch.Tensor, gpu_amplitudes: torch.Tensor
    ) -> None:
        psd = power_spectral_density(gpu_amplitudes, gpu_phases)
        assert psd.device.type == "cuda"
        assert (psd >= 0).all()

    def test_synchronization_energy_gpu(
        self, gpu_phases: torch.Tensor, gpu_amplitudes: torch.Tensor
    ) -> None:
        e = synchronization_energy(gpu_phases, gpu_amplitudes)
        assert isinstance(e.item(), float)

    def test_extract_concept_probabilities_gpu(
        self, gpu_phases: torch.Tensor, gpu_amplitudes: torch.Tensor
    ) -> None:
        n_concepts = 8
        concept_freqs = torch.linspace(0.5, 5.0, n_concepts, device=DEVICE)
        concept_bw = torch.full((n_concepts,), 0.5, device=DEVICE)
        probs = extract_concept_probabilities(
            gpu_amplitudes, gpu_phases, concept_freqs, concept_bw
        )
        assert probs.device.type == "cuda"
        assert probs.shape[-1] == n_concepts
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(probs.shape[:-1], device=DEVICE),
            atol=1e-5,
        )


# ===========================================================================
# NN LAYERS ON GPU
# ===========================================================================


class TestResonanceLayerGPU:
    """ResonanceLayer on CUDA via .to(device)."""

    def test_forward_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        layer = ResonanceLayer(n_oscillators=32, n_dims=64, n_steps=5, dt=0.01).to(
            DEVICE
        )
        x = torch.randn(8, 64, device=DEVICE)
        out = layer(x)
        assert out.device.type == "cuda"
        assert out.shape == (8, 32)
        assert not torch.isnan(out).any()

    def test_gradient_flow_gpu(self) -> None:
        torch.manual_seed(SEED)
        layer = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3, dt=0.01).to(
            DEVICE
        )
        x = torch.randn(4, 32, device=DEVICE, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"

    def test_order_parameter_gpu(self) -> None:
        torch.manual_seed(SEED)
        layer = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3, dt=0.01).to(
            DEVICE
        )
        x = torch.randn(4, 32, device=DEVICE)
        r = layer.get_order_parameter(x)
        assert (r >= 0).all() and (r <= 1.0 + 1e-6).all()


class TestPRINetModelGPU:
    """Full PRINet model on CUDA."""

    def test_forward_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=16, n_dims=32, n_concepts=5, n_layers=2, n_steps=3
        ).to(DEVICE)
        x = torch.randn(4, 32, device=DEVICE)
        log_probs = model(x)
        assert log_probs.device.type == "cuda"
        assert log_probs.shape == (4, 5)
        assert not torch.isnan(log_probs).any()

    def test_training_step_gpu(self) -> None:
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=16, n_dims=32, n_concepts=5, n_layers=2, n_steps=3
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 32, device=DEVICE)
        targets = torch.randint(0, 5, (4,), device=DEVICE)
        log_probs = model(x)
        loss = torch.nn.functional.nll_loss(log_probs, targets)
        loss.backward()
        opt.step()
        assert loss.device.type == "cuda"

    def test_oscillatory_weight_init_gpu(self) -> None:
        torch.manual_seed(SEED)
        model = PRINetModel(n_resonances=16, n_dims=32, n_concepts=5, n_layers=2).to(
            DEVICE
        )
        oscillatory_weight_init(model)
        x = torch.randn(4, 32, device=DEVICE)
        out = model(x)
        assert not torch.isnan(out).any()


class TestSynchronizedGradientDescentGPU:
    """SynchronizedGradientDescent optimizer with GPU parameters."""

    def test_step_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        model = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3, dt=0.01).to(
            DEVICE
        )
        opt = SynchronizedGradientDescent(
            model.parameters(), lr=0.01, sync_penalty=0.1, critical_order=0.5
        )
        x = torch.randn(4, 32, device=DEVICE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        opt.step(order_parameter=0.3)
        # Verify parameters are still on GPU
        for p in model.parameters():
            assert p.device.type == "cuda"


class TestRIPOptimizerGPU:
    """RIPOptimizer with GPU parameters."""

    def test_step_on_gpu(self) -> None:
        torch.manual_seed(SEED)
        model = ResonanceLayer(n_oscillators=16, n_dims=32, n_steps=3, dt=0.01).to(
            DEVICE
        )
        opt = RIPOptimizer(model.parameters(), lr=0.01)
        x = torch.randn(4, 32, device=DEVICE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        opt.step()
        for p in model.parameters():
            assert p.device.type == "cuda"


# ===========================================================================
# UTILS / SOLVERS ON GPU
# ===========================================================================


class TestBatchedRK45SolverGPU:
    """Batched adaptive RK45 solver on CUDA."""

    def test_solve_on_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=16, coupling_strength=2.0, device=DEVICE)
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        solver = BatchedRK45Solver(atol=1e-6, rtol=1e-3)
        result = solver.solve(osc, state, t_span=(0.0, 0.1))
        assert result.final_state.phase.device.type == "cuda"

    def test_solve_with_trajectory_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=8, coupling_strength=2.0, device=DEVICE)
        state = OscillatorState.create_random(
            n_oscillators=8, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        solver = BatchedRK45Solver(atol=1e-6, rtol=1e-3)
        result = solver.solve(osc, state, t_span=(0.0, 0.1), record_trajectory=True)
        assert len(result.trajectory) > 0
        for s in result.trajectory:
            assert s.phase.device.type == "cuda"


class TestFixedStepRK4SolverGPU:
    """Fixed-step RK4 solver on CUDA."""

    def test_solve_on_gpu(self) -> None:
        osc = StuartLandauOscillator(
            n_oscillators=16, coupling_strength=1.0, device=DEVICE
        )
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        solver = FixedStepRK4Solver(dt=0.01)
        result = solver.solve(osc, state, n_steps=10)
        assert result.final_state.phase.device.type == "cuda"
        assert result.final_state.amplitude.device.type == "cuda"


class TestSparseCouplingMatrixGPU:
    """Sparse coupling matrix generation on CUDA."""

    def test_on_gpu(self) -> None:
        K = sparse_coupling_matrix(
            32, sparsity=0.8, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        assert K.device.type == "cuda"
        assert K.shape == (32, 32)
        assert torch.diag(K).abs().max().item() < 1e-7

    def test_symmetric_on_gpu(self) -> None:
        K = sparse_coupling_matrix(
            16, sparsity=0.5, symmetric=True, device=DEVICE, seed=SEED
        )
        assert torch.allclose(K, K.T, atol=1e-7)


class TestGradientCheckpointGPU:
    """Gradient checkpoint integration on CUDA."""

    def test_checkpoint_on_gpu(self) -> None:
        osc = KuramotoOscillator(n_oscillators=16, coupling_strength=2.0, device=DEVICE)
        state = OscillatorState.create_random(
            n_oscillators=16, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        result = gradient_checkpoint_integration(osc, state, n_steps=10, dt=0.01)
        assert result.phase.device.type == "cuda"


# ===========================================================================
# GPU-CPU RESULT CONSISTENCY
# ===========================================================================


class TestGPUCPUConsistency:
    """Verify GPU and CPU produce numerically close results."""

    def test_kuramoto_step_consistency(self) -> None:
        """CPU and GPU produce same results from same initial state."""
        # Create state on CPU then copy to GPU for identical inputs
        state_cpu = OscillatorState.create_random(
            n_oscillators=16, device=torch.device("cpu"), dtype=DTYPE, seed=SEED
        )
        state_gpu = OscillatorState(
            phase=state_cpu.phase.to(DEVICE),
            amplitude=state_cpu.amplitude.to(DEVICE),
            frequency=state_cpu.frequency.to(DEVICE),
        )

        osc_cpu = KuramotoOscillator(
            n_oscillators=16, coupling_strength=2.0, device=torch.device("cpu")
        )
        osc_gpu = KuramotoOscillator(
            n_oscillators=16, coupling_strength=2.0, device=DEVICE
        )

        new_cpu = osc_cpu.step(state_cpu, dt=0.01)
        new_gpu = osc_gpu.step(state_gpu, dt=0.01)

        assert torch.allclose(new_cpu.phase, new_gpu.phase.cpu(), atol=1e-5)
        assert torch.allclose(new_cpu.amplitude, new_gpu.amplitude.cpu(), atol=1e-5)

    def test_order_parameter_consistency(self) -> None:
        torch.manual_seed(SEED)
        phase = torch.rand(64) * 2 * math.pi
        r_cpu = kuramoto_order_parameter(phase)
        r_gpu = kuramoto_order_parameter(phase.to(DEVICE))
        assert abs(r_cpu.item() - r_gpu.item()) < 1e-5

    def test_decomposition_consistency(self) -> None:
        torch.manual_seed(SEED)
        tensor = torch.randn(6, 6, 6)
        pt_cpu = PolyadicTensor(shape=(6, 6, 6), rank=6, device=torch.device("cpu"))
        pt_cpu.decompose(tensor)
        recon_cpu = pt_cpu.reconstruct()

        pt_gpu = PolyadicTensor(shape=(6, 6, 6), rank=6, device=DEVICE)
        pt_gpu.decompose(tensor.to(DEVICE))
        recon_gpu = pt_gpu.reconstruct()

        assert torch.allclose(recon_cpu, recon_gpu.cpu(), atol=1e-4)

    def test_prinet_model_consistency(self) -> None:
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=16, n_dims=32, n_concepts=5, n_layers=2, n_steps=3
        )
        x = torch.randn(2, 32)
        out_cpu = model(x)

        torch.manual_seed(SEED)
        model_gpu = PRINetModel(
            n_resonances=16, n_dims=32, n_concepts=5, n_layers=2, n_steps=3
        ).to(DEVICE)
        out_gpu = model_gpu(x.to(DEVICE))

        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-4)


# ===========================================================================
# GPU MEMORY AND PERFORMANCE SMOKE TESTS
# ===========================================================================


class TestGPUPerformance:
    """Basic GPU performance and memory smoke tests."""

    def test_large_oscillator_on_gpu(self) -> None:
        """1024 oscillators should run without OOM on 8GB GPU."""
        osc = KuramotoOscillator(
            n_oscillators=1024,
            coupling_strength=2.0,
            device=DEVICE,
            mean_field=True,
        )
        state = OscillatorState.create_random(
            n_oscillators=1024, device=DEVICE, dtype=DTYPE, seed=SEED
        )
        for _ in range(10):
            state = osc.step(state, dt=0.01)
        r = kuramoto_order_parameter(state.phase)
        assert 0.0 <= r.item() <= 1.0

    def test_gpu_memory_freed(self) -> None:
        """Verify tensors are freed after going out of scope."""
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        # Allocate large tensor
        t = torch.randn(4096, 4096, device=DEVICE)
        mem_during = torch.cuda.memory_allocated()
        assert mem_during > mem_before
        del t
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()
        assert mem_after <= mem_during

    def test_batched_forward_gpu(self) -> None:
        """Batched PRINet forward with larger batch on GPU."""
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=32, n_dims=64, n_concepts=10, n_layers=2, n_steps=5
        ).to(DEVICE)
        x = torch.randn(64, 64, device=DEVICE)
        out = model(x)
        assert out.shape == (64, 10)
        assert not torch.isnan(out).any()
