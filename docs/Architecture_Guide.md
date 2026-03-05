# PRINet Architecture Guide

## Overview

**PRINet** (Polyadic Resonance Intelligence Network) is a neural architecture that
processes information through multi-dimensional harmonic resonances rather than
traditional token embeddings. It combines oscillator dynamics, polyadic tensor
decomposition, and phase synchronization into a PyTorch-compatible framework.

## Package Structure

```
prinet/
├── core/                        # Fundamental dynamics
│   ├── decomposition.py         # Polyadic tensor decomposition (Tucker, CP)
│   ├── measurement.py           # Synchronization metrics (order parameter, PLV)
│   ├── subconscious.py          # State & control signal dataclasses
│   ├── subconscious_daemon.py   # Background ONNX inference thread
│   └── propagation/             # Oscillator models & dynamics subpackage
│       ├── oscillator_models.py # Kuramoto, Stuart-Landau, Hopf oscillators
│       ├── oscillator_state.py  # OscillatorState dataclass
│       ├── coupling.py          # Mean-field, sparse k-NN, small-world coupling
│       ├── integrators.py       # RK4, RK45, exponential, multi-rate integrators
│       ├── inhibition.py        # Feedforward/feedback inhibition
│       ├── networks.py          # DeltaThetaGammaNetwork, DiscreteDeltaThetaGamma
│       ├── temporal.py          # TemporalSequenceProcessor
│       └── sweep_utils.py       # Phase-space sweep helpers
├── nn/                          # PyTorch-compatible layers
│   ├── layers.py                # ResonanceLayer, PRINetModel, OscillatoryAttention,
│   │                            #   DiscreteDeltaThetaGammaLayer, GatedPhaseActivation
│   ├── hybrid.py                # HybridPRINet, HybridPRINetV2, PhaseTracker,
│   │                            #   TemporalSlotAttentionMOT, CLEVR-N variants
│   ├── activations.py           # dSiLU, HolomorphicActivation, PhaseActivation
│   ├── optimizers.py            # SynchronizedGradientDescent, SCALROptimizer,
│   │                            #   RIPOptimizer, AlternatingOptimizer
│   ├── hep.py                   # Holomorphic Equilibrium Propagation
│   ├── slot_attention.py        # Slot attention integration
│   ├── subconscious_model.py    # SubconsciousController MLP
│   ├── adaptive_allocation.py   # Region-aware allocation
│   ├── training_hooks.py        # Telemetry, active control, state collection
│   ├── ablation_variants.py     # Ablation study model variants
│   └── mot_evaluation.py        # Multi-object tracking evaluation
├── utils/                       # GPU kernels, solvers & tools
│   ├── triton_kernels.py        # Fused Triton + PyTorch fallback kernels
│   ├── fused_kernels.py         # CUDA JIT fused discrete step kernel
│   ├── cuda_kernels.py          # ODE solvers (RK4, RK45)
│   ├── npu_backend.py           # ONNX Runtime backend abstraction
│   ├── oscillosim.py            # OscilloSim interactive simulator (v2.0)
│   ├── datasets.py              # CLEVR-N, MNIST-N data loaders
│   ├── profiler.py              # PRINetProfiler instrumentation
│   ├── temporal_metrics.py      # Phase-locking value, coherence metrics
│   ├── temporal_training.py     # TemporalTrainer curriculum
│   ├── benchmark_reporting.py   # JSON benchmark reporting
│   ├── figure_generation.py     # NeurIPS figure generators
│   ├── table_generation.py      # NeurIPS LaTeX table generators
│   ├── adversarial_tools.py     # Adversarial robustness tools
│   └── y4q1_tools.py            # Year 4 Q1 experiment utilities
└── _deprecation.py              # API freeze & deprecation utilities
```

## Core Concepts

### 1. Oscillator Dynamics (`core.propagation`)

PRINet's foundation is coupled oscillator networks. Each "neuron" is an oscillator
with a phase θ ∈ [0, 2π) and amplitude A > 0. Information is encoded in the
**synchronization patterns** between oscillators.

**Oscillator Models:**

| Model | Equation | Use Case |
|-------|----------|----------|
| `KuramotoOscillator` | dθ/dt = ω + (K/N)·Σ sin(θⱼ − θᵢ) | Phase-only binding |
| `StuartLandauOscillator` | dz/dt = (μ + iω)z − \|z\|²z + K·coupling | Amplitude + phase |
| `HopfOscillator` | dz/dt = (α + iω)z − β\|z\|²z | Limit cycles |

**Coupling Modes** (see `API_Reference_Coupling_Topologies.md`):

| Mode | Complexity | Description |
|------|-----------|-------------|
| `"mean_field"` | O(N) | Global order parameter coupling |
| `"sparse_knn"` | O(N log N) | Phase-based k-nearest neighbor |
| `"full"` | O(N²) | All-to-all pairwise |
| `"csr"` | O(nnz) | Arbitrary sparse matrix (OscilloSim) |
| `"ring"` | O(N·k) | 1D ring lattice (chimera-capable) |
| `"small_world"` | O(N·k) | Watts-Strogatz small-world topology |
| `"auto"` | varies | Selects based on N |

**Hierarchical Multi-Rate Dynamics:**

The `DiscreteDeltaThetaGamma` network implements three frequency bands:
- **Delta** (slow): context/episode boundaries
- **Theta** (medium): sequential/working memory
- **Gamma** (fast): feature binding

Phase-amplitude coupling (PAC) between bands follows neuroscience principles:
gamma amplitude is modulated by theta phase.

### 2. Polyadic Tensor Decomposition (`core.decomposition`)

Input data is decomposed via Tucker (HOSVD) or CP decomposition into resonance
modes — multi-dimensional harmonic patterns that oscillators can interact with.

```python
from prinet import PolyadicTensor, CPDecomposition

tensor = PolyadicTensor(data=torch.randn(4, 8, 16))
cp = CPDecomposition(rank=10)
factors = cp.decompose(tensor)
```

### 3. Synchronization Measurement (`core.measurement`)

The **Kuramoto order parameter** r ∈ [0, 1] measures global phase coherence:

$$r = \frac{1}{N} \left| \sum_{j=1}^{N} e^{i\theta_j} \right|$$

- r ≈ 1: full synchronization (consensus/decision)
- r ≈ 0: desynchronized (exploration/uncertainty)

Sparse variants use k-NN indices for O(N·k) approximations of O(N²) metrics.

### 4. Neural Network Layers (`nn.layers`, `nn.hybrid`)

**Core Layers:**

| Layer | Description |
|-------|-------------|
| `ResonanceLayer` | Wraps oscillator dynamics in nn.Module |
| `PRINetModel` | Full end-to-end model (decomposition → dynamics → classification) |
| `HierarchicalResonanceLayer` | Multi-band (delta/theta/gamma) dynamics |
| `OscillatoryAttention` | Standard attention + phase-coherence bias |
| `DiscreteDeltaThetaGammaLayer` | Discrete multi-rate integration layer |
| `GatedPhaseActivation` | Phase-gated activation function |

**Hybrid Architectures:**

| Architecture | Description |
|-------------|-------------|
| `HybridPRINet` | LOBM → Phase-to-Rate → Transformer GRIM |
| `HybridPRINetV2` | Adaptive token count, conv stem, improved GRIM |
| `InterleavedHybridPRINet` | Alternating oscillatory + attention blocks |
| `PhaseTracker` | Multi-object tracking via phase identity |
| `TemporalSlotAttentionMOT` | Slot attention + temporal oscillatory binding for MOT |

**V2 Pipeline:**
```
Input → [Conv Stem] → [LOBM₁...LOBMₙ] → Phase-to-Rate → [GRIM] → Classifier
         (optional)    oscillatory         conversion      Transformer
                       binding             soft WTA        integration
```

### 5. Optimizers (`nn.optimizers`)

| Optimizer | Description |
|-----------|-------------|
| `SynchronizedGradientDescent` | SGD + phase synchronization barrier |
| `SCALROptimizer` | Synchronization-Conditioned Adaptive Learning Rate |
| `RIPOptimizer` | Resonance-Informed Parameter optimizer |
| `AlternatingOptimizer` | Separate LR/schedule for oscillatory vs rate-coded params |

### 6. Subconscious Control Loop

A background daemon monitors training and adaptively adjusts hyperparameters:

```
Training Loop → StateCollector → SubconsciousDaemon (ONNX)
                                       ↓
              ← ControlSignals ← ControlSignalBuffer
              (K range, LR mult, regime preference, alert level)
```

The `SubconsciousController` is a small MLP (32→128→128→8) that maps system
state to control signals. It runs on CPU/NPU/DirectML via ONNX Runtime, never
blocking the GPU training pipeline.

### 7. GPU Kernels (`utils.triton_kernels`)

Fused Triton kernels provide 2-10× speedup over PyTorch equivalents:
- **Fused Mean-Field RK4**: 9 kernel launches vs ~28+ PyTorch ops
- **Sparse k-NN Coupling**: single gather→trig→reduce kernel
- **PAC Modulation**: fused phase-amplitude coupling
- **Fused Discrete Step**: combined multi-rate integration

All kernels have pure PyTorch fallbacks for platforms without Triton.

## Design Principles

1. **Phase = Information**: Oscillator synchronization patterns encode object
   bindings, temporal sequences, and relational structure.

2. **Multi-Scale**: Delta/theta/gamma frequency bands capture different
   temporal scales — from episode context to feature binding.

3. **Hybrid**: Oscillatory dynamics for binding + Transformer attention for
   global integration. Best of both paradigms.

4. **Non-Blocking Control**: Subconscious daemon runs inference in background,
   training loop never waits for control signals.

5. **Graceful Degradation**: Triton → PyTorch fallback, NPU → CPU fallback.
   Everything works on any platform.

## API Stability

As of v3.0.0, the public API is frozen. All symbols in `prinet.__all__` are
covered by semantic versioning:
- **Patch** (3.0.x): bug fixes only, no API changes
- **Minor** (3.x.0): new features, backward-compatible
- **Major** (x.0.0): breaking changes with deprecation cycle

See `prinet._deprecation.FROZEN_PUBLIC_API` for the complete contract.
