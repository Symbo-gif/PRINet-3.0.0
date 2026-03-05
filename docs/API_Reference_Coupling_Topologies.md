# PRINet Coupling Topologies & Triton Kernels — API Reference

> **Version:** 3.0.0 | **Module:** `prinet.core.propagation`, `prinet.utils.triton_kernels`

This document provides the complete API reference for PRINet's oscillator coupling subsystem, including:

1. **Coupling topology modes** — how to select and configure coupling between oscillators.
2. **Triton fused kernels** — GPU-accelerated kernels for mean-field and sparse k-NN coupling.
3. **Custom topology examples** — extending PRINet with custom neighbour structures.

---

## Table of Contents

- [1. Coupling Modes Overview](#1-coupling-modes-overview)
- [2. KuramotoOscillator API](#2-kuramotooscillator-api)
  - [2.1 Constructor Parameters](#21-constructor-parameters)
  - [2.2 Coupling Mode Selection](#22-coupling-mode-selection)
  - [2.3 Coupling Mode Semantics](#23-coupling-mode-semantics)
- [3. Triton Fused Kernels API](#3-triton-fused-kernels-api)
  - [3.1 triton\_fused\_mean\_field\_rk4\_step()](#31-triton_fused_mean_field_rk4_step)
  - [3.2 triton\_sparse\_knn\_coupling()](#32-triton_sparse_knn_coupling)
  - [3.3 PyTorch Fallback Functions](#33-pytorch-fallback-functions)
  - [3.4 triton\_available()](#34-triton_available)
- [4. Architecture & Performance](#4-architecture--performance)
  - [4.1 Kernel Launch Structure](#41-kernel-launch-structure)
  - [4.2 Block Size Tuning](#42-block-size-tuning)
  - [4.3 Benchmark Results](#43-benchmark-results)
- [5. Custom Coupling Topologies](#5-custom-coupling-topologies)
  - [5.1 Using Sparse k-NN with Custom Neighbours](#51-using-sparse-k-nn-with-custom-neighbours)
  - [5.2 Small-World Topology](#52-small-world-topology)
  - [5.3 Ring Topology](#53-ring-topology)
  - [5.4 Hierarchical Topology](#54-hierarchical-topology)
  - [5.5 Feeding Custom Topologies to Triton Kernels](#55-feeding-custom-topologies-to-triton-kernels)
- [6. Migration Guide](#6-migration-guide)

---

## 1. Coupling Modes Overview

PRINet's oscillator models support three coupling topologies, each representing a different trade-off between accuracy and computational cost:

| Mode | Complexity | Description | Best For |
|------|-----------|-------------|----------|
| `"mean_field"` | O(N) | Global order parameter coupling | N > 10K, uniform coupling |
| `"sparse_knn"` | O(N log N) | k nearest phase-neighbours | Training, gradient flows, chimera states |
| `"full"` | O(N²) | All-to-all pairwise coupling | Small N (< 500), exact dynamics |
| `"csr"` | O(nnz) | Arbitrary sparse matrix (OscilloSim) | Custom topologies, moderate N |
| `"ring"` | O(N·k) | 1D ring lattice spatial neighbours | Chimera states, spatial coupling |
| `"small_world"` | O(N·k) | Watts-Strogatz small-world topology | Realistic brain-like coupling |
| `"auto"` (default) | varies | Selects based on N | General use |

The `"auto"` mode selects:
- `"mean_field"` when `mean_field=True` is set explicitly.
- `"sparse_knn"` when `N ≥ 1000`.
- `"full"` when `N < 1000`.

---

## 2. KuramotoOscillator API

### 2.1 Constructor Parameters

```python
from prinet.core.propagation import KuramotoOscillator

model = KuramotoOscillator(
    n_oscillators: int,           # Number of oscillators N
    coupling_strength: float,     # Coupling constant K
    decay_rate: float = 0.1,      # Amplitude decay λ
    freq_adaptation_rate: float = 0.01,  # Frequency adaptation γ
    mean_field: bool = False,     # Force mean-field mode
    coupling_mode: str = "auto",  # "auto", "full", "sparse_knn", "mean_field"
    sparse_k: int | None = None,  # Custom k for sparse_knn (default: ⌈log₂N⌉)
    device: str | torch.device = "cpu",
)
```

### 2.2 Coupling Mode Selection

```python
# Mean-field (O(N)) — fastest, loss of local structure
model = KuramotoOscillator(1024, K=2.0, coupling_mode="mean_field")

# Sparse k-NN (O(N log N)) — preserves local structure
model = KuramotoOscillator(1024, K=2.0, coupling_mode="sparse_knn")

# Sparse k-NN with custom k
model = KuramotoOscillator(1024, K=2.0, coupling_mode="sparse_knn", sparse_k=20)

# Full pairwise (O(N²)) — exact
model = KuramotoOscillator(256, K=2.0, coupling_mode="full")

# Auto selection (recommended for most use cases)
model = KuramotoOscillator(1024, K=2.0, coupling_mode="auto")
```

### 2.3 Coupling Mode Semantics

**Mean-Field Coupling** (`"mean_field"`):
Computes the complex order parameter Z = (1/N) Σ rᵢ exp(iφᵢ), extracts magnitude R = |Z| and phase ψ = arg(Z), then applies:

- dφᵢ/dt = ωᵢ + K · R · sin(ψ − φᵢ)
- drᵢ/dt = −λ · rᵢ + K · R · cos(ψ − φᵢ)
- dωᵢ/dt = γ · K · R · sin(ψ − φᵢ) / N

**Sparse k-NN Coupling** (`"sparse_knn"`):
For each oscillator i, finds the k nearest neighbours in phase (using sort-based O(N log N) selection), then couples only to those neighbours with K_eff = K/k:

- dφᵢ/dt = ωᵢ + Σⱼ∈nbr(i) K_eff · rⱼ · sin(φⱼ − φᵢ)
- drᵢ/dt = −λ · rᵢ + Σⱼ∈nbr(i) K_eff · rⱼ · cos(φⱼ − φᵢ)
- dωᵢ/dt = γ · Σⱼ∈nbr(i) K_eff · rⱼ · sin(φⱼ − φᵢ) / k

**Full Pairwise Coupling** (`"full"`):
Constructs the N×N phase difference matrix and computes all-to-all sin/cos coupling.

---

## 3. Triton Fused Kernels API

All Triton kernel functions are available from `prinet.utils.triton_kernels` and also exported at the top-level `prinet` package.

```python
from prinet import (
    triton_available,
    triton_fused_mean_field_rk4_step,
    triton_sparse_knn_coupling,
    pytorch_mean_field_rk4_step,
    pytorch_sparse_knn_coupling,
)
```

### 3.1 triton\_fused\_mean\_field\_rk4\_step()

Performs a complete 4th-order Runge-Kutta integration step with mean-field Kuramoto coupling, using Triton-accelerated kernels.

```python
def triton_fused_mean_field_rk4_step(
    phase: Tensor,          # Shape (N,), float32, CUDA
    amplitude: Tensor,      # Shape (N,), float32, CUDA
    frequency: Tensor,      # Shape (N,), float32, CUDA
    K: float,               # Coupling strength
    decay: float,           # Amplitude decay rate λ
    gamma: float,           # Frequency adaptation rate γ
    dt: float,              # Timestep
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (new_phase, new_amplitude, new_frequency)."""
```

**Details:**
- Fuses 4 RK4 stages × 2 Triton passes + 1 final sum = **9 kernel launches** (vs. ~28+ PyTorch ops).
- Two-pass mean-field: block-local reduction → atomic\_add for global Z → element-wise derivatives.
- Phase output wrapped to [0, 2π); amplitude clamped to ≥ 0.
- Uses algebraic identity to avoid atan2: R·sin(ψ−φ) = Zy·cos(φ) − Zx·sin(φ).

**Example:**
```python
import torch

phase = torch.rand(16384, device="cuda") * 6.2832
amp = torch.ones(16384, device="cuda")
freq = torch.randn(16384, device="cuda")

new_p, new_a, new_f = triton_fused_mean_field_rk4_step(
    phase, amp, freq, K=2.0, decay=0.1, gamma=0.01, dt=0.01
)
```

**Raises:** `RuntimeError` if Triton unavailable or tensors not on CUDA.

### 3.2 triton\_sparse\_knn\_coupling()

Computes sparse k-NN Kuramoto coupling derivatives in a single fused gather→trig→reduce kernel.

```python
def triton_sparse_knn_coupling(
    phase: Tensor,          # Shape (N,), float32, CUDA
    amplitude: Tensor,      # Shape (N,), float32, CUDA
    frequency: Tensor,      # Shape (N,), float32, CUDA
    nbr_idx: Tensor,        # Shape (N, k), int64, CUDA
    K: float,               # Coupling strength
    decay: float,           # Decay rate λ
    gamma: float,           # Frequency adaptation γ
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (dphi, dr, domega) each shape (N,)."""
```

**Details:**
- Single kernel launch: gathers k neighbour phases/amplitudes, computes sin/cos differences, reduces.
- K_eff = K/k applied per edge (constant total coupling per oscillator).
- The `nbr_idx` tensor can represent **any** coupling topology (see Section 5).

**Example:**
```python
N, k = 16384, 14
phase = torch.rand(N, device="cuda") * 6.2832
amp = torch.ones(N, device="cuda")
freq = torch.randn(N, device="cuda")
nbr = torch.randint(0, N, (N, k), device="cuda")

dphi, dr, domega = triton_sparse_knn_coupling(
    phase, amp, freq, nbr, K=2.0, decay=0.1, gamma=0.01
)
```

### 3.3 PyTorch Fallback Functions

Identical API to the Triton versions, but using pure PyTorch operations. Use these when Triton is unavailable, for CPU execution, or as correctness references.

```python
pytorch_mean_field_rk4_step(phase, amp, freq, K, decay, gamma, dt)
pytorch_sparse_knn_coupling(phase, amp, freq, nbr_idx, K, decay, gamma)
```

### 3.4 triton\_available()

```python
def triton_available() -> bool:
    """True if triton is importable AND CUDA is available."""
```

---

## 4. Architecture & Performance

### 4.1 Kernel Launch Structure

**Mean-Field RK4 (per step):**

```
For each RK4 stage (4 total):
  ├─ _reduce_order_param_kernel  → block-local Z reduction + atomic_add
  └─ _mf_derivatives_kernel      → element-wise dphi, dr, domega
_rk4_weighted_sum_kernel          → final x + (dt/6)(k1 + 2k2 + 2k3 + k4)
                                    + phase wrap + amplitude clamp
Total: 9 kernel launches
```

The reduction kernel accumulates `amp·cos(phase)` and `amp·sin(phase)` into a global Z buffer using `tl.atomic_add`. The derivative kernel reads Z and computes per-oscillator derivatives using the algebraic identity (avoids atan2).

**Sparse k-NN (per call):**

```
_sparse_knn_coupling_kernel  → gather nbr phases/amps + sin/cos loop + reduce
Total: 1 kernel launch
```

### 4.2 Block Size Tuning

Tuned for NVIDIA RTX 4060 (24 SMs, Ada Lovelace):

| Kernel | Block Size | Rationale |
|--------|-----------|-----------|
| Reduction | 256 | Higher occupancy with atomics |
| Derivatives | 512 | Memory-bound, fits L2 cache |
| RK4 Sum | 1024 | Streaming read, high throughput |
| SpMV | 512 | Balanced gather + compute |

### 4.3 Benchmark Results

Measured on RTX 4060 with CUDA events (median µs):

**Mean-Field RK4:**

| N | PyTorch µs | Triton µs | Speedup |
|---|-----------|----------|---------|
| 256 | 1756 | 595 | 2.95× |
| 1,024 | 1758 | 594 | 2.96× |
| 16,384 | 1772 | 598 | 2.96× |
| 65,536 | 1730 | 596 | 2.90× |

**Sparse k-NN (k=14):**

| N | PyTorch µs | Triton µs | Speedup |
|---|-----------|----------|---------|
| 256 | 228 | 42 | 5.44× |
| 1,024 | 311 | 81 | 3.85× |
| 16,384 | 329 | 76 | 4.34× |
| 65,536 | 352 | 184 | 1.91× |

### 4.4 Scientific Regime Comparison (Q4.9)

Solo device × regime throughput (osc·step/s, RTX 4060 + Ryzen 8c/16t):

| Regime | GPU | CPU | N (GPU/CPU) |
|--------|-----|-----|-------------|
| mean_field | 31.8 M | 11.3 M | 65K / 16K |
| sparse_knn | 2.0 M | 886 K | 8K / 4K |
| full | 354 K | 374 K | 512 / 512 |
| csr | 12.9 M | 18.4 M | 4K / 4K |

Goldilocks coupling zones (K value at which dr/dK is maximised):

| Regime | K_critical | K_goldilocks | Zone Width |
|--------|-----------|-------------|------------|
| mean_field | 4.0 | 5.0 | 2.0 |
| sparse_knn | 5.0 | 5.0 | 1.0 |
| full | 1.5 | 4.0 | 3.5 |
| csr | 5.0 | 8.0 | 15.0 |

Finite-size scaling $K_c(N) = K_c(\infty) + a/\sqrt{N}$:

| Regime | $K_c(\infty)$ | $a$ | $R^2$ |
|--------|---------------|-----|-------|
| mean_field | 6.37 | −47.5 | 0.657 |
| sparse_knn | 7.95 | −48.8 | 0.732 |
| full | 4.19 | −4.9 | 0.341 |
| csr | 12.60 | −68.1 | 0.465 |

Full details: `benchmarks/results/y3q49_*.json` (8 files, 104 experimental configurations).

---

## 5. Custom Coupling Topologies

The `triton_sparse_knn_coupling()` kernel accepts **any** neighbour index tensor of shape `(N, k)`. This makes it a general-purpose engine for custom topologies — you only need to build the `nbr_idx` tensor.

### 5.1 Using Sparse k-NN with Custom Neighbours

```python
import torch
from prinet import triton_sparse_knn_coupling

N, k = 8192, 10
device = "cuda"

# Build YOUR custom neighbour index (N, k) — any topology
nbr_idx = build_my_custom_topology(N, k, device)  # int64, values in [0, N)

# Use the Triton kernel directly
phase = torch.rand(N, device=device) * 6.2832
amp = torch.ones(N, device=device)
freq = torch.randn(N, device=device)

dphi, dr, domega = triton_sparse_knn_coupling(
    phase, amp, freq, nbr_idx, K=2.0, decay=0.1, gamma=0.01
)
```

### 5.2 Small-World Topology

Watts-Strogatz small-world: ring lattice with random rewiring.

```python
def small_world_topology(
    N: int, k: int, p_rewire: float = 0.1, device: str = "cuda",
) -> torch.Tensor:
    """Build a Watts-Strogatz small-world neighbour index.

    Args:
        N: Number of nodes.
        k: Neighbours per node (must be even).
        p_rewire: Probability of rewiring each edge.
        device: Target device.

    Returns:
        Tensor of shape (N, k) with neighbour indices.
    """
    assert k % 2 == 0, "k must be even for ring lattice"
    nbr = torch.zeros(N, k, dtype=torch.int64, device=device)

    # Start with ring lattice
    for i in range(N):
        neighbours = [(i + j) % N for j in range(-k // 2, k // 2 + 1) if j != 0]
        nbr[i] = torch.tensor(neighbours[:k], dtype=torch.int64, device=device)

    # Rewire with probability p
    mask = torch.rand(N, k, device=device) < p_rewire
    random_targets = torch.randint(0, N, (N, k), device=device)
    nbr = torch.where(mask, random_targets, nbr)

    return nbr
```

### 5.3 Ring Topology

Connect each oscillator to its k nearest spatial neighbours (1D ring):

```python
def ring_topology(N: int, k: int, device: str = "cuda") -> torch.Tensor:
    """Ring lattice: each node connects to k nearest ring neighbours."""
    half = k // 2
    idx = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
    offsets = torch.cat([
        torch.arange(-half, 0, device=device),
        torch.arange(1, half + 1, device=device),
    ])  # (k,)
    nbr = (idx + offsets) % N  # (N, k)
    return nbr.to(torch.int64)
```

### 5.4 Hierarchical Topology

Multi-scale coupling: each oscillator connects to neighbours at multiple distance scales.

```python
def hierarchical_topology(
    N: int, levels: int = 4, k_per_level: int = 4, device: str = "cuda",
) -> torch.Tensor:
    """Multi-scale hierarchical coupling.

    At level l, connects to k_per_level neighbours at distance ~2^l.
    Total k = levels * k_per_level.
    """
    k = levels * k_per_level
    nbr = torch.zeros(N, k, dtype=torch.int64, device=device)
    idx = torch.arange(N, device=device)

    col = 0
    for level in range(levels):
        stride = 2 ** level
        for j in range(k_per_level):
            offset = stride * (j - k_per_level // 2)
            if offset == 0:
                offset = stride
            nbr[:, col] = (idx + offset) % N
            col += 1

    return nbr
```

### 5.5 Feeding Custom Topologies to Triton Kernels

Any method that produces a `(N, k)` int64 tensor on CUDA works:

```python
# From NetworkX graph
import networkx as nx

G = nx.watts_strogatz_graph(N, k, 0.1)
adj = torch.zeros(N, k, dtype=torch.int64, device="cuda")
for i, nbrs in enumerate(G.adjacency()):
    node, adj_dict = nbrs
    adj[node, :len(adj_dict)] = torch.tensor(
        list(adj_dict.keys())[:k], dtype=torch.int64
    )

# From scipy sparse matrix
from scipy.sparse import csr_matrix

A = my_sparse_adjacency()  # CSR
for i in range(N):
    cols = A.indices[A.indptr[i]:A.indptr[i+1]]
    adj[i, :len(cols)] = torch.tensor(cols[:k], dtype=torch.int64)
adj = adj.to("cuda")

# Then use with Triton kernel
dphi, dr, dom = triton_sparse_knn_coupling(phase, amp, freq, adj, K, decay, gamma)
```

---

## 6. Migration Guide

**From `KuramotoOscillator` to standalone Triton kernels:**

If you currently use `KuramotoOscillator.step()` and want direct Triton kernel access for custom pipelines:

```python
# Before (OscillatorModel API):
model = KuramotoOscillator(N, K, coupling_mode="mean_field", device="cuda")
state = OscillatorState.create_random(N, device="cuda")
new_state, _ = model.step(state, dt=0.01)

# After (direct Triton kernel):
from prinet import triton_fused_mean_field_rk4_step

new_phase, new_amp, new_freq = triton_fused_mean_field_rk4_step(
    state.phase, state.amplitude, state.frequency,
    K=K, decay=0.1, gamma=0.01, dt=0.01,
)
```

The `KuramotoOscillator` class remains the recommended high-level API. The Triton kernels are exposed for:
- Custom simulation loops that bypass the OscillatorModel overhead.
- Research requiring direct access to derivative computations.
- Hybrid pipelines mixing different coupling topologies per step.

---

## 7. Year 4 Planned Integration: OscilloSim Ring & Small-World Modes

> **Status:** Planned for Year 4 Q1 (Months 37–39)  
> **Reference:** `Docs/Planning_Documentation/Year-4-Plan.md` tasks T.1–T.2

### 7.1 Motivation

Year 3 Q4.9 confirmed that **no chimera states emerge with random-neighbour
topology** (local order parameter r > 0.7 everywhere). The literature is
clear: chimera emergence requires spatially structured coupling, specifically
ring or lattice topology with non-local coupling (Abrams & Strogatz 2004).

The `ring_topology()` and `small_world_topology()` helper functions already
exist (Sections 5.3 and 5.2 above) — they just need to be wired into
`OscilloSim` as first-class `coupling_mode` options.

### 7.2 Planned OscilloSim API Extension

```python
from prinet.utils.oscillosim import OscilloSim

# Ring topology (Year 4 Q1 — T.1)
sim = OscilloSim(
    n_oscillators=1024,
    coupling_strength=5.0,
    coupling_mode="ring",      # NEW: 1D ring lattice
    ring_k=10,                 # neighbours per node in ring
    device="cuda",
)

# Small-world topology (Year 4 Q1 — T.1)
sim = OscilloSim(
    n_oscillators=1024,
    coupling_strength=5.0,
    coupling_mode="small_world",  # NEW: Watts-Strogatz
    ring_k=10,
    p_rewire=0.1,                 # rewiring probability
    device="cuda",
)
```

### 7.3 Planned Coupling Mode Table (v2.1.0)

| Mode | Complexity | Description | Best For |
|------|-----------|-------------|----------|
| `"mean_field"` | O(N) | Global order parameter coupling | N > 10K |
| `"sparse_knn"` | O(N·k) | k nearest phase-neighbours | 1K ≤ N < 100K |
| `"full"` | O(N²) | All-to-all pairwise | N < 500 |
| `"csr"` | O(nnz) | Arbitrary sparse matrix | Custom topologies |
| `"ring"` | O(N·k) | 1D ring lattice neighbours | **Chimera states** |
| `"small_world"` | O(N·k) | Watts-Strogatz ring + rewiring | **Small-world dynamics** |

### 7.4 Chimera Detection Utilities (Planned)

```python
from prinet.utils import local_order_parameter, bimodality_index

# Compute local order parameter for each oscillator
local_r = local_order_parameter(phases, nbr_idx)  # shape (N,)

# Quantify bimodality of local r distribution
bm = bimodality_index(local_r)
# bm > 0.2 → chimera state detected (coexistence of coherent + incoherent)
```

### 7.5 Gate Criteria

| Gate | Metric | Threshold |
|------|--------|-----------|
| Y4.1 | Chimera bimodality index (ring) | > 0.2 at N ≥ 256 |
| Y4.2 | Distinct chimera regimes | ≥ 2 K ranges with chimera |
