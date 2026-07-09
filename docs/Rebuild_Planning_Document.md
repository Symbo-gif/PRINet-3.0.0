# PRINet 4.0 — From-Scratch Rebuild Planning Document

**Status:** Planning — ready to seed a new project
**Source of truth:** PRINet 3.0.0 (`Symbo-gif/PRINet-3.0.0`, ~25.5k lines of Python across 43 modules, 37 test files / ~1,670 tests, 62 benchmark scripts, Sphinx docs, NeurIPS paper artefacts)
**Goal:** Define a complete, actionable plan to rebuild PRINet from scratch in the optimal implementation language, preserving all scientific functionality, numerical results, and reproducibility guarantees while improving performance, safety, and maintainability.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What PRINet Is — Functional Inventory](#2-what-prinet-is--functional-inventory)
3. [Requirements the Rebuild Must Satisfy](#3-requirements-the-rebuild-must-satisfy)
4. [Language Evaluation and Recommendation](#4-language-evaluation-and-recommendation)
5. [Target Architecture](#5-target-architecture)
6. [Module-by-Module Rebuild Mapping](#6-module-by-module-rebuild-mapping)
7. [Numerical Parity and Correctness Strategy](#7-numerical-parity-and-correctness-strategy)
8. [Performance Strategy](#8-performance-strategy)
9. [Testing Strategy](#9-testing-strategy)
10. [Tooling, CI/CD, and Packaging](#10-tooling-cicd-and-packaging)
11. [Documentation Plan](#11-documentation-plan)
12. [Phased Implementation Roadmap](#12-phased-implementation-roadmap)
13. [Risk Register and Mitigations](#13-risk-register-and-mitigations)
14. [Repository Layout for the New Project](#14-repository-layout-for-the-new-project)
15. [Definition of Done](#15-definition-of-done)

---

## 1. Executive Summary

PRINet (Phase-Resonance Interference Network) is a scientific ML framework built on
coupled-oscillator dynamics (Kuramoto, Stuart–Landau, Hopf), hierarchical δ/θ/γ band
networks with phase–amplitude coupling (PAC), polyadic tensor decomposition, and
PyTorch-compatible trainable layers used for temporal object binding and multi-object
tracking (MOT). It ships GPU acceleration (Triton + CUDA JIT kernels), a large benchmark
suite, a subconscious ONNX/NPU controller, and a fully reproducible NeurIPS paper
pipeline.

**Recommendation:** Rebuild as a **two-layer system**:

- **Core in Rust** — all oscillator dynamics, integrators, coupling topologies, sparse
  ops, phase metrics, tensor decomposition, and GPU kernels (via `wgpu`/CUDA through the
  Burn tensor framework or `cudarc` for hand-written kernels). Rust delivers memory
  safety, deterministic performance, first-class SIMD/parallelism, zero-cost
  abstractions, and painless cross-platform distribution (including the Windows + MSVC
  targets PRINet 3.0 must JIT-compile for today).
- **Thin Python API layer** (PyO3/maturin wheel) — preserves the existing user-facing
  ergonomics, PyTorch interop (via DLPack zero-copy tensor exchange), notebooks,
  matplotlib figure generation, and the reproducibility pipeline. The scientific
  community consuming PRINet lives in Python; the rebuild must not abandon them.

This hybrid is the "optimal language" answer for this codebase: pure Python is the
current bottleneck (kernel-launch overhead, JIT fragility, GIL); pure Rust would orphan
the research audience; Julia was seriously evaluated (§4) but loses on deployment,
ONNX/NPU story, and team ramp-up. The plan below is complete enough to open a new
repository and begin Phase 0 immediately.

---

## 2. What PRINet Is — Functional Inventory

A rebuild must reproduce every capability below. Line counts refer to PRINet 3.0.0.

### 2.1 `prinet.core` — fundamental dynamics (~5,000 lines)

| Capability | Details |
|---|---|
| Oscillator models | Kuramoto (mean-field O(N), full pairwise O(N²), sparse k-NN O(N·k)), Stuart–Landau (complex amplitude dynamics), Hopf (bifurcation-driven limit cycles) |
| Oscillator state | Phase wrapping to [0, 2π), atan2-safe phase differences, NaN/Inf guards, derivative clamping (±1e4), amplitude clamping [1e-6, 10] |
| Integrators | Euler, RK4, adaptive RK45, exponential integrator (direct matrix-exp O(D³) and Krylov O(D·m²) with adaptive subspace dimension), multi-rate sub-stepped RK4 for θ→γ frequency separation |
| Coupling | Mean-field, ring, small-world, sparse k-NN on the phase circle (O(N log N) sort-based index), conduction delays, directed/weighted topologies |
| PAC | `A_fast = A₀·[1 + m·cos(φ_slow + offset)]` slow→fast modulation |
| Hierarchical networks | ThetaGamma (2-band, ~7-item binding capacity), DeltaThetaGamma (3-band continuous ODE), DiscreteDeltaThetaGamma (trainable discrete-time `nn.Module` with learnable coupling matrices, PAC depths, frequency offsets) |
| Temporal propagation | Complex-phasor phase blending + EMA amplitude blending across frames |
| Inhibition / phase-to-rate | Feedforward inhibition (fast gating), feedback inhibition (winner-take-all top-k with straight-through estimator, ~10% sparsity), DentateGyrusConverter pipeline |
| Synchronization metrics | Kuramoto order parameter, mean phase coherence, phase coherence matrix, PSD, sparse k-NN metric variants, chimera metrics (local order parameter, bimodality index, chimera index, strength of incoherence, discontinuity measure) |
| Tensor decomposition | Tucker/HOSVD (`PolyadicTensor`) and CP/PARAFAC (`CPDecomposition`) |
| Subconscious controller | State/control dataclasses (STATE_DIM=32, CONTROL_DIM=8), background daemon thread with control-signal buffering, ONNX inference via NPU/DirectML/CPU backends |

### 2.2 `prinet.nn` — trainable layers and models (~7,400 lines)

| Capability | Details |
|---|---|
| Layers | `ResonanceLayer` (Kuramoto as differentiable layer), `PRINetModel`, `HierarchicalResonanceLayer`, `OscillatoryAttention`, `PhaseToRateConverter` (soft/hard/annealed) |
| Models | `HybridPRINet` / `HybridPRINetV2` (oscillator front-end → phase-to-rate → transformer back-end → classifier), `PhaseTracker` (MOT: detection encoder → DiscreteDeltaThetaGamma dynamics → Hungarian phase-similarity matching; 4,991 params vs SlotAttention's 83,904) |
| Baselines | `SlotAttentionModule`, `TemporalSlotAttentionMOT` for head-to-head comparison |
| Optimizers | SynchronizedGradientDescent (barrier penalty on order parameter), SCALR (Lyapunov-stability-adaptive coupling), RIP (black-box resonance perturbation), AlternatingOptimizer |
| Activations | dSiLU, HolomorphicActivation (complex tanh), PhaseActivation, GatedPhaseActivation |
| HEP | Holomorphic Equilibrium Propagation trainer (free + ±β nudge phases, gradient ≈ (1/2β)(E⁺ − E⁻), no BPTT, 3× forward cost) |
| Training infrastructure | Training hooks/telemetry (loss EMA, gradient norms, p50/p95 latency), daemon integration, adaptive oscillator allocation (rule-based + learned MLP), ablation variants (frozen/static/no-GRU) |
| MOT evaluation | MOTA/MOTP/IDF1, identity switches, identity preservation, synthetic sequence generators |

### 2.3 `prinet.utils` — acceleration and tooling (~11,900 lines)

| Capability | Details |
|---|---|
| OscilloSim | 1M+-oscillator simulator with sparse CSR coupling and chimera detection |
| GPU kernels | Triton fused mean-field RK4 (9 kernel launches vs 28+ PyTorch ops, 2× at N=1M), sparse k-NN coupling (3× at N=16K), PAC modulation, hierarchical order parameters; CUDA C++ JIT fused discrete step (phase advance + PAC gating + Stuart–Landau in one launch, Windows/MSVC compatible); all with pure-framework fallbacks |
| Solvers | Batched RK45/RK4 with torch.compile, pre-allocated stage buffers, GPU-resident scalars, gradient checkpointing |
| Sparse ops | CSR coupling matrices, SpMV coupling steps, k-NN index construction, mixed-precision training, async CPU/GPU pipeline, oscillator pruning |
| NPU backend | VitisAI (Ryzen AI) / DirectML / CPU ONNX Runtime session abstraction |
| Datasets | CIFAR-10, Fashion-MNIST loaders (cached), temporal CLEVR-N generator |
| Reporting | Benchmark JSON report/leaderboard generation, 15 NeurIPS figures (300 DPI matplotlib), 11 LaTeX table fragments, torch.profiler wrapper |
| Experiment tooling | Fair PT-vs-SA training framework (identical loss/optimizer/augmentation/parameter budgets), Hungarian similarity loss, multi-seed training with bootstrap CIs and Welch t-tests, FGSM/PGD adversarial evaluation, FLOPs counting |

### 2.4 Ancillary assets

- `reproduce.py`: regenerates all 15 figures + 11 tables from ~172 stored JSON artefacts in ~8 s, no GPU/training needed; SHA-256 manifest.
- `models/subconscious_controller.onnx` (104 KB pre-trained controller).
- 62 benchmark scripts (scaling, chimera phase diagrams, MOT, ablations, kernel perf, quarterly Y2–Y4 deliverables); results written to `docs/test_and_benchmark_results/` (lowercase, gitignored).
- 3 tutorial notebooks; NeurIPS paper sources; Sphinx + ReadTheDocs docs; CI with CPU matrix (Python 3.11–3.13), Windows job, mypy strict, opt-in self-hosted GPU job, reproducibility check, wheel smoke test, OIDC release.

---

## 3. Requirements the Rebuild Must Satisfy

### 3.1 Functional requirements

- **F1** — Feature parity with §2: every public class/function in the 175+-symbol API has an equivalent.
- **F2** — Numerical parity: simulation trajectories, metrics, and benchmark results match 3.0.0 within documented tolerances (§7).
- **F3** — Differentiability: all trainable components support reverse-mode autodiff and interop with PyTorch training loops (users keep their optimizers, schedulers, data pipelines).
- **F4** — The reproducibility pipeline (`reproduce.py` equivalent) regenerates all paper figures/tables byte-comparably from the same JSON artefacts.
- **F5** — ONNX subconscious controller keeps running on CPU, DirectML, and Ryzen AI NPU.

### 3.2 Non-functional requirements

- **N1** — Performance: ≥ current Triton/CUDA fused-kernel throughput at N=1M oscillators; ≥2× current pure-PyTorch fallback paths on CPU (Rust SIMD + rayon).
- **N2** — Platforms: Linux, Windows (no MSVC JIT-at-runtime requirement — kernels precompiled), macOS (CPU + Metal via wgpu); CUDA GPUs; graceful CPU fallback everywhere.
- **N3** — Safety: no `unsafe` outside audited kernel-FFI modules; no data races (enforced by the compiler); NaN/Inf guards preserved as runtime checks toggleable per build profile.
- **N4** — Distribution: `pip install prinet` yields prebuilt wheels (manylinux, Windows, macOS arm64/x86_64) with no compiler needed on the user machine.
- **N5** — Maintainability: strict typing throughout (Rust type system + fully typed Python stubs), one canonical implementation per algorithm (today mean-field RK4 exists in ≥3 places: Triton, CUDA JIT, PyTorch fallback).
- **N6** — License: MIT, matching 3.0.0.

---

## 4. Language Evaluation and Recommendation

### 4.1 Candidates considered

| Criterion (weight) | Python 3.13 (status quo, rewritten) | **Rust + PyO3 (recommended)** | Julia | C++20 + pybind11 | Mojo |
|---|---|---|---|---|---|
| Numerical performance, CPU (25%) | Poor without C extensions; GIL limits parallel sweeps | Excellent: SIMD, rayon, no GC pauses | Excellent (JIT), but startup/TTFX cost | Excellent | Excellent (immature) |
| GPU story (20%) | Triton is good but Python-locked; CUDA JIT is fragile on Windows | Good: Burn (wgpu/CUDA/Metal backends), `cudarc` for raw CUDA, precompiled — no runtime nvcc/MSVC | Very good (CUDA.jl, KernelAbstractions.jl) | Good but high build complexity | Promising, unproven |
| Autodiff / ML training (15%) | Best-in-class (PyTorch) | Good: Burn autodiff for core; PyTorch interop via DLPack keeps training in torch | Good (Zygote/Enzyme), ecosystem thinner for MOT/vision | Poor without LibTorch | Immature |
| Ecosystem fit for users — notebooks, matplotlib, ONNX Runtime, motmetrics (15%) | Best | Best (kept via Python layer) | Weak ONNX/NPU/DirectML support; smaller community | Kept via bindings, but slower iteration | Very weak |
| Memory/type safety (10%) | Weak (runtime errors) | Best (compile-time) | Moderate | Weak (UB risk) | Moderate |
| Cross-platform packaging incl. Windows (10%) | Painful for the CUDA JIT kernel today | Excellent (maturin wheels, no user-side toolchain) | Painful (sysimages, large artifacts) | Painful | N/A on most platforms |
| Team ramp-up / hiring (5%) | Easy | Moderate | Moderate-hard | Hard | Hard |

### 4.2 Decision

**Rust core + thin Python package (PyO3/maturin), PyTorch interop via DLPack.**

Rationale specific to this codebase:

1. The hot loops are *simple math over large flat arrays* (phase advance, trig, sparse
   gathers, reductions). This is exactly where Rust + SIMD + rayon shines and where
   Python overhead dominates today (the entire Triton effort exists to work around it).
2. The single worst maintenance burden in 3.0.0 — the Windows MSVC + nvcc runtime JIT
   for `fused_kernels.py` — disappears: kernels are compiled once at wheel-build time.
3. Triple-implementation drift (Triton / CUDA / PyTorch fallback of the same step) is
   replaced by one Rust kernel written against Burn's backend-generic tensor API (or
   KernelAbstractions-style CubeCL kernels) that compiles to CUDA, Metal, Vulkan, and CPU.
4. Users lose nothing: the Python layer re-exports the same API, tensors cross the
   boundary zero-copy via DLPack, and training loops remain plain PyTorch. Trainable
   layers are exposed as `torch.autograd.Function`s whose forward/backward call into
   Rust.
5. Julia was the closest runner-up (superb for the ODE/dynamics science), but fails
   N4 (packaging/deployment), F5 (ONNX Runtime + DirectML + VitisAI NPU support), and
   would fork the user base away from PyTorch.

### 4.3 Key technology choices

| Concern | Choice | Fallback/notes |
|---|---|---|
| Tensor/autodiff core | Burn (backend-generic: `burn-ndarray` CPU, `burn-cuda`, `burn-wgpu`) | `ndarray` + hand-rolled adjoints for the small set of custom ops if Burn autodiff is insufficient |
| Hand-tuned GPU kernels | CubeCL (Burn's kernel DSL — write once, run CUDA/Metal/Vulkan) | `cudarc` for raw CUDA where CubeCL underperforms |
| CPU parallelism | `rayon` + `std::simd`/`wide` | — |
| Sparse linear algebra | `sprs` (CSR) + custom SpMV kernels | — |
| Python bindings | PyO3 + maturin; DLPack for zero-copy torch interop | — |
| ONNX inference | `ort` crate (ONNX Runtime bindings — supports DirectML, VitisAI EPs) | Python `onnxruntime` in the Python layer if EP coverage lags |
| Hungarian matching | `pathfinding`/`lapjv` crate | scipy in Python layer for parity testing |
| Serialization | `serde_json` (benchmark artefacts keep the exact 3.0.0 JSON schema) | — |
| Figures/tables | Stay in Python (matplotlib + existing generators, ported nearly verbatim) | — |
| Property testing | `proptest` (Rust) + `hypothesis` (Python parity suite) | — |

---

## 5. Target Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Python package: prinet (pure Python + compiled extension)          │
│  • Public API (drop-in symbols matching 3.0.0)                     │
│  • torch.nn.Module wrappers + torch.autograd.Function bridges      │
│  • matplotlib figures, LaTeX tables, notebooks, reproduce pipeline │
│  • datasets (torchvision), motmetrics, benchmark drivers           │
└───────────────▲────────────────────────────────────────────────────┘
                │ PyO3 FFI + DLPack (zero-copy tensors)
┌───────────────┴────────────────────────────────────────────────────┐
│ Rust workspace: prinet-core                                        │
│                                                                    │
│  prinet-dynamics   oscillator models, state, integrators, PAC,     │
│                    coupling topologies, hierarchical networks,     │
│                    temporal propagation, inhibition                │
│  prinet-metrics    order parameters, coherence, chimera metrics,   │
│                    PSD, sparse variants                            │
│  prinet-tensor     Tucker/HOSVD, CP decomposition                  │
│  prinet-kernels    CubeCL fused kernels (mean-field RK4, k-NN      │
│                    coupling, PAC, discrete step) + CPU SIMD paths  │
│  prinet-sim        OscilloSim engine (CSR sparse, 1M+ oscillators, │
│                    pruning, async pipelines)                       │
│  prinet-train      Burn modules mirroring nn/ (Resonance layers,   │
│                    DiscreteDeltaThetaGamma, optimizers, HEP) —     │
│                    used natively AND exposed to torch via bridges  │
│  prinet-daemon     subconscious controller: state buffer, control  │
│                    signals, `ort` ONNX inference, backend detect   │
│  prinet-py         PyO3 binding crate (the only crate that links   │
│                    Python)                                         │
└────────────────────────────────────────────────────────────────────┘
```

Design rules:

- **One algorithm, one implementation.** Backend dispatch (CPU/CUDA/wgpu) happens inside
  `prinet-kernels`, never by duplicating math at call sites.
- **The Python layer contains no numerics.** It holds API ergonomics, torch glue,
  plotting, and orchestration only.
- **State is explicit.** `OscillatorState { phase, amplitude, frequency, coupling }` is a
  plain struct-of-arrays; no hidden globals; deterministic seeding threaded through every
  stochastic entry point.
- **Feature flags:** `cuda`, `wgpu`, `npu` (ort execution providers), `strict-checks`
  (NaN/Inf guards on), mirrored as wheel variants where needed.

---

## 6. Module-by-Module Rebuild Mapping

| PRINet 3.0.0 module | New home | Notes / changes |
|---|---|---|
| `core/propagation/oscillator_state.py` | `prinet-dynamics::state` | Struct-of-arrays; phase wrap, safe diff, clamps as inlined helpers; k-NN phase index via sort (rayon parallel sort) |
| `core/propagation/oscillator_models.py` | `prinet-dynamics::models` | Kuramoto/StuartLandau/Hopf as a `Dynamics` trait; coupling mode is an enum, not string |
| `core/propagation/integrators.py` | `prinet-dynamics::integrate` | Euler/RK4/RK45/exponential(direct+Krylov)/multi-rate as an `Integrator` trait; matrix-exp via `faer` or Burn linalg |
| `core/propagation/coupling.py` | `prinet-dynamics::pac` | Direct port of PAC formula |
| `core/propagation/networks.py` | `prinet-dynamics::bands` (continuous) + `prinet-train::bands` (trainable discrete) | Split simulation-only vs trainable variants explicitly |
| `core/propagation/temporal.py` | `prinet-dynamics::temporal` | Complex-phasor blending |
| `core/propagation/inhibition.py` | `prinet-train::inhibition` | STE top-k needs autodiff → lives in Burn-land with a custom backward |
| `core/propagation/sweep_utils.py` | `prinet-sim::sweep` | Grid sweeps parallelized over rayon (embarrassingly parallel — big win vs GIL) |
| `core/measurement.py` | `prinet-metrics` | Full + sparse variants |
| `core/decomposition.py` | `prinet-tensor` | HOSVD via SVD (`faer`); CP via ALS |
| `core/subconscious*.py` | `prinet-daemon` | Real OS thread (no GIL contention), lock-free control-signal ring buffer, `ort` inference |
| `nn/layers.py`, `nn/hybrid.py`, `nn/slot_attention.py`, `nn/subconscious_model.py`, `nn/adaptive_allocation.py`, `nn/ablation_variants.py` | `prinet-train` (Burn) + Python `prinet.nn` torch wrappers | Torch wrappers implement `autograd.Function` calling Rust forward/backward; SlotAttention baseline may remain pure-torch (it exists only for comparison) |
| `nn/optimizers.py` | `prinet-train::optim` + torch-side counterparts | SCALR/RIP/SyncGD need order-parameter feedback → Rust computes metrics, Python optimizer classes stay thin |
| `nn/activations.py` | `prinet-train::activations` + torch functional equivalents | Trivial |
| `nn/hep.py` | `prinet-train::hep` | Energy + ±β nudge trainer; no BPTT, maps cleanly |
| `nn/mot_evaluation.py`, `nn/training_hooks.py` | Python layer (`prinet.eval`, `prinet.hooks`) | Orchestration-heavy, perf-uncritical; hot inner metric loops call Rust |
| `utils/oscillosim.py` | `prinet-sim` | CSR coupling, chimera metrics from `prinet-metrics` |
| `utils/cuda_kernels.py`, `utils/triton_kernels.py`, `utils/fused_kernels.py` | `prinet-kernels` | **Collapse three implementations into one CubeCL kernel set**: fused mean-field RK4, sparse k-NN coupling, PAC modulation, fused discrete step, hierarchical order-parameter reduction; CPU SIMD path is the fallback |
| `utils/npu_backend.py` | `prinet-daemon::backend` | `detect_best_backend()` over ort EPs (VitisAI → DirectML → CPU) |
| `utils/datasets.py` | Python layer | Keep torchvision loaders as-is |
| `utils/benchmark_reporting.py`, `figure_generation.py`, `table_generation.py`, `profiler.py` | Python layer | Near-verbatim port; JSON schema unchanged; profiler wraps both torch.profiler and Rust `tracing` spans |
| `utils/temporal_training.py`, `temporal_metrics.py`, `adversarial_tools.py`, `y4q1_tools.py` | Python layer (`prinet.experiments`) | Rename the y4q1 grab-bag into coherent `experiments.ablation`, `experiments.stats`, `experiments.adversarial` modules; hot loops (Hungarian similarity loss, metric computation) call Rust |
| `_deprecation.py` | Python layer | Keep the API-freeze machinery |
| `reproduce.py` | `tools/reproduce.py` | Unchanged behavior; artefact paths keep `docs/test_and_benchmark_results/` (lowercase) convention |
| `benchmarks/*` (62 scripts) | `benchmarks/` reorganized into 9 category packages (§2.3 groupings) with a single `benchrunner` CLI | Consolidate the quarterly y2q/y3q/y4q naming into named-by-topic modules; keep JSON output schema identical so old artefacts stay valid |

---

## 7. Numerical Parity and Correctness Strategy

This is the highest-risk area of any rewrite of a scientific codebase. Plan:

1. **Golden-trajectory corpus (built from 3.0.0 before any new code).**
   For every dynamics primitive (each oscillator model × coupling mode × integrator),
   record seeded inputs and outputs at float64: initial state, parameters, per-step
   trajectories (first 100 steps), final metrics. Store as versioned JSON/NPZ in a
   `parity/` directory of the new repo. Target ~500 golden cases.
2. **Tolerance policy.** Trajectories: `rtol=1e-6, atol=1e-8` (float64 reference);
   chaotic regimes compared via statistics (order-parameter time series, Lyapunov-safe
   windows) not pointwise values beyond the shadowing horizon. Metrics/decompositions:
   `rtol=1e-10` float64.
3. **Differential testing in CI.** A `parity` test job installs both `prinet==3.0.0`
   (PyPI) and the new build in one venv and asserts equivalence on the corpus plus
   hypothesis-generated random cases.
4. **Benchmark-result parity.** Re-run the 62-benchmark suite; scientific conclusions
   (chimera phase-diagram boundaries, IP scores, ablation orderings, statistical test
   outcomes) must be unchanged; raw numbers documented in a migration report.
5. **Bit-level reproducibility.** All RNG behind a single `Seed` type (Philox/PCG64
   counter-based) so multi-seed experiments (bootstrap CIs, Welch tests) are exactly
   reproducible across CPU/GPU and across runs.
6. **Known numerical hazards to preserve deliberately:** phase-wrap semantics `% 2π`
   (not `atan2` renormalization) in state updates; amplitude clamp bounds [1e-6, 10];
   derivative clamp ±1e4; coupling normalization 1/N vs 1/k per mode; φ₁(λ) → 1 limit
   handling in the exponential integrator; STE forward-hard/backward-soft gradient
   identity in feedback inhibition.

---

## 8. Performance Strategy

Targets (measured against 3.0.0 on the same hardware):

| Workload | 3.0.0 baseline | Rebuild target |
|---|---|---|
| Mean-field RK4, N=1M, GPU | Triton fused (2× over torch) | ≥ parity with Triton; single CubeCL kernel set |
| Sparse k-NN coupling, N=16K, k=14, GPU | Triton 3× over torch | ≥ parity |
| Fused discrete step (3-band + PAC), GPU | CUDA JIT 1.5× over torch | ≥ parity, without runtime JIT |
| CPU fallback paths (all of the above) | pure PyTorch | ≥ 2× (SIMD + rayon) |
| Parameter sweeps (`sweep_coupling_params`) | serial Python loop | ≥ 8× on 16-core CPU (rayon grid parallelism) |
| Subconscious daemon latency | Python thread + GIL | lower p95 (native thread, lock-free buffer) |
| PhaseTracker training step (torch bridge) | pure torch | parity ±10% (bridge overhead must stay <5%; verify with DLPack round-trip microbench in Phase 1) |

Tactics: pre-allocated stage buffers (as in 3.0.0's Q2 work), device-resident scalars,
kernel fusion (phase advance + PAC + amplitude update in one launch), CSR SpMV for
sparse topologies, mixed precision (f32 compute / f64 accumulate for order-parameter
reductions), gradient checkpointing for long integrations, `criterion` +
`pytest-benchmark` regression gates in CI (fail on >10% regression).

---

## 9. Testing Strategy

| Layer | Framework | Content |
|---|---|---|
| Rust unit | `cargo test` | Per-module math correctness, edge cases (N=1, zero coupling, extreme K, empty k-NN), clamp/guard behavior |
| Rust property | `proptest` | Phase always in [0,2π); order parameter ∈ [0,1]; energy monotonicity where guaranteed; integrator order-of-convergence checks (RK4 error ∝ h⁴) |
| Kernel equivalence | `cargo test --features cuda,wgpu` | Every GPU kernel vs CPU reference within tolerance, all shapes/dtypes |
| Python API | `pytest` (port the 37 existing test files) | Keep the existing 1,670 tests as the acceptance suite — they define the API contract; adapt imports only |
| Parity | `pytest` differential job (§7.3) | Old vs new on golden corpus + hypothesis fuzzing |
| Gradient checks | `pytest` + `torch.autograd.gradcheck` | Every `autograd.Function` bridge, float64 |
| GPU integration | opt-in marker `gpu` (self-hosted runner, `[gpu]` commit-message trigger as today) | Kernel perf + correctness on real CUDA |
| Reproducibility | CI job running `tools/reproduce.py` | Figures/tables regenerate; SHA-256 manifest matches |
| Benchmarks | `criterion` (Rust) + `pytest-benchmark` (Python) | Regression thresholds enforced |

---

## 10. Tooling, CI/CD, and Packaging

- **Build:** Cargo workspace + maturin; `pip install -e .` uses maturin develop; wheels
  for manylinux2014 (x86_64, aarch64), Windows x86_64, macOS universal2. CUDA-enabled
  wheels published as `prinet[cuda]` variant (or separate `prinet-cuda` package,
  decided in Phase 0 spike).
- **Lint/format:** `rustfmt` + `clippy -D warnings`; Python: `ruff` (replaces
  black+isort+bandit roles), `mypy --strict` on the Python layer, generated `.pyi` stubs
  for the extension module.
- **CI (GitHub Actions), mirroring 3.0.0's proven setup:**
  1. `rust.yml` — fmt, clippy, cargo test (Linux/Windows/macOS), criterion smoke.
  2. `python.yml` — pytest matrix (3.11–3.13 × Linux/Windows), mypy, ruff, coverage→codecov.
  3. `parity.yml` — differential tests vs `prinet==3.0.0`.
  4. `gpu.yml` — self-hosted CUDA runner, opt-in via `[gpu]` commit tag.
  5. `repro.yml` — reproducibility pipeline check.
  6. `release.yml` — maturin wheel matrix, sdist, PyPI via OIDC trusted publishing;
     crates.io publish for the Rust crates.
- **Docs build:** RTD config as today (§11).
- **Security:** `cargo audit` + `pip-audit` in CI; secret scanning; no runtime code
  generation (removes the current bandit-relevant JIT surface).

---

## 11. Documentation Plan

- Keep Sphinx + ReadTheDocs (`prinet.readthedocs.io`), Markdown guides alongside RST.
- Rust API docs via `cargo doc` published to docs.rs; Sphinx links to them.
- Port and update: Architecture Guide, Getting Started Tutorial, Capacity Analysis,
  Coupling Topologies API reference, per-package READMEs (the 3.0.0 convention of a
  README per source directory is good — keep it).
- New documents: **Migration Guide (3.0 → 4.0)** with symbol-by-symbol mapping table and
  tolerance notes; **Kernel Architecture** doc explaining the single-source CubeCL
  design; **Parity Report** publishing the §7 results.
- Notebooks: port all 3, plus one new notebook demonstrating the torch bridge.
- Benchmark reports continue to be written to `docs/test_and_benchmark_results/`
  (lowercase, gitignored — never a capital-D `Docs/` path).

---

## 12. Phased Implementation Roadmap

Dependency-ordered; each phase ends with green CI and a tagged pre-release. (No calendar
estimates included by design; sequence and exit criteria only.)

**Phase 0 — Foundation and de-risking spikes**
- New repo scaffold (§14), Cargo workspace, maturin wheel building on all 3 OS targets in CI.
- Spikes: (a) DLPack zero-copy torch↔Rust round-trip overhead; (b) CubeCL fused
  mean-field RK4 prototype vs 3.0.0 Triton at N=1M; (c) `ort` DirectML/VitisAI EP check
  on target hardware. **Go/no-go gate:** spikes meet §8 targets; else revisit §4.3 choices
  (raw `cudarc` kernels, or keep Triton via the Python layer for GPU paths).
- Build the golden-trajectory parity corpus from 3.0.0 (§7.1).

**Phase 1 — Dynamics core**
- `prinet-dynamics`: state, Kuramoto/Stuart–Landau/Hopf, Euler/RK4/RK45, PAC, coupling
  topologies, k-NN index; `prinet-metrics` full set. CPU only.
- Exit: parity suite green for all non-trainable dynamics; property tests green.

**Phase 2 — Advanced numerics and simulation engine**
- Exponential + multi-rate integrators, hierarchical band networks (continuous),
  temporal propagator, sweep utilities, `prinet-tensor` (Tucker/CP), `prinet-sim`
  (OscilloSim, CSR sparse, chimera metrics, pruning).
- Exit: OscilloSim parity at N up to 1M on CPU; sweep speedup target met.

**Phase 3 — GPU kernels**
- `prinet-kernels`: fused mean-field RK4, sparse k-NN coupling, PAC modulation, fused
  discrete step, hierarchical reductions; CUDA + wgpu backends; kernel-equivalence tests.
- Exit: §8 GPU targets met on the self-hosted runner.

**Phase 4 — Trainable layer stack and torch bridge**
- `prinet-train`: DiscreteDeltaThetaGamma, ResonanceLayer, inhibition/STE, activations,
  HEP, optimizers. Python `prinet.nn` with `autograd.Function` bridges; gradcheck suite.
- Port PhaseTracker, HybridPRINetV2, SlotAttention baselines, ablation variants,
  adaptive allocation.
- Exit: PhaseTracker trains on temporal CLEVR-N to ≥3.0.0 IP scores; gradcheck green;
  bridge overhead <5%.

**Phase 5 — Daemon, evaluation, and experiment tooling**
- `prinet-daemon` (ONNX controller, backend detection, native thread + ring buffer),
  training hooks, MOT evaluation, temporal metrics/training framework, adversarial
  tools, stats utilities.
- Exit: daemon latency target met; MOT metrics match motmetrics reference.

**Phase 6 — Benchmarks, reproduction, docs, release**
- Reorganized benchmark suite + `benchrunner` CLI; port figure/table generators and
  `tools/reproduce.py`; full 62-benchmark re-run and Parity Report; Migration Guide;
  notebooks; docs site.
- Exit: reproducibility check byte-identical; all CI jobs green; publish `4.0.0-rc1`
  wheels; deprecation shims in a final `prinet 3.1` release pointing users to 4.0.

**Parallelization:** Phases 2 and 3 can proceed concurrently after Phase 1; Python-layer
porting (datasets, reporting, figure generation) can start in Phase 1 since it is
independent of the Rust core.

---

## 13. Risk Register and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Burn/CubeCL kernels can't match Triton perf at N=1M | Medium | High | Phase 0 spike gates the decision; fallback to `cudarc` hand-written CUDA (still precompiled, still single-source per backend) or retain Triton behind the Python layer for GPU-only paths |
| 2 | Torch-bridge autograd overhead erodes training perf | Medium | High | Phase 0 DLPack spike; batch boundary crossings (one call per integration, not per step); custom backward implemented in Rust |
| 3 | Numerical drift breaks published results | Medium | Critical | Golden corpus + differential CI from day one (§7); statistical comparison in chaotic regimes; Parity Report before release |
| 4 | `ort` lacks VitisAI EP parity on Ryzen AI | Medium | Medium | F5 fallback: keep NPU path in Python `onnxruntime` (it's a small, perf-uncritical component) |
| 5 | Rust autodiff (Burn) gaps for exotic ops (STE, complex tanh, HEP) | Medium | Medium | These have simple closed-form adjoints; implement custom backward ops — already required for STE anyway |
| 6 | Scope creep from 62 legacy benchmark scripts | High | Medium | Benchmarks are consumers, not core: port the 9 categories to shared drivers; keep old JSON artefacts valid so nothing needs re-measuring for reproduction |
| 7 | Team Rust ramp-up slows delivery | Medium | Medium | Phase ordering puts the simplest, best-specified code first (dynamics core); the acceptance test suite (existing 1,670 tests) provides tight guardrails |
| 8 | Windows CUDA wheel toolchain complexity | Low | Medium | maturin + GitHub-hosted Windows runners already proven for major projects (polars, tokenizers); no user-side compiler needed (improvement over 3.0.0) |
| 9 | macOS has no CUDA | Low | Low | wgpu/Metal backend covers it; GPU parity targets apply to CUDA only |

---

## 14. Repository Layout for the New Project

```
prinet4/
├── Cargo.toml                    # workspace
├── crates/
│   ├── prinet-dynamics/
│   ├── prinet-metrics/
│   ├── prinet-tensor/
│   ├── prinet-kernels/
│   ├── prinet-sim/
│   ├── prinet-train/
│   ├── prinet-daemon/
│   └── prinet-py/                # PyO3 extension crate
├── python/
│   └── prinet/                   # pure-Python layer
│       ├── __init__.py           # public API (3.0.0-compatible symbols)
│       ├── nn/                   # torch wrappers + baselines
│       ├── eval/                 # MOT evaluation, temporal metrics
│       ├── experiments/          # ablation, stats, adversarial, training frameworks
│       ├── reporting/            # benchmark JSON, figures, tables, profiler
│       ├── datasets.py
│       └── _prinet_core.pyi      # generated stubs
├── parity/                       # golden-trajectory corpus + differential tests
├── benchmarks/                   # 9 category packages + benchrunner CLI
├── tests/                        # ported pytest acceptance suite (37 files)
├── tools/
│   └── reproduce.py
├── models/                       # subconscious_controller.onnx
├── notebooks/
├── docs/                         # Sphinx + Markdown guides (+ this document's successor)
├── paper/                        # unchanged artefacts
├── pyproject.toml                # maturin build backend
└── .github/workflows/            # rust, python, parity, gpu, repro, release
```

---

## 15. Definition of Done

The rebuild is complete when **all** of the following hold:

1. Every symbol in PRINet 3.0.0's public API (175+ exports) has a working equivalent,
   with the Migration Guide documenting any renames.
2. The full ported pytest acceptance suite (1,670 tests) passes on Linux + Windows,
   Python 3.11–3.13.
3. The parity suite passes at the tolerances in §7; the Parity Report is published in docs.
4. GPU performance targets in §8 are met on the reference CUDA runner; CPU fallback
   targets met on the reference 16-core machine; benchmark regression gates active in CI.
5. `tools/reproduce.py` regenerates all 15 figures and 11 tables from the existing JSON
   artefacts with a matching SHA-256 manifest.
6. Prebuilt wheels install cleanly (`pip install prinet`) on manylinux, Windows, and
   macOS without a local compiler; the wheel smoke test passes in CI.
7. Subconscious controller inference works on CPU, DirectML, and (where hardware is
   available) Ryzen AI NPU.
8. Documentation site builds and deploys; Rust crates documented on docs.rs; the three
   notebooks plus the torch-bridge notebook run end-to-end.
9. `cargo clippy -D warnings`, `cargo audit`, `ruff`, `mypy --strict`, and `pip-audit`
   are all clean.
10. A final `prinet 3.1` maintenance release ships deprecation pointers to 4.0.

---

*Prepared from a full audit of PRINet 3.0.0: 43 source modules (~25.5k lines), 37 test
files, 62 benchmark scripts, CI workflows, docs, and reproducibility pipeline.*
