# benchmarks/

Experiment benchmarks and performance evaluation scripts for PRINet.

## Overview

This directory contains **58 benchmark scripts** that execute the experiments reported in the NeurIPS 2026 paper. All benchmark results are stored as JSON artefacts in [`results/`](results/) for reproducibility.

## Running Benchmarks

```bash
# Run a specific benchmark
python benchmarks/y4q3_benchmarks.py

# Run the CLEVR-N capacity sweep
python benchmarks/run_clevr_n_sweep.py

# Reproduce all paper figures/tables from stored results (no GPU needed)
python reproduce.py
```

## Benchmark Scripts

### Core Performance

| Script | Description |
|---|---|
| `gpu_benchmarks.py` | GPU throughput and latency benchmarks |
| `triton_fused_kernel_benchmark.py` | Triton fused kernel performance comparison |
| `activation_profile.py` | Activation function profiling |
| `holomorphic_profile.py` | Holomorphic activation profiling |
| `on_vs_onlogn_benchmark.py` | O(N) vs O(N log N) coupling scaling |
| `on_stress_benchmark.py` | O(N) stress testing at scale |

### Oscillator Dynamics

| Script | Description |
|---|---|
| `oscillator_scaling.py` | Oscillator count scaling benchmarks |
| `oscillobench.py` | OscilloSim comprehensive benchmarks |
| `pairwise_coupling_scaling.py` | Pairwise coupling matrix scaling |
| `coupling_complexity_benchmark.py` | Coupling topology complexity analysis |
| `scientific_coupling_benchmark.py` | Scientific coupling experiments |
| `phase_diagram.py` | Phase diagram generation |
| `desync_catastrophe.py` | Desynchronization catastrophe detection |
| `phase_to_rate_benchmark.py` | Phase-to-rate conversion benchmarks |

### Multi-Object Tracking

| Script | Description |
|---|---|
| `clevr_n.py` | CLEVR-N binding benchmark |
| `run_clevr_n_sweep.py` | CLEVR-N parameter sweep driver |
| `subconscious_benchmark.py` | SubconsciousController benchmarks |

### Heterogeneous Computing

| Script | Description |
|---|---|
| `heterogeneous_gpu_cpu_benchmark.py` | GPU+CPU pipeline benchmarks |
| `multirate_triton_benchmark.py` | Multi-rate Triton kernel benchmarks |
| `scalr_vs_adam_benchmark.py` | SCALR vs Adam optimizer comparison |

### Statistical Hardening (Phase 1–4)

| Script | Description |
|---|---|
| `phase1_statistical_hardening.py` | Multi-seed CV, bootstrap, Cliff's delta |
| `phase1_bf_extension.py` | Extended Bayes Factor analysis |
| `phase1_bf_tost_resolution.py` | TOST equivalence testing |
| `phase2_scaling_analysis.py` | Object/sequence/occlusion/velocity/noise scaling |
| `phase3_scientific_experiments.py` | FLOPs audit, gradient flow, representation geometry |
| `phase4_theoretical_verification.py` | Convergence bounds, parameter scaling proofs |

### Research Phase Benchmarks

| Phase | Scripts |
|---|---|
| Q2 | `q2_benchmarks.py` |
| Q4 | `q4_benchmarks.py` |
| Y2 Q1–Q4 | `y2q1_benchmarks.py` through `y2q4_benchmarks.py` |
| Y3 Q1–Q4 | `y3q1_benchmarks.py` through `y3q4_benchmarks.py`, `y3q45_comprehensive_benchmarks.py`, `y3q49_scientific_regime_benchmark.py` |
| Y4 Q1–Q4 | `y4q1_benchmarks.py` through `y4q4_benchmarks.py`, plus `y4q1_2` through `y4q1_9` variants |

### Individual Benchmark Runners

| Script | Description |
|---|---|
| `run_q17_individual.py` | Q17 individual benchmark runner |
| `run_q18_individual.py` | Q18 individual benchmark runner |
| `run_q19_individual.py` | Q19 individual benchmark runner |
| `concurrent_2regime_benchmark.py` | 2-regime concurrent execution |
| `concurrent_3regime_benchmark.py` | 3-regime concurrent execution |
| `goldilocks_sustained_benchmark.py` | Goldilocks zone sustained GPU benchmarks |
| `goldilocks_sustained_cpu_benchmark.py` | Goldilocks zone sustained CPU benchmarks |
| `mnist_subset.py` | MNIST subset evaluation |

## Results

All benchmark outputs are stored in [`results/`](results/) — see that directory's README for details.

## License

MIT — see [LICENSE](../LICENSE).
