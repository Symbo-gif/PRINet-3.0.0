# prinet.utils

GPU-accelerated solvers, simulation tools, datasets, and utilities for PRINet.

## Modules

| Module | Description |
|---|---|
| `oscillosim.py` | **OscilloSim v2.0** — GPU-accelerated Kuramoto oscillator simulator supporting 8 coupling modes, scaling to 1M+ oscillators at 1.95B osc·step/s |
| `cuda_kernels.py` | CUDA JIT-compiled kernels for oscillator simulation (36.9× speedup over PyTorch baseline) |
| `triton_kernels.py` | Triton fused kernels: mean-field RK4, sparse k-NN coupling, PAC modulation, hierarchical order parameter |
| `fused_kernels.py` | Fused kernel utility wrappers and availability detection |
| `npu_backend.py` | DirectML/NPU backend support for Windows (Neural Processing Unit acceleration) |
| `datasets.py` | `CLEVR-N` synthetic dataset loader for multi-object tracking benchmarks |
| `figure_generation.py` | NeurIPS-style publication figure generators (14 figures) from benchmark JSON artefacts |
| `table_generation.py` | LaTeX table generators (11 tables) from benchmark JSON artefacts |
| `benchmark_reporting.py` | Benchmark result parsing, report generation, and leaderboard creation |
| `temporal_metrics.py` | Temporal analysis metrics: `TemporalMetrics`, identity switches, track fragmentation, binding robustness |
| `temporal_training.py` | Temporal training loops: `TemporalTrainer`, multi-seed training, Hungarian matching loss |
| `adversarial_tools.py` | Adversarial robustness testing: FGSM, PGD attack implementations |
| `profiler.py` | Performance profiling: FLOPs counting, wall-time measurement, bottleneck analysis |
| `y4q1_tools.py` | Y4Q1 benchmark-specific utilities for extended training and ablation studies |

## OscilloSim Coupling Modes

| Mode | Description |
|---|---|
| `global` | All-to-all Kuramoto coupling |
| `ring` | Nearest-neighbour lattice |
| `small_world` | Watts-Strogatz rewiring |
| `nonlocal` | Distance-weighted (Abrams-Strogatz) |
| `community` | Block-structured modules |
| `hierarchical` | Multi-scale nested coupling |
| `evolutionary` | Time-varying weights |
| `custom` | User-defined adjacency matrix |

## GPU Acceleration Stack

```
Triton Fused Kernels  ←  highest performance (Linux)
         ↓ fallback
CUDA JIT Kernels      ←  36.9× speedup (NVIDIA GPUs)
         ↓ fallback
DirectML / NPU        ←  Windows NPU acceleration
         ↓ fallback
PyTorch CPU           ←  universal fallback
```

## License

MIT — see [LICENSE](../../../LICENSE).
