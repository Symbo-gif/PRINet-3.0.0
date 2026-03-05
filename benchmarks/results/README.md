# benchmarks/results/

Stored JSON benchmark artefacts — **152 files** providing full reproducibility for all figures and tables in the NeurIPS 2026 paper.

## Overview

Every experimental claim in the paper is backed by a JSON artefact in this directory. The [`reproduce.py`](../../reproduce.py) script reads these files to regenerate all paper figures and LaTeX tables without requiring any training or GPU access.

```bash
# Verify all required artefacts are present and regenerate outputs
python reproduce.py
```

## Artefact Categories

### Phase 1: Statistical Hardening (8 artefacts)

| File | Contents |
|---|---|
| `phase1_clevr_seeds.json` | CLEVR-N 10-seed cross-validation (CV = 0.099%) |
| `phase1_chimera_seeds.json` | Chimera bootstrap analysis (BC = 0.643 ± 0.051) |
| `phase1_cliffs_delta.json` | Cliff's delta effect sizes |
| `phase1_bayes_factor.json` | Bayesian Factor analysis (BF₁₀ = 0.521) |
| `phase1_holm_bonferroni.json` | Holm-Bonferroni multiple comparison correction |
| `phase1_bf_extension_15seed.json` | Extended 15-seed Bayes Factor |
| `phase1_bf_tost_resolution.json` | TOST equivalence testing (δ=0.5%, p=0.004) |
| `phase1_unified_summary.json` | Phase 1 unified summary |

### Phase 2: Scaling Analysis (7 artefacts)

| File | Contents |
|---|---|
| `phase2_object_scaling.json` | Object count scaling N=5–20 |
| `phase2_sequence_scaling.json` | Sequence length scaling T=50–1000 |
| `phase2_occlusion_recovery.json` | Occlusion recovery time τ_r |
| `phase2_velocity_stress.json` | Velocity stress 1×–10× |
| `phase2_noise_sweep.json` | Noise robustness σ=0–1 |
| `phase2_chimera_nsweep.json` | Chimera N=128–1024 |
| `phase2_unified_summary.json` | Phase 2 unified summary |

### Phase 3: Scientific Experiments (3 artefacts)

| File | Contents |
|---|---|
| `phase3_profiling.json` | FLOPs/latency profiling (PT 44.7× fewer FLOPs) |
| `phase3_gradient_flow.json` | Gradient propagation analysis (PT 15× stronger) |
| `phase3_representation_geometry.json` | Embedding geometry (k-NN purity 0.687 vs 0.207) |

### Phase 4: Theoretical Verification (2 artefacts)

| File | Contents |
|---|---|
| `phase4_convergence_verification.json` | Convergence bound verification (K_eff/K_c = 15–207×) |
| `phase4_parameter_scaling.json` | Parameter scaling (PT 379 vs SA 54,336 = 143×) |

### Legacy Benchmark Artefacts (~130 artefacts)

Artefacts from the full development history, prefixed by research phase:

- `benchmark_y4q1_*.json` — Y4Q1 ablation, chimera, temporal MOT, training curves
- `y4q1_7_*.json` — Preregistration, occlusion stress, training curves
- `y4q1_8_*.json` — Adversarial tests, coupling experiments, curriculum learning
- `y4q1_9_*.json` — Fine occlusion, object scaling, embedding analysis
- `y4q2_*.json`, `y4q3_*.json`, `y4q4_*.json` — Publication generation artefacts
- `y3q*_*.json` — Year 3 research phase artefacts

### Integrity

| File | Purpose |
|---|---|
| `sha256_manifest.json` | SHA-256 checksums for all artefacts — use to verify integrity |

## License

MIT — see [LICENSE](../../LICENSE).
