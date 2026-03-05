# paper/figures/

Publication figures for the NeurIPS 2026 paper — 14 figures in both PDF (vector) and PNG (raster) format.

## Figures

| Figure | Filename | Description |
|---|---|---|
| Fig. 2 | `fig2_clevr_n_capacity` | CLEVR-N binding capacity: PhaseTracker vs Slot Attention |
| Fig. 3 | `fig3_chimera_metrics` | Chimera state bimodality coefficient metrics |
| Fig. 4 | `fig4_chimera_k_alpha_heatmap` | K–α parameter sensitivity heatmap |
| Fig. 5 | `fig5_mot_occlusion_comparison` | Multi-object tracking occlusion recovery comparison |
| Fig. 6 | `fig6_oscillosim_scaling` | OscilloSim GPU throughput scaling |
| Fig. 7 | `fig7_ablation_results` | Ablation study: attention-only vs oscillators-only |
| Fig. 8 | `fig8_parameter_efficiency` | Parameter count comparison (PT 4,991 vs SA 83,904) |
| Fig. 9 | `fig9_training_curves` | Training convergence curves (PhaseTracker and Slot Attention) |
| Fig. 10 | `fig10_statistical_summary` | Statistical test summary (TOST, Bayes Factor, Cliff's delta) |
| Fig. 11 | `fig11_flops_scaling` | FLOPs efficiency scaling analysis |
| Fig. 12 | `fig12_supercritical_regime` | Supercritical Kuramoto regime parameter space |
| Fig. 13 | `fig13_representation_geometry` | Embedding geometry: k-NN purity, intrinsic dimensionality |
| Fig. 14 | `fig14_gradient_flow` | Gradient propagation analysis through layers |
| Fig. 15 | `fig15_noise_velocity` | Noise and velocity robustness evaluation |

## Regeneration

All figures can be regenerated from stored benchmark artefacts:

```bash
python reproduce.py --figures-only
```

## License

MIT — see [LICENSE](../../LICENSE).
