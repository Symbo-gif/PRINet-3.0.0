# paper/

Publication assets for the NeurIPS 2026 submission.

## Contents

| File | Description |
|---|---|
| `main.tex` | Main paper source (~800 lines, 8 sections) |
| `main.pdf` | Compiled main paper (10 pages) |
| `supplementary.tex` | Supplementary material (~740 lines, sections S1–S24) |
| `supplementary.pdf` | Compiled supplementary material (7 pages) |
| `neurips_2026.sty` | NeurIPS 2026 LaTeX style file |

## Subdirectories

| Directory | Description |
|---|---|
| [`figures/`](figures/) | 14 publication figures in PDF and PNG format |
| [`tables/`](tables/) | 11 LaTeX table fragments included by `main.tex` and `supplementary.tex` |

## Compilation

Requires a LaTeX distribution (e.g., TeX Live, MiKTeX) with standard packages.

```bash
cd paper/
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex          # second pass for references
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode supplementary.tex  # second pass for references
```

## Reproducing Figures and Tables

All figures and tables can be regenerated from benchmark JSON artefacts without training or GPU access:

```bash
# From the project root
python reproduce.py
```

This reads from `benchmarks/results/*.json` and writes to `paper/figures/` and `paper/tables/`.

## Paper Structure

### Main Paper (main.tex)

1. **Introduction** — Oscillatory binding motivation and contributions
2. **Related Work** — Slot attention, oscillator models, chimera states
3. **Methods** — PhaseTracker architecture, OscilloSim, coupling modes
4. **Trained Model Comparison** — PhaseTracker vs Slot Attention (7-seed)
5. **Parameter Efficiency Analysis** — O(1) vs O(d²) formal argument
6. **Scientific Analysis** — Gradient flow, representation geometry, supercritical regime
7. **Chimera States** — Confirmation in differentiable neural framework
8. **Discussion** — Limitations, complementary failure modes, future work

### Supplementary (supplementary.tex)

Sections S1–S24 covering extended statistical analyses, additional figures, hyperparameter details, and proof sketches.

## Citation

```bibtex
@article{maillet2026prinet,
  title={Phase-Resonance Binding: A Parameter-Efficient Alternative to
         Slot Attention for Temporal Object Tracking},
  author={Maillet, Michael and Davison, Damien and Davison, Sacha},
  journal={Advances in Neural Information Processing Systems},
  year={2026},
  note={NeurIPS 2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
