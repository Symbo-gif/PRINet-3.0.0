# notebooks/

Jupyter notebook tutorials for PRINet.

## Notebooks

| Notebook | Description |
|---|---|
| [`01_oscillosim_quickstart.ipynb`](01_oscillosim_quickstart.ipynb) | **OscilloSim Quick Start** — Create and visualize oscillator networks, explore coupling modes, measure synchronization and chimera states |
| [`02_clevr_n_binding.ipynb`](02_clevr_n_binding.ipynb) | **CLEVR-N Binding** — Train a PhaseTracker on the CLEVR-N synthetic dataset, evaluate identity preservation, compare with Slot Attention |
| [`03_custom_coupling.ipynb`](03_custom_coupling.ipynb) | **Custom Coupling Topologies** — Build custom adjacency matrices, implement novel coupling modes, parameter sweep exploration |

## Getting Started

```bash
# Install PRINet from source with development dependencies
git clone https://github.com/michaelmaillet/prinet.git
cd prinet
pip install -e ".[dev]"

# Launch Jupyter
jupyter notebook notebooks/
```

## Prerequisites

- Python ≥ 3.11
- PyTorch ≥ 2.0
- Matplotlib (included in `prinet[dev]`)
- Optional: CUDA GPU for accelerated simulation

## License

MIT — see [LICENSE](../LICENSE).
