# prinet

**PRINet — Phase-Resonance Interference Network**

Top-level Python package providing oscillatory binding, phase-resonance dynamics, and parameter-efficient neural network layers for temporal object tracking.

## Subpackages

| Subpackage | Description |
|---|---|
| [`core/`](core/) | Fundamental dynamics: polyadic tensor decomposition, oscillator models, phase-amplitude coupling, propagation kernels, and the subconscious controller |
| [`nn/`](nn/) | PyTorch `nn.Module` layers, architectures (PRINet, HybridPRINetV2, PhaseTracker), optimizers (SCALR, RIP), and training utilities |
| [`utils/`](utils/) | OscilloSim simulator, CUDA/Triton kernels, datasets, benchmark reporting, figure/table generation, and profiling tools |

## Key Modules

| File | Purpose |
|---|---|
| `__init__.py` | Public API — exports 175+ symbols from all subpackages |
| `_deprecation.py` | Deprecation warning utilities |
| `py.typed` | PEP 561 marker for static type checker support |

## Quick Start

```python
import torch
from prinet.nn import PhaseTracker, PRINetModel
from prinet.utils.oscillosim import OscilloSim

# Phase-resonance object tracker (4,991 params vs SlotAttention's 83,904)
tracker = PhaseTracker(n_slots=4, n_features=32, n_oscillators=16)

# GPU-accelerated Kuramoto simulator
sim = OscilloSim(n_oscillators=512, coupling_mode="ring", K=4.0)
```

## Version

```python
>>> import prinet
>>> prinet.__version__
'3.0.0'
```

## License

MIT — see [LICENSE](../../LICENSE).
