# prinet.core

Fundamental oscillatory dynamics and mathematical primitives for the PRINet architecture.

## Modules

| Module | Description |
|---|---|
| `decomposition.py` | `CPDecomposition` and `PolyadicTensor` — canonical polyadic tensor decomposition for multi-dimensional harmonic representation |
| `measurement.py` | `PhaseAmplitudeCoupling`, `kuramoto_order_parameter`, `mean_phase_coherence`, `phase_coherence_matrix` — synchronization and coupling metrics |
| `subconscious.py` | `SubconsciousController` — adaptive control layer for oscillatory systems |
| `subconscious_daemon.py` | `CognitiveDaemon` / `SubconsciousDaemon` — background monitoring of oscillator state and control signal buffering |

## Subpackages

| Subpackage | Description |
|---|---|
| [`propagation/`](propagation/) | Oscillator models, ODE integrators, coupling topologies, temporal dynamics, and parameter sweep utilities |

## Key Exports

```python
from prinet.core import (
    CPDecomposition,
    PolyadicTensor,
    PhaseAmplitudeCoupling,
    KuramotoOscillator,
    StuartLandauOscillator,
    HopfOscillator,
    kuramoto_order_parameter,
    mean_phase_coherence,
    phase_to_rate,
    sweep_coupling_params,
)
```

## License

MIT — see [LICENSE](../../../LICENSE).
