# prinet.core.propagation

Oscillator propagation kernels — the computational engine for phase dynamics in PRINet.

## Modules

| Module | Description |
|---|---|
| `oscillator_models.py` | Oscillator model implementations: `KuramotoOscillator`, `StuartLandauOscillator`, `HopfOscillator` |
| `oscillator_state.py` | `OscillatorState` — state representation and management for oscillator networks |
| `integrators.py` | ODE integrators: `ExponentialIntegrator`, `MultiRateIntegrator`, RK4, Euler methods |
| `coupling.py` | Coupling topology builders: ring, small-world (Watts-Strogatz), nonlocal, community, hierarchical, evolutionary, global, and custom adjacency matrices |
| `networks.py` | Network structure construction: `DeltaThetaGammaNetwork`, `ThetaGammaNetwork`, `DiscreteDeltaThetaGamma` |
| `inhibition.py` | Oscillatory inhibition mechanisms: `FeedforwardInhibition`, `FeedbackInhibition`, `DentateGyrusConverter` |
| `temporal.py` | Temporal dynamics: `TemporalPhasePropagator`, `inter_frame_phase_correlation` — phase evolution across time steps |
| `sweep_utils.py` | Parameter sweep utilities: `sweep_coupling_params` for systematic exploration of coupling strength and topology |

## Architecture

The propagation engine follows a modular design:

```
Oscillator Models  →  Coupling Topology  →  ODE Integrator  →  State Update
   (Kuramoto)         (ring, SW, ...)       (RK4, Euler)       (OscillatorState)
```

Multi-rate integration supports separate delta, theta, and gamma frequency bands operating at different timescales for hierarchical processing.

## License

MIT — see [LICENSE](../../../../LICENSE).
