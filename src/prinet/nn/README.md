# prinet.nn

PyTorch-compatible neural network modules, architectures, and optimizers for PRINet.

## Modules

| Module | Description |
|---|---|
| `layers.py` | Core layers: `ResonanceLayer`, `PhaseTracker`, `PhaseAmplitudeCouplingLayer`, `HierarchicalResonanceLayer`, `GatedPhaseActivation`, `PRINetModel` |
| `hybrid.py` | `HybridPRINetV2` — hybrid architecture combining oscillatory binding with attention |
| `ablation_variants.py` | `AblationHybridPRINetV2` — ablation study variants: attention-only, oscillators-only, frozen, static |
| `activations.py` | Custom activation functions: `dSiLU`, `HolomorphicActivation`, `PhaseActivation` |
| `adaptive_allocation.py` | Adaptive resource allocation layer for dynamic compute budgeting |
| `hep.py` | Hidden Error Prediction (HEP) mechanisms for predictive coding |
| `optimizers.py` | `SCALROptimizer` (Scaled Cosine Annealed Learning Rate), `RIPOptimizer`, `AlternatingOptimizer` |
| `slot_attention.py` | `TemporalSlotAttentionMOT`, `SlotAttentionModule`, `SlotAttentionCLEVRN` — Slot Attention baselines for comparison |
| `subconscious_model.py` | `SubconsciousController` neural architecture for adaptive oscillator control |
| `mot_evaluation.py` | Multi-object tracking evaluation metrics (MOTA, MOTP, identity metrics) |
| `training_hooks.py` | Training utilities: callbacks, hooks, `TelemetryLogger`, `StateCollector` |

## Key Architectures

| Architecture | Parameters | Description |
|---|---|---|
| `PhaseTracker` | 4,991 | Phase-resonance temporal object tracker — primary contribution |
| `HybridPRINetV2` | ~5,000 | Hybrid oscillator + attention architecture |
| `TemporalSlotAttentionMOT` | 83,904 | Temporal Slot Attention baseline |
| `PRINetModel` | Configurable | End-to-end resonance inference model |

## Usage

```python
import torch
from prinet.nn import PhaseTracker, SCALROptimizer

model = PhaseTracker(n_slots=4, n_features=32, n_oscillators=16)
optimizer = SCALROptimizer(model.parameters(), lr=1e-3)

frames = torch.randn(8, 4, 32)  # (batch, slots, features)
bindings = model(frames)
```

## License

MIT — see [LICENSE](../../../LICENSE).
