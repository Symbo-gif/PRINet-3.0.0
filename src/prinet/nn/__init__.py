"""PRINet Neural Network Modules.

Provides PyTorch-compatible layers, models, and optimizers for building
and training Phase-Rotation-Interaction Networks.

Modules:
    layers: ResonanceLayer, PRINetModel, and initialization utilities.
    optimizers: SynchronizedGradientDescent, SCALROptimizer, RIPOptimizer.
    hep: Holomorphic Equilibrium Propagation trainer.
    activations: dSiLU, HolomorphicActivation, PhaseActivation.
"""

from prinet.nn.ablation_variants import (
    PhaseTrackerFrozen,
    PhaseTrackerStatic,
    SlotAttentionFrozen,
    SlotAttentionNoGRU,
    create_ablation_tracker,
)
from prinet.nn.activations import (
    GatedPhaseActivation,
    HolomorphicActivation,
    PhaseActivation,
    dSiLU,
)
from prinet.nn.adaptive_allocation import (
    AdaptiveOscillatorAllocator,
    DynamicPhaseTracker,
    OscillatorBudget,
    estimate_complexity,
)
from prinet.nn.hep import (
    HolomorphicEnergy,
    HolomorphicEPTrainer,
)
from prinet.nn.hybrid import (
    AlternatingOptimizer,
    HybridCLEVRN,
    HybridPRINet,
    HybridPRINetV2,
    HybridPRINetV2CLEVRN,
    InterleavedHybridPRINet,
    PhaseTracker,
    TemporalHybridPRINet,
)
from prinet.nn.layers import (
    DenseAutoencoder,
    DGLayer,
    DiscreteDeltaThetaGammaLayer,
    HierarchicalResonanceLayer,
    OscillatoryAttention,
    PhaseAmplitudeCouplingLayer,
    PhaseToRateAutoencoder,
    PhaseToRateConverter,
    PRINetModel,
    ResonanceLayer,
    SparsityRegularizationLoss,
    compile_model,
    oscillatory_weight_init,
)
from prinet.nn.mot_evaluation import (
    AttentionTracker,
    Detection,
    TrackingResult,
    evaluate_tracking,
    generate_crowded_mot_sequence,
    generate_linear_mot_sequence,
    generate_temporal_reasoning_sequence,
)
from prinet.nn.optimizers import (
    RIPOptimizer,
    SCALROptimizer,
    SynchronizedGradientDescent,
)
from prinet.nn.slot_attention import (
    SlotAttentionCLEVRN,
    SlotAttentionModule,
    TemporalSlotAttentionMOT,
)
from prinet.nn.subconscious_model import SubconsciousController, retrain_controller
from prinet.nn.training_hooks import (
    ActiveControlTrainer,
    StateCollector,
    TelemetryLogger,
    apply_k_range_narrowing,
    apply_lr_adjustment,
    apply_regime_bias,
)

__all__ = [
    "ResonanceLayer",
    "PRINetModel",
    "compile_model",
    "oscillatory_weight_init",
    "SynchronizedGradientDescent",
    "SCALROptimizer",
    "RIPOptimizer",
    "HolomorphicEnergy",
    "HolomorphicEPTrainer",
    "dSiLU",
    "HolomorphicActivation",
    "PhaseActivation",
    # Q3: Hierarchical layers
    "GatedPhaseActivation",
    "PhaseToRateConverter",
    "SparsityRegularizationLoss",
    "HierarchicalResonanceLayer",
    "PhaseAmplitudeCouplingLayer",
    # Q3 late: DG & autoencoder
    "DGLayer",
    "PhaseToRateAutoencoder",
    "DenseAutoencoder",
    # Q4: Subconscious controller
    "SubconsciousController",
    # Q4: Hybrid architecture
    "HybridPRINet",
    "HybridCLEVRN",
    "AlternatingOptimizer",
    "StateCollector",
    # Y2 Q1: Discrete multi-rate & oscillatory attention
    "DiscreteDeltaThetaGammaLayer",
    "OscillatoryAttention",
    "InterleavedHybridPRINet",
    # Y2 Q2: Temporal & active control
    "TemporalHybridPRINet",
    "ActiveControlTrainer",
    "retrain_controller",
    # Y2 Q3: Architecture v2 & MOT
    "HybridPRINetV2",
    "HybridPRINetV2CLEVRN",
    "PhaseTracker",
    # Y2 Q1: Subconscious training integration
    "TelemetryLogger",
    "apply_lr_adjustment",
    "apply_k_range_narrowing",
    "apply_regime_bias",
    # Y3 Q2: Adaptive allocation & MOT evaluation
    "AdaptiveOscillatorAllocator",
    "DynamicPhaseTracker",
    "OscillatorBudget",
    "estimate_complexity",
    "AttentionTracker",
    "Detection",
    "TrackingResult",
    "evaluate_tracking",
    "generate_linear_mot_sequence",
    "generate_crowded_mot_sequence",
    "generate_temporal_reasoning_sequence",
    # Y3 Q4: Slot Attention comparison (P.5)
    "SlotAttentionModule",
    "SlotAttentionCLEVRN",
    # Y4 Q1: Temporal Slot Attention MOT (T.3)
    "TemporalSlotAttentionMOT",
    # Y4 Q1.7: Ablation variants
    "PhaseTrackerFrozen",
    "PhaseTrackerStatic",
    "SlotAttentionNoGRU",
    "SlotAttentionFrozen",
    "create_ablation_tracker",
]
