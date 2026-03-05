"""PRINet Core Dynamics.

This package contains the fundamental mathematical implementations for
Phase-Rotation-Interaction Networks, including polyadic tensor decomposition,
oscillator dynamics (Kuramoto/Stuart-Landau), and synchronization measurement.

Modules:
    decomposition: Polyadic tensor decomposition (Tucker/CP).
    propagation: ODE-based oscillator dynamics and abstract models.
    measurement: Synchronization metrics (order parameter, coherence).
"""

from prinet.core.decomposition import CPDecomposition, PolyadicTensor
from prinet.core.measurement import (
    build_phase_knn,
    inter_frame_phase_correlation,
    kuramoto_order_parameter,
    mean_phase_coherence,
    phase_coherence_matrix,
    sparse_mean_phase_coherence,
    sparse_synchronization_energy,
)
from prinet.core.propagation import (
    DeltaThetaGammaNetwork,
    DentateGyrusConverter,
    DiscreteDeltaThetaGamma,
    ExponentialIntegrator,
    FeedbackInhibition,
    FeedforwardInhibition,
    HopfOscillator,
    KuramotoOscillator,
    MultiRateIntegrator,
    OscillatorModel,
    OscillatorState,
    PhaseAmplitudeCoupling,
    StuartLandauOscillator,
    TemporalPhasePropagator,
    ThetaGammaNetwork,
    detect_oscillation,
    phase_to_rate,
    sweep_coupling_params,
)
from prinet.core.subconscious import (
    CONTROL_DIM,
    STATE_DIM,
    ControlSignalBuffer,
    ControlSignals,
    SubconsciousState,
)
from prinet.core.subconscious_daemon import (
    SubconsciousDaemon,
    collect_system_state,
)

__all__ = [
    "PolyadicTensor",
    "CPDecomposition",
    "OscillatorModel",
    "OscillatorState",
    "KuramotoOscillator",
    "StuartLandauOscillator",
    "HopfOscillator",
    "ExponentialIntegrator",
    "kuramoto_order_parameter",
    "mean_phase_coherence",
    "phase_coherence_matrix",
    "build_phase_knn",
    "sparse_mean_phase_coherence",
    "sparse_synchronization_energy",
    # Q3: Hierarchical dynamics
    "DeltaThetaGammaNetwork",
    "DiscreteDeltaThetaGamma",
    "ThetaGammaNetwork",
    "PhaseAmplitudeCoupling",
    "MultiRateIntegrator",
    "detect_oscillation",
    "phase_to_rate",
    "sweep_coupling_params",
    # Q3: Inhibition & DG conversion
    "FeedforwardInhibition",
    "FeedbackInhibition",
    "DentateGyrusConverter",
    # Y2 Q2: Temporal phase propagation
    "TemporalPhasePropagator",
    "inter_frame_phase_correlation",
    # Q4: Subconscious layer
    "SubconsciousState",
    "ControlSignals",
    "ControlSignalBuffer",
    "STATE_DIM",
    "CONTROL_DIM",
    "SubconsciousDaemon",
    "collect_system_state",
]
