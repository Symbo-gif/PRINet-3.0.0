"""PRINet core propagation package.

This package replaces the monolithic ``propagation.py`` module (114 KB,
3290 lines) with 8 focused submodules.  All public symbols are
re-exported here so that existing code using::

    from prinet.core.propagation import KuramotoOscillator

continues to work without modification (R.1 backward compatibility).

Submodules
----------
oscillator_state
    :class:`OscillatorState`, :class:`OscillatorSyncError`, helpers.
oscillator_models
    :class:`OscillatorModel`, :class:`KuramotoOscillator`,
    :class:`StuartLandauOscillator`, :class:`HopfOscillator`.
integrators
    :class:`ExponentialIntegrator`, :class:`MultiRateIntegrator`.
coupling
    :class:`PhaseAmplitudeCoupling`.
networks
    :class:`ThetaGammaNetwork`, :class:`DeltaThetaGammaNetwork`,
    :class:`DiscreteDeltaThetaGamma`.
temporal
    :class:`TemporalPhasePropagator`.
inhibition
    :class:`FeedforwardInhibition`, :class:`FeedbackInhibition`,
    :class:`DentateGyrusConverter`.
sweep_utils
    :func:`phase_to_rate`, :func:`detect_oscillation`,
    :func:`sweep_coupling_params`.
"""

from .coupling import PhaseAmplitudeCoupling
from .inhibition import (
    DentateGyrusConverter,
    FeedbackInhibition,
    FeedforwardInhibition,
)
from .integrators import (
    ExponentialIntegrator,
    MultiRateIntegrator,
)
from .networks import (
    DeltaThetaGammaNetwork,
    DiscreteDeltaThetaGamma,
    ThetaGammaNetwork,
)
from .oscillator_models import (
    HopfOscillator,
    KuramotoOscillator,
    OscillatorModel,
    StuartLandauOscillator,
)
from .oscillator_state import (
    _DERIV_CLAMP,
    _SPARSE_EPS,
    _TWO_PI,
    OscillatorState,
    OscillatorSyncError,
    _build_phase_knn_index,
    _clamp_finite,
    _safe_phase_diff,
    _wrap_phase,
)
from .sweep_utils import (
    detect_oscillation,
    phase_to_rate,
    sweep_coupling_params,
)
from .temporal import TemporalPhasePropagator

__all__ = [
    # oscillator_state
    "OscillatorState",
    "OscillatorSyncError",
    # oscillator_models
    "OscillatorModel",
    "KuramotoOscillator",
    "StuartLandauOscillator",
    "HopfOscillator",
    # integrators
    "ExponentialIntegrator",
    "MultiRateIntegrator",
    # coupling
    "PhaseAmplitudeCoupling",
    # networks
    "ThetaGammaNetwork",
    "DeltaThetaGammaNetwork",
    "DiscreteDeltaThetaGamma",
    # temporal
    "TemporalPhasePropagator",
    # inhibition
    "FeedforwardInhibition",
    "FeedbackInhibition",
    "DentateGyrusConverter",
    # sweep_utils
    "detect_oscillation",
    "phase_to_rate",
    "sweep_coupling_params",
    # private helpers (backward compat)
    "_TWO_PI",
    "_SPARSE_EPS",
    "_DERIV_CLAMP",
    "_wrap_phase",
    "_safe_phase_diff",
    "_clamp_finite",
    "_build_phase_knn_index",
]
