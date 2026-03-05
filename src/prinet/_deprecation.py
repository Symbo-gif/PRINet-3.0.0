"""PRINet deprecation utilities for API freeze enforcement.

Provides decorators and helpers for marking deprecated functions,
classes, and parameters with clear migration paths.

Example:
    >>> from prinet._deprecation import deprecated
    >>> @deprecated("1.0.0", "Use new_func() instead.")
    ... def old_func():
    ...     pass
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    since: str,
    message: str,
    removal: str | None = None,
) -> Callable[[F], F]:
    """Mark a function or class as deprecated.

    Args:
        since: Version when the deprecation was introduced (e.g. "1.0.0").
        message: Migration guidance for users.
        removal: Version when the symbol will be removed (e.g. "2.0.0").

    Returns:
        Decorator that emits DeprecationWarning on first call.

    Example:
        >>> @deprecated("1.0.0", "Use new_api() instead.", removal="2.0.0")
        ... def old_api():
        ...     pass
    """

    def decorator(func: F) -> F:
        removal_note = f" Will be removed in {removal}." if removal else ""
        warn_msg = (
            f"{func.__qualname__} is deprecated since v{since}. "
            f"{message}{removal_note}"
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_parameter(
    param_name: str,
    since: str,
    message: str,
) -> Callable[[F], F]:
    """Warn when a deprecated keyword argument is passed.

    Args:
        param_name: Name of the deprecated parameter.
        since: Version when the deprecation was introduced.
        message: Migration guidance for users.

    Returns:
        Decorator that emits DeprecationWarning when param_name is in kwargs.

    Example:
        >>> @deprecated_parameter("old_param", "1.0.0", "Use new_param instead.")
        ... def my_func(new_param=1, **kwargs):
        ...     pass
    """

    def decorator(func: F) -> F:
        warn_msg = (
            f"Parameter '{param_name}' of {func.__qualname__} is deprecated "
            f"since v{since}. {message}"
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs:
                warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Frozen public API surface for prinet 1.0.0
# This list is the contract: any symbol in __all__ of the top-level
# package at the time of the 1.0.0 release. Removing or renaming
# any of these without a deprecation cycle is a breaking change.
FROZEN_PUBLIC_API: frozenset[str] = frozenset(
    [
        # Core — Foundation
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
        # Core — Hierarchical
        "DeltaThetaGammaNetwork",
        "ThetaGammaNetwork",
        "PhaseAmplitudeCoupling",
        "MultiRateIntegrator",
        "detect_oscillation",
        "phase_to_rate",
        "sweep_coupling_params",
        # Core — Inhibition & DG
        "FeedforwardInhibition",
        "FeedbackInhibition",
        "DentateGyrusConverter",
        # Core — Subconscious
        "SubconsciousState",
        "ControlSignals",
        "ControlSignalBuffer",
        "STATE_DIM",
        "CONTROL_DIM",
        "SubconsciousDaemon",
        "collect_system_state",
        # Core — Temporal
        "TemporalPhasePropagator",
        "inter_frame_phase_correlation",
        # Neural Network — Foundation
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
        "GatedPhaseActivation",
        # Neural Network — Hierarchical
        "PhaseToRateConverter",
        "SparsityRegularizationLoss",
        "HierarchicalResonanceLayer",
        "PhaseAmplitudeCouplingLayer",
        "DGLayer",
        "PhaseToRateAutoencoder",
        "DenseAutoencoder",
        # Neural Network — Subconscious & Hybrid
        "SubconsciousController",
        "HybridPRINet",
        "HybridCLEVRN",
        "AlternatingOptimizer",
        "StateCollector",
        # Neural Network — Y2 Q1-Q3
        "DiscreteDeltaThetaGamma",
        "DiscreteDeltaThetaGammaLayer",
        "OscillatoryAttention",
        "InterleavedHybridPRINet",
        "TemporalHybridPRINet",
        "ActiveControlTrainer",
        "retrain_controller",
        "HybridPRINetV2",
        "HybridPRINetV2CLEVRN",
        "PhaseTracker",
        "TelemetryLogger",
        # Utils — Solvers
        "BatchedRK45Solver",
        "FixedStepRK4Solver",
        "SolverResult",
        "sparse_coupling_matrix",
        "gradient_checkpoint_integration",
        # Utils — NPU / Backend
        "BackendType",
        "detect_best_backend",
        "npu_available",
        "directml_available",
        "create_session",
        "backend_info",
        # Utils — Benchmark
        "generate_benchmark_report",
        "generate_leaderboard",
        "generate_scalr_metrics_report",
        # Utils — Triton / PyTorch kernels
        "triton_available",
        "triton_fused_mean_field_rk4_step",
        "triton_sparse_knn_coupling",
        "pytorch_mean_field_rk4_step",
        "pytorch_sparse_knn_coupling",
        "triton_pac_modulation",
        "triton_hierarchical_order_param",
        "pytorch_pac_modulation",
        "pytorch_hierarchical_order_param",
        "pytorch_multi_rate_rk4_step",
        "pytorch_multi_rate_derivatives",
        "pytorch_fused_sub_step_rk4",
        "pytorch_cross_band_coupling",
        "triton_fused_discrete_step",
        "pytorch_fused_discrete_step",
    ]
)


def verify_api_surface(module_all: list[str]) -> tuple[set[str], set[str]]:
    """Compare a module's __all__ against the frozen API contract.

    Args:
        module_all: The __all__ list from the top-level prinet package.

    Returns:
        Tuple of (missing_from_module, extra_in_module) symbol sets.
        Both empty means the API matches the frozen contract.

    Example:
        >>> import prinet
        >>> missing, extra = verify_api_surface(prinet.__all__)
        >>> assert not missing, f"API regression: {missing}"
    """
    current = set(module_all)
    frozen = set(FROZEN_PUBLIC_API)
    missing = frozen - current
    extra = current - frozen
    return missing, extra
