"""Subconscious State & Control-Signal Dataclasses for PRINet.

Defines the fixed-schema data containers used by the subconscious
meta-controller:

* :class:`SubconsciousState` — a compressed snapshot of the running
  system (order parameters, training dynamics, hardware telemetry)
  that is serialized to a 32-dimensional float tensor for ONNX
  inference.
* :class:`ControlSignals` — the 8-dimensional output produced by the
  subconscious controller, containing suggestions for the main training
  loop (K range, LR multiplier, regime preference, alert level,
  coupling-mode suggestion).
* :class:`ControlSignalBuffer` — a thread-safe, lock-protected wrapper
  that allows the daemon thread to *write* control signals while the
  main training loop *reads* the latest value without blocking.

Example:
    >>> import time, numpy as np
    >>> state = SubconsciousState.default()
    >>> z = state.to_tensor()
    >>> print(z.shape, z.dtype)
    (32,) float32
    >>> ctrl = ControlSignals.from_tensor(np.zeros(8, dtype=np.float32))
    >>> print(ctrl.alert_level)
    0.0
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM: int = 32
"""Fixed dimensionality of the flattened state vector fed to the controller."""

CONTROL_DIM: int = 8
"""Fixed dimensionality of the control-signal output tensor."""

_REGIME_MAP: dict[str, int] = {
    "mean_field": 0,
    "sparse_knn": 1,
    "full": 2,
}
"""Maps regime name strings to integer indices for tensor encoding."""

_REGIME_INV: dict[int, str] = {v: k for k, v in _REGIME_MAP.items()}


# ---------------------------------------------------------------------------
# SubconsciousState
# ---------------------------------------------------------------------------


@dataclass
class SubconsciousState:
    """Compressed system snapshot for the subconscious controller.

    All numeric fields are scalar floats (or a short list for per-band
    values).  The :meth:`to_tensor` method packs them into a
    ``numpy.ndarray`` of shape ``(32,)`` with ``dtype=float32``.

    Attributes:
        r_per_band: Per-band Kuramoto order parameters ``[r_δ, r_θ, r_γ]``.
        r_global: Global (all-band) order parameter.
        loss_ema: Exponential moving average of the training loss.
        loss_variance: Variance of the training loss over recent window.
        grad_norm_ema: EMA of the gradient L2 norm.
        lr_current: Current learning rate.
        scalr_alpha: Current SCALR coupling-strength modifier.
        gpu_temp: GPU temperature in degrees Celsius.
        gpu_util: GPU utilization fraction ``[0, 1]``.
        vram_pct: GPU VRAM utilization fraction ``[0, 1]``.
        cpu_util: CPU utilization fraction ``[0, 1]``.
        step_latency_p50: Median step latency in seconds.
        step_latency_p95: 95th-percentile step latency in seconds.
        throughput: Training throughput (samples / second).
        epoch: Current epoch index.
        regime: Active coupling regime name
            (``"mean_field"`` | ``"sparse_knn"`` | ``"full"``).
        timestamp: Unix timestamp of the snapshot.
    """

    # Per-band order parameters (3 floats)
    r_per_band: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    r_global: float = 0.0

    # Training dynamics
    loss_ema: float = 0.0
    loss_variance: float = 0.0
    grad_norm_ema: float = 0.0
    lr_current: float = 1e-3
    scalr_alpha: float = 1.0

    # Hardware telemetry
    gpu_temp: float = 0.0
    gpu_util: float = 0.0
    vram_pct: float = 0.0
    cpu_util: float = 0.0

    # Timing
    step_latency_p50: float = 0.0
    step_latency_p95: float = 0.0
    throughput: float = 0.0

    # Meta
    epoch: int = 0
    regime: str = "mean_field"
    timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """Flatten this state to a fixed-size numpy array for ONNX input.

        Layout (32 floats):
            ``[r_δ, r_θ, r_γ, r_global, loss_ema, loss_variance,
              grad_norm_ema, lr_current, scalr_alpha,
              gpu_temp/100, gpu_util, vram_pct, cpu_util,
              step_latency_p50, step_latency_p95, throughput/1e4,
              epoch/1e4, regime_idx/2, timestamp_frac,
              <13 zeros padding>]``

        Returns:
            ``numpy.ndarray`` of shape ``(32,)`` with ``dtype=float32``.
        """
        regime_idx = float(_REGIME_MAP.get(self.regime, 0))
        # Normalize timestamp to fractional hours since midnight
        ts_frac = (self.timestamp % 86400.0) / 86400.0

        vec = np.array(
            [
                # Per-band order params (3)
                self.r_per_band[0] if len(self.r_per_band) > 0 else 0.0,
                self.r_per_band[1] if len(self.r_per_band) > 1 else 0.0,
                self.r_per_band[2] if len(self.r_per_band) > 2 else 0.0,
                # Global order param (1)
                self.r_global,
                # Training dynamics (5)
                self.loss_ema,
                self.loss_variance,
                self.grad_norm_ema,
                self.lr_current,
                self.scalr_alpha,
                # Hardware (4) — normalized
                self.gpu_temp / 100.0,
                self.gpu_util,
                self.vram_pct,
                self.cpu_util,
                # Timing (3) — normalized
                self.step_latency_p50,
                self.step_latency_p95,
                self.throughput / 1e4,
                # Meta (3) — normalized
                float(self.epoch) / 1e4,
                regime_idx / 2.0,
                ts_frac,
            ],
            dtype=np.float32,
        )
        # Pad to STATE_DIM
        padded = np.zeros(STATE_DIM, dtype=np.float32)
        padded[: len(vec)] = vec
        return padded

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default() -> SubconsciousState:
        """Create a default (zero-initialized) state snapshot.

        Returns:
            A ``SubconsciousState`` with safe defaults.
        """
        return SubconsciousState(timestamp=time.time())

    def clone(self) -> SubconsciousState:
        """Return a shallow copy of this state.

        Returns:
            New ``SubconsciousState`` with the same field values.
        """
        return SubconsciousState(
            **{f.name: getattr(self, f.name) for f in fields(self)}
        )


# ---------------------------------------------------------------------------
# ControlSignals
# ---------------------------------------------------------------------------


@dataclass
class ControlSignals:
    """Control suggestions produced by the subconscious controller.

    The main training loop reads these at epoch boundaries and may
    apply them with damping.  All signals are bounded and finite by
    design.

    Attributes:
        suggested_K_min: Lower bound for coupling strength K.
        suggested_K_max: Upper bound for coupling strength K.
        lr_multiplier: Multiplicative LR adjustment (centered at 1.0).
        regime_mf_weight: Weight for mean-field regime preference.
        regime_sk_weight: Weight for sparse-kNN regime preference.
        regime_full_weight: Weight for full-coupling regime preference.
        alert_level: Alert level ``[0, 1]``.  0 = nominal, 1 = intervene.
        coupling_mode_suggestion: Index into coupling-type enum.
    """

    suggested_K_min: float = 0.5
    suggested_K_max: float = 5.0
    lr_multiplier: float = 1.0
    regime_mf_weight: float = 0.33
    regime_sk_weight: float = 0.33
    regime_full_weight: float = 0.34
    alert_level: float = 0.0
    coupling_mode_suggestion: float = 0.0

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """Pack control signals into a numpy array.

        Returns:
            ``numpy.ndarray`` of shape ``(8,)`` with ``dtype=float32``.
        """
        return np.array(
            [
                self.suggested_K_min,
                self.suggested_K_max,
                self.lr_multiplier,
                self.regime_mf_weight,
                self.regime_sk_weight,
                self.regime_full_weight,
                self.alert_level,
                self.coupling_mode_suggestion,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_tensor(arr: np.ndarray) -> ControlSignals:
        """Unpack a numpy array into a ``ControlSignals`` instance.

        Args:
            arr: Array of shape ``(8,)`` or ``(1, 8)``.

        Returns:
            Populated ``ControlSignals`` dataclass.

        Raises:
            ValueError: If *arr* has fewer than 8 elements.
        """
        flat = np.asarray(arr, dtype=np.float32).ravel()
        if flat.shape[0] < CONTROL_DIM:
            msg = f"Expected at least {CONTROL_DIM} elements, " f"got {flat.shape[0]}."
            raise ValueError(msg)
        return ControlSignals(
            suggested_K_min=float(flat[0]),
            suggested_K_max=float(flat[1]),
            lr_multiplier=float(np.clip(flat[2], 0.1, 10.0)),
            regime_mf_weight=float(flat[3]),
            regime_sk_weight=float(flat[4]),
            regime_full_weight=float(flat[5]),
            alert_level=float(np.clip(flat[6], 0.0, 1.0)),
            coupling_mode_suggestion=float(flat[7]),
        )

    @staticmethod
    def default() -> ControlSignals:
        """Return safe default control signals (no-op).

        Returns:
            A ``ControlSignals`` that requests no changes.
        """
        return ControlSignals()

    @property
    def preferred_regime(self) -> str:
        """Return the regime with the highest preference weight.

        Returns:
            One of ``"mean_field"``, ``"sparse_knn"``, or ``"full"``.
        """
        weights = {
            "mean_field": self.regime_mf_weight,
            "sparse_knn": self.regime_sk_weight,
            "full": self.regime_full_weight,
        }
        return max(weights, key=weights.get)  # type: ignore[arg-type]

    def is_finite(self) -> bool:
        """Check that all control signals are finite.

        Returns:
            ``True`` if every field is finite (no NaN or Inf).
        """
        return bool(np.all(np.isfinite(self.to_tensor())))


# ---------------------------------------------------------------------------
# ControlSignalBuffer (thread-safe)
# ---------------------------------------------------------------------------


class ControlSignalBuffer:
    """Thread-safe buffer for storing the latest control signals.

    The daemon writes via :meth:`update` and the main training loop
    reads via :meth:`latest`.  Both operations acquire a
    :class:`threading.Lock` and are therefore safe from data races.

    Example:
        >>> buf = ControlSignalBuffer()
        >>> buf.update(ControlSignals(alert_level=0.8))
        >>> ctrl = buf.latest()
        >>> print(ctrl.alert_level)
        0.8
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signals: ControlSignals = ControlSignals.default()

    def update(self, signals: ControlSignals) -> None:
        """Atomically replace the stored control signals.

        Args:
            signals: New control signals to store.
        """
        with self._lock:
            self._signals = signals

    def latest(self) -> ControlSignals:
        """Read the most recent control signals.

        Returns:
            The latest ``ControlSignals`` (safe defaults if never updated).
        """
        with self._lock:
            return self._signals
