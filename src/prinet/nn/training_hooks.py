"""Training loop hooks for subconscious daemon integration.

Provides :class:`StateCollector` — a callback that gathers training
metrics and system telemetry, submits them to the
:class:`~prinet.core.subconscious_daemon.SubconsciousDaemon`, and
reads adaptive control signals back into the training loop.

This implements Task 5.6 from the Q4 plan: "Wire daemon into
HybridPRINet training — StateCollector at epoch boundaries,
ControlSignalBuffer for adaptive lr/K/regime suggestions."

Example:
    >>> from prinet import SubconsciousDaemon, SubconsciousController
    >>> daemon = SubconsciousDaemon("models/subconscious_controller.onnx")
    >>> daemon.start()
    >>> hook = StateCollector(daemon)
    >>> # In training loop:
    >>> hook.on_epoch_end(epoch=1, loss=0.5, r_per_band=[0.8, 0.6, 0.4])
    >>> ctrl = hook.latest_control()
    >>> daemon.stop()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import torch
from torch import Tensor


class StateCollector:
    """Training loop hook that bridges the model and subconscious daemon.

    Collects training metrics (loss, gradients, order parameters) and
    system telemetry (GPU temp, VRAM, latency) then submits a packed
    :class:`~prinet.core.subconscious.SubconsciousState` to the daemon.

    Can be called at epoch boundaries or after every N steps.

    Args:
        daemon: Running :class:`~prinet.core.subconscious_daemon.SubconsciousDaemon`.
        loss_ema_alpha: EMA smoothing factor for loss tracking.
        latency_window: Number of recent step latencies to keep for
            p50/p95 computation.
    """

    def __init__(
        self,
        daemon: Any,
        loss_ema_alpha: float = 0.1,
        latency_window: int = 100,
    ) -> None:
        self._daemon = daemon
        self._loss_ema: float = 0.0
        self._loss_var: float = 0.0
        self._loss_alpha = loss_ema_alpha
        self._grad_norm_ema: float = 0.0
        self._step_latencies: deque[float] = deque(maxlen=latency_window)
        self._last_step_time: float = time.monotonic()
        self._step_count: int = 0
        self._epoch: int = 0

    def on_step_start(self) -> None:
        """Call at the beginning of each training step to record timing."""
        self._last_step_time = time.monotonic()

    def on_step_end(
        self,
        loss: float | Tensor,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Call at the end of each training step to accumulate metrics.

        Args:
            loss: Current step loss (scalar Tensor or float).
            model: Optional model to compute gradient norm from.
        """
        # Record latency
        elapsed = (time.monotonic() - self._last_step_time) * 1000.0  # ms
        self._step_latencies.append(elapsed)

        # Update loss EMA / variance
        loss_val = float(loss.item() if isinstance(loss, Tensor) else loss)
        alpha = self._loss_alpha
        self._loss_ema = alpha * loss_val + (1 - alpha) * self._loss_ema
        diff = loss_val - self._loss_ema
        self._loss_var = alpha * (diff * diff) + (1 - alpha) * self._loss_var

        # Update gradient norm EMA
        if model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5
            self._grad_norm_ema = alpha * grad_norm + (1 - alpha) * self._grad_norm_ema

        self._step_count += 1

    def on_epoch_end(
        self,
        epoch: int,
        loss: float | Tensor | None = None,
        r_per_band: Optional[list[float]] = None,
        r_global: Optional[float] = None,
        lr_current: float = 0.0,
        scalr_alpha: float = 1.0,
        regime: str = "mean_field",
    ) -> None:
        """Call at the end of each epoch to submit state to daemon.

        This is the primary integration point — it packs all collected
        metrics and system telemetry into a
        :class:`~prinet.core.subconscious.SubconsciousState` and
        submits it to the daemon.

        Args:
            epoch: Current epoch number.
            loss: Epoch loss (overrides accumulated EMA if provided).
            r_per_band: Per-band order parameters ``[r_delta, r_theta, r_gamma]``.
            r_global: Global order parameter. If ``None``, computed as mean.
            lr_current: Current learning rate.
            scalr_alpha: SCALR alpha parameter.
            regime: Current coupling regime name.
        """
        from prinet.core.subconscious import SubconsciousState
        from prinet.core.subconscious_daemon import collect_system_state

        self._epoch = epoch

        # Override loss EMA if explicit loss provided
        if loss is not None:
            loss_val = float(loss.item() if isinstance(loss, Tensor) else loss)
            self._loss_ema = loss_val

        # Default order parameters
        rpb = r_per_band if r_per_band is not None else [0.5, 0.5, 0.5]
        r_g = r_global if r_global is not None else sum(rpb) / len(rpb)

        # Compute latency percentiles
        if self._step_latencies:
            sorted_lat = sorted(self._step_latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95 = sorted_lat[min(int(n * 0.95), n - 1)]
            throughput = 1000.0 / (sum(sorted_lat) / n) if n > 0 else 0.0
        else:
            p50 = p95 = 0.0
            throughput = 0.0

        # Collect system telemetry
        try:
            sys_state = collect_system_state(
                r_per_band=rpb,
                r_global=r_g,
                loss_ema=self._loss_ema,
                loss_variance=self._loss_var,
                grad_norm_ema=self._grad_norm_ema,
                lr_current=lr_current,
                scalr_alpha=scalr_alpha,
                epoch=epoch,
                regime=regime,
            )
            # Augment with latency/throughput (not supported by collect_system_state)
            sys_state.step_latency_p50 = p50
            sys_state.step_latency_p95 = p95
            sys_state.throughput = throughput
        except Exception:
            # Fallback: construct state manually without system telemetry
            sys_state = SubconsciousState(
                r_per_band=rpb,
                r_global=r_g,
                loss_ema=self._loss_ema,
                loss_variance=self._loss_var,
                grad_norm_ema=self._grad_norm_ema,
                lr_current=lr_current,
                scalr_alpha=scalr_alpha,
                gpu_temp=0.0,
                gpu_util=0.0,
                vram_pct=0.0,
                cpu_util=0.0,
                step_latency_p50=p50,
                step_latency_p95=p95,
                throughput=throughput,
                epoch=epoch,
                regime=regime,
            )

        self._daemon.submit_state(sys_state)

    def latest_control(self) -> Any:
        """Read the latest control signals from the daemon.

        Returns:
            :class:`~prinet.core.subconscious.ControlSignals` from daemon.
        """
        return self._daemon.get_control()

    @property
    def loss_ema(self) -> float:
        """Current exponentially-weighted moving average of loss."""
        return self._loss_ema

    @property
    def loss_variance(self) -> float:
        """Current EMA of loss variance."""
        return self._loss_var

    @property
    def grad_norm_ema(self) -> float:
        """Current EMA of gradient norm."""
        return self._grad_norm_ema

    @property
    def step_count(self) -> int:
        """Total number of training steps recorded."""
        return self._step_count


# =========================================================================
# Year 2 Q1 — Workstream C: Subconscious Control Policies
# =========================================================================


def apply_lr_adjustment(
    control: Any,
    optimizer: torch.optim.Optimizer,
    max_adjustment: float = 0.05,
) -> float:
    """Policy C.a: Adjust learning rate based on ``lr_multiplier``.

    When ``alert_level > 0.7``, applies the daemon's suggested
    lr_multiplier with damping (max ±5% per epoch by default).

    Args:
        control: :class:`~prinet.core.subconscious.ControlSignals`.
        optimizer: Torch optimizer whose param group LRs are adjusted.
        max_adjustment: Maximum fractional LR change per call.

    Returns:
        The actual multiplier applied (1.0 if no adjustment).
    """
    if control is None:
        return 1.0

    alert = getattr(control, "alert_level", 0.0)
    if alert < 0.7:
        return 1.0

    raw_mult = getattr(control, "lr_multiplier", 1.0)
    # Damp: clamp to [1 - max_adj, 1 + max_adj]
    mult = max(1.0 - max_adjustment, min(1.0 + max_adjustment, raw_mult))

    for pg in optimizer.param_groups:
        pg["lr"] *= mult

    return mult


def apply_k_range_narrowing(
    control: Any,
    model: torch.nn.Module,
    field_name: str = "coupling_strength",
    max_adjustment: float = 0.05,
) -> tuple[float, float]:
    """Policy C.b: Narrow coupling strength K range.

    When ``r_global < 0.3`` (desynchronized), narrows the coupling
    range toward the daemon's suggested ``[K_min, K_max]``.

    Searches for any ``nn.Parameter`` whose name matches
    ``field_name`` (e.g., ``W_delta``, ``W_theta``, ``W_gamma``)
    and clamps its values within the suggested range.

    Args:
        control: :class:`~prinet.core.subconscious.ControlSignals`.
        model: Model containing coupling parameters.
        field_name: Substring to match against parameter names.
        max_adjustment: Maximum K adjustment per call.

    Returns:
        ``(K_min, K_max)`` — the suggested range (or ``(0.0, 0.0)``
        if no adjustment was made).
    """
    if control is None:
        return 0.0, 0.0

    k_min = getattr(control, "suggested_K_min", 0.0)
    k_max = getattr(control, "suggested_K_max", 10.0)

    if k_min >= k_max:
        return 0.0, 0.0

    adjusted = False
    for name, param in model.named_parameters():
        if field_name in name and param.requires_grad:
            with torch.no_grad():
                # Soft clamp: move at most max_adjustment per call
                low = param.data.clamp(min=k_min - max_adjustment)
                param.data.copy_(low.clamp(max=k_max + max_adjustment))
                adjusted = True

    return (k_min, k_max) if adjusted else (0.0, 0.0)


def apply_regime_bias(
    control: Any,
) -> str:
    """Policy C.c: Suggest coupling regime based on subconscious weights.

    Reads ``regime_mf_weight``, ``regime_sk_weight``, ``regime_full_weight``
    from control signals and returns the regime with the highest weight.

    Args:
        control: :class:`~prinet.core.subconscious.ControlSignals`.

    Returns:
        Regime name: ``"mean_field"``, ``"sparse_knn"``, or ``"full"``.
    """
    if control is None:
        return "mean_field"

    w_mf = getattr(control, "regime_mf_weight", 0.5)
    w_sk = getattr(control, "regime_sk_weight", 0.3)
    w_full = getattr(control, "regime_full_weight", 0.2)

    weights = {"mean_field": w_mf, "sparse_knn": w_sk, "full": w_full}
    return max(weights, key=weights.get)  # type: ignore[arg-type]


class TelemetryLogger:
    """Observation-mode telemetry for subconscious training integration.

    Logs daemon state + control signal pairs alongside training metrics
    without applying any control adjustments. Used to gather telemetry
    datasets for future controller retraining (Workstream C.2).

    Args:
        capacity: Maximum number of records to keep in memory.

    Example:
        >>> logger = TelemetryLogger()
        >>> logger.record(epoch=1, loss=0.5, control=ctrl)
        >>> logger.to_json("telemetry.json")
    """

    def __init__(self, capacity: int = 10000) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def record(
        self,
        epoch: int,
        loss: float,
        r_per_band: Optional[list[float]] = None,
        r_global: float = 0.0,
        control: Any = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a telemetry snapshot.

        Args:
            epoch: Current epoch.
            loss: Current loss value.
            r_per_band: Per-band order parameters.
            r_global: Global order parameter.
            control: ControlSignals from daemon (or None).
            extra: Additional key-value pairs to log.
        """
        entry: dict[str, Any] = {
            "epoch": epoch,
            "loss": loss,
            "r_per_band": r_per_band or [0.0, 0.0, 0.0],
            "r_global": r_global,
            "timestamp": time.time(),
        }

        if control is not None:
            entry["control"] = {
                "lr_multiplier": getattr(control, "lr_multiplier", 1.0),
                "alert_level": getattr(control, "alert_level", 0.0),
                "suggested_K_min": getattr(control, "suggested_K_min", 0.0),
                "suggested_K_max": getattr(control, "suggested_K_max", 10.0),
                "regime_mf_weight": getattr(control, "regime_mf_weight", 0.5),
                "regime_sk_weight": getattr(control, "regime_sk_weight", 0.3),
                "regime_full_weight": getattr(control, "regime_full_weight", 0.2),
                "coupling_mode_suggestion": getattr(
                    control, "coupling_mode_suggestion", 0.0
                ),
            }

        if extra:
            entry.update(extra)

        self._records.append(entry)

    def to_json(self, path: str) -> None:
        """Write accumulated telemetry to a JSON file.

        Args:
            path: Output file path.
        """
        import json

        with open(path, "w") as f:
            json.dump(list(self._records), f, indent=2)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return all accumulated records as a list."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


# =========================================================================
# Year 2 Q2 — Workstream F: Active Subconscious Control
# =========================================================================


class ActiveControlTrainer:
    """Training wrapper that applies subconscious control policies.

    Integrates the three control policies (lr adjustment, K-range
    narrowing, regime bias) into a training loop with a hard limit
    on per-signal adjustment magnitude (max 5% per epoch by default).

    Used for A/B testing: active mode applies policies each epoch;
    passive/observation mode logs them without applying.

    Args:
        model: The model being trained.
        optimizer: The optimizer.
        daemon: Running SubconsciousDaemon (or None for passive mode).
        max_adjustment: Maximum per-signal adjustment fraction.
        active: Whether to actually apply control signals.
        telemetry_logger: Optional TelemetryLogger for recording data.

    Example:
        >>> trainer = ActiveControlTrainer(model, optimizer, daemon)
        >>> trainer.on_epoch_end(epoch=1, loss=0.5, r_per_band=[0.8, 0.6, 0.4])
        >>> print(f"Policy applied: {trainer.last_policy_applied}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        daemon: Any = None,
        max_adjustment: float = 0.05,
        active: bool = True,
        telemetry_logger: Optional[TelemetryLogger] = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._daemon = daemon
        self._max_adj = max_adjustment
        self._active = active
        self._logger = telemetry_logger or TelemetryLogger()
        self._collector: Optional[StateCollector] = None
        self._last_policy: dict[str, Any] = {}

        if daemon is not None:
            self._collector = StateCollector(daemon)

    @property
    def active(self) -> bool:
        """Whether active control is enabled."""
        return self._active

    @property
    def last_policy_applied(self) -> dict[str, Any]:
        """Details of the last policy application."""
        return self._last_policy

    @property
    def telemetry(self) -> TelemetryLogger:
        """Access the telemetry logger."""
        return self._logger

    def on_step_start(self) -> None:
        """Call at the start of each training step."""
        if self._collector is not None:
            self._collector.on_step_start()

    def on_step_end(self, loss: float | Tensor) -> None:
        """Call at the end of each training step."""
        if self._collector is not None:
            self._collector.on_step_end(loss, model=self._model)

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        r_per_band: Optional[list[float]] = None,
        r_global: Optional[float] = None,
        lr_current: float = 0.0,
    ) -> dict[str, Any]:
        """End-of-epoch hook: log telemetry and apply control policies.

        Args:
            epoch: Current epoch number.
            loss: Epoch loss.
            r_per_band: Per-band order parameters.
            r_global: Global order parameter.
            lr_current: Current learning rate.

        Returns:
            Dict summarizing applied policies. Keys: ``lr_mult``,
            ``k_range``, ``regime``, ``active``.
        """
        rpb = r_per_band or [0.5, 0.5, 0.5]
        rg = r_global if r_global is not None else sum(rpb) / len(rpb)

        control = None
        if self._collector is not None:
            self._collector.on_epoch_end(
                epoch=epoch,
                loss=loss,
                r_per_band=rpb,
                r_global=rg,
                lr_current=lr_current,
            )
            control = self._collector.latest_control()

        # Log telemetry regardless of active/passive
        self._logger.record(
            epoch=epoch,
            loss=loss,
            r_per_band=rpb,
            r_global=rg,
            control=control,
        )

        policy: dict[str, Any] = {
            "active": self._active,
            "lr_mult": 1.0,
            "k_range": (0.0, 0.0),
            "regime": "mean_field",
        }

        if self._active and control is not None:
            # Apply policies with damping
            lr_mult = apply_lr_adjustment(
                control,
                self._optimizer,
                max_adjustment=self._max_adj,
            )
            k_range = apply_k_range_narrowing(
                control,
                self._model,
                max_adjustment=self._max_adj,
            )
            regime = apply_regime_bias(control)

            policy["lr_mult"] = lr_mult
            policy["k_range"] = k_range
            policy["regime"] = regime

        self._last_policy = policy
        return policy
