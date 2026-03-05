"""Subconscious Daemon Thread for PRINet.

Implements :class:`SubconsciousDaemon`, a ``threading.Thread``
(daemon=True) that:

1. Receives :class:`~prinet.core.subconscious.SubconsciousState`
   snapshots via a bounded :class:`queue.Queue`.
2. Runs ONNX inference through the backend selected by
   :func:`~prinet.utils.npu_backend.detect_best_backend`.
3. Publishes :class:`~prinet.core.subconscious.ControlSignals` to a
   thread-safe :class:`~prinet.core.subconscious.ControlSignalBuffer`.

The main GPU training loop interacts **only** through two methods:

* :meth:`SubconsciousDaemon.submit_state` — non-blocking enqueue.
* :meth:`SubconsciousDaemon.get_control` — non-blocking read of the
  latest control signals.

Because the daemon runs on a separate thread, it never blocks the
GPU training pipeline.  State collection and ONNX inference happen
at ~10–30 s intervals (100–1000× slower than GPU steps), ensuring
minimal CPU overhead and stable feedback dynamics.

Example:
    >>> daemon = SubconsciousDaemon("models/subconscious_controller.onnx")
    >>> daemon.start()
    >>> daemon.submit_state(SubconsciousState.default())
    >>> ctrl = daemon.get_control()
    >>> daemon.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from prinet.core.subconscious import (
    CONTROL_DIM,
    STATE_DIM,
    ControlSignalBuffer,
    ControlSignals,
    SubconsciousState,
)
from prinet.utils.npu_backend import BackendType, create_session, detect_best_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INTERVAL: float = 15.0
"""Default polling interval in seconds."""

_DEFAULT_QUEUE_SIZE: int = 100
"""Maximum pending state snapshots before oldest are dropped."""

_WARMUP_SENTINEL = np.zeros((1, STATE_DIM), dtype=np.float32)
"""Dummy input used for the warm-up inference pass."""


# ---------------------------------------------------------------------------
# SubconsciousDaemon
# ---------------------------------------------------------------------------


class SubconsciousDaemon(threading.Thread):
    """Background daemon that runs the subconscious controller model.

    The daemon is a standard Python :class:`threading.Thread` with
    ``daemon=True``, so it is automatically terminated when the main
    process exits.

    Args:
        model_path: Path to the ONNX model file.
        backend: Execution-provider backend.  ``None`` for auto-detect.
        interval: Maximum seconds to wait for a new state before
            looping (acts as the effective polling interval).
        queue_size: Bounded queue capacity for pending states.
        warmup: If ``True``, run a single dummy inference on start
            to warm up the ONNX session and trigger JIT optimizations.
        dlq_maxlen: Maximum number of failed inference records retained
            in the dead-letter queue.  Oldest entries are evicted when
            the queue is full.
        max_errors_before_escalation: Number of consecutive / total
            errors that must accumulate before *error_escalation_callback*
            is invoked.  ``0`` disables escalation.
        error_escalation_callback: Optional callable invoked with a
            ``dict`` payload when the error threshold is crossed.  The
            payload contains keys ``"error_count"`` and ``"dlq_tail"``
            (the most recent DLQ entry).  Called on the daemon thread.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: BackendType | None = None,
        interval: float = _DEFAULT_INTERVAL,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        *,
        warmup: bool = True,
        dlq_maxlen: int = 100,
        max_errors_before_escalation: int = 10,
        error_escalation_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="PRINet-Subconscious")
        self._model_path = str(model_path)
        self._backend = backend
        self._interval = interval
        self._warmup = warmup

        self._state_queue: queue.Queue[SubconsciousState] = queue.Queue(
            maxsize=queue_size,
        )
        self._control_buffer = ControlSignalBuffer()
        self._stop_event = threading.Event()

        # Telemetry counters
        self._inferences: int = 0
        self._errors: int = 0
        self._start_time: float = 0.0

        # Dead-letter queue for failed inference records
        self._dead_letter_queue: deque[dict[str, Any]] = deque(maxlen=dlq_maxlen)
        self._max_errors_before_escalation: int = max_errors_before_escalation
        self._error_escalation_callback: Optional[Callable[[dict[str, Any]], None]] = (
            error_escalation_callback
        )

        # Lazily created in run() so Session lives on the daemon thread
        self._session: object | None = None  # ort.InferenceSession
        self._input_name: str = "state_vector"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_state(self, state: SubconsciousState) -> None:
        """Non-blocking enqueue of a system state snapshot.

        If the queue is full, the **oldest** state is dropped silently
        and the new state is enqueued.

        Args:
            state: The current system snapshot.
        """
        try:
            self._state_queue.put_nowait(state)
        except queue.Full:
            try:
                self._state_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._state_queue.put_nowait(state)
            except queue.Full:
                pass  # extremely unlikely; silently skip

    def get_control(self) -> ControlSignals:
        """Read the latest control signals (non-blocking).

        Returns:
            The most recent :class:`ControlSignals`, or safe defaults
            if no inference has been completed yet.
        """
        return self._control_buffer.latest()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the daemon to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for thread termination.
        """
        self._stop_event.set()
        self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("SubconsciousDaemon did not stop within %.1fs.", timeout)

    @property
    def inference_count(self) -> int:
        """Number of successful inferences completed."""
        return self._inferences

    @property
    def error_count(self) -> int:
        """Number of inference errors encountered."""
        return self._errors

    @property
    def dead_letter_queue(self) -> list[dict[str, Any]]:
        """Snapshot of the dead-letter queue (most-recent first).

        Each entry is a ``dict`` with keys:
        ``"error"`` (str), ``"error_count"`` (int),
        ``"timestamp"`` (float).
        """
        return list(reversed(self._dead_letter_queue))

    @property
    def dlq_size(self) -> int:
        """Number of entries currently in the dead-letter queue."""
        return len(self._dead_letter_queue)

    @property
    def uptime(self) -> float:
        """Seconds since the daemon started running."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the daemon loop (called by :meth:`start`).

        Do **not** call this directly — use ``daemon.start()`` instead.
        """
        self._start_time = time.monotonic()
        logger.info(
            "SubconsciousDaemon starting (model=%s, interval=%.1fs).",
            self._model_path,
            self._interval,
        )

        try:
            self._init_session()
        except Exception:
            logger.exception("Failed to create ORT session — daemon aborting.")
            return

        while not self._stop_event.is_set():
            try:
                state = self._state_queue.get(timeout=self._interval)
            except queue.Empty:
                continue

            self._run_inference(state)

        logger.info(
            "SubconsciousDaemon stopped after %d inferences (%d errors, "
            "%.1fs uptime).",
            self._inferences,
            self._errors,
            self.uptime,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_session(self) -> None:
        """Create the ONNX session and optionally warm it up."""
        backend = self._backend or detect_best_backend()
        self._session = create_session(self._model_path, backend)

        # Discover the input tensor name from the model metadata
        inputs = self._session.get_inputs()  # type: ignore[union-attr]
        if inputs:
            self._input_name = inputs[0].name

        if self._warmup:
            logger.debug("Warm-up inference…")
            try:
                self._session.run(None, {self._input_name: _WARMUP_SENTINEL})  # type: ignore[union-attr]
            except Exception:
                logger.debug(
                    "Warm-up inference raised (may be expected).", exc_info=True
                )

    def _run_inference(self, state: SubconsciousState) -> None:
        """Run a single inference pass and update the control buffer."""
        try:
            z = state.to_tensor().reshape(1, STATE_DIM)
            outputs = self._session.run(None, {self._input_name: z})  # type: ignore[union-attr]
            raw = outputs[0]  # shape (1, CONTROL_DIM) or (CONTROL_DIM,)
            signals = ControlSignals.from_tensor(raw)

            if not signals.is_finite():
                logger.warning("Non-finite control signals detected — using defaults.")
                signals = ControlSignals.default()

            self._control_buffer.update(signals)
            self._inferences += 1
        except Exception as exc:
            self._errors += 1
            logger.exception("Inference error in SubconsciousDaemon.")
            entry: dict[str, Any] = {
                "error": str(exc),
                "error_count": self._errors,
                "timestamp": time.monotonic(),
            }
            self._dead_letter_queue.append(entry)
            if (
                self._max_errors_before_escalation > 0
                and self._errors >= self._max_errors_before_escalation
                and self._error_escalation_callback is not None
            ):
                try:
                    self._error_escalation_callback(
                        {"error_count": self._errors, "dlq_tail": entry}
                    )
                except Exception:
                    logger.debug("Error escalation callback raised.", exc_info=True)


# ---------------------------------------------------------------------------
# Convenience: collect_system_state helper
# ---------------------------------------------------------------------------


def collect_system_state(
    *,
    r_per_band: Optional[list[float]] = None,
    r_global: float = 0.0,
    loss_ema: float = 0.0,
    loss_variance: float = 0.0,
    grad_norm_ema: float = 0.0,
    lr_current: float = 1e-3,
    scalr_alpha: float = 1.0,
    epoch: int = 0,
    regime: str = "mean_field",
) -> SubconsciousState:
    """Build a :class:`SubconsciousState` with automatic hardware telemetry.

    Reads GPU temperature, utilization, VRAM, and CPU utilization via
    :mod:`psutil` and :mod:`torch.cuda` when available.  Falls back to
    zeros if telemetry is unavailable.

    Args:
        r_per_band: Per-band order parameters ``[r_δ, r_θ, r_γ]``.
        r_global: Global order parameter.
        loss_ema: EMA of training loss.
        loss_variance: Variance of training loss.
        grad_norm_ema: EMA of gradient L2 norm.
        lr_current: Current learning rate.
        scalr_alpha: Current SCALR alpha.
        epoch: Current epoch index.
        regime: Active coupling regime.

    Returns:
        A fully-populated :class:`SubconsciousState`.
    """
    gpu_temp = 0.0
    gpu_util = 0.0
    vram_pct = 0.0
    cpu_util = 0.0

    # GPU telemetry via torch.cuda
    try:
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            vram_pct = 1.0 - (mem[0] / max(mem[1], 1))
    except Exception:
        pass

    # CPU utilization via psutil
    try:
        import psutil  # type: ignore[import-untyped]

        cpu_util = psutil.cpu_percent(interval=None) / 100.0
    except Exception:
        pass

    # GPU temp + utilization via pynvml (if available)
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_temp = float(pynvml.nvmlDeviceGetTemperature(handle, 0))
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(util.gpu) / 100.0
    except Exception:
        pass

    return SubconsciousState(
        r_per_band=r_per_band if r_per_band is not None else [0.0, 0.0, 0.0],
        r_global=r_global,
        loss_ema=loss_ema,
        loss_variance=loss_variance,
        grad_norm_ema=grad_norm_ema,
        lr_current=lr_current,
        scalr_alpha=scalr_alpha,
        gpu_temp=gpu_temp,
        gpu_util=gpu_util,
        vram_pct=vram_pct,
        cpu_util=cpu_util,
        step_latency_p50=0.0,
        step_latency_p95=0.0,
        throughput=0.0,
        epoch=epoch,
        regime=regime,
        timestamp=time.time(),
    )
