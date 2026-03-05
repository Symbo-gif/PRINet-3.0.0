"""torch.profiler wrapper utilities for PRINet training-loop analysis.

Provides :func:`profile_training_loop` and :class:`PRINetProfiler` for
bottleneck profiling of PRINet models. Generates Chrome-trace JSON,
a plain-text table summary, and a structured :class:`ProfileReport`.

Typical usage::

    from prinet.utils.profiler import profile_training_loop

    report = profile_training_loop(
        model=my_model,
        dataloader=train_loader,
        n_steps=50,
        out_dir="profiler_output",
    )
    print(report.top_ops_table)

"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# ProfileReport
# ---------------------------------------------------------------------------


@dataclass
class ProfileReport:
    """Structured summary of a profiling run.

    Attributes:
        total_wall_ms: Total wall-clock time over profiled steps.
        n_steps: Number of steps profiled.
        avg_step_ms: Average wall time per step.
        top_ops: List of ``(op_name, self_cpu_ms, self_cuda_ms)`` tuples,
            sorted by total time descending.
        top_ops_table: Plain-text table of *top_ops*.
        bottleneck_op: Name of the single most expensive operator.
        json_trace_path: Path to Chrome-trace JSON, or ``None``.
        raw: Raw nested-dict data for downstream consumption.
    """

    total_wall_ms: float = 0.0
    n_steps: int = 0
    avg_step_ms: float = 0.0
    top_ops: list[tuple[str, float, float]] = field(default_factory=list)
    top_ops_table: str = ""
    bottleneck_op: str = ""
    json_trace_path: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PRINetProfiler context manager
# ---------------------------------------------------------------------------


class PRINetProfiler:
    """Context-manager wrapper around :class:`torch.profiler.profile`.

    Handles warm-up steps (excluded from report), active profiling steps,
    and optional Chrome-trace export.

    Args:
        out_dir: Directory for Chrome-trace JSON output.
            Pass ``None`` to skip file output.
        warmup_steps: Number of initial steps to skip before recording.
        active_steps: Number of steps to record.
        record_shapes: Forward tensor-shape information to profiler.
        profile_memory: Include memory allocation profiling.
        with_flops: Estimate FLOPs per operator (requires *record_shapes*).

    Example:
        >>> profiler = PRINetProfiler(out_dir=None, active_steps=5)
        >>> with profiler:
        ...     for _ in range(5):
        ...         y = model(x)
        ...         profiler.step()
        >>> report = profiler.report()
    """

    def __init__(
        self,
        out_dir: Optional[str | Path] = None,
        warmup_steps: int = 2,
        active_steps: int = 20,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_flops: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        self._prof: Any = None
        self._start_wall: float = 0.0
        self._end_wall: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PRINetProfiler":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=1,
        )
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        self._prof.__enter__()
        self._start_wall = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end_wall = time.perf_counter()
        if self._prof is not None:
            self._prof.__exit__(*args)

    def step(self) -> None:
        """Signal the profiler that one step has completed."""
        if self._prof is not None:
            self._prof.step()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, top_n: int = 20) -> ProfileReport:
        """Build a :class:`ProfileReport` from the completed profile.

        Args:
            top_n: Number of top operators to include in the table.

        Returns:
            Populated :class:`ProfileReport`.
        """
        if self._prof is None:
            return ProfileReport()

        total_wall_ms = (self._end_wall - self._start_wall) * 1000.0
        n_steps = self.active_steps

        # Sort key events by total time
        key_avgs = self._prof.key_averages()
        sorted_avgs = sorted(
            key_avgs,
            key=lambda e: (
                getattr(e, "self_cpu_time_total", 0)
                + getattr(
                    e, "self_cuda_time_total", getattr(e, "self_device_time_total", 0)
                )
            ),
            reverse=True,
        )[:top_n]

        top_ops: list[tuple[str, float, float]] = []
        lines = [
            f"{'Op':<55} {'CPU ms':>10} {'CUDA ms':>10}",
            "-" * 78,
        ]
        for evt in sorted_avgs:
            cpu_ms = getattr(evt, "self_cpu_time_total", 0) / 1000.0
            cuda_raw = getattr(
                evt, "self_cuda_time_total", getattr(evt, "self_device_time_total", 0)
            )
            cuda_ms = (cuda_raw or 0) / 1000.0
            top_ops.append((evt.key, cpu_ms, cuda_ms))
            lines.append(f"{evt.key:<55} {cpu_ms:>10.3f} {cuda_ms:>10.3f}")

        table = "\n".join(lines)
        bottleneck = top_ops[0][0] if top_ops else ""

        # Optional trace export
        trace_path: Optional[str] = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            trace_file = self.out_dir / "prinet_trace.json"
            self._prof.export_chrome_trace(str(trace_file))
            trace_path = str(trace_file)

        raw: dict[str, Any] = {
            "total_wall_ms": total_wall_ms,
            "n_steps": n_steps,
            "avg_step_ms": total_wall_ms / max(n_steps, 1),
            "top_ops": [{"op": op, "cpu_ms": c, "cuda_ms": g} for op, c, g in top_ops],
        }

        return ProfileReport(
            total_wall_ms=total_wall_ms,
            n_steps=n_steps,
            avg_step_ms=total_wall_ms / max(n_steps, 1),
            top_ops=top_ops,
            top_ops_table=table,
            bottleneck_op=bottleneck,
            json_trace_path=trace_path,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def profile_training_loop(
    model: nn.Module,
    dataloader: Any,
    n_steps: int = 50,
    out_dir: Optional[str | Path] = None,
    warmup_steps: int = 5,
    device: Optional[torch.device] = None,
    loss_fn: Optional[Any] = None,
    top_n: int = 20,
) -> ProfileReport:
    """Profile a single epoch of training on *dataloader*.

    Runs the model forward (and backward if *loss_fn* is supplied) for
    up to *n_steps* batches, then returns a :class:`ProfileReport`.

    Args:
        model: The model to profile.
        dataloader: Iterable yielding ``(inputs, labels)`` batches.
        n_steps: Maximum number of steps to profile (active steps).
        out_dir: Directory for Chrome-trace export; ``None`` to skip.
        warmup_steps: Steps excluded from the report (JIT warm-up).
        device: Target device; auto-detected if ``None``.
        loss_fn: Optional loss callable ``(logits, labels) → scalar``.
            If provided, backward pass is included in profiling.
        top_n: Number of top operators in the report table.

    Returns:
        :class:`ProfileReport` with timing breakdown.

    Example:
        >>> import torchvision, torchvision.transforms as T
        >>> ds = torchvision.datasets.FakeData(
        ...     size=128, image_size=(3, 32, 32), num_classes=10,
        ...     transform=T.ToTensor(),
        ... )
        >>> dl = torch.utils.data.DataLoader(ds, batch_size=32)
        >>> from prinet.nn.hybrid import HybridPRINetV2
        >>> m = HybridPRINetV2(n_input=64, n_classes=10, use_conv_stem=True)
        >>> report = profile_training_loop(m, dl, n_steps=4, warmup_steps=1)
        >>> assert report.avg_step_ms >= 0
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    active = min(
        n_steps, len(dataloader) if hasattr(dataloader, "__len__") else n_steps
    )
    profiler = PRINetProfiler(
        out_dir=out_dir,
        warmup_steps=warmup_steps,
        active_steps=active,
        record_shapes=True,
        profile_memory=False,
    )

    data_iter = iter(dataloader)
    total_steps = warmup_steps + active

    with profiler:
        for _ in range(total_steps):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, labels = next(data_iter)

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda"), dtype=torch.float16
            ):
                logits = model(inputs)
                if loss_fn is not None:
                    loss = loss_fn(logits, labels)
                    loss.backward()

            profiler.step()

    return profiler.report(top_n=top_n)
