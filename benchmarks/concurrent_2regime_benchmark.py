"""2-Regime Concurrent GPU Benchmark: CUDA Streams, Single Process.

Runs two coupling regimes **concurrently** on a single GPU using
separate CUDA streams, then compares each regime's throughput and
latency against solo baselines to quantify interference.

Architecture
------------
Single Python process, one CUDA context, two CUDA streams:

* ``stream_mf``   : O(N) mean-field,      N = 1,048,576
* ``stream_full`` : O(N²) full pairwise,   N = 1,024

These represent the two extremes of coupling complexity — maximum N
with the cheapest coupling, and minimum N with the most expensive
coupling. Removing the middle O(N log N) sparse k-NN regime
isolates the memory-bandwidth-bound vs compute-bound interference
pattern, which Perplexity MCP research suggests should yield better
kernel overlap due to complementary resource profiles.

Protocol
--------
1. **Solo baselines** — each regime runs alone for 60 s with 30 s telemetry.
2. **Concurrent run** — both streams fire one step each per loop
   iteration for 5 minutes, with 30 s telemetry windows.
3. **Comparison** — throughput ratio (shared / solo), latency inflation,
   NaN detection, cross-comparison with 3-regime results.

Metrics Captured (per 30 s window, per regime)
----------------------------------------------
* Steps completed / throughput (osc·steps/s)
* Mean, std, P50, P95, P99 per-step latency (ms)
* Cumulative order parameter *r*
* Peak VRAM (GB)
* GPU temperature, clock, utilisation

Post-Run Analysis
-----------------
* Per-regime throughput ratio vs solo
* Latency inflation per regime
* NaN detection
* Comparison with 3-regime concurrent results
* Global GPU telemetry time-series

Outputs
-------
* ``Docs/test_and_benchmark_results/concurrent_2regime_benchmark.json``
* ``Docs/test_and_benchmark_results/concurrent_2regime_report.md``
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter  # noqa: E402
from prinet.core.propagation import KuramotoOscillator, OscillatorState  # noqa: E402

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

SEED: int = 42
DT: float = 0.01
COUPLING_K: float = 2.0

SOLO_DURATION_S: float = 60.0        # Solo baseline per regime
CONCURRENT_DURATION_S: float = 300.0  # 5 minutes concurrent
WINDOW_S: float = 30.0               # Telemetry interval
WARMUP_STEPS: int = 50               # Steps before timing
BATCH_STEPS: int = 50                # Steps per CUDA event batch (solo)

# Two-regime configuration: extremes only
REGIMES: list[dict[str, Any]] = [
    {
        "label": "O(N) Mean-Field",
        "short": "mf",
        "complexity": "O(N)",
        "coupling_mode": "mean_field",
        "N": 1_048_576,
        "mean_field": True,
        "sparse_k": None,
    },
    {
        "label": "O(N²) Full Pairwise",
        "short": "full",
        "complexity": "O(N²)",
        "coupling_mode": "full",
        "N": 1_024,
        "mean_field": False,
        "sparse_k": None,
    },
]

# RTX 4060 theoretical peaks
GPU_FP32_TFLOPS: float = 15.11
GPU_MEMBW_GBS: float = 272.0

RESULTS_DIR: Path = (
    Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
)


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _reset_gpu() -> None:
    """Free GPU caches and reset peak memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _flush_l2(device: torch.device) -> None:
    """Flush GPU L2 cache with a dummy allocation."""
    try:
        dummy = torch.empty(8_000_000, device=device, dtype=torch.float32)
        dummy.fill_(0.0)
        del dummy
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def _gpu_temp() -> Optional[int]:
    """Read GPU temperature via nvidia-smi.

    Returns:
        Temperature in Celsius, or None if unavailable.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _gpu_clock() -> Optional[int]:
    """Read current GPU SM clock (MHz) via nvidia-smi.

    Returns:
        Clock speed in MHz, or None if unavailable.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _gpu_util() -> Optional[int]:
    """Read GPU utilisation percentage via nvidia-smi.

    Returns:
        GPU utilisation percentage, or None if unavailable.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _gpu_vram_used_gb() -> Optional[float]:
    """Read current GPU VRAM usage in GB via nvidia-smi.

    Returns:
        VRAM used in GB, or None if unavailable.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip()) / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile from data list.

    Args:
        data: List of numeric values.
        pct: Percentile (0–100).

    Returns:
        Interpolated percentile value.
    """
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _coeff_var(data: list[float]) -> float:
    """Coefficient of variation (std / mean). 0 = perfect stability.

    Args:
        data: List of numeric values.

    Returns:
        CV value; 0 if insufficient data.
    """
    if not data or len(data) < 2:
        return 0.0
    m = sum(data) / len(data)
    if m == 0:
        return 0.0
    v = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return (v**0.5) / m


def _make_model(
    cfg: dict[str, Any],
    device: torch.device,
) -> KuramotoOscillator:
    """Create a KuramotoOscillator from regime config.

    Args:
        cfg: Regime configuration dict with keys N, coupling_mode, etc.
        device: CUDA device.

    Returns:
        Configured KuramotoOscillator instance.
    """
    kwargs: dict[str, Any] = {
        "n_oscillators": cfg["N"],
        "coupling_strength": COUPLING_K,
        "mean_field": cfg["mean_field"],
        "coupling_mode": cfg["coupling_mode"],
        "device": device,
    }
    if cfg["sparse_k"] is not None:
        kwargs["sparse_k"] = cfg["sparse_k"]
    return KuramotoOscillator(**kwargs)


def _estimate_flops_per_step(N: int, mode: str) -> float:
    """Rough FLOPs estimate per RK4 step.

    Args:
        N: Number of oscillators.
        mode: Coupling mode string.

    Returns:
        Estimated floating-point operations per step.
    """
    if mode == "mean_field":
        return 172.0 * N
    else:  # full
        return 80.0 * N * N


def _estimate_bytes_per_step(N: int, mode: str) -> float:
    """Rough bytes moved per RK4 step.

    Args:
        N: Number of oscillators.
        mode: Coupling mode string.

    Returns:
        Estimated bytes transferred per step.
    """
    if mode == "mean_field":
        return 248.0 * N
    else:  # full
        return 32.0 * N + 16.0 * N * N


def _is_nan(value: float) -> bool:
    """Check if a float is NaN.

    Args:
        value: Value to check.

    Returns:
        True if NaN.
    """
    return math.isnan(value)


# ══════════════════════════════════════════════════════════════════
# Solo Baseline Run
# ══════════════════════════════════════════════════════════════════


def run_solo(
    cfg: dict[str, Any],
    device: torch.device,
    duration: float = SOLO_DURATION_S,
    window: float = WINDOW_S,
) -> dict[str, Any]:
    """Run a single regime solo for baseline throughput measurement.

    Args:
        cfg: Regime configuration dict.
        device: CUDA device.
        duration: Total run time in seconds.
        window: Telemetry collection interval in seconds.

    Returns:
        Dict with ``summary`` and ``windows`` keys.
    """
    N = cfg["N"]
    label = cfg["label"]
    mode = cfg["coupling_mode"]
    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(f"\n{'─' * 70}")
    print(f"  SOLO: {label}  |  N = {N:,d}  |  {duration:.0f}s")
    print(f"{'─' * 70}")

    _reset_gpu()
    _flush_l2(device)

    model = _make_model(cfg, device)
    state = OscillatorState.create_random(N, device=device, seed=SEED)

    # Warm-up
    print(f"  Warm-up: {WARMUP_STEPS} steps ... ", end="", flush=True)
    s = state.clone()
    for _ in range(WARMUP_STEPS):
        s = model.step(s, dt=DT)
    torch.cuda.synchronize()
    _reset_gpu()
    print("done.")

    # Sustained solo run
    windows: list[dict[str, Any]] = []
    total_steps = 0
    wall_start = time.perf_counter()
    s = state.clone()
    window_idx = 0

    while True:
        elapsed_total = time.perf_counter() - wall_start
        if elapsed_total >= duration:
            break

        window_start = time.perf_counter()
        window_steps = 0
        step_times_ms: list[float] = []

        while True:
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            for _ in range(BATCH_STEPS):
                s = model.step(s, dt=DT)
            ev_e.record()
            torch.cuda.synchronize()
            batch_ms = ev_s.elapsed_time(ev_e)
            per_step_ms = batch_ms / BATCH_STEPS
            step_times_ms.extend([per_step_ms] * BATCH_STEPS)
            window_steps += BATCH_STEPS

            if time.perf_counter() - window_start >= window:
                break
            if time.perf_counter() - wall_start >= duration:
                break

        total_steps += window_steps
        window_elapsed = time.perf_counter() - window_start

        peak_vram = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram / 1024**3
        pct_vram = peak_vram / total_vram * 100

        r_val = kuramoto_order_parameter(s.phase).item()
        r_is_nan = _is_nan(r_val)

        temp_end = _gpu_temp()
        clock_end = _gpu_clock()
        gpu_util_val = _gpu_util()

        throughput = (N * window_steps) / window_elapsed
        avg_ms = sum(step_times_ms) / len(step_times_ms)
        std_ms = (
            sum((t - avg_ms) ** 2 for t in step_times_ms)
            / max(len(step_times_ms) - 1, 1)
        ) ** 0.5
        p50 = _percentile(step_times_ms, 50)
        p95 = _percentile(step_times_ms, 95)
        p99 = _percentile(step_times_ms, 99)

        window_data: dict[str, Any] = {
            "window": window_idx,
            "elapsed_s": round(time.perf_counter() - wall_start, 1),
            "window_steps": window_steps,
            "throughput": round(throughput, 0),
            "avg_step_ms": round(avg_ms, 4),
            "std_step_ms": round(std_ms, 4),
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "p99_ms": round(p99, 4),
            "order_param_r": round(r_val, 6) if not r_is_nan else "NaN",
            "r_is_nan": r_is_nan,
            "peak_vram_gb": round(peak_vram_gb, 3),
            "pct_vram": round(pct_vram, 1),
            "gpu_temp_c": temp_end,
            "gpu_clock_mhz": clock_end,
            "gpu_util_pct": gpu_util_val,
        }
        windows.append(window_data)

        r_str = f"{r_val:.4f}" if not r_is_nan else "NaN"
        temp_str = f"{temp_end}°C" if temp_end is not None else "—"
        print(
            f"  [{window_idx:>2}] {window_data['elapsed_s']:>5.0f}s  "
            f"steps={window_steps:>6,d}  "
            f"T={throughput:>14,.0f}  "
            f"avg={avg_ms:.3f}ms  "
            f"p95={p95:.3f}ms  "
            f"r={r_str}  "
            f"{temp_str}"
        )
        window_idx += 1

    total_elapsed = time.perf_counter() - wall_start

    throughputs = [w["throughput"] for w in windows]
    avg_tp = sum(throughputs) / len(throughputs) if throughputs else 0
    tp_cv = _coeff_var(throughputs)
    all_avgs = [w["avg_step_ms"] for w in windows]
    all_p95 = [w["p95_ms"] for w in windows]
    all_r = [
        w["order_param_r"]
        for w in windows
        if not w["r_is_nan"]
    ]
    nan_count = sum(1 for w in windows if w["r_is_nan"])

    summary: dict[str, Any] = {
        "label": label,
        "short": cfg["short"],
        "complexity": cfg["complexity"],
        "coupling_mode": mode,
        "N": N,
        "total_steps": total_steps,
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_throughput": round(avg_tp, 0),
        "throughput_cv": round(tp_cv, 4),
        "avg_step_ms": round(sum(all_avgs) / len(all_avgs), 4) if all_avgs else 0,
        "p50_ms": round(_percentile(all_avgs, 50), 4),
        "p95_ms": round(_percentile(all_p95, 95), 4),
        "p99_ms": round(_percentile(all_p95, 99), 4),
        "r_start": all_r[0] if all_r else "NaN",
        "r_end": all_r[-1] if all_r else "NaN",
        "r_mean": round(sum(all_r) / len(all_r), 6) if all_r else "NaN",
        "nan_count": nan_count,
    }

    print(f"\n  ── Solo {label} Summary ──")
    print(f"  Total steps:    {total_steps:,d} in {total_elapsed:.1f}s")
    print(f"  Avg throughput: {avg_tp:,.0f} osc·steps/s  (CV={tp_cv:.4f})")
    print(f"  Avg step:       {summary['avg_step_ms']:.3f} ms")
    if nan_count:
        print(f"  NaN r windows:  {nan_count}/{len(windows)}")

    del model, state, s
    _reset_gpu()

    return {"summary": summary, "windows": windows}


# ══════════════════════════════════════════════════════════════════
# Concurrent 2-Regime Run (CUDA Streams)
# ══════════════════════════════════════════════════════════════════


def run_concurrent(
    device: torch.device,
    duration: float = CONCURRENT_DURATION_S,
    window: float = WINDOW_S,
) -> dict[str, Any]:
    """Run both regimes concurrently on separate CUDA streams.

    Each loop iteration launches one step of each regime into its own
    stream (memory-bound mean-field first, then compute-bound full
    pairwise for optimal kernel overlap). Telemetry is collected every
    *window* seconds.

    Args:
        device: CUDA device.
        duration: Total concurrent run time in seconds.
        window: Telemetry collection interval in seconds.

    Returns:
        Dict keyed by regime short name with ``summary`` and ``windows``,
        plus ``global_windows`` for GPU-wide telemetry.
    """
    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(f"\n{'═' * 70}")
    print(f"  CONCURRENT 2-REGIME RUN  |  {duration:.0f}s  |  CUDA Streams")
    print(f"{'═' * 70}")

    _reset_gpu()
    _flush_l2(device)

    # ── Create streams ──
    stream_mf = torch.cuda.Stream(device)
    stream_full = torch.cuda.Stream(device)

    streams: dict[str, torch.cuda.Stream] = {
        "mf": stream_mf,
        "full": stream_full,
    }

    # ── Create models & states ──
    models: dict[str, KuramotoOscillator] = {}
    states: dict[str, OscillatorState] = {}
    for cfg in REGIMES:
        short = cfg["short"]
        models[short] = _make_model(cfg, device)
        states[short] = OscillatorState.create_random(
            cfg["N"], device=device, seed=SEED
        )

    # ── Warm-up each regime sequentially ──
    print(
        f"  Warm-up: {WARMUP_STEPS} steps each (sequential) ... ",
        end="",
        flush=True,
    )
    for cfg in REGIMES:
        short = cfg["short"]
        s = states[short].clone()
        for _ in range(WARMUP_STEPS):
            s = models[short].step(s, dt=DT)
        torch.cuda.synchronize()
    _reset_gpu()
    print("done.")

    # ── Re-create fresh states after warmup ──
    current_states: dict[str, OscillatorState] = {}
    for cfg in REGIMES:
        short = cfg["short"]
        current_states[short] = states[short].clone()

    # ── Tracking per regime ──
    per_regime: dict[str, dict[str, Any]] = {}
    for cfg in REGIMES:
        short = cfg["short"]
        per_regime[short] = {
            "cfg": cfg,
            "windows": [],
            "total_steps": 0,
            "window_steps": 0,
            "step_times_ms": [],
        }

    # ── Global tracking ──
    global_windows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    window_start = time.perf_counter()
    window_idx = 0

    print(f"\n  Running concurrent for {duration:.0f}s ...")
    print(
        f"  {'Time':>6s}  {'MF steps':>10s}  "
        f"{'Full steps':>10s}  {'GPU°C':>5s}  {'Util%':>5s}  {'VRAM GB':>8s}"
    )
    print(
        f"  {'─' * 6}  {'─' * 10}  "
        f"{'─' * 10}  {'─' * 5}  {'─' * 5}  {'─' * 8}"
    )

    while True:
        elapsed_total = time.perf_counter() - wall_start
        if elapsed_total >= duration:
            break

        # ── Launch one step per regime into its stream ──
        # Mean-field O(N) first — memory-bandwidth-bound, saturates bus
        ev_mf_s = torch.cuda.Event(enable_timing=True)
        ev_mf_e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream_mf):
            ev_mf_s.record(stream_mf)
            current_states["mf"] = models["mf"].step(
                current_states["mf"], dt=DT
            )
            ev_mf_e.record(stream_mf)

        # Full O(N²) second — compute-bound, can overlap on residual SMs
        ev_full_s = torch.cuda.Event(enable_timing=True)
        ev_full_e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream_full):
            ev_full_s.record(stream_full)
            current_states["full"] = models["full"].step(
                current_states["full"], dt=DT
            )
            ev_full_e.record(stream_full)

        # Synchronize both streams to get timing
        stream_mf.synchronize()
        stream_full.synchronize()

        # Record per-step times
        mf_ms = ev_mf_s.elapsed_time(ev_mf_e)
        full_ms = ev_full_s.elapsed_time(ev_full_e)

        per_regime["mf"]["step_times_ms"].append(mf_ms)
        per_regime["full"]["step_times_ms"].append(full_ms)

        per_regime["mf"]["window_steps"] += 1
        per_regime["full"]["window_steps"] += 1

        # ── Check if window has elapsed ──
        window_elapsed = time.perf_counter() - window_start
        if window_elapsed >= window or (
            time.perf_counter() - wall_start >= duration
        ):
            # Collect telemetry for this window
            peak_vram = torch.cuda.max_memory_allocated()
            peak_vram_gb = peak_vram / 1024**3
            pct_vram = peak_vram / total_vram * 100
            temp = _gpu_temp()
            clock = _gpu_clock()
            gpu_util_val = _gpu_util()
            vram_used = _gpu_vram_used_gb()
            total_wall_elapsed = time.perf_counter() - wall_start

            global_window: dict[str, Any] = {
                "window": window_idx,
                "elapsed_s": round(total_wall_elapsed, 1),
                "gpu_temp_c": temp,
                "gpu_clock_mhz": clock,
                "gpu_util_pct": gpu_util_val,
                "peak_vram_gb": round(peak_vram_gb, 3),
                "pct_vram": round(pct_vram, 1),
                "vram_used_gb": round(vram_used, 3) if vram_used else None,
            }

            # Per-regime telemetry
            regime_window_data: dict[str, dict[str, Any]] = {}
            for cfg in REGIMES:
                short = cfg["short"]
                pr = per_regime[short]
                ws = pr["window_steps"]
                times = pr["step_times_ms"]
                N = cfg["N"]

                if times:
                    throughput = (N * ws) / window_elapsed
                    avg_ms = sum(times) / len(times)
                    std_ms = (
                        sum((t - avg_ms) ** 2 for t in times)
                        / max(len(times) - 1, 1)
                    ) ** 0.5
                    p50 = _percentile(times, 50)
                    p95 = _percentile(times, 95)
                    p99 = _percentile(times, 99)
                else:
                    throughput = avg_ms = std_ms = p50 = p95 = p99 = 0.0

                r_val = kuramoto_order_parameter(
                    current_states[short].phase
                ).item()
                r_is_nan = _is_nan(r_val)

                rw: dict[str, Any] = {
                    "window": window_idx,
                    "elapsed_s": round(total_wall_elapsed, 1),
                    "window_steps": ws,
                    "throughput": round(throughput, 0),
                    "avg_step_ms": round(avg_ms, 4),
                    "std_step_ms": round(std_ms, 4),
                    "p50_ms": round(p50, 4),
                    "p95_ms": round(p95, 4),
                    "p99_ms": round(p99, 4),
                    "order_param_r": (
                        round(r_val, 6) if not r_is_nan else "NaN"
                    ),
                    "r_is_nan": r_is_nan,
                }
                regime_window_data[short] = rw
                pr["windows"].append(rw)
                pr["total_steps"] += ws
                pr["window_steps"] = 0
                pr["step_times_ms"] = []

            global_window["regimes"] = regime_window_data
            global_windows.append(global_window)

            # Live output
            temp_str = f"{temp}°C" if temp is not None else "—"
            util_str = f"{gpu_util_val}%" if gpu_util_val is not None else "—"
            vram_str = f"{peak_vram_gb:.2f}" if peak_vram_gb else "—"
            print(
                f"  {total_wall_elapsed:>5.0f}s  "
                f"{regime_window_data['mf']['window_steps']:>10,d}  "
                f"{regime_window_data['full']['window_steps']:>10,d}  "
                f"{temp_str:>5s}  {util_str:>5s}  {vram_str:>8s}"
            )

            # Print per-regime detail
            for cfg in REGIMES:
                short = cfg["short"]
                rw = regime_window_data[short]
                r_s = (
                    f"{rw['order_param_r']}"
                    if not rw["r_is_nan"]
                    else "NaN"
                )
                print(
                    f"    {cfg['label']:30s}  "
                    f"T={rw['throughput']:>14,.0f}  "
                    f"avg={rw['avg_step_ms']:.3f}ms  "
                    f"p95={rw['p95_ms']:.3f}ms  "
                    f"r={r_s}"
                )

            window_start = time.perf_counter()
            window_idx += 1

    total_elapsed = time.perf_counter() - wall_start

    # ── Aggregate per-regime summaries ──
    results: dict[str, Any] = {"global_windows": global_windows}
    for cfg in REGIMES:
        short = cfg["short"]
        pr = per_regime[short]
        wins = pr["windows"]
        total_steps = pr["total_steps"]

        throughputs = [w["throughput"] for w in wins]
        avg_tp = sum(throughputs) / len(throughputs) if throughputs else 0
        tp_cv = _coeff_var(throughputs)
        all_avgs = [w["avg_step_ms"] for w in wins]
        all_p95 = [w["p95_ms"] for w in wins]
        all_r = [
            w["order_param_r"]
            for w in wins
            if not w["r_is_nan"]
        ]
        nan_count = sum(1 for w in wins if w["r_is_nan"])

        summary: dict[str, Any] = {
            "label": cfg["label"],
            "short": short,
            "complexity": cfg["complexity"],
            "coupling_mode": cfg["coupling_mode"],
            "N": cfg["N"],
            "total_steps": total_steps,
            "total_elapsed_s": round(total_elapsed, 2),
            "avg_throughput": round(avg_tp, 0),
            "throughput_cv": round(tp_cv, 4),
            "avg_step_ms": (
                round(sum(all_avgs) / len(all_avgs), 4) if all_avgs else 0
            ),
            "p50_ms": round(_percentile(all_avgs, 50), 4),
            "p95_ms": round(_percentile(all_p95, 95), 4),
            "p99_ms": round(_percentile(all_p95, 99), 4),
            "r_start": all_r[0] if all_r else "NaN",
            "r_end": all_r[-1] if all_r else "NaN",
            "r_mean": (
                round(sum(all_r) / len(all_r), 6) if all_r else "NaN"
            ),
            "nan_count": nan_count,
        }
        results[short] = {"summary": summary, "windows": wins}

    # Print concurrent summary
    print(f"\n  ── Concurrent Run Summary ({total_elapsed:.1f}s) ──")
    for cfg in REGIMES:
        short = cfg["short"]
        s = results[short]["summary"]
        print(
            f"  {s['label']:30s}  "
            f"steps={s['total_steps']:>8,d}  "
            f"T={s['avg_throughput']:>14,.0f}  "
            f"CV={s['throughput_cv']:.4f}  "
            f"avg={s['avg_step_ms']:.3f}ms  "
            f"NaN={s['nan_count']}"
        )

    # Cleanup
    for short in list(models.keys()):
        del models[short]
    for short in list(states.keys()):
        del states[short]
    for short in list(current_states.keys()):
        del current_states[short]
    _reset_gpu()

    return results


# ══════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════


def generate_report(
    solo_results: dict[str, dict[str, Any]],
    concurrent_results: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    """Generate comprehensive Markdown report comparing solo vs concurrent.

    Args:
        solo_results: Dict keyed by regime short name with solo baselines.
        concurrent_results: Dict from ``run_concurrent()``.
        meta: Metadata dict (GPU info, config, etc.).

    Returns:
        Markdown report string.
    """
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    w("# PRINet 2-Regime Concurrent GPU Benchmark (CUDA Streams)")
    w()
    w("## Configuration")
    w()
    w(f"**Generated**: {ts}")
    w(f"**GPU**: {meta.get('gpu_name', 'N/A')} "
      f"({meta.get('gpu_vram_gb', 0):.1f} GB VRAM, "
      f"{meta.get('gpu_sms', 0)} SMs)")
    w(f"**PyTorch**: {meta.get('pytorch_version')}")
    w(f"**CUDA**: {meta.get('cuda_version', 'N/A')}")
    w(f"**Coupling**: K={meta.get('coupling_K')}, "
      f"dt={meta.get('dt')}, seed={meta.get('seed')}")
    w(f"**Solo Duration**: {meta.get('solo_duration_s', 60)}s per regime")
    w(f"**Concurrent Duration**: "
      f"{meta.get('concurrent_duration_s', 300)}s (5 minutes)")
    w(f"**Telemetry**: Every {meta.get('window_s', 30)}s")
    w(f"**GPU Theoretical**: FP32 {GPU_FP32_TFLOPS} TFLOPS, "
      f"BW {GPU_MEMBW_GBS} GB/s")
    w()
    w("### Protocol")
    w()
    w("Single Python process, one CUDA context, **two** streams (extremes only):")
    w()
    w("| Stream | Regime | Complexity | N | Resource Profile |")
    w("|--------|--------|-----------|---|-----------------|")
    w("| `stream_mf` | Mean-Field | O(N) | 1,048,576 | Memory-BW bound |")
    w("| `stream_full` | Full Pairwise | O(N²) | 1,024 | Compute bound |")
    w()
    w("This benchmark isolates the two extreme coupling regimes to measure")
    w("interference between a large memory-bandwidth-bound workload and a")
    w("small compute-bound workload, without the middle-weight O(N log N)")
    w("sparse k-NN regime present in the 3-regime benchmark.")
    w()
    w("1. Warm up each regime sequentially (50 steps each)")
    w("2. Launch memory-bound step first, compute-bound step second per iteration")
    w("3. Synchronize both streams after each iteration for timing")
    w("4. Collect per-regime + global telemetry every 30 s")
    w()
    w("---")
    w()

    # ── Executive Summary ──
    w("## Executive Summary")
    w()
    w("### Solo Baselines (60 s each)")
    w()
    w("| Metric | O(N) Mean-Field | O(N²) Full Pairwise |")
    w("|--------|----------------|---------------------|")

    shorts = ["mf", "full"]
    summary_fields: list[tuple[str, str, str]] = [
        ("N", "N", "{:,d}"),
        ("Total Steps", "total_steps", "{:,d}"),
        ("Avg Throughput (osc·s/s)", "avg_throughput", "{:,.0f}"),
        ("Throughput CV", "throughput_cv", "{:.4f}"),
        ("Avg Step (ms)", "avg_step_ms", "{:.3f}"),
        ("P95 Latency (ms)", "p95_ms", "{:.3f}"),
        ("Order Param r (final)", "r_end", "{}"),
        ("NaN Windows", "nan_count", "{}"),
    ]
    for row_label, key, fmt in summary_fields:
        vals = []
        for short in shorts:
            v = solo_results[short]["summary"].get(key, "—")
            if v == "—" or v is None:
                vals.append("—")
            elif isinstance(v, str):
                vals.append(v)
            else:
                vals.append(fmt.format(v))
        w(f"| {row_label} | {vals[0]} | {vals[1]} |")
    w()

    w("### Concurrent Run (5 min, 2 streams)")
    w()
    w("| Metric | O(N) Mean-Field | O(N²) Full Pairwise |")
    w("|--------|----------------|---------------------|")

    conc_fields: list[tuple[str, str, str]] = [
        ("Total Steps", "total_steps", "{:,d}"),
        ("Avg Throughput (osc·s/s)", "avg_throughput", "{:,.0f}"),
        ("Throughput CV", "throughput_cv", "{:.4f}"),
        ("Avg Step (ms)", "avg_step_ms", "{:.3f}"),
        ("P95 Latency (ms)", "p95_ms", "{:.3f}"),
        ("Order Param r (final)", "r_end", "{}"),
        ("NaN Windows", "nan_count", "{}"),
    ]
    for row_label, key, fmt in conc_fields:
        vals = []
        for short in shorts:
            v = concurrent_results[short]["summary"].get(key, "—")
            if v == "—" or v is None:
                vals.append("—")
            elif isinstance(v, str):
                vals.append(v)
            else:
                vals.append(fmt.format(v))
        w(f"| {row_label} | {vals[0]} | {vals[1]} |")
    w()

    # ── Key Metric: Throughput Ratio ──
    w("### Throughput Ratio (Shared / Solo)")
    w()
    w("| Regime | Solo Throughput | Shared Throughput "
      "| Ratio (shared/solo) | Interpretation |")
    w("|--------|----------------|-------------------"
      "|-------------------|---------------|")

    for cfg in REGIMES:
        short = cfg["short"]
        solo_tp = solo_results[short]["summary"]["avg_throughput"]
        shared_tp = concurrent_results[short]["summary"]["avg_throughput"]
        if solo_tp > 0:
            ratio = shared_tp / solo_tp
            if ratio > 0.90:
                interpretation = "minimal interference"
            elif ratio > 0.70:
                interpretation = "moderate interference"
            elif ratio > 0.50:
                interpretation = "heavy interference"
            else:
                interpretation = "severe contention"
        else:
            ratio = 0.0
            interpretation = "no baseline"
        w(
            f"| {cfg['label']} | {solo_tp:,.0f} | {shared_tp:,.0f} | "
            f"{ratio:.3f}× | {interpretation} |"
        )
    w()

    # ── Latency Inflation ──
    w("### Latency Inflation (Shared vs Solo)")
    w()
    w("| Regime | Solo Avg (ms) | Shared Avg (ms) | Inflation "
      "| Solo P95 (ms) | Shared P95 (ms) | P95 Inflation |")
    w("|--------|--------------|----------------|---------- "
      "|--------------|----------------|--------------|")

    for cfg in REGIMES:
        short = cfg["short"]
        solo_avg = solo_results[short]["summary"]["avg_step_ms"]
        shared_avg = concurrent_results[short]["summary"]["avg_step_ms"]
        solo_p95 = solo_results[short]["summary"]["p95_ms"]
        shared_p95 = concurrent_results[short]["summary"]["p95_ms"]
        avg_infl = (
            f"{shared_avg / solo_avg:.2f}×" if solo_avg > 0 else "—"
        )
        p95_infl = (
            f"{shared_p95 / solo_p95:.2f}×" if solo_p95 > 0 else "—"
        )
        w(
            f"| {cfg['label']} | {solo_avg:.3f} | {shared_avg:.3f} | "
            f"{avg_infl} | {solo_p95:.3f} | {shared_p95:.3f} | {p95_infl} |"
        )
    w()

    # ── NaN Detection ──
    w("### NaN Detection")
    w()
    any_nan = False
    for cfg in REGIMES:
        short = cfg["short"]
        solo_nan = solo_results[short]["summary"]["nan_count"]
        shared_nan = concurrent_results[short]["summary"]["nan_count"]
        if solo_nan or shared_nan:
            any_nan = True
            w(
                f"- **{cfg['label']}**: Solo NaN windows = {solo_nan}, "
                f"Shared NaN windows = {shared_nan}"
            )
    if not any_nan:
        w("No NaN order parameters detected in either regime (solo or shared).")
    w()

    # ── Cross-Comparison with 3-Regime Results ──
    w("### Cross-Comparison: 2-Regime vs 3-Regime Concurrent")
    w()
    w("Reference values from the 3-regime concurrent benchmark (same session/GPU):")
    w()
    w("| Metric | 3-Regime MF | 2-Regime MF | 3-Regime Full | 2-Regime Full |")
    w("|--------|-----------|-----------|-------------|-------------|")

    # Hard-coded 3-regime reference values from the prior run
    ref_3r: dict[str, dict[str, Any]] = {
        "mf": {
            "solo_tp": 171_576_254,
            "shared_tp": 55_175_325,
            "ratio": 0.322,
            "avg_ms": 12.108,
            "p95_ms": 16.377,
        },
        "full": {
            "solo_tp": 373_928,
            "shared_tp": 53_882,
            "ratio": 0.144,
            "avg_ms": 8.416,
            "p95_ms": 11.317,
        },
    }

    mf_2r_tp = concurrent_results["mf"]["summary"]["avg_throughput"]
    full_2r_tp = concurrent_results["full"]["summary"]["avg_throughput"]
    mf_2r_ratio = (
        mf_2r_tp / solo_results["mf"]["summary"]["avg_throughput"]
        if solo_results["mf"]["summary"]["avg_throughput"] > 0
        else 0
    )
    full_2r_ratio = (
        full_2r_tp / solo_results["full"]["summary"]["avg_throughput"]
        if solo_results["full"]["summary"]["avg_throughput"] > 0
        else 0
    )
    mf_2r_avg = concurrent_results["mf"]["summary"]["avg_step_ms"]
    full_2r_avg = concurrent_results["full"]["summary"]["avg_step_ms"]

    w(
        f"| Shared Throughput | {ref_3r['mf']['shared_tp']:,.0f} "
        f"| {mf_2r_tp:,.0f} "
        f"| {ref_3r['full']['shared_tp']:,.0f} "
        f"| {full_2r_tp:,.0f} |"
    )
    w(
        f"| TP Ratio (shared/solo) | {ref_3r['mf']['ratio']:.3f}× "
        f"| {mf_2r_ratio:.3f}× "
        f"| {ref_3r['full']['ratio']:.3f}× "
        f"| {full_2r_ratio:.3f}× |"
    )
    w(
        f"| Avg Latency (ms) | {ref_3r['mf']['avg_ms']:.3f} "
        f"| {mf_2r_avg:.3f} "
        f"| {ref_3r['full']['avg_ms']:.3f} "
        f"| {full_2r_avg:.3f} |"
    )
    w(
        f"| P95 Latency (ms) | {ref_3r['mf']['p95_ms']:.3f} "
        f"| {concurrent_results['mf']['summary']['p95_ms']:.3f} "
        f"| {ref_3r['full']['p95_ms']:.3f} "
        f"| {concurrent_results['full']['summary']['p95_ms']:.3f} |"
    )
    w()

    delta_mf = mf_2r_ratio - ref_3r["mf"]["ratio"]
    delta_full = full_2r_ratio - ref_3r["full"]["ratio"]
    w(
        f"**MF improvement** (2R vs 3R): "
        f"{delta_mf:+.3f} TP ratio "
        f"({'better' if delta_mf > 0 else 'worse' if delta_mf < 0 else 'same'})"
    )
    w(
        f"**Full improvement** (2R vs 3R): "
        f"{delta_full:+.3f} TP ratio "
        f"({'better' if delta_full > 0 else 'worse' if delta_full < 0 else 'same'})"
    )
    w()

    # ── Solo Baseline Detail ──
    for cfg in REGIMES:
        short = cfg["short"]
        s = solo_results[short]["summary"]
        wins = solo_results[short]["windows"]
        w(f"## Solo Baseline: {s['label']} — N = {s['N']:,d}")
        w()
        w(f"**Complexity**: {s['complexity']}  |  "
          f"**Coupling**: `{s['coupling_mode']}`")
        w(f"**Total**: {s['total_steps']:,d} steps in "
          f"{s['total_elapsed_s']:.1f}s")
        w()

        w("### Telemetry Time-Series (30s windows)")
        w()
        w("| Win | Elapsed | Steps | Throughput | Avg (ms) | P50 (ms) "
          "| P95 (ms) | P99 (ms) | r | Temp |")
        w("|-----|---------|-------|-----------|----------|---------- "
          "|----------|----------|---|------|")
        for wi in wins:
            temp_s = (
                f"{wi['gpu_temp_c']}°C"
                if wi["gpu_temp_c"] is not None
                else "—"
            )
            r_s = (
                f"{wi['order_param_r']}"
                if not wi["r_is_nan"]
                else "NaN"
            )
            w(
                f"| {wi['window']:>2} | {wi['elapsed_s']:.0f}s "
                f"| {wi['window_steps']:,d} | "
                f"{wi['throughput']:,.0f} | {wi['avg_step_ms']:.3f} | "
                f"{wi['p50_ms']:.3f} | {wi['p95_ms']:.3f} "
                f"| {wi['p99_ms']:.3f} | "
                f"{r_s} | {temp_s} |"
            )
        w()

    # ── Concurrent Detail ──
    w("## Concurrent Run Detail (5 min, 2 streams)")
    w()

    # Global telemetry
    w("### Global GPU Telemetry")
    w()
    w("| Win | Elapsed | Temp (°C) | Clock (MHz) | GPU Util % "
      "| VRAM (GB) | VRAM % |")
    w("|-----|---------|----------|------------|----------- "
      "|----------|--------|")
    for gw in concurrent_results["global_windows"]:
        temp_s = (
            f"{gw['gpu_temp_c']}"
            if gw["gpu_temp_c"] is not None
            else "—"
        )
        clk_s = (
            f"{gw['gpu_clock_mhz']}"
            if gw["gpu_clock_mhz"] is not None
            else "—"
        )
        util_s = (
            f"{gw['gpu_util_pct']}"
            if gw["gpu_util_pct"] is not None
            else "—"
        )
        w(
            f"| {gw['window']:>2} | {gw['elapsed_s']:.0f}s | "
            f"{temp_s} | {clk_s} | "
            f"{util_s} | {gw['peak_vram_gb']:.3f} | "
            f"{gw['pct_vram']:.1f}% |"
        )
    w()

    # Per-regime concurrent telemetry
    for cfg in REGIMES:
        short = cfg["short"]
        s = concurrent_results[short]["summary"]
        wins = concurrent_results[short]["windows"]
        w(f"### Concurrent: {s['label']} — N = {s['N']:,d}")
        w()
        w(f"**Total**: {s['total_steps']:,d} steps in "
          f"{s['total_elapsed_s']:.1f}s")
        w()

        w("| Win | Elapsed | Steps | Throughput | Avg (ms) | P50 (ms) "
          "| P95 (ms) | P99 (ms) | r |")
        w("|-----|---------|-------|-----------|----------|---------- "
          "|----------|----------|---|")
        for wi in wins:
            r_s = (
                f"{wi['order_param_r']}"
                if not wi["r_is_nan"]
                else "NaN"
            )
            w(
                f"| {wi['window']:>2} | {wi['elapsed_s']:.0f}s "
                f"| {wi['window_steps']:,d} | "
                f"{wi['throughput']:,.0f} | {wi['avg_step_ms']:.3f} | "
                f"{wi['p50_ms']:.3f} | {wi['p95_ms']:.3f} "
                f"| {wi['p99_ms']:.3f} | "
                f"{r_s} |"
            )
        w()

    # ── Key Insights ──
    w("## Key Insights")
    w()

    # Compute insights dynamically
    ratios: dict[str, float] = {}
    for cfg in REGIMES:
        short = cfg["short"]
        solo_tp = solo_results[short]["summary"]["avg_throughput"]
        shared_tp = concurrent_results[short]["summary"]["avg_throughput"]
        ratios[short] = shared_tp / solo_tp if solo_tp > 0 else 0

    most_affected = min(ratios, key=ratios.get)  # type: ignore[arg-type]
    least_affected = max(ratios, key=ratios.get)  # type: ignore[arg-type]

    most_label = next(
        c["label"] for c in REGIMES if c["short"] == most_affected
    )
    least_label = next(
        c["label"] for c in REGIMES if c["short"] == least_affected
    )

    w(
        f"1. **Most affected regime**: {most_label} — "
        f"throughput ratio {ratios[most_affected]:.3f}× "
        f"({(1 - ratios[most_affected]) * 100:.1f}% interference)."
    )
    w()
    w(
        f"2. **Least affected regime**: {least_label} — "
        f"throughput ratio {ratios[least_affected]:.3f}× "
        f"({(1 - ratios[least_affected]) * 100:.1f}% interference)."
    )
    w()

    w(
        "3. **2-stream vs 3-stream hypothesis**: Removing the middle-weight "
        "O(N log N) "
    )
    w(
        "   sparse k-NN stream should reduce memory bus contention. With only "
        "a memory-"
    )
    w(
        "   bandwidth-bound kernel (O(N) at 1M) and a compute-bound kernel "
        "(O(N²) at 1K), "
    )
    w(
        "   the complementary resource profiles allow better kernel overlap on "
        "Ada Lovelace SMs."
    )
    w()

    # Compare with 3-regime
    w(
        f"4. **2R vs 3R comparison**: MF TP ratio "
        f"{mf_2r_ratio:.3f}× (2R) vs {ref_3r['mf']['ratio']:.3f}× (3R) "
        f"= {delta_mf:+.3f}. "
        f"Full TP ratio {full_2r_ratio:.3f}× (2R) vs "
        f"{ref_3r['full']['ratio']:.3f}× (3R) = {delta_full:+.3f}."
    )
    w()

    any_shared_nan = any(
        concurrent_results[c["short"]]["summary"]["nan_count"] > 0
        for c in REGIMES
    )
    if any_shared_nan:
        w(
            "5. **NaN order parameters detected** in concurrent run — "
            "investigate numerical stability."
        )
    else:
        w(
            "5. **No NaN order parameters** introduced by concurrent "
            "execution — numerical stability preserved."
        )
    w()

    w(
        "6. **Latency inflation** remains the primary cost of concurrency. "
        "Per-step "
    )
    w(
        "   latency increases when both workloads compete for GPU resources, "
        "but with "
    )
    w(
        "   only 2 streams the scheduler has fewer queues to arbitrate."
    )
    w()

    # Total throughput comparison
    total_shared = sum(
        concurrent_results[c["short"]]["summary"]["avg_throughput"]
        for c in REGIMES
    )
    max_solo = max(
        solo_results[c["short"]]["summary"]["avg_throughput"]
        for c in REGIMES
    )
    w(
        f"7. **Aggregate concurrent throughput**: "
        f"{total_shared:,.0f} osc·s/s total "
    )
    w(
        f"   across both regimes vs {max_solo:,.0f} osc·s/s best solo. "
        f"Ratio: {total_shared / max_solo:.2f}×."
    )
    w()

    w("---")
    w()
    w(
        f"*Report generated by `concurrent_2regime_benchmark.py` on {ts}*"
    )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run solo baselines then concurrent 2-regime benchmark."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    print(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    print(
        "║  PRINet 2-Regime Concurrent GPU Benchmark (CUDA Streams)    ║"
    )
    print(
        "║  O(N) + O(N²) — Solo Baselines + 5 min Concurrent          ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════╝"
    )
    print(
        f"\nGPU: {props.name}, "
        f"{props.total_memory / 1024**3:.1f} GB VRAM, "
        f"{props.multi_processor_count} SMs"
    )
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Config: K={COUPLING_K}, dt={DT}, seed={SEED}")
    print(f"Solo: {SOLO_DURATION_S:.0f}s per regime")
    print(f"Concurrent: {CONCURRENT_DURATION_S:.0f}s total")
    print(f"Telemetry: every {WINDOW_S:.0f}s")
    print()

    for i, cfg in enumerate(REGIMES):
        print(
            f"  [{i + 1}] {cfg['label']:30s} "
            f"N = {cfg['N']:>10,d}  ({cfg['short']})"
        )
    print()

    meta: dict[str, Any] = {
        "benchmark": "2-Regime Concurrent GPU Benchmark (CUDA Streams)",
        "gpu_name": props.name,
        "gpu_vram_gb": round(props.total_memory / 1024**3, 1),
        "gpu_sms": props.multi_processor_count,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "coupling_K": COUPLING_K,
        "dt": DT,
        "seed": SEED,
        "solo_duration_s": SOLO_DURATION_S,
        "concurrent_duration_s": CONCURRENT_DURATION_S,
        "window_s": WINDOW_S,
        "warmup_steps": WARMUP_STEPS,
        "batch_steps": BATCH_STEPS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Solo Baselines
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PHASE 1: SOLO BASELINES (60s each)")
    print("═" * 70)

    solo_results: dict[str, dict[str, Any]] = {}
    for cfg in REGIMES:
        result = run_solo(cfg, device)
        solo_results[cfg["short"]] = result
        # Brief cooldown between regimes
        print("\n  Cooldown: 5s ...")
        _reset_gpu()
        time.sleep(5)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Concurrent Run
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PHASE 2: CONCURRENT 2-REGIME RUN (5 min)")
    print("═" * 70)

    _reset_gpu()
    time.sleep(5)  # Cooldown before concurrent
    concurrent_results = run_concurrent(device)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  PHASE 3: COMPARISON (Solo vs Concurrent)")
    print("═" * 70)

    print(
        f"\n  {'Regime':<30s} {'Solo TP':>14s} {'Shared TP':>14s} "
        f"{'Ratio':>8s} {'Solo ms':>10s} {'Shared ms':>10s} {'Infl':>6s}"
    )
    print(
        f"  {'─' * 30} {'─' * 14} {'─' * 14} "
        f"{'─' * 8} {'─' * 10} {'─' * 10} {'─' * 6}"
    )
    for cfg in REGIMES:
        short = cfg["short"]
        solo_tp = solo_results[short]["summary"]["avg_throughput"]
        shared_tp = concurrent_results[short]["summary"]["avg_throughput"]
        ratio = shared_tp / solo_tp if solo_tp > 0 else 0
        solo_ms = solo_results[short]["summary"]["avg_step_ms"]
        shared_ms = concurrent_results[short]["summary"]["avg_step_ms"]
        infl = shared_ms / solo_ms if solo_ms > 0 else 0
        print(
            f"  {cfg['label']:<30s} "
            f"{solo_tp:>14,.0f} {shared_tp:>14,.0f} {ratio:>8.3f}× "
            f"{solo_ms:>10.3f} {shared_ms:>10.3f} {infl:>6.2f}×"
        )

    # ═══════════════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════════════
    full_results: dict[str, Any] = {
        "meta": meta,
        "solo": {short: solo_results[short] for short in solo_results},
        "concurrent": concurrent_results,
    }

    json_path = RESULTS_DIR / "concurrent_2regime_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n✓ JSON: {json_path}")

    # ── Generate report ──
    report = generate_report(solo_results, concurrent_results, meta)
    report_path = RESULTS_DIR / "concurrent_2regime_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Report: {report_path}")

    print("\n✓ 2-Regime Concurrent GPU Benchmark complete.")


if __name__ == "__main__":
    main()
