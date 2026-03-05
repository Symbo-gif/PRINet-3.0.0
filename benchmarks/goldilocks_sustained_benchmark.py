"""Goldilocks Sustained GPU Benchmark: O(N) vs O(N log N) vs O(N²).

Each coupling regime is run at its empirically-determined optimal
("Goldilocks") throughput size for a sustained **5 minutes**, collecting
time-series telemetry every 30 seconds.

Goldilocks Sizes (from prior benchmarks)
-----------------------------------------
* O(N)         mean_field   N = 1,048,576    (1 M oscillators)
* O(N log N)   sparse_knn   N =    65,536    (65 K oscillators)
* O(N²)        full         N =     1,024    (1 K oscillators)

Hardware Target
---------------
* GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM, 24 SMs, Ada Lovelace
* Theoretical: FP32 ~15.11 TFLOPS, BW ~272 GB/s

Metrics Captured (per 30-second window)
---------------------------------------
* Steps completed / throughput (osc·steps/s)
* Mean, std, P50, P95, P99 per-step latency (ms)
* Cumulative order parameter r
* Peak VRAM (GB / %)
* GPU temperature (°C), clock (MHz)
* Coefficient of variation (throughput stability)

Post-Run Analysis
-----------------
* Throughput time-series & stability
* Latency percentile profiles
* Thermal trajectory over 5 min
* Regime comparison table
* Goldilocks characterisation

Outputs
-------
* ``Docs/test_and_benchmark_results/goldilocks_sustained_benchmark.json``
* ``Docs/test_and_benchmark_results/goldilocks_sustained_report.md``
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

DURATION_S: float = 300.0          # 5 minutes per regime
WINDOW_S: float = 30.0             # Telemetry interval
WARMUP_STEPS: int = 50             # Steps before timing begins
BATCH_STEPS: int = 50              # Steps between sync points (amortise event overhead)

# Goldilocks configurations
REGIMES: list[dict[str, Any]] = [
    {
        "label": "O(N) Mean-Field",
        "complexity": "O(N)",
        "coupling_mode": "mean_field",
        "N": 1_048_576,
        "mean_field": True,
        "sparse_k": None,
    },
    {
        "label": "O(N log N) Sparse k-NN",
        "complexity": "O(N log N)",
        "coupling_mode": "sparse_knn",
        "N": 65_536,
        "mean_field": False,
        "sparse_k": None,  # defaults to ceil(log2(N)) = 16
    },
    {
        "label": "O(N²) Full Pairwise",
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
    """Flush GPU L2 cache."""
    try:
        dummy = torch.empty(8_000_000, device=device, dtype=torch.float32)
        dummy.fill_(0.0)
        del dummy
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def _gpu_temp() -> Optional[int]:
    """Read GPU temperature via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _gpu_clock() -> Optional[int]:
    """Read current GPU SM clock (MHz)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile from sorted data."""
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
    """Coefficient of variation (std / mean). 0 = perfect stability."""
    if not data or len(data) < 2:
        return 0.0
    m = sum(data) / len(data)
    if m == 0:
        return 0.0
    v = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return (v ** 0.5) / m


def _make_model(
    cfg: dict[str, Any], device: torch.device,
) -> KuramotoOscillator:
    """Create a KuramotoOscillator from regime config."""
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

    Mean-field O(N):   172·N   (4 derivative evals × ~40N + combine ~12N)
    Sparse k-NN:       172·N + 4·N·k·(~15)  (k-NN sort + local coupling)
    Full O(N²):        4·N²·~20 + overhead ≈ 80·N²
    """
    if mode == "mean_field":
        return 172.0 * N
    elif mode == "sparse_knn":
        k = max(1, math.ceil(math.log2(N)))
        return 172.0 * N + 4 * N * k * 15
    else:  # full
        return 80.0 * N * N


def _estimate_bytes_per_step(N: int, mode: str) -> float:
    """Rough bytes moved per RK4 step.

    Mean-field: 248·N
    Sparse:     248·N + 4·N·k·12
    Full:       4·(2·N·4 + N²·4) ≈ 32N + 16N²
    """
    if mode == "mean_field":
        return 248.0 * N
    elif mode == "sparse_knn":
        k = max(1, math.ceil(math.log2(N)))
        return 248.0 * N + 4 * N * k * 12
    else:  # full
        return 32.0 * N + 16.0 * N * N


# ══════════════════════════════════════════════════════════════════
# Sustained Run for One Regime
# ══════════════════════════════════════════════════════════════════


def run_sustained(
    cfg: dict[str, Any],
    device: torch.device,
    duration: float = DURATION_S,
    window: float = WINDOW_S,
) -> dict[str, Any]:
    """Run a single regime for *duration* seconds, collecting telemetry.

    Args:
        cfg: Regime configuration dict.
        device: CUDA device.
        duration: Total run time in seconds.
        window: Telemetry collection interval in seconds.

    Returns:
        Dict with time-series windows and aggregate statistics.
    """
    N = cfg["N"]
    label = cfg["label"]
    mode = cfg["coupling_mode"]
    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(f"\n{'═' * 70}")
    print(f"  {label}  |  N = {N:,d}  |  {duration:.0f}s sustained")
    print(f"{'═' * 70}")

    _reset_gpu()
    _flush_l2(device)

    # Create model & initial state
    model = _make_model(cfg, device)
    state = OscillatorState.create_random(N, device=device, seed=SEED)

    # ── Warm-up ──
    print(f"  Warm-up: {WARMUP_STEPS} steps ... ", end="", flush=True)
    s = state.clone()
    for _ in range(WARMUP_STEPS):
        s = model.step(s, dt=DT)
    torch.cuda.synchronize()
    _reset_gpu()
    print("done.")

    # ── Sustained run ──
    windows: list[dict[str, Any]] = []
    total_steps = 0
    wall_start = time.perf_counter()
    s = state.clone()

    window_idx = 0

    while True:
        elapsed_total = time.perf_counter() - wall_start
        if elapsed_total >= duration:
            break

        # ── Start of window ──
        window_start = time.perf_counter()
        window_steps = 0
        step_times_ms: list[float] = []

        # Snapshot telemetry at window start
        temp_start = _gpu_temp()
        clock_start = _gpu_clock()

        while True:
            # Batch of steps with CUDA event timing
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

            # Check if window expired
            if time.perf_counter() - window_start >= window:
                break
            # Check if total duration expired
            if time.perf_counter() - wall_start >= duration:
                break

        total_steps += window_steps
        window_elapsed = time.perf_counter() - window_start

        # Telemetry
        peak_vram = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram / 1024 ** 3
        pct_vram = peak_vram / total_vram * 100

        r_val = kuramoto_order_parameter(s.phase).item()

        temp_end = _gpu_temp()
        clock_end = _gpu_clock()

        throughput = (N * window_steps) / window_elapsed
        avg_ms = sum(step_times_ms) / len(step_times_ms)
        std_ms = (sum((t - avg_ms) ** 2 for t in step_times_ms) / max(len(step_times_ms) - 1, 1)) ** 0.5

        p50 = _percentile(step_times_ms, 50)
        p95 = _percentile(step_times_ms, 95)
        p99 = _percentile(step_times_ms, 99)

        # FLOPs / bandwidth estimates
        flops_per_step = _estimate_flops_per_step(N, mode)
        bytes_per_step = _estimate_bytes_per_step(N, mode)
        achieved_tflops = (flops_per_step * window_steps) / window_elapsed / 1e12
        achieved_bw_gbs = (bytes_per_step * window_steps) / window_elapsed / 1e9
        pct_peak_flops = achieved_tflops / GPU_FP32_TFLOPS * 100
        pct_peak_bw = achieved_bw_gbs / GPU_MEMBW_GBS * 100

        window_data: dict[str, Any] = {
            "window": window_idx,
            "elapsed_s": round(time.perf_counter() - wall_start, 1),
            "window_steps": window_steps,
            "window_s": round(window_elapsed, 2),
            "throughput": round(throughput, 0),
            "avg_step_ms": round(avg_ms, 4),
            "std_step_ms": round(std_ms, 4),
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "p99_ms": round(p99, 4),
            "order_param_r": round(r_val, 6),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "pct_vram": round(pct_vram, 1),
            "gpu_temp_c": temp_end,
            "gpu_clock_mhz": clock_end,
            "achieved_tflops": round(achieved_tflops, 4),
            "pct_peak_compute": round(pct_peak_flops, 2),
            "achieved_bw_gbs": round(achieved_bw_gbs, 1),
            "pct_peak_bw": round(pct_peak_bw, 1),
        }
        windows.append(window_data)

        # Live output
        temp_str = f"{temp_end}°C" if temp_end is not None else "—"
        clk_str = f"{clock_end}MHz" if clock_end is not None else "—"
        print(
            f"  [{window_idx:>2}] {window_data['elapsed_s']:>5.0f}s  "
            f"steps={window_steps:>6,d}  "
            f"T={throughput:>14,.0f}  "
            f"avg={avg_ms:.3f}ms  "
            f"p95={p95:.3f}ms  "
            f"r={r_val:.4f}  "
            f"VRAM={peak_vram_gb:.2f}GB({pct_vram:.0f}%)  "
            f"{temp_str}  {clk_str}"
        )
        window_idx += 1

    total_elapsed = time.perf_counter() - wall_start

    # ── Aggregate statistics ──
    throughputs = [w["throughput"] for w in windows]
    avg_tp = sum(throughputs) / len(throughputs)
    all_avgs = [w["avg_step_ms"] for w in windows]
    all_p95 = [w["p95_ms"] for w in windows]
    all_r = [w["order_param_r"] for w in windows]
    temps = [w["gpu_temp_c"] for w in windows if w["gpu_temp_c"] is not None]

    # Throughput stability: coefficient of variation
    tp_cv = _coeff_var(throughputs)

    # Thermal analysis
    if temps:
        temp_min = min(temps)
        temp_max = max(temps)
        temp_mean = sum(temps) / len(temps)
        # First-half vs second-half throughput (thermal throttle detection)
        half = len(throughputs) // 2
        first_half_tp = sum(throughputs[:half]) / max(half, 1)
        second_half_tp = sum(throughputs[half:]) / max(len(throughputs) - half, 1)
        tp_drift_pct = (second_half_tp - first_half_tp) / max(first_half_tp, 1) * 100
    else:
        temp_min = temp_max = temp_mean = 0
        first_half_tp = second_half_tp = avg_tp
        tp_drift_pct = 0.0

    # Order parameter trend
    r_start = all_r[0] if all_r else 0.0
    r_end = all_r[-1] if all_r else 0.0
    r_mean = sum(all_r) / len(all_r) if all_r else 0.0

    # Compute estimates (aggregate)
    flops_per_step = _estimate_flops_per_step(N, mode)
    bytes_per_step = _estimate_bytes_per_step(N, mode)
    total_flops = flops_per_step * total_steps
    total_bytes = bytes_per_step * total_steps
    avg_tflops = total_flops / total_elapsed / 1e12
    avg_bw_gbs = total_bytes / total_elapsed / 1e9
    arith_intensity = flops_per_step / bytes_per_step

    summary = {
        "label": label,
        "complexity": cfg["complexity"],
        "coupling_mode": mode,
        "N": N,
        "total_steps": total_steps,
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_throughput": round(avg_tp, 0),
        "throughput_cv": round(tp_cv, 4),
        "throughput_stability": "excellent" if tp_cv < 0.02 else "good" if tp_cv < 0.05 else "moderate" if tp_cv < 0.10 else "unstable",
        "avg_step_ms": round(sum(all_avgs) / len(all_avgs), 4),
        "global_p50_ms": round(_percentile(all_avgs, 50), 4),
        "global_p95_ms": round(_percentile(all_p95, 95), 4),
        "global_p99_ms": round(_percentile(all_p95, 99), 4),
        "r_start": round(r_start, 6),
        "r_end": round(r_end, 6),
        "r_mean": round(r_mean, 6),
        "peak_vram_gb": round(max(w["peak_vram_gb"] for w in windows), 3),
        "pct_vram": round(max(w["pct_vram"] for w in windows), 1),
        "temp_min_c": temp_min,
        "temp_max_c": temp_max,
        "temp_mean_c": round(temp_mean, 1) if temps else None,
        "first_half_throughput": round(first_half_tp, 0),
        "second_half_throughput": round(second_half_tp, 0),
        "throughput_drift_pct": round(tp_drift_pct, 2),
        "thermal_throttle": tp_drift_pct < -5.0,
        "achieved_tflops": round(avg_tflops, 4),
        "pct_peak_compute": round(avg_tflops / GPU_FP32_TFLOPS * 100, 2),
        "achieved_bw_gbs": round(avg_bw_gbs, 1),
        "pct_peak_bw": round(avg_bw_gbs / GPU_MEMBW_GBS * 100, 1),
        "arith_intensity": round(arith_intensity, 2),
    }

    print(f"\n  ── {label} Summary ──")
    print(f"  Total steps:    {total_steps:,d} in {total_elapsed:.1f}s")
    print(f"  Avg throughput: {avg_tp:,.0f} osc·steps/s  (CV={tp_cv:.4f})")
    print(f"  Avg step:       {summary['avg_step_ms']:.3f} ms  P95={summary['global_p95_ms']:.3f} ms")
    print(f"  Order param:    r_start={r_start:.4f} → r_end={r_end:.4f}")
    print(f"  VRAM:           {summary['peak_vram_gb']:.2f} GB ({summary['pct_vram']:.0f}%)")
    if temps:
        print(f"  Temp:           {temp_min}–{temp_max}°C (mean {temp_mean:.0f}°C)")
    print(f"  TP drift:       {tp_drift_pct:+.2f}% (2nd half vs 1st)  {'THROTTLE' if summary['thermal_throttle'] else 'stable'}")
    print(f"  Compute:        {avg_tflops:.4f} TFLOPS ({avg_tflops/GPU_FP32_TFLOPS*100:.1f}%)")
    print(f"  Bandwidth:      {avg_bw_gbs:.1f} GB/s ({avg_bw_gbs/GPU_MEMBW_GBS*100:.1f}%)")

    del model, state, s
    _reset_gpu()

    return {
        "summary": summary,
        "windows": windows,
    }


# ══════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════


def generate_report(
    results: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    """Generate comprehensive Markdown report."""
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    w("# PRINet Goldilocks Sustained GPU Benchmark")
    w()
    w("## Configuration")
    w()
    w(f"**Generated**: {ts}")
    w(f"**GPU**: {meta.get('gpu_name', 'N/A')} ({meta.get('gpu_vram_gb', 0):.1f} GB VRAM)")
    w(f"**PyTorch**: {meta.get('pytorch_version')}")
    w(f"**Coupling**: K={meta.get('coupling_K')}, dt={meta.get('dt')}, seed={meta.get('seed')}")
    w(f"**Duration**: {meta.get('duration_s', 300)}s per regime (5 minutes)")
    w(f"**Telemetry**: Every {meta.get('window_s', 30)}s")
    w(f"**GPU Theoretical**: FP32 {GPU_FP32_TFLOPS} TFLOPS, BW {GPU_MEMBW_GBS} GB/s")
    w()
    w("Each coupling regime is run at its empirically-determined **Goldilocks** size —")
    w("the N that maximises throughput on this hardware — for a sustained 5-minute burn.")
    w()
    w("---")
    w()

    # ── Executive Summary Table ──
    w("## Executive Summary")
    w()
    w("| Metric | O(N) Mean-Field | O(N log N) Sparse k-NN | O(N²) Full Pairwise |")
    w("|--------|----------------|----------------------|---------------------|")

    summaries = [r["summary"] for r in results]
    field_rows = [
        ("Goldilocks N", "N", "{:,d}"),
        ("Total Steps (5 min)", "total_steps", "{:,d}"),
        ("Avg Throughput (osc·s/s)", "avg_throughput", "{:,.0f}"),
        ("Throughput CV", "throughput_cv", "{:.4f}"),
        ("Throughput Stability", "throughput_stability", "{}"),
        ("Avg Step (ms)", "avg_step_ms", "{:.3f}"),
        ("P95 Latency (ms)", "global_p95_ms", "{:.3f}"),
        ("Order Param r (final)", "r_end", "{:.4f}"),
        ("VRAM (GB)", "peak_vram_gb", "{:.2f}"),
        ("VRAM %", "pct_vram", "{:.0f}%"),
        ("Temp Range (°C)", None, None),  # special
        ("TP Drift (2nd/1st half)", "throughput_drift_pct", "{:+.2f}%"),
        ("Thermal Throttle", "thermal_throttle", "{}"),
        ("Achieved TFLOPS", "achieved_tflops", "{:.4f}"),
        ("% Peak Compute", "pct_peak_compute", "{:.1f}%"),
        ("Achieved BW (GB/s)", "achieved_bw_gbs", "{:.1f}"),
        ("% Peak BW", "pct_peak_bw", "{:.1f}%"),
        ("Arithmetic Intensity", "arith_intensity", "{:.2f}"),
    ]

    for row_label, key, fmt in field_rows:
        vals: list[str] = []
        for s in summaries:
            if key is None:  # temp range
                tmin = s.get("temp_min_c", "—")
                tmax = s.get("temp_max_c", "—")
                vals.append(f"{tmin}–{tmax}")
            else:
                v = s.get(key, "—")
                if v == "—" or v is None:
                    vals.append("—")
                elif isinstance(v, bool):
                    vals.append("Yes" if v else "No")
                else:
                    vals.append(fmt.format(v))
        w(f"| {row_label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    w()

    # ── Per-Regime Time-Series ──
    for res in results:
        s = res["summary"]
        windows = res["windows"]
        w(f"## {s['label']} — N = {s['N']:,d}")
        w()
        w(f"**Complexity**: {s['complexity']}  |  **Coupling**: `{s['coupling_mode']}`")
        w(f"**Total**: {s['total_steps']:,d} steps in {s['total_elapsed_s']:.1f}s")
        w()

        # Time-series table
        w("### Telemetry Time-Series (30s windows)")
        w()
        w("| Window | Elapsed | Steps | Throughput | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | r | VRAM % | Temp | Clock |")
        w("|--------|---------|-------|-----------|----------|----------|----------|----------|---|--------|------|-------|")
        for wi in windows:
            temp = f"{wi['gpu_temp_c']}°C" if wi["gpu_temp_c"] is not None else "—"
            clk = f"{wi['gpu_clock_mhz']}" if wi["gpu_clock_mhz"] is not None else "—"
            w(
                f"| {wi['window']:>2} | {wi['elapsed_s']:.0f}s | {wi['window_steps']:,d} | "
                f"{wi['throughput']:,.0f} | {wi['avg_step_ms']:.3f} | "
                f"{wi['p50_ms']:.3f} | {wi['p95_ms']:.3f} | {wi['p99_ms']:.3f} | "
                f"{wi['order_param_r']:.4f} | {wi['pct_vram']:.0f}% | {temp} | {clk} |"
            )
        w()

        # Per-regime analysis
        w("### Analysis")
        w()
        w(f"- **Throughput stability**: CV = {s['throughput_cv']:.4f} → **{s['throughput_stability']}**")
        w(f"- **Latency profile**: avg {s['avg_step_ms']:.3f}ms, P95 {s['global_p95_ms']:.3f}ms")
        w(f"- **Throughput drift**: {s['throughput_drift_pct']:+.2f}% (1st half: {s['first_half_throughput']:,.0f}, 2nd half: {s['second_half_throughput']:,.0f})")
        if s.get("thermal_throttle"):
            w(f"- **THERMAL THROTTLE DETECTED**: >5% throughput drop in 2nd half")
        else:
            w(f"- **Thermal**: Stable ({s['temp_min_c']}–{s['temp_max_c']}°C)")
        w(f"- **Order parameter**: r = {s['r_start']:.4f} → {s['r_end']:.4f} (mean {s['r_mean']:.4f})")
        w(f"- **Compute utilisation**: {s['achieved_tflops']:.4f} TFLOPS ({s['pct_peak_compute']:.1f}% of {GPU_FP32_TFLOPS})")
        w(f"- **Memory bandwidth**: {s['achieved_bw_gbs']:.1f} GB/s ({s['pct_peak_bw']:.1f}% of {GPU_MEMBW_GBS})")
        w(f"- **Arithmetic intensity**: {s['arith_intensity']:.2f} FLOPs/byte " +
          ("— **memory-bound**" if s['arith_intensity'] < 5 else "— **compute-bound**"))
        w()

    # ── Cross-Regime Comparison ──
    w("## Goldilocks Characterisation")
    w()
    w("### What is a Goldilocks Zone?")
    w()
    w("The Goldilocks zone for each coupling regime is the N value where the GPU")
    w("achieves **maximum sustained throughput** — the point where:")
    w("- All SMs are saturated with enough parallelism")
    w("- Memory usage fits comfortably within physical VRAM")
    w("- Per-step latency is minimised relative to problem size")
    w("- The ratio of useful computation to overhead is maximised")
    w()

    w("### Regime Comparison")
    w()

    # Find max throughput regime
    max_tp = max(s["avg_throughput"] for s in summaries)
    min_tp = min(s["avg_throughput"] for s in summaries)
    most_stable = min(summaries, key=lambda s: s["throughput_cv"])

    for s in summaries:
        w(f"#### {s['label']} at N = {s['N']:,d}")
        w()
        strength = "highest throughput" if s["avg_throughput"] == max_tp else ""
        if s["throughput_cv"] == most_stable["throughput_cv"]:
            strength = (strength + ", " if strength else "") + "most stable"
        sync_note = "converges" if s["r_end"] > 0.3 else "does not synchronise"
        w(f"- Throughput: **{s['avg_throughput']:,.0f}** osc·steps/s ({strength if strength else 'baseline'})")
        w(f"- Steps completed in 5 min: **{s['total_steps']:,d}**")
        w(f"- VRAM footprint: {s['peak_vram_gb']:.2f} GB ({s['pct_vram']:.0f}%)")
        w(f"- Synchronisation: {sync_note} (r = {s['r_end']:.4f})")
        bound = "memory-bandwidth-bound" if s["arith_intensity"] < 5 else "compute-bound"
        w(f"- Bottleneck: **{bound}** (AI = {s['arith_intensity']:.2f})")
        w()

    # Speedup ratios
    w("### Throughput Ratios")
    w()
    labels = [s["label"].split()[0].strip("O()") for s in summaries]
    tps = [s["avg_throughput"] for s in summaries]
    w("| Comparison | Speedup |")
    w("|-----------|---------|")
    if len(tps) >= 3:
        w(f"| O(N) / O(N log N) | {tps[0]/max(tps[1],1):.1f}× |")
        w(f"| O(N) / O(N²) | {tps[0]/max(tps[2],1):.1f}× |")
        w(f"| O(N log N) / O(N²) | {tps[1]/max(tps[2],1):.1f}× |")
    w()

    # ── Key Insights ──
    w("## Key Insights")
    w()
    w("1. **Goldilocks zones are real**: Each regime has a clear optimal size where")
    w("   throughput peaks before either memory pressure or insufficient parallelism")
    w("   degrades performance.")
    w()
    w("2. **Throughput stability over 5 minutes**: The coefficient of variation (CV)")
    w("   reveals how consistent each regime is under sustained load. Lower CV means")
    w("   more predictable, production-ready performance.")
    w()
    w("3. **Thermal behaviour**: 5-minute sustained runs expose any thermal throttling")
    w("   that short benchmarks miss. The throughput drift metric (2nd half vs 1st half)")
    w("   quantifies this directly.")
    w()
    w("4. **Physics accuracy vs throughput trade-off**: Higher throughput generally means")
    w("   lower physics fidelity. O(N²) preserves all pairwise interactions but runs at")
    w("   a fraction of O(N)'s throughput. O(N log N) offers a middle ground.")
    w()
    w("5. **Memory vs compute bottleneck**: The arithmetic intensity (FLOPs/byte) reveals")
    w("   whether each regime is limited by memory bandwidth or compute capacity, directly")
    w("   informing optimisation strategy.")
    w()

    w("---")
    w()
    w(f"*Report generated by `goldilocks_sustained_benchmark.py` on {ts}*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all three regimes for 5 minutes each at Goldilocks sizes."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PRINet Goldilocks Sustained GPU Benchmark                  ║")
    print("║  O(N) vs O(N log N) vs O(N²) — 5 min each                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nGPU: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM, {props.multi_processor_count} SMs")
    print(f"PyTorch: {torch.__version__}")
    print(f"Duration: {DURATION_S:.0f}s per regime ({DURATION_S/60:.0f} min)")
    print(f"Telemetry: every {WINDOW_S:.0f}s")
    print(f"Config: K={COUPLING_K}, dt={DT}, seed={SEED}, batch={BATCH_STEPS}")
    print()

    for i, cfg in enumerate(REGIMES):
        print(f"  [{i+1}] {cfg['label']:30s} N = {cfg['N']:>10,d}")
    print()

    meta: dict[str, Any] = {
        "benchmark": "Goldilocks Sustained GPU Benchmark",
        "gpu_name": props.name,
        "gpu_vram_gb": round(props.total_memory / 1024 ** 3, 1),
        "gpu_sms": props.multi_processor_count,
        "pytorch_version": torch.__version__,
        "coupling_K": COUPLING_K,
        "dt": DT,
        "seed": SEED,
        "duration_s": DURATION_S,
        "window_s": WINDOW_S,
        "batch_steps": BATCH_STEPS,
        "warmup_steps": WARMUP_STEPS,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run each regime
    all_results: list[dict[str, Any]] = []
    for cfg in REGIMES:
        result = run_sustained(cfg, device)
        all_results.append(result)
        # Brief cooldown between regimes
        print(f"\n  Cooldown: 10s ...")
        _reset_gpu()
        time.sleep(10)

    # ── Final comparison ──
    print(f"\n{'═' * 70}")
    print("  FINAL COMPARISON")
    print(f"{'═' * 70}")
    print(f"\n  {'Regime':<30s} {'N':>10s} {'Throughput':>14s} {'CV':>8s} {'Steps':>12s} {'r(end)':>8s} {'Temp':>8s}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 14} {'─' * 8} {'─' * 12} {'─' * 8} {'─' * 8}")
    for r in all_results:
        s = r["summary"]
        temp = f"{s['temp_min_c']}–{s['temp_max_c']}" if s.get("temp_min_c") else "—"
        print(
            f"  {s['label']:<30s} {s['N']:>10,d} "
            f"{s['avg_throughput']:>14,.0f} {s['throughput_cv']:>8.4f} "
            f"{s['total_steps']:>12,d} {s['r_end']:>8.4f} {temp:>8s}"
        )

    # ── Save JSON ──
    full_results = {
        "meta": meta,
        "results": all_results,
    }
    json_path = RESULTS_DIR / "goldilocks_sustained_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n✓ JSON: {json_path}")

    # ── Generate report ──
    report = generate_report(all_results, meta)
    report_path = RESULTS_DIR / "goldilocks_sustained_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Report: {report_path}")

    print("\n✓ Goldilocks Sustained Benchmark complete.")


if __name__ == "__main__":
    main()
