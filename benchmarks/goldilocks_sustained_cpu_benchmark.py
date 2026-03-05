"""Goldilocks Sustained CPU / RAM Benchmark: O(N) vs O(N log N) vs O(N²).

Each coupling regime is run at its empirically-determined CPU optimal
("Goldilocks") throughput size for a sustained **5 minutes**, collecting
time-series telemetry every 30 seconds.

Goldilocks Sizes (from prior CPU benchmarks)
----------------------------------------------
* O(N)         mean_field   N =   262,144    (262 K oscillators)
* O(N log N)   sparse_knn   N =    16,384    (16 K oscillators)
* O(N²)        full         N =     1,024    (1 K oscillators)

Sources:
  O(N):         on_stress_benchmark Phase 4 MT — 23.2M at N=262K (16 cores)
  O(N log N):   on_vs_onlogn_benchmark CPU MT — 1.34M at N=16K (peak)
  O(N²):        GPU Goldilocks was N=1K; no prior CPU data at this regime

Hardware Target
---------------
* CPU: AMD Ryzen (16 logical cores, 8 physical), 32 GB RAM
* Torch threads: default (all cores) for sustained multi-threaded runs

Metrics Captured (per 30-second window)
---------------------------------------
* Steps completed / throughput (osc·steps/s)
* Mean, std, P50, P95, P99 per-step latency (ms)
* Cumulative order parameter r
* Process RSS (GB / % of total RAM)
* Per-core CPU utilisation (%)
* System-wide memory pressure
* Coefficient of variation (throughput stability)

Post-Run Analysis
-----------------
* Throughput time-series & stability (CV)
* Latency percentile profiles
* RSS trajectory over 5 min (memory leak detection)
* CPU utilisation heat map (per-window)
* Regime comparison table
* Goldilocks characterisation

Outputs
-------
* ``Docs/test_and_benchmark_results/goldilocks_sustained_cpu_benchmark.json``
* ``Docs/test_and_benchmark_results/goldilocks_sustained_cpu_report.md``
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
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
WARMUP_STEPS: int = 30             # Steps before timing begins
BATCH_STEPS: int = 20              # Steps between time samples

# CPU Goldilocks configurations (from prior benchmarks)
REGIMES: list[dict[str, Any]] = [
    {
        "label": "O(N) Mean-Field",
        "complexity": "O(N)",
        "coupling_mode": "mean_field",
        "N": 262_144,
        "mean_field": True,
        "sparse_k": None,
        "source": "on_stress Phase 4 MT: 23.2M @ N=262K",
    },
    {
        "label": "O(N log N) Sparse k-NN",
        "complexity": "O(N log N)",
        "coupling_mode": "sparse_knn",
        "N": 16_384,
        "mean_field": False,
        "sparse_k": None,  # defaults to ceil(log2(N)) = 14
        "source": "on_vs_onlogn CPU MT: 1.34M @ N=16K",
    },
    {
        "label": "O(N²) Full Pairwise",
        "complexity": "O(N²)",
        "coupling_mode": "full",
        "N": 1_024,
        "mean_field": False,
        "sparse_k": None,
        "source": "GPU Goldilocks N=1K (no prior CPU data)",
    },
]

TOTAL_RAM_GB: float = psutil.virtual_memory().total / (1024 ** 3)

RESULTS_DIR: Path = (
    Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
)


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _gc_collect() -> None:
    """Force garbage collection."""
    gc.collect()
    gc.collect()


def _rss_gb() -> float:
    """Current process RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _system_ram_pct() -> float:
    """System-wide RAM usage %."""
    return psutil.virtual_memory().percent


def _cpu_percent_per_core() -> list[float]:
    """Per-logical-core CPU % (non-blocking snapshot, prev interval)."""
    return psutil.cpu_percent(percpu=True)


def _cpu_percent_overall(interval: float = 0.1) -> float:
    """Overall CPU % (blocking brief sample)."""
    return psutil.cpu_percent(interval=interval)


def _cpu_freq() -> Optional[float]:
    """Current CPU frequency in MHz."""
    freq = psutil.cpu_freq()
    return freq.current if freq else None


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

    Mean-field O(N):   172·N
    Sparse k-NN:       172·N + 4·N·k·15
    Full O(N²):        80·N²
    """
    if mode == "mean_field":
        return 172.0 * N
    elif mode == "sparse_knn":
        k = max(1, math.ceil(math.log2(N)))
        return 172.0 * N + 4 * N * k * 15
    else:
        return 80.0 * N * N


def _estimate_bytes_per_step(N: int, mode: str) -> float:
    """Rough bytes moved per RK4 step.

    Mean-field: 248·N
    Sparse:     248·N + 4·N·k·12
    Full:       32N + 16N²
    """
    if mode == "mean_field":
        return 248.0 * N
    elif mode == "sparse_knn":
        k = max(1, math.ceil(math.log2(N)))
        return 248.0 * N + 4 * N * k * 12
    else:
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
    """Run a single regime for *duration* seconds on CPU, collecting telemetry.

    Args:
        cfg: Regime configuration dict.
        device: CPU device.
        duration: Total run time in seconds.
        window: Telemetry collection interval in seconds.

    Returns:
        Dict with time-series windows and aggregate statistics.
    """
    N = cfg["N"]
    label = cfg["label"]
    mode = cfg["coupling_mode"]

    print(f"\n{'═' * 72}")
    print(f"  {label}  |  N = {N:,d}  |  {duration:.0f}s sustained  |  CPU")
    print(f"{'═' * 72}")

    _gc_collect()

    # Snapshot baseline RSS
    rss_baseline = _rss_gb()

    # Prime CPU % measurement (first call is always 0)
    psutil.cpu_percent(percpu=True)

    # Create model & initial state
    model = _make_model(cfg, device)
    state = OscillatorState.create_random(N, device=device, seed=SEED)

    rss_after_alloc = _rss_gb()
    model_rss_gb = rss_after_alloc - rss_baseline

    # ── Warm-up ──
    print(f"  Warm-up: {WARMUP_STEPS} steps ... ", end="", flush=True)
    s = state.clone()
    for _ in range(WARMUP_STEPS):
        s = model.step(s, dt=DT)
    print("done.")
    print(f"  Model RSS: {model_rss_gb:.3f} GB  |  Process RSS: {_rss_gb():.3f} GB")
    print(f"  System RAM: {_system_ram_pct():.1f}%  |  Cores: {psutil.cpu_count()} logical")
    print()

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

        # Reset per-core CPU counters
        psutil.cpu_percent(percpu=True)

        while True:
            # Batch of steps with perf_counter timing
            t0 = time.perf_counter()
            for _ in range(BATCH_STEPS):
                s = model.step(s, dt=DT)
            t1 = time.perf_counter()
            batch_ms = (t1 - t0) * 1000.0
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

        # ── Telemetry ──
        rss_now = _rss_gb()
        rss_pct = rss_now / TOTAL_RAM_GB * 100
        sys_ram_pct = _system_ram_pct()

        # CPU utilisation (per-core snapshot from the window)
        per_core = _cpu_percent_per_core()
        cpu_avg = sum(per_core) / len(per_core) if per_core else 0.0
        cpu_max = max(per_core) if per_core else 0.0
        cores_active = sum(1 for c in per_core if c > 50.0)

        cpu_freq = _cpu_freq()

        r_val = kuramoto_order_parameter(s.phase).item()

        throughput = (N * window_steps) / window_elapsed
        avg_ms = sum(step_times_ms) / len(step_times_ms)
        std_ms = (
            sum((t - avg_ms) ** 2 for t in step_times_ms)
            / max(len(step_times_ms) - 1, 1)
        ) ** 0.5

        p50 = _percentile(step_times_ms, 50)
        p95 = _percentile(step_times_ms, 95)
        p99 = _percentile(step_times_ms, 99)

        # FLOPs / bandwidth estimates
        flops_per_step = _estimate_flops_per_step(N, mode)
        bytes_per_step = _estimate_bytes_per_step(N, mode)
        achieved_gflops = (flops_per_step * window_steps) / window_elapsed / 1e9
        achieved_bw_gbs = (bytes_per_step * window_steps) / window_elapsed / 1e9

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
            "rss_gb": round(rss_now, 3),
            "rss_pct": round(rss_pct, 1),
            "sys_ram_pct": round(sys_ram_pct, 1),
            "cpu_avg_pct": round(cpu_avg, 1),
            "cpu_max_pct": round(cpu_max, 1),
            "cores_active": cores_active,
            "cpu_freq_mhz": round(cpu_freq, 0) if cpu_freq else None,
            "achieved_gflops": round(achieved_gflops, 3),
            "achieved_bw_gbs": round(achieved_bw_gbs, 2),
        }
        windows.append(window_data)

        # Live output
        freq_str = f"{cpu_freq:.0f}MHz" if cpu_freq else "—"
        print(
            f"  [{window_idx:>2}] {window_data['elapsed_s']:>5.0f}s  "
            f"steps={window_steps:>6,d}  "
            f"T={throughput:>12,.0f}  "
            f"avg={avg_ms:.3f}ms  "
            f"p95={p95:.3f}ms  "
            f"r={r_val:.4f}  "
            f"RSS={rss_now:.2f}GB({rss_pct:.0f}%)  "
            f"CPU={cpu_avg:.0f}%({cores_active}c)  "
            f"{freq_str}"
        )
        window_idx += 1

    total_elapsed = time.perf_counter() - wall_start

    # ── Aggregate statistics ──
    throughputs = [w["throughput"] for w in windows]
    avg_tp = sum(throughputs) / len(throughputs)
    all_avgs = [w["avg_step_ms"] for w in windows]
    all_p95 = [w["p95_ms"] for w in windows]
    all_r = [w["order_param_r"] for w in windows]
    all_rss = [w["rss_gb"] for w in windows]
    all_cpu_avg = [w["cpu_avg_pct"] for w in windows]

    tp_cv = _coeff_var(throughputs)

    # RSS trajectory (memory leak detection)
    rss_start = all_rss[0] if all_rss else 0.0
    rss_end = all_rss[-1] if all_rss else 0.0
    rss_delta = rss_end - rss_start
    rss_peak = max(all_rss) if all_rss else 0.0
    rss_stable = abs(rss_delta) < 0.05  # < 50 MB drift = stable

    # CPU analysis
    cpu_mean = sum(all_cpu_avg) / len(all_cpu_avg) if all_cpu_avg else 0.0

    # Throughput drift (first half vs second half)
    half = len(throughputs) // 2
    first_half_tp = sum(throughputs[:half]) / max(half, 1)
    second_half_tp = sum(throughputs[half:]) / max(len(throughputs) - half, 1)
    tp_drift_pct = (second_half_tp - first_half_tp) / max(first_half_tp, 1) * 100

    # Order parameter trend
    r_start = all_r[0] if all_r else 0.0
    r_end = all_r[-1] if all_r else 0.0
    r_mean = sum(all_r) / len(all_r) if all_r else 0.0

    # Compute estimates
    flops_per_step = _estimate_flops_per_step(N, mode)
    bytes_per_step = _estimate_bytes_per_step(N, mode)
    total_flops = flops_per_step * total_steps
    total_bytes = bytes_per_step * total_steps
    avg_gflops = total_flops / total_elapsed / 1e9
    avg_bw_gbs = total_bytes / total_elapsed / 1e9

    summary = {
        "label": label,
        "complexity": cfg["complexity"],
        "coupling_mode": mode,
        "N": N,
        "source": cfg["source"],
        "total_steps": total_steps,
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_throughput": round(avg_tp, 0),
        "throughput_cv": round(tp_cv, 4),
        "throughput_stability": (
            "excellent" if tp_cv < 0.02
            else "good" if tp_cv < 0.05
            else "moderate" if tp_cv < 0.10
            else "unstable"
        ),
        "avg_step_ms": round(sum(all_avgs) / len(all_avgs), 4),
        "global_p50_ms": round(_percentile(all_avgs, 50), 4),
        "global_p95_ms": round(_percentile(all_p95, 95), 4),
        "global_p99_ms": round(_percentile(all_p95, 99), 4),
        "r_start": round(r_start, 6),
        "r_end": round(r_end, 6),
        "r_mean": round(r_mean, 6),
        "rss_start_gb": round(rss_start, 3),
        "rss_end_gb": round(rss_end, 3),
        "rss_peak_gb": round(rss_peak, 3),
        "rss_delta_gb": round(rss_delta, 4),
        "rss_stable": rss_stable,
        "rss_pct": round(rss_peak / TOTAL_RAM_GB * 100, 1),
        "cpu_mean_pct": round(cpu_mean, 1),
        "first_half_throughput": round(first_half_tp, 0),
        "second_half_throughput": round(second_half_tp, 0),
        "throughput_drift_pct": round(tp_drift_pct, 2),
        "avg_gflops": round(avg_gflops, 3),
        "avg_bw_gbs": round(avg_bw_gbs, 2),
    }

    print(f"\n  ── {label} Summary ──")
    print(f"  Total steps:    {total_steps:,d} in {total_elapsed:.1f}s")
    print(f"  Avg throughput: {avg_tp:,.0f} osc·steps/s  (CV={tp_cv:.4f})")
    print(f"  Avg step:       {summary['avg_step_ms']:.3f} ms  P95={summary['global_p95_ms']:.3f} ms")
    print(f"  Order param:    r_start={r_start:.4f} → r_end={r_end:.4f}")
    print(f"  RSS:            {rss_start:.3f} → {rss_end:.3f} GB (Δ={rss_delta:+.4f} GB) {'STABLE' if rss_stable else 'DRIFT'}")
    print(f"  Peak RSS:       {rss_peak:.3f} GB ({rss_peak/TOTAL_RAM_GB*100:.1f}% of {TOTAL_RAM_GB:.1f} GB)")
    print(f"  CPU avg:        {cpu_mean:.1f}%")
    print(f"  TP drift:       {tp_drift_pct:+.2f}% (2nd half vs 1st)")
    print(f"  Compute:        {avg_gflops:.3f} GFLOPS")
    print(f"  Bandwidth:      {avg_bw_gbs:.2f} GB/s")

    del model, state, s
    _gc_collect()

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

    w("# PRINet Goldilocks Sustained CPU / RAM Benchmark")
    w()
    w("## Configuration")
    w()
    w(f"**Generated**: {ts}")
    w(f"**CPU**: {meta.get('cpu_name', 'N/A')} ({meta.get('logical_cores')} logical, "
      f"{meta.get('physical_cores')} physical)")
    w(f"**RAM**: {meta.get('total_ram_gb', 0):.1f} GB")
    w(f"**PyTorch**: {meta.get('pytorch_version')} "
      f"(threads: {meta.get('torch_threads')}, interop: {meta.get('torch_interop_threads')})")
    w(f"**Coupling**: K={meta.get('coupling_K')}, dt={meta.get('dt')}, seed={meta.get('seed')}")
    w(f"**Duration**: {meta.get('duration_s', 300)}s per regime (5 minutes)")
    w(f"**Telemetry**: Every {meta.get('window_s', 30)}s")
    w()
    w("Each coupling regime is run at its empirically-determined **CPU Goldilocks** size —")
    w("the N that maximises CPU throughput from prior benchmarks — for a sustained 5-minute burn.")
    w()

    # Source of Goldilocks sizes
    w("### Goldilocks Size Selection")
    w()
    w("| Regime | N | Source |")
    w("|--------|---|--------|")
    for r in results:
        s = r["summary"]
        w(f"| {s['label']} | {s['N']:,d} | {s['source']} |")
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
        ("RSS (GB)", "rss_peak_gb", "{:.3f}"),
        ("RSS % of RAM", "rss_pct", "{:.1f}%"),
        ("RSS Stable", "rss_stable", "{}"),
        ("RSS Δ (GB)", "rss_delta_gb", "{:+.4f}"),
        ("CPU Avg %", "cpu_mean_pct", "{:.1f}%"),
        ("TP Drift (2nd/1st)", "throughput_drift_pct", "{:+.2f}%"),
        ("Achieved GFLOPS", "avg_gflops", "{:.3f}"),
        ("Achieved BW (GB/s)", "avg_bw_gbs", "{:.2f}"),
    ]

    for row_label, key, fmt in field_rows:
        vals: list[str] = []
        for s in summaries:
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
        win = res["windows"]
        w(f"## {s['label']} — N = {s['N']:,d}")
        w()
        w(f"**Complexity**: {s['complexity']}  |  **Coupling**: `{s['coupling_mode']}`")
        w(f"**Source**: {s['source']}")
        w(f"**Total**: {s['total_steps']:,d} steps in {s['total_elapsed_s']:.1f}s")
        w()

        w("### Telemetry Time-Series (30s windows)")
        w()
        w("| Win | Elapsed | Steps | Throughput | Avg (ms) | P50 (ms) | P95 (ms) "
          "| P99 (ms) | r | RSS GB | RSS% | RAM% | CPU% | Cores |")
        w("|-----|---------|-------|-----------|----------|----------|----------"
          "|----------|---|--------|------|------|------|-------|")
        for wi in win:
            w(
                f"| {wi['window']:>2} | {wi['elapsed_s']:.0f}s "
                f"| {wi['window_steps']:,d} | {wi['throughput']:,.0f} "
                f"| {wi['avg_step_ms']:.3f} | {wi['p50_ms']:.3f} "
                f"| {wi['p95_ms']:.3f} | {wi['p99_ms']:.3f} "
                f"| {wi['order_param_r']:.4f} | {wi['rss_gb']:.3f} "
                f"| {wi['rss_pct']:.0f}% | {wi['sys_ram_pct']:.0f}% "
                f"| {wi['cpu_avg_pct']:.0f}% | {wi['cores_active']} |"
            )
        w()

        # Per-regime analysis
        w("### Analysis")
        w()
        w(f"- **Throughput stability**: CV = {s['throughput_cv']:.4f} → "
          f"**{s['throughput_stability']}**")
        w(f"- **Latency profile**: avg {s['avg_step_ms']:.3f}ms, "
          f"P95 {s['global_p95_ms']:.3f}ms")
        w(f"- **Throughput drift**: {s['throughput_drift_pct']:+.2f}% "
          f"(1st half: {s['first_half_throughput']:,.0f}, "
          f"2nd half: {s['second_half_throughput']:,.0f})")
        w(f"- **Memory (RSS)**: {s['rss_start_gb']:.3f} → {s['rss_end_gb']:.3f} GB "
          f"(Δ = {s['rss_delta_gb']:+.4f} GB) — "
          f"**{'stable, no leak' if s['rss_stable'] else 'DRIFT DETECTED'}**")
        w(f"- **Peak RSS**: {s['rss_peak_gb']:.3f} GB ({s['rss_pct']:.1f}% of "
          f"{TOTAL_RAM_GB:.1f} GB)")
        w(f"- **CPU utilisation**: avg {s['cpu_mean_pct']:.1f}%")
        w(f"- **Order parameter**: r = {s['r_start']:.4f} → {s['r_end']:.4f} "
          f"(mean {s['r_mean']:.4f})")
        w(f"- **Compute**: {s['avg_gflops']:.3f} GFLOPS")
        w(f"- **Memory bandwidth**: {s['avg_bw_gbs']:.2f} GB/s")
        w()

    # ── Cross-Regime Comparison ──
    w("## Goldilocks Characterisation")
    w()
    w("### CPU Goldilocks Zones")
    w()
    w("The CPU Goldilocks zone differs from GPU because:")
    w("- CPU parallelism is limited to the thread count (16 logical cores here)")
    w("- Memory hierarchy (L1/L2/L3 → RAM) creates a different bandwidth profile")
    w("- Thread synchronisation overhead penalises tiny N values")
    w("- Cache locality matters more — regimes that stride memory widely lose efficiency")
    w()

    for s in summaries:
        w(f"#### {s['label']} at N = {s['N']:,d}")
        w()
        w(f"- Throughput: **{s['avg_throughput']:,.0f}** osc·steps/s")
        w(f"- Steps completed in 5 min: **{s['total_steps']:,d}**")
        w(f"- RSS footprint: {s['rss_peak_gb']:.3f} GB ({s['rss_pct']:.1f}%)")
        sync_note = "converges" if s["r_end"] > 0.3 else "does not synchronise"
        w(f"- Synchronisation: {sync_note} (r = {s['r_end']:.4f})")
        w(f"- CPU utilisation: {s['cpu_mean_pct']:.1f}% average")
        w()

    # Speedup ratios
    w("### Throughput Ratios")
    w()
    tps = [s["avg_throughput"] for s in summaries]
    w("| Comparison | Speedup |")
    w("|-----------|---------|")
    if len(tps) >= 3:
        w(f"| O(N) / O(N log N) | {tps[0]/max(tps[1],1):.1f}× |")
        w(f"| O(N) / O(N²) | {tps[0]/max(tps[2],1):.1f}× |")
        w(f"| O(N log N) / O(N²) | {tps[1]/max(tps[2],1):.1f}× |")
    w()

    # ── CPU vs GPU comparison ──
    w("### CPU vs GPU Goldilocks Comparison")
    w()
    w("| Metric | CPU Goldilocks | GPU Goldilocks |")
    w("|--------|---------------|---------------|")
    w("| O(N) Goldilocks N | 262,144 | 1,048,576 |")
    w("| O(N log N) Goldilocks N | 16,384 | 65,536 |")
    w("| O(N²) Goldilocks N | 1,024 | 1,024 |")
    w("| O(N) peak throughput | (this run) | 443M osc·s/s |")
    w("| O(N log N) peak throughput | (this run) | 23.6M osc·s/s |")
    w("| O(N²) peak throughput | (this run) | 679K osc·s/s |")
    w()
    w("The GPU can fill its SMs with larger N before saturating, so the GPU Goldilocks")
    w("is 4× larger for O(N) and O(N log N). O(N²) is the same because N=1K is already")
    w("the practical limit due to quadratic memory scaling.")
    w()

    # ── Key Insights ──
    w("## Key Insights")
    w()
    w("1. **CPU Goldilocks zones are smaller than GPU**: The CPU saturates at lower N")
    w("   because it has fewer parallel execution units (16 cores vs 24 SMs × 128 CUDA cores).")
    w()
    w("2. **Memory stability over 5 minutes**: The RSS trajectory reveals whether the")
    w("   implementation has memory leaks under sustained load. Stable RSS =")
    w("   production-ready memory management.")
    w()
    w("3. **CPU utilisation**: Multi-threaded PyTorch should max out all cores. If CPU%")
    w("   is below 100%, there is serialisation overhead or memory-boundedness.")
    w()
    w("4. **Throughput consistency**: CV < 0.05 means the CPU maintains steady performance")
    w("   without GC pauses or thermal throttling impacting throughput.")
    w()
    w("5. **RAM vs VRAM**: CPU operates in system RAM (32 GB here) — vastly more than")
    w("   GPU VRAM (8 GB). This means CPU can handle larger N before memory pressure,")
    w("   but per-byte bandwidth is much lower than GPU HBM/GDDR6.")
    w()

    w("---")
    w()
    w(f"*Report generated by `goldilocks_sustained_cpu_benchmark.py` on {ts}*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all three regimes for 5 minutes each at CPU Goldilocks sizes."""
    device = torch.device("cpu")

    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    cpu_name = platform.processor() or "Unknown"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PRINet Goldilocks Sustained CPU / RAM Benchmark            ║")
    print("║  O(N) vs O(N log N) vs O(N²) — 5 min each — CPU only       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nCPU: {cpu_name}")
    print(f"Cores: {logical} logical, {physical} physical")
    print(f"RAM: {TOTAL_RAM_GB:.1f} GB total ({_system_ram_pct():.1f}% in use)")
    print(f"PyTorch: {torch.__version__} (threads={torch.get_num_threads()}, "
          f"interop={torch.get_num_interop_threads()})")
    print(f"Duration: {DURATION_S:.0f}s per regime ({DURATION_S/60:.0f} min)")
    print(f"Telemetry: every {WINDOW_S:.0f}s")
    print(f"Config: K={COUPLING_K}, dt={DT}, seed={SEED}, batch={BATCH_STEPS}")
    print()

    for i, cfg in enumerate(REGIMES):
        print(f"  [{i+1}] {cfg['label']:30s} N = {cfg['N']:>10,d}  ({cfg['source']})")
    print()

    meta: dict[str, Any] = {
        "benchmark": "Goldilocks Sustained CPU / RAM Benchmark",
        "cpu_name": cpu_name,
        "logical_cores": logical,
        "physical_cores": physical,
        "total_ram_gb": round(TOTAL_RAM_GB, 1),
        "pytorch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
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
        # Brief cooldown between regimes (GC + let threads settle)
        print(f"\n  Cooldown: 5s ...")
        _gc_collect()
        time.sleep(5)

    # ── Final comparison ──
    print(f"\n{'═' * 72}")
    print("  FINAL COMPARISON (CPU)")
    print(f"{'═' * 72}")
    print(
        f"\n  {'Regime':<30s} {'N':>10s} {'Throughput':>14s} "
        f"{'CV':>8s} {'Steps':>12s} {'r(end)':>8s} "
        f"{'RSS GB':>8s} {'CPU%':>6s}"
    )
    print(
        f"  {'─' * 30} {'─' * 10} {'─' * 14} "
        f"{'─' * 8} {'─' * 12} {'─' * 8} "
        f"{'─' * 8} {'─' * 6}"
    )
    for r in all_results:
        s = r["summary"]
        print(
            f"  {s['label']:<30s} {s['N']:>10,d} "
            f"{s['avg_throughput']:>14,.0f} {s['throughput_cv']:>8.4f} "
            f"{s['total_steps']:>12,d} {s['r_end']:>8.4f} "
            f"{s['rss_peak_gb']:>8.3f} {s['cpu_mean_pct']:>5.1f}%"
        )

    # ── Save JSON ──
    full_results = {
        "meta": meta,
        "results": all_results,
    }
    json_path = RESULTS_DIR / "goldilocks_sustained_cpu_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n✓ JSON: {json_path}")

    # ── Generate report ──
    report = generate_report(all_results, meta)
    report_path = RESULTS_DIR / "goldilocks_sustained_cpu_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ Report: {report_path}")

    print("\n✓ Goldilocks Sustained CPU / RAM Benchmark complete.")


if __name__ == "__main__":
    main()
