"""Year 4 Q1.5 Benchmarks — Session-Length Dynamics.

Runs PhaseTracker and TemporalSlotAttentionMOT continuously for
wall-clock durations of 5, 10, 30, and 60 minutes, sampling temporal
dynamics, computational sustainability, and tracking quality metrics
at regular intervals throughout each session.

Scientific significance is assessed via one-way ANOVA across session
durations, pairwise Cohen's d effect sizes, and bootstrap 95% CIs.

Metrics from Perplexity-researched literature:
- **Order parameter r(t)**: Kuramoto synchronisation magnitude,
  sampled every 30 seconds.
- **Windowed order parameter variance**: Stability of r(t) within
  sliding windows — increasing variance → degradation.
- **Phase Locking Value (PLV)**: Pairwise phase synchrony between
  tracked objects (independent of signal amplitude).
- **Phase drift (frequency spread)**: Std of instantaneous frequencies
  across oscillators — increasing spread → desynchronisation.
- **Cumulative phase-slip curve**: Running total of phase slips over
  session.  Acceleration coefficient reveals degradation kinetics.
- **Throughput (FPS)**: Frames per second per sampling interval.
  Degradation indicates thermal throttling or memory contention.
- **Memory growth**: GPU/CPU memory tracked across session.
- **Identity Preservation (IP)**: Running IP progression.
- **Binding Persistence (BP)**: Per-interval binding strength.
- **Coherence decay rate (λ)**: Per-interval exponential decay fit.

Benchmarks:
1. **5-minute session** — baseline session (3 seeds).
2. **10-minute session** — short session (3 seeds).
3. **30-minute session** — medium session (3 seeds).
4. **60-minute session** — extended session (3 seeds).
5. **Cross-duration statistical comparison** — ANOVA, Cohen's d, η².

Generates JSON result files in ``benchmarks/results/``.

Usage:
    python benchmarks/y4q1_5_benchmarks.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

# =========================================================================
# Setup
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

# Sampling parameters — interval between metric snapshots
SAMPLE_INTERVAL_FRAMES = 50  # sample metrics every 50 frames
FRAMES_PER_BATCH = 10  # frames generated per tracking iteration


def _save(name: str, data: dict) -> None:
    path = RESULTS_DIR / f"benchmark_y4q1_5_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


def _cleanup() -> None:
    """Free GPU/CPU memory between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _get_gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def _get_process_memory_mb() -> float:
    """Return current process RSS in MB (cross-platform)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _make_smooth_sequence(
    n_frames: int, n_objects: int, det_dim: int = 4, seed: int = 42,
) -> list[torch.Tensor]:
    """Generate slowly-drifting detections (smooth temporal trajectory)."""
    torch.manual_seed(seed)
    base = torch.randn(n_objects, det_dim)
    frames: list[torch.Tensor] = []
    for t in range(n_frames):
        noise = 0.02 * (t % 200) * torch.randn(n_objects, det_dim)
        frames.append(base + noise)
    return frames


# =========================================================================
# Tracker factories
# =========================================================================


def _build_phase_tracker(seed: int = 42) -> "PhaseTracker":
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(
        detection_dim=4,
        n_delta=4,
        n_theta=8,
        n_gamma=16,
        n_discrete_steps=5,
        match_threshold=0.1,
    )


def _build_slot_tracker(seed: int = 42) -> "TemporalSlotAttentionMOT":
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(
        detection_dim=4,
        num_slots=6,
        slot_dim=64,
        num_iterations=3,
        match_threshold=0.1,
    )


# =========================================================================
# Core session runner
# =========================================================================


def run_timed_session(
    duration_minutes: float,
    seed: int = 42,
    n_objects: int = 4,
    warmup_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run PhaseTracker + SlotAttention continuously for *duration_minutes*.

    During the session, metrics are sampled every ``SAMPLE_INTERVAL_FRAMES``
    frames.  A warm-up period is run but excluded from reported metrics.

    Args:
        duration_minutes: Target session duration in minutes.
        seed: Random seed for reproducibility.
        n_objects: Number of tracked objects.
        warmup_seconds: Seconds of warm-up before measurement starts.

    Returns:
        Dict with per-interval and aggregate metrics for both trackers.
    """
    from prinet.utils.y4q1_tools import (
        binding_persistence,
        coherence_decay_rate,
        order_parameter_series,
        phase_locking_value,
        phase_slip_rate,
    )

    duration_seconds = duration_minutes * 60.0
    det_dim = 4
    frame_seed_offset = seed * 10000

    pt = _build_phase_tracker(seed=seed)
    sa = _build_slot_tracker(seed=seed)

    # ---- Warm-up phase ----
    print(f"    Warm-up ({warmup_seconds:.0f}s) ...", end=" ", flush=True)
    warmup_start = time.perf_counter()
    warmup_frame_count = 0
    while time.perf_counter() - warmup_start < warmup_seconds:
        frames = _make_smooth_sequence(
            FRAMES_PER_BATCH, n_objects, det_dim,
            seed=frame_seed_offset + warmup_frame_count,
        )
        with torch.no_grad():
            pt.track_sequence(frames)
            sa.track_sequence(frames)
        warmup_frame_count += FRAMES_PER_BATCH
    print(f"done ({warmup_frame_count} frames)")
    _cleanup()

    # ---- Measurement phase ----
    print(f"    Measuring ({duration_minutes:.0f}min) ...", flush=True)

    # Accumulators for interval-level metrics
    interval_data: list[dict[str, Any]] = []
    total_frames = 0
    session_start = time.perf_counter()

    # Running accumulators between intervals
    pt_phases_interval: list[torch.Tensor] = []
    pt_matches_interval: list[torch.Tensor] = []
    sa_matches_interval: list[torch.Tensor] = []
    pt_sims_interval: list[float] = []
    sa_sims_interval: list[float] = []
    pt_rho_interval: list[float] = []
    interval_frame_count = 0
    interval_wall_start = time.perf_counter()

    frame_counter = warmup_frame_count  # continue seed offset

    while True:
        elapsed = time.perf_counter() - session_start
        if elapsed >= duration_seconds:
            break

        # Generate a batch of frames
        frames = _make_smooth_sequence(
            FRAMES_PER_BATCH, n_objects, det_dim,
            seed=frame_seed_offset + frame_counter,
        )
        frame_counter += FRAMES_PER_BATCH

        with torch.no_grad():
            pt_result = pt.track_sequence(frames)
            sa_result = sa.track_sequence(frames)

        # Accumulate interval data
        for ph in pt_result.get("phase_history", []):
            pt_phases_interval.append(ph)
        for m in pt_result.get("identity_matches", []):
            pt_matches_interval.append(m)
        for m in sa_result.get("identity_matches", []):
            sa_matches_interval.append(m)
        pt_sims_interval.extend(pt_result.get("per_frame_similarity", []))
        sa_sims_interval.extend(sa_result.get("per_frame_similarity", []))
        pt_rho_interval.extend(
            pt_result.get("per_frame_phase_correlation", [])
        )
        interval_frame_count += FRAMES_PER_BATCH
        total_frames += FRAMES_PER_BATCH

        # Check if we've accumulated enough for a sample interval
        if interval_frame_count >= SAMPLE_INTERVAL_FRAMES:
            interval_wall_end = time.perf_counter()
            interval_wall_sec = interval_wall_end - interval_wall_start
            elapsed_total = interval_wall_end - session_start

            entry: dict[str, Any] = {
                "elapsed_seconds": round(elapsed_total, 2),
                "interval_frames": interval_frame_count,
                "interval_wall_sec": round(interval_wall_sec, 4),
                "fps": round(
                    interval_frame_count / max(interval_wall_sec, 1e-9), 2
                ),
                "gpu_memory_mb": round(_get_gpu_memory_mb(), 2),
                "process_memory_mb": round(_get_process_memory_mb(), 2),
            }

            # PhaseTracker metrics
            if len(pt_phases_interval) >= 2:
                traj = torch.stack(pt_phases_interval)  # (T, N, n_osc)
                if traj.dim() == 3:
                    T, N, K = traj.shape
                    traj_flat = traj.reshape(T, N * K)
                else:
                    traj_flat = traj

                opr = order_parameter_series(traj)
                entry["pt_mean_r"] = round(opr["mean_r"], 6)
                entry["pt_std_r"] = round(opr["std_r"], 6)

                # Phase slip rate
                mean_phase = traj.mean(dim=-1) if traj.dim() == 3 else traj
                psr = phase_slip_rate(mean_phase)
                entry["pt_slip_fraction"] = round(psr["slip_fraction"], 6)

                # PLV (first two objects if available)
                if mean_phase.shape[1] >= 2:
                    plv = phase_locking_value(
                        mean_phase[:, 0], mean_phase[:, 1])
                    entry["pt_plv"] = round(plv["plv"], 6)

            # IP for this interval
            if pt_matches_interval:
                pt_matched = sum(
                    int((m >= 0).sum().item()) for m in pt_matches_interval
                )
                pt_possible = sum(
                    m.shape[0] for m in pt_matches_interval
                )
                entry["pt_ip"] = round(
                    pt_matched / max(pt_possible, 1), 6
                )

            if sa_matches_interval:
                sa_matched = sum(
                    int((m >= 0).sum().item()) for m in sa_matches_interval
                )
                sa_possible = sum(
                    m.shape[0] for m in sa_matches_interval
                )
                entry["sa_ip"] = round(
                    sa_matched / max(sa_possible, 1), 6
                )

            # Mean similarity
            if pt_sims_interval:
                entry["pt_mean_sim"] = round(
                    sum(pt_sims_interval) / len(pt_sims_interval), 6
                )
            if sa_sims_interval:
                entry["sa_mean_sim"] = round(
                    sum(sa_sims_interval) / len(sa_sims_interval), 6
                )
            if pt_rho_interval:
                entry["pt_mean_rho"] = round(
                    sum(pt_rho_interval) / len(pt_rho_interval), 6
                )

            # Binding persistence
            if pt_matches_interval:
                bp = binding_persistence(pt_matches_interval, n_objects)
                entry["pt_bp"] = round(bp["mean_persistence"], 6)
            if sa_matches_interval:
                bp_sa = binding_persistence(sa_matches_interval, 6)
                entry["sa_bp"] = round(bp_sa["mean_persistence"], 6)

            interval_data.append(entry)

            # Report progress
            pct = min(elapsed_total / duration_seconds * 100, 100)
            pt_ip_str = f"{entry.get('pt_ip', 0):.3f}"
            sa_ip_str = f"{entry.get('sa_ip', 0):.3f}"
            print(
                f"      [{pct:5.1f}%] {elapsed_total:7.1f}s  "
                f"FPS={entry['fps']:7.1f}  "
                f"PT_IP={pt_ip_str}  SA_IP={sa_ip_str}  "
                f"GPU={entry['gpu_memory_mb']:.0f}MB",
                flush=True,
            )

            # Reset interval accumulators
            pt_phases_interval = []
            pt_matches_interval = []
            sa_matches_interval = []
            pt_sims_interval = []
            sa_sims_interval = []
            pt_rho_interval = []
            interval_frame_count = 0
            interval_wall_start = time.perf_counter()

    session_wall = time.perf_counter() - session_start

    # ---- Aggregate metrics ----
    agg: dict[str, Any] = {
        "duration_minutes": duration_minutes,
        "seed": seed,
        "total_frames": total_frames,
        "total_wall_seconds": round(session_wall, 2),
        "n_intervals": len(interval_data),
        "n_objects": n_objects,
    }

    if interval_data:
        fps_list = [d["fps"] for d in interval_data]
        gpu_mem_list = [d["gpu_memory_mb"] for d in interval_data]
        pt_ip_list = [d.get("pt_ip", 0) for d in interval_data]
        sa_ip_list = [d.get("sa_ip", 0) for d in interval_data]
        pt_r_list = [d.get("pt_mean_r", 0) for d in interval_data]
        pt_slip_list = [d.get("pt_slip_fraction", 0) for d in interval_data]

        import numpy as np

        agg["mean_fps"] = round(float(np.mean(fps_list)), 2)
        agg["std_fps"] = round(float(np.std(fps_list)), 2)
        agg["fps_degradation_pct"] = round(
            ((fps_list[-1] - fps_list[0]) / max(abs(fps_list[0]), 1e-9))
            * 100,
            2,
        ) if len(fps_list) >= 2 else 0.0

        agg["gpu_mem_initial_mb"] = round(gpu_mem_list[0], 2)
        agg["gpu_mem_peak_mb"] = round(float(np.max(gpu_mem_list)), 2)
        agg["gpu_mem_final_mb"] = round(gpu_mem_list[-1], 2)
        agg["gpu_mem_growth_mb"] = round(
            float(np.max(gpu_mem_list)) - gpu_mem_list[0], 2
        )

        agg["pt_mean_ip"] = round(float(np.mean(pt_ip_list)), 6)
        agg["pt_std_ip"] = round(float(np.std(pt_ip_list)), 6)
        agg["sa_mean_ip"] = round(float(np.mean(sa_ip_list)), 6)
        agg["sa_std_ip"] = round(float(np.std(sa_ip_list)), 6)

        agg["pt_mean_order_param"] = round(float(np.mean(pt_r_list)), 6)
        agg["pt_std_order_param"] = round(float(np.std(pt_r_list)), 6)

        agg["pt_mean_slip_fraction"] = round(
            float(np.mean(pt_slip_list)), 6
        )
        agg["pt_max_slip_fraction"] = round(
            float(np.max(pt_slip_list)), 6
        )

        # Throughput trend
        if len(fps_list) >= 2:
            x = np.arange(len(fps_list), dtype=float)
            agg["fps_trend_slope"] = round(
                float(np.polyfit(x, np.array(fps_list), 1)[0]), 6
            )

        # IP trend
        if len(pt_ip_list) >= 2:
            x = np.arange(len(pt_ip_list), dtype=float)
            agg["pt_ip_trend_slope"] = round(
                float(np.polyfit(x, np.array(pt_ip_list), 1)[0]), 8
            )
            agg["sa_ip_trend_slope"] = round(
                float(np.polyfit(x, np.array(sa_ip_list), 1)[0]), 8
            )

        # Order parameter trend
        if len(pt_r_list) >= 2:
            x = np.arange(len(pt_r_list), dtype=float)
            agg["pt_r_trend_slope"] = round(
                float(np.polyfit(x, np.array(pt_r_list), 1)[0]), 8
            )

    del pt, sa
    _cleanup()

    return {
        "aggregate": agg,
        "intervals": interval_data,
    }


# =========================================================================
# Benchmark wrappers for each duration
# =========================================================================


def _run_duration_benchmark(
    duration_minutes: float,
    n_seeds: int = 3,
    label: str = "",
) -> dict[str, Any]:
    """Run *n_seeds* sessions at *duration_minutes* and aggregate."""
    from prinet.utils.y4q1_tools import bootstrap_ci

    print(f"\n=== Session-Length Benchmark: {label} ===")

    seeds = list(range(42, 42 + n_seeds))
    per_seed: list[dict] = []
    ip_pt_all: list[float] = []
    ip_sa_all: list[float] = []
    fps_all: list[float] = []
    r_all: list[float] = []
    slip_all: list[float] = []

    for seed in seeds:
        print(f"  Seed {seed}:")
        result = run_timed_session(
            duration_minutes=duration_minutes,
            seed=seed,
            n_objects=4,
            warmup_seconds=min(30.0, duration_minutes * 60 * 0.1),
        )
        agg = result["aggregate"]
        per_seed.append({
            "seed": seed,
            "aggregate": agg,
            "n_intervals": agg["n_intervals"],
        })

        ip_pt_all.append(agg.get("pt_mean_ip", 0))
        ip_sa_all.append(agg.get("sa_mean_ip", 0))
        fps_all.append(agg.get("mean_fps", 0))
        r_all.append(agg.get("pt_mean_order_param", 0))
        slip_all.append(agg.get("pt_mean_slip_fraction", 0))

        _cleanup()

    # Bootstrap CI across seeds
    ci_pt_ip = bootstrap_ci(ip_pt_all) if len(ip_pt_all) >= 2 else {
        "mean": ip_pt_all[0] if ip_pt_all else 0, "ci_lower": 0, "ci_upper": 0
    }
    ci_sa_ip = bootstrap_ci(ip_sa_all) if len(ip_sa_all) >= 2 else {
        "mean": ip_sa_all[0] if ip_sa_all else 0, "ci_lower": 0, "ci_upper": 0
    }
    ci_fps = bootstrap_ci(fps_all) if len(fps_all) >= 2 else {
        "mean": fps_all[0] if fps_all else 0, "ci_lower": 0, "ci_upper": 0
    }
    ci_r = bootstrap_ci(r_all) if len(r_all) >= 2 else {
        "mean": r_all[0] if r_all else 0, "ci_lower": 0, "ci_upper": 0
    }

    summary = {
        "duration_minutes": duration_minutes,
        "n_seeds": n_seeds,
        "per_seed": per_seed,
        "pt_ip_ci": ci_pt_ip,
        "sa_ip_ci": ci_sa_ip,
        "fps_ci": ci_fps,
        "order_param_ci": ci_r,
        "mean_slip_fraction": round(
            sum(slip_all) / max(len(slip_all), 1), 6
        ),
    }

    print(
        f"\n  Summary: PT_IP={ci_pt_ip['mean']:.4f} "
        f"[{ci_pt_ip.get('ci_lower', 0):.4f}, "
        f"{ci_pt_ip.get('ci_upper', 0):.4f}]  "
        f"SA_IP={ci_sa_ip['mean']:.4f}  "
        f"FPS={ci_fps['mean']:.1f}  "
        f"r={ci_r['mean']:.4f}"
    )

    return summary


def benchmark_5min() -> dict:
    """5-minute session benchmark (3 seeds)."""
    result = _run_duration_benchmark(5.0, n_seeds=3, label="5-Minute Session")
    _save("5min", {
        "benchmark": "session_length_5min",
        "description": (
            "5-minute continuous tracking session with PhaseTracker and "
            "SlotAttention. 3 seeds. Metrics sampled every "
            f"{SAMPLE_INTERVAL_FRAMES} frames."
        ),
        "results": result,
    })
    return result


def benchmark_10min() -> dict:
    """10-minute session benchmark (3 seeds)."""
    result = _run_duration_benchmark(10.0, n_seeds=3, label="10-Minute Session")
    _save("10min", {
        "benchmark": "session_length_10min",
        "description": (
            "10-minute continuous tracking session. 3 seeds."
        ),
        "results": result,
    })
    return result


def benchmark_30min() -> dict:
    """30-minute session benchmark (3 seeds)."""
    result = _run_duration_benchmark(30.0, n_seeds=3, label="30-Minute Session")
    _save("30min", {
        "benchmark": "session_length_30min",
        "description": (
            "30-minute continuous tracking session. 3 seeds."
        ),
        "results": result,
    })
    return result


def benchmark_60min() -> dict:
    """60-minute session benchmark (3 seeds)."""
    result = _run_duration_benchmark(60.0, n_seeds=3, label="60-Minute Session")
    _save("60min", {
        "benchmark": "session_length_60min",
        "description": (
            "60-minute continuous tracking session. 3 seeds."
        ),
        "results": result,
    })
    return result


def benchmark_cross_duration_comparison(
    results: dict[str, dict],
) -> None:
    """Cross-duration statistical comparison (ANOVA + pairwise Cohen's d).

    Compares key metrics across 5/10/30/60 minute sessions.
    """
    from prinet.utils.y4q1_tools import session_length_statistical_comparison

    print("\n=== Cross-Duration Statistical Comparison ===")

    # Collect per-seed means for each duration
    metrics_to_compare = [
        "pt_mean_ip", "sa_mean_ip", "mean_fps",
        "pt_mean_order_param", "pt_mean_slip_fraction",
    ]

    comparisons: dict[str, dict] = {}

    for metric_key in metrics_to_compare:
        by_duration: dict[str, list[float]] = {}
        for dur_label, dur_result in results.items():
            vals = []
            for ps in dur_result.get("per_seed", []):
                v = ps.get("aggregate", {}).get(metric_key, 0)
                vals.append(v)
            if vals:
                by_duration[dur_label] = vals

        if len(by_duration) >= 2:
            comparison = session_length_statistical_comparison(by_duration)
            comparisons[metric_key] = comparison
            print(
                f"  {metric_key}: F={comparison['anova_f']:.3f}, "
                f"p={comparison['anova_p']:.4f}, "
                f"eta2={comparison['eta_squared']:.4f}, "
                f"sig={'YES' if comparison['significant'] else 'no'}"
            )
            for pair, d in comparison.get("pairwise_d", {}).items():
                print(f"    {pair}: d={d:.3f}")

    _save("cross_duration", {
        "benchmark": "cross_duration_statistical_comparison",
        "description": (
            "One-way ANOVA + pairwise Cohen's d comparing key metrics "
            "across 5, 10, 30, 60-minute sessions (3 seeds each). "
            "Tests whether session duration significantly affects "
            "tracking quality, throughput, or stability."
        ),
        "results": comparisons,
    })


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    print("=" * 70)
    print("Y4 Q1.5: Session-Length Dynamics Benchmarks")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Sampling every {SAMPLE_INTERVAL_FRAMES} frames")
    print()

    all_results: dict[str, dict] = {}

    # Run each duration back-to-back
    r5 = benchmark_5min()
    all_results["5min"] = r5

    r10 = benchmark_10min()
    all_results["10min"] = r10

    r30 = benchmark_30min()
    all_results["30min"] = r30

    r60 = benchmark_60min()
    all_results["60min"] = r60

    # Cross-duration comparison
    benchmark_cross_duration_comparison(all_results)

    # Summary table
    print("\n" + "=" * 70)
    print("SESSION-LENGTH SUMMARY")
    print("=" * 70)
    print(f"{'Duration':>10} {'PT_IP':>8} {'SA_IP':>8} {'FPS':>8} "
          f"{'r(t)':>8} {'Slips':>8} {'GPU_MB':>8}")
    print("-" * 70)
    for dur_label, dur_result in all_results.items():
        pt_ip = dur_result.get("pt_ip_ci", {}).get("mean", 0)
        sa_ip = dur_result.get("sa_ip_ci", {}).get("mean", 0)
        fps = dur_result.get("fps_ci", {}).get("mean", 0)
        r_val = dur_result.get("order_param_ci", {}).get("mean", 0)
        slip = dur_result.get("mean_slip_fraction", 0)
        # Get mean GPU mem from first seed
        gpu_mem = 0
        for ps in dur_result.get("per_seed", []):
            gpu_mem = ps.get("aggregate", {}).get("gpu_mem_peak_mb", 0)
            break
        print(
            f"{dur_label:>10} {pt_ip:8.4f} {sa_ip:8.4f} {fps:8.1f} "
            f"{r_val:8.4f} {slip:8.6f} {gpu_mem:8.1f}"
        )

    print("\n" + "=" * 70)
    print("All 5 session-length benchmarks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
