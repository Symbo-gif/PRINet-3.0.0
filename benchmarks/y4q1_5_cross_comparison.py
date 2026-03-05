"""Y4 Q1.5 — Standalone Cross-Duration Comparison.

Loads the 4 saved session-length JSON result files and performs
the cross-duration statistical comparison (ANOVA + pairwise Cohen's d)
without re-running any benchmarks.

Also prints a summary table identical to the main script's output.

Usage:
    python benchmarks/y4q1_5_cross_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# -- Ensure project root is on sys.path for imports --
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from prinet.utils.y4q1_tools import session_length_statistical_comparison

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DURATIONS = ["5min", "10min", "30min", "60min"]


def _load(label: str) -> dict:
    path = RESULTS_DIR / f"benchmark_y4q1_5_{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("results", data)


def _save(data: dict) -> None:
    out = RESULTS_DIR / "benchmark_y4q1_5_cross_duration.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved: {out}")


def main() -> None:
    print("=" * 70)
    print("Y4 Q1.5: Cross-Duration Statistical Comparison (standalone)")
    print("=" * 70)

    # Load all 4 duration results
    all_results: dict[str, dict] = {}
    for label in DURATIONS:
        try:
            all_results[label] = _load(label)
            n_seeds = len(all_results[label].get("per_seed", []))
            print(f"  Loaded {label}: {n_seeds} seeds")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(all_results) < 2:
        print("ERROR: Need at least 2 duration results for comparison.")
        return

    # ---- Cross-Duration ANOVA ----
    print("\n=== Cross-Duration Statistical Comparison ===")

    metrics_to_compare = [
        "pt_mean_ip", "sa_mean_ip", "mean_fps",
        "pt_mean_order_param", "pt_mean_slip_fraction",
    ]

    comparisons: dict[str, dict] = {}

    for metric_key in metrics_to_compare:
        by_duration: dict[str, list[float]] = {}
        for dur_label, dur_result in all_results.items():
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

    # Save comparison JSON
    _save({
        "benchmark": "cross_duration_statistical_comparison",
        "description": (
            "One-way ANOVA + pairwise Cohen's d comparing key metrics "
            "across 5, 10, 30, 60-minute sessions (3 seeds each). "
            "Tests whether session duration significantly affects "
            "tracking quality, throughput, or stability."
        ),
        "results": comparisons,
    })

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print("SESSION-LENGTH SUMMARY")
    print("=" * 70)
    print(
        f"{'Duration':>10} {'PT_IP':>8} {'SA_IP':>8} {'FPS':>8} "
        f"{'r(t)':>8} {'Slips':>8} {'GPU_MB':>8}"
    )
    print("-" * 70)
    for dur_label, dur_result in all_results.items():
        pt_ip = dur_result.get("pt_ip_ci", {}).get("mean", 0)
        sa_ip = dur_result.get("sa_ip_ci", {}).get("mean", 0)
        fps = dur_result.get("fps_ci", {}).get("mean", 0)
        r_val = dur_result.get("order_param_ci", {}).get("mean", 0)
        slip = dur_result.get("mean_slip_fraction", 0)
        gpu_mem = 0
        for ps in dur_result.get("per_seed", []):
            gpu_mem = ps.get("aggregate", {}).get("gpu_mem_peak_mb", 0)
            break
        print(
            f"{dur_label:>10} {pt_ip:8.4f} {sa_ip:8.4f} {fps:8.1f} "
            f"{r_val:8.4f} {slip:8.6f} {gpu_mem:8.1f}"
        )

    # ---- Interpretation ----
    print("\n=== Scientific Interpretation ===")

    # Check for significant degradation across durations
    fps_comp = comparisons.get("mean_fps", {})
    ip_comp = comparisons.get("pt_mean_ip", {})
    r_comp = comparisons.get("pt_mean_order_param", {})

    if fps_comp.get("significant"):
        print(
            f"  FPS: SIGNIFICANT duration effect "
            f"(F={fps_comp['anova_f']:.2f}, p={fps_comp['anova_p']:.4f}, "
            f"eta2={fps_comp['eta_squared']:.3f})"
        )
    else:
        print(
            f"  FPS: No significant duration effect "
            f"(F={fps_comp.get('anova_f', 0):.2f}, "
            f"p={fps_comp.get('anova_p', 1):.4f})"
        )

    if ip_comp.get("significant"):
        print(
            f"  PT Identity Preservation: SIGNIFICANT duration effect "
            f"(F={ip_comp['anova_f']:.2f}, p={ip_comp['anova_p']:.4f}, "
            f"eta2={ip_comp['eta_squared']:.3f})"
        )
    else:
        print(
            f"  PT Identity Preservation: No significant duration effect "
            f"(F={ip_comp.get('anova_f', 0):.2f}, "
            f"p={ip_comp.get('anova_p', 1):.4f})"
        )

    if r_comp.get("significant"):
        print(
            f"  Order param r(t): SIGNIFICANT duration effect "
            f"(F={r_comp['anova_f']:.2f}, p={r_comp['anova_p']:.4f}, "
            f"eta2={r_comp['eta_squared']:.3f})"
        )
    else:
        print(
            f"  Order param r(t): No significant duration effect "
            f"(F={r_comp.get('anova_f', 0):.2f}, "
            f"p={r_comp.get('anova_p', 1):.4f})"
        )

    print("\n" + "=" * 70)
    print("Cross-duration comparison complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
