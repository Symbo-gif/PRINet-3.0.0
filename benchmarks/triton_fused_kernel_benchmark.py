"""Fused-Kernel Benchmark — Triton vs PyTorch Kuramoto step latency.

Task 0.4: Measure latency reduction from Triton-fused kernels for
batched RK4 steps and sparse k-NN coupling at multiple oscillator
counts, including small N where kernel-launch overhead dominates.

Benchmarks:
    1. Mean-field RK4 step: Triton fused vs PyTorch reference.
    2. Sparse k-NN coupling: Triton fused vs PyTorch reference.

Each configuration is timed with CUDA events for microsecond precision,
repeated over sufficient iterations with warm-up. Results are saved to
JSON and printed as a formatted Markdown table.

Saves results to:
    Docs/test_and_benchmark_results/triton_fused_kernel_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.utils.triton_kernels import (
    pytorch_mean_field_rk4_step,
    pytorch_sparse_knn_coupling,
    triton_available,
    triton_fused_mean_field_rk4_step,
    triton_sparse_knn_coupling,
)

SEED = 42
DEVICE = torch.device("cuda")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"

# ── Timing parameters ─────────────────────────────────────────────
WARMUP_ITERS = 50
BENCH_ITERS = 200
SMALL_N_EXTRA_ITERS = 500  # More iterations at small N for stable timing


# ══════════════════════════════════════════════════════════════════
# Timing Utilities
# ══════════════════════════════════════════════════════════════════


def cuda_timer(
    fn: Callable[[], None],
    n_warmup: int = WARMUP_ITERS,
    n_iter: int = BENCH_ITERS,
) -> float:
    """Time a CUDA kernel call using CUDA events.

    Returns:
        Median latency per call in microseconds.
    """
    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)  # ms → µs

    timings.sort()
    # Return median (robust to outliers)
    return timings[len(timings) // 2]


# ══════════════════════════════════════════════════════════════════
# 1. Mean-Field RK4 Benchmark
# ══════════════════════════════════════════════════════════════════


def benchmark_mean_field_rk4() -> list[dict]:
    """Benchmark Triton vs PyTorch mean-field RK4 step at multiple N."""
    sizes = [256, 512, 1024, 4096, 8192, 16384, 65536]
    K, decay, gamma, dt = 2.0, 0.1, 0.01, 0.01
    results: list[dict] = []

    print("\n" + "=" * 72)
    print("  MEAN-FIELD RK4: Triton Fused vs PyTorch Reference")
    print("=" * 72)
    print(f"{'N':>8} | {'PyTorch µs':>12} | {'Triton µs':>12} | {'Speedup':>8} | Note")
    print("-" * 72)

    for N in sizes:
        torch.manual_seed(SEED)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        phase = torch.rand(N, device=DEVICE, generator=gen) * 6.2832
        amp = torch.ones(N, device=DEVICE)
        freq = torch.randn(N, device=DEVICE, generator=gen)

        n_iter = SMALL_N_EXTRA_ITERS if N <= 1024 else BENCH_ITERS

        # PyTorch reference
        pt_us = cuda_timer(
            lambda ph=phase, am=amp, fr=freq: pytorch_mean_field_rk4_step(
                ph, am, fr, K, decay, gamma, dt,
            ),
            n_iter=n_iter,
        )

        # Triton fused
        tr_us = cuda_timer(
            lambda ph=phase, am=amp, fr=freq: triton_fused_mean_field_rk4_step(
                ph, am, fr, K, decay, gamma, dt,
            ),
            n_iter=n_iter,
        )

        speedup = pt_us / tr_us if tr_us > 0 else float("inf")
        note = "✓ faster" if speedup > 1.0 else "slower"

        entry = {
            "kernel": "mean_field_rk4",
            "N": N,
            "pytorch_us": round(pt_us, 2),
            "triton_us": round(tr_us, 2),
            "speedup": round(speedup, 3),
            "iters": n_iter,
        }
        results.append(entry)
        print(f"{N:>8} | {pt_us:>12.2f} | {tr_us:>12.2f} | {speedup:>7.3f}× | {note}")

    return results


# ══════════════════════════════════════════════════════════════════
# 2. Sparse k-NN Coupling Benchmark
# ══════════════════════════════════════════════════════════════════


def benchmark_sparse_knn() -> list[dict]:
    """Benchmark Triton vs PyTorch sparse k-NN coupling at multiple N."""
    configs = [
        (256, 8),
        (512, 14),
        (1024, 14),
        (4096, 14),
        (8192, 14),
        (16384, 14),
        (65536, 14),
    ]
    K, decay, gamma = 2.0, 0.1, 0.01
    results: list[dict] = []

    print("\n" + "=" * 72)
    print("  SPARSE k-NN COUPLING: Triton Fused vs PyTorch Reference")
    print("=" * 72)
    print(f"{'N':>8} {'k':>3} | {'PyTorch µs':>12} | {'Triton µs':>12} | {'Speedup':>8} | Note")
    print("-" * 72)

    for N, k in configs:
        torch.manual_seed(SEED)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        phase = torch.rand(N, device=DEVICE, generator=gen) * 6.2832
        amp = torch.ones(N, device=DEVICE)
        freq = torch.randn(N, device=DEVICE, generator=gen)
        nbr = torch.randint(0, N, (N, k), device=DEVICE, generator=gen)

        n_iter = SMALL_N_EXTRA_ITERS if N <= 1024 else BENCH_ITERS

        # PyTorch reference
        pt_us = cuda_timer(
            lambda ph=phase, am=amp, fr=freq, nb=nbr: pytorch_sparse_knn_coupling(
                ph, am, fr, nb, K, decay, gamma,
            ),
            n_iter=n_iter,
        )

        # Triton fused
        tr_us = cuda_timer(
            lambda ph=phase, am=amp, fr=freq, nb=nbr: triton_sparse_knn_coupling(
                ph, am, fr, nb, K, decay, gamma,
            ),
            n_iter=n_iter,
        )

        speedup = pt_us / tr_us if tr_us > 0 else float("inf")
        note = "✓ faster" if speedup > 1.0 else "slower"

        entry = {
            "kernel": "sparse_knn",
            "N": N,
            "k": k,
            "pytorch_us": round(pt_us, 2),
            "triton_us": round(tr_us, 2),
            "speedup": round(speedup, 3),
            "iters": n_iter,
        }
        results.append(entry)
        print(f"{N:>8} {k:>3} | {pt_us:>12.2f} | {tr_us:>12.2f} | {speedup:>7.3f}× | {note}")

    return results


# ══════════════════════════════════════════════════════════════════
# 3. Combined Results & Report
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run all fused-kernel benchmarks and save results."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Exiting.")
        sys.exit(1)
    if not triton_available():
        print("ERROR: Triton not available. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Triton fused-kernel benchmark — Warm-up: {WARMUP_ITERS}, Iters: {BENCH_ITERS}")

    mf_results = benchmark_mean_field_rk4()
    sp_results = benchmark_sparse_knn()

    all_results = {
        "metadata": {
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": _get_triton_version(),
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "timing_method": "CUDA events (median µs)",
        },
        "mean_field_rk4": mf_results,
        "sparse_knn": sp_results,
    }

    # ── Summary ───────────────────────────────────────────────────
    mf_speedups = [r["speedup"] for r in mf_results]
    sp_speedups = [r["speedup"] for r in sp_results]

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"Mean-field RK4 speedup range: {min(mf_speedups):.3f}× – {max(mf_speedups):.3f}×")
    print(f"Sparse k-NN    speedup range: {min(sp_speedups):.3f}× – {max(sp_speedups):.3f}×")

    best_mf = max(mf_results, key=lambda r: r["speedup"])
    best_sp = max(sp_results, key=lambda r: r["speedup"])
    print(f"Best MF speedup: {best_mf['speedup']:.3f}× at N={best_mf['N']}")
    print(f"Best SP speedup: {best_sp['speedup']:.3f}× at N={best_sp['N']}")

    # ── Pass criteria ─────────────────────────────────────────────
    # Triton should show speedup at large N (≥16K)
    large_mf = [r for r in mf_results if r["N"] >= 16384]
    large_sp = [r for r in sp_results if r["N"] >= 16384]
    mf_pass = any(r["speedup"] > 1.0 for r in large_mf)
    sp_pass = any(r["speedup"] > 1.0 for r in large_sp)

    all_results["summary"] = {
        "mf_speedup_range": [min(mf_speedups), max(mf_speedups)],
        "sp_speedup_range": [min(sp_speedups), max(sp_speedups)],
        "best_mf": {"N": best_mf["N"], "speedup": best_mf["speedup"]},
        "best_sp": {"N": best_sp["N"], "speedup": best_sp["speedup"]},
        "mf_pass_at_large_N": mf_pass,
        "sp_pass_at_large_N": sp_pass,
        "overall_pass": mf_pass or sp_pass,
    }

    status = "PASS" if all_results["summary"]["overall_pass"] else "FAIL"
    print(f"\nOverall: {status}")

    # ── Save to JSON ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "triton_fused_kernel_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _get_triton_version() -> str:
    """Get Triton version string."""
    try:
        import triton
        return triton.__version__
    except ImportError:
        return "N/A"


if __name__ == "__main__":
    main()
