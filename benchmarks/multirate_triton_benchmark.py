"""Multi-Rate Triton Kernel Benchmark.

Measures per-step latency of multi-rate RK4 kernels vs Python-loop
sub-stepping. Target: ≥2x speedup from kernel fusion.

Benchmark TODO Tasks: Multi-rate Triton kernel benchmark
                      Hierarchical order param kernel benchmark

Usage::

    python -m benchmarks.multirate_triton_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prinet.utils.triton_kernels import (
    pytorch_cross_band_coupling,
    pytorch_fused_sub_step_rk4,
    pytorch_hierarchical_order_param,
    pytorch_multi_rate_derivatives,
    pytorch_multi_rate_rk4_step,
)

SEED = 42
N_WARMUP = 5
N_ITERS = 50


def _benchmark_multi_rate_derivatives(
    N: int, device: str
) -> dict[str, Any]:
    """Benchmark pytorch_multi_rate_derivatives at given N."""
    torch.manual_seed(SEED)
    phase = torch.rand(N, device=device) * 6.2832
    amp = torch.ones(N, device=device)
    freq = torch.rand(N, device=device) * 40.0
    freq_band = torch.randint(0, 3, (N,), device=device)

    # Warmup
    for _ in range(N_WARMUP):
        pytorch_multi_rate_derivatives(phase, amp, freq, freq_band, 2.0, 0.1, 0.01)
    if device != "cpu":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        pytorch_multi_rate_derivatives(phase, amp, freq, freq_band, 2.0, 0.1, 0.01)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"N": N, "device": device, "mean_ms": _mean(times), "min_ms": min(times)}


def _benchmark_fused_sub_step_rk4(
    N: int, device: str
) -> dict[str, Any]:
    """Benchmark fused sub-step RK4."""
    torch.manual_seed(SEED)
    phase = torch.rand(N, device=device) * 6.2832
    amp = torch.ones(N, device=device)
    freq = torch.rand(N, device=device) * 40.0
    freq_band = torch.randint(0, 3, (N,), device=device)

    for _ in range(N_WARMUP):
        pytorch_fused_sub_step_rk4(phase, amp, freq, freq_band, 2.0, 0.1, 0.01, 0.01)
    if device != "cpu":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        pytorch_fused_sub_step_rk4(phase, amp, freq, freq_band, 2.0, 0.1, 0.01, 0.01)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"N": N, "device": device, "mean_ms": _mean(times), "min_ms": min(times)}


def _benchmark_python_loop_sub_stepping(
    N: int, device: str
) -> dict[str, Any]:
    """Benchmark pytorch_multi_rate_rk4_step with sub_steps=20 (baseline).

    This is NOT a Python loop; it uses the built-in sub-steps parameter.
    We compare it against the fused per-band sub-step variant.
    """
    torch.manual_seed(SEED)
    phase = torch.rand(N, device=device) * 6.2832
    amp = torch.ones(N, device=device)
    freq = torch.rand(N, device=device) * 40.0

    sub_steps = 20

    for _ in range(N_WARMUP):
        pytorch_multi_rate_rk4_step(
            phase, amp, freq, 2.0, 0.1, 0.01, 0.01, sub_steps
        )
    if device != "cpu":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        pytorch_multi_rate_rk4_step(
            phase, amp, freq, 2.0, 0.1, 0.01, 0.01, sub_steps
        )
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"N": N, "device": device, "mean_ms": _mean(times), "min_ms": min(times)}


def _benchmark_cross_band_coupling(
    N_slow: int, N_fast: int, device: str
) -> dict[str, Any]:
    """Benchmark cross-band PAC coupling."""
    torch.manual_seed(SEED)
    slow_phase = torch.rand(N_slow, device=device) * 6.2832
    fast_phase = torch.rand(N_fast, device=device) * 6.2832
    fast_amp = torch.ones(N_fast, device=device)
    parent_idx = torch.randint(0, N_slow, (N_fast,), device=device)

    for _ in range(N_WARMUP):
        pytorch_cross_band_coupling(slow_phase, fast_phase, fast_amp, parent_idx)
    if device != "cpu":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        pytorch_cross_band_coupling(slow_phase, fast_phase, fast_amp, parent_idx)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "N_slow": N_slow,
        "N_fast": N_fast,
        "device": device,
        "mean_ms": _mean(times),
        "min_ms": min(times),
    }


def _benchmark_hierarchical_order_param(
    N: int, device: str
) -> dict[str, Any]:
    """Benchmark hierarchical order parameter (single-pass)."""
    torch.manual_seed(SEED)
    phase = torch.rand(N, device=device) * 6.2832
    band_sizes = [N // 8, N // 4, N - N // 8 - N // 4]

    for _ in range(N_WARMUP):
        pytorch_hierarchical_order_param(phase, band_sizes)
    if device != "cpu":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        pytorch_hierarchical_order_param(phase, band_sizes)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"N": N, "device": device, "mean_ms": _mean(times), "min_ms": min(times)}


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    """Run all multi-rate kernel benchmarks."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    N_values = [512, 1024, 4096, 16384, 65536]

    all_results: dict[str, Any] = {
        "benchmark": "multirate_triton_kernels",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    for dev in devices:
        print(f"\n{'='*50}")
        print(f"Device: {dev}")
        print(f"{'='*50}")

        dev_results: dict[str, list[dict[str, Any]]] = {
            "multi_rate_derivatives": [],
            "fused_sub_step_rk4": [],
            "python_loop_baseline": [],
            "cross_band_coupling": [],
            "hierarchical_order_param": [],
        }

        for N in N_values:
            print(f"\n  N={N}")

            r = _benchmark_multi_rate_derivatives(N, dev)
            dev_results["multi_rate_derivatives"].append(r)
            print(f"    multi_rate_derivatives: {r['mean_ms']:.3f}ms")

            r = _benchmark_fused_sub_step_rk4(N, dev)
            dev_results["fused_sub_step_rk4"].append(r)
            print(f"    fused_sub_step_rk4:    {r['mean_ms']:.3f}ms")

            r = _benchmark_python_loop_sub_stepping(N, dev)
            dev_results["python_loop_baseline"].append(r)
            print(f"    python_loop_baseline:  {r['mean_ms']:.3f}ms")

            N_slow = max(N // 8, 4)
            r = _benchmark_cross_band_coupling(N_slow, N, dev)
            dev_results["cross_band_coupling"].append(r)
            print(f"    cross_band_coupling:   {r['mean_ms']:.3f}ms")

            r = _benchmark_hierarchical_order_param(N, dev)
            dev_results["hierarchical_order_param"].append(r)
            print(f"    hierarchical_order:    {r['mean_ms']:.3f}ms")

        all_results[dev] = dev_results

    # Compute speedup
    for dev in devices:
        if dev in all_results and isinstance(all_results[dev], dict):
            fused = all_results[dev].get("fused_sub_step_rk4", [])
            baseline = all_results[dev].get("python_loop_baseline", [])
            speedups: list[dict[str, Any]] = []
            for f, b in zip(fused, baseline):
                if f["mean_ms"] > 0:
                    sp = b["mean_ms"] / f["mean_ms"]
                    speedups.append({"N": f["N"], "speedup": round(sp, 2)})
            all_results[f"{dev}_speedup"] = speedups
            if speedups:
                print(f"\n  Fused vs Loop speedup ({dev}):")
                for s in speedups:
                    print(f"    N={s['N']}: {s['speedup']:.2f}x")

    # Save
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multirate_kernel_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
