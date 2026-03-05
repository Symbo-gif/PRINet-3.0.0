"""Subconscious Layer Overhead Benchmark.

Measures:
    1. ONNX inference latency per backend (CPU / DirectML / NPU).
    2. Daemon thread overhead on a synthetic GPU-style workload.
    3. State serialization throughput.

Usage::

    python benchmarks/subconscious_benchmark.py [--duration 30] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from prinet.core.subconscious import CONTROL_DIM, STATE_DIM, SubconsciousState
from prinet.core.subconscious_daemon import SubconsciousDaemon, collect_system_state
from prinet.nn.subconscious_model import SubconsciousController
from prinet.utils.npu_backend import (
    BackendType,
    backend_info,
    create_session,
    detect_best_backend,
)


# ======================================================================
# Helpers
# ======================================================================


def _dummy_workload(n: int) -> float:
    """CPU-bound workload simulating a training step."""
    total = 0.0
    for i in range(n):
        total += math.sin(float(i)) * math.cos(float(i))
    return total


def _export_model(path: Path) -> Path:
    """Export a fresh SubconsciousController to ONNX."""
    model = SubconsciousController()
    return model.export_to_onnx(path)


# ======================================================================
# Benchmark: ONNX inference latency
# ======================================================================


def bench_inference_latency(
    model_path: str | Path,
    backend: BackendType,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> dict[str, float]:
    """Measure ONNX inference latency on a given backend.

    Args:
        model_path: Path to the ONNX model.
        backend: Execution provider to target.
        n_warmup: Number of warm-up runs (not timed).
        n_iters: Number of timed runs.

    Returns:
        Dict with ``mean_ms``, ``p50_ms``, ``p95_ms``, ``min_ms``, ``max_ms``.
    """
    try:
        session = create_session(model_path, backend)
    except Exception as e:
        return {"error": str(e)}

    z = np.random.randn(1, STATE_DIM).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warm up
    for _ in range(n_warmup):
        session.run(None, {input_name: z})

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: z})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    p50_idx = int(len(latencies_ms) * 0.50)
    p95_idx = min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)

    return {
        "backend": backend,
        "n_iters": n_iters,
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": latencies_ms[p50_idx],
        "p95_ms": latencies_ms[p95_idx],
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "stdev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
    }


# ======================================================================
# Benchmark: daemon overhead
# ======================================================================


def bench_daemon_overhead(
    model_path: str | Path,
    work_iterations: int = 200_000,
    n_trials: int = 5,
    submit_interval: float = 0.5,
) -> dict[str, float]:
    """Measure daemon overhead on a synthetic CPU workload.

    Args:
        model_path: Path to the ONNX model.
        work_iterations: Size of the dummy workload per trial.
        n_trials: Number of measurement trials.
        submit_interval: Seconds between state submissions.

    Returns:
        Dict with ``baseline_s``, ``with_daemon_s``, ``overhead_pct``.
    """
    # Baseline: no daemon
    baseline_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _dummy_workload(work_iterations)
        baseline_times.append(time.perf_counter() - t0)

    baseline_mean = statistics.mean(baseline_times)

    # With daemon
    daemon = SubconsciousDaemon(
        model_path, backend="cpu", interval=0.2, warmup=True
    )
    daemon.start()
    time.sleep(0.5)

    daemon_times: list[float] = []
    for _ in range(n_trials):
        daemon.submit_state(SubconsciousState.default())
        t0 = time.perf_counter()
        _dummy_workload(work_iterations)
        daemon_times.append(time.perf_counter() - t0)
        time.sleep(submit_interval)

    daemon_mean = statistics.mean(daemon_times)
    inferences = daemon.inference_count
    daemon.stop(timeout=5.0)

    overhead_pct = ((daemon_mean - baseline_mean) / max(baseline_mean, 1e-9)) * 100.0

    return {
        "work_iterations": work_iterations,
        "n_trials": n_trials,
        "baseline_mean_s": baseline_mean,
        "with_daemon_mean_s": daemon_mean,
        "overhead_pct": overhead_pct,
        "daemon_inferences": inferences,
    }


# ======================================================================
# Benchmark: state serialization throughput
# ======================================================================


def bench_state_serialization(n_iters: int = 10_000) -> dict[str, float]:
    """Measure SubconsciousState.to_tensor() throughput.

    Args:
        n_iters: Number of serialization iterations.

    Returns:
        Dict with ``total_s``, ``per_call_us``, ``throughput_hz``.
    """
    state = SubconsciousState(
        r_per_band=[0.8, 0.5, 0.3],
        r_global=0.6,
        loss_ema=1.5,
        loss_variance=0.1,
        grad_norm_ema=2.0,
        lr_current=1e-4,
        scalr_alpha=0.7,
        gpu_temp=72.0,
        gpu_util=0.85,
        vram_pct=0.60,
        cpu_util=0.40,
        step_latency_p50=0.012,
        step_latency_p95=0.025,
        throughput=5000.0,
        epoch=42,
        regime="sparse_knn",
        timestamp=time.time(),
    )

    t0 = time.perf_counter()
    for _ in range(n_iters):
        state.to_tensor()
    total = time.perf_counter() - t0

    return {
        "n_iters": n_iters,
        "total_s": total,
        "per_call_us": (total / n_iters) * 1e6,
        "throughput_hz": n_iters / max(total, 1e-9),
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Subconscious Layer Benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of inference iterations per backend.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Subconscious Layer Benchmark")
    print("=" * 60)

    # Export model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _export_model(Path(tmpdir) / "controller.onnx")
        print(f"\nModel exported: {model_path}")

        info = backend_info()
        print(f"Backend info: {json.dumps(info, indent=2)}")

        results: dict[str, object] = {"backend_info": info}

        # 1. Inference latency per backend
        print("\n--- Inference Latency ---")
        best_backend = detect_best_backend()
        for be in ["cpu", best_backend]:
            print(f"  Backend: {be}")
            lat = bench_inference_latency(model_path, be, n_iters=args.n_iters)  # type: ignore[arg-type]
            for k, v in lat.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
            results[f"inference_{be}"] = lat

        # 2. Daemon overhead
        print("\n--- Daemon Overhead ---")
        overhead = bench_daemon_overhead(model_path)
        for k, v in overhead.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        results["daemon_overhead"] = overhead

        # 3. State serialization
        print("\n--- State Serialization ---")
        ser = bench_state_serialization()
        for k, v in ser.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        results["state_serialization"] = ser

        # Write results
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
