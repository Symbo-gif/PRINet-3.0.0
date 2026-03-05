"""Benchmark: 10,000 oscillator scalability (Task 1.7).

Success criterion: Simulate 10,000 Kuramoto oscillators for 1,000
timesteps in under 10 seconds on CPU.

Also measures scaling from 100 → 10,000 oscillators.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState
from prinet.utils.cuda_kernels import FixedStepRK4Solver

SEED = 42


def benchmark_oscillator_scaling() -> list[dict]:
    """Run oscillator simulation at multiple scales."""
    sizes = [100, 500, 1_000, 5_000, 10_000]
    results: list[dict] = []

    for n in sizes:
        torch.manual_seed(SEED)
        # Use mean-field approximation for large systems (O(N) vs O(N²))
        use_mean_field = n >= 1_000

        model = KuramotoOscillator(
            n_oscillators=n,
            coupling_strength=2.0 / n,
            decay_rate=0.1,
            mean_field=use_mean_field,
        )

        state = OscillatorState.create_random(n, seed=SEED)

        n_steps = 1_000
        dt = 0.01

        t0 = time.perf_counter()
        final_state, trajectory = model.integrate(
            state, n_steps=n_steps, dt=dt, method="rk4"
        )
        elapsed = time.perf_counter() - t0

        r = kuramoto_order_parameter(final_state.phase).item()
        entry = {
            "n_oscillators": n,
            "n_steps": n_steps,
            "dt": dt,
            "sparsity": "mean_field" if use_mean_field else "dense",
            "wall_seconds": round(elapsed, 4),
            "final_order_param": round(r, 6),
            "passed": elapsed < 10.0 if n == 10_000 else True,
        }
        results.append(entry)
        status = "PASS" if entry["passed"] else "FAIL"
        print(
            f"  N={n:>6d}  t={elapsed:>8.3f}s  r={r:.4f}  [{status}]"
        )

    return results


def main() -> None:
    print("=" * 60)
    print("  BENCHMARK: Oscillator Scaling (Task 1.7)")
    print("  Target: 10k oscillators, 1k steps < 10 s")
    print("=" * 60)
    print()

    results = benchmark_oscillator_scaling()

    # Summary
    target = next(r for r in results if r["n_oscillators"] == 10_000)
    print()
    if target["passed"]:
        print(
            f"SUCCESS: 10,000 oscillators completed in "
            f"{target['wall_seconds']:.3f}s (< 10s)"
        )
    else:
        print(
            f"FAIL: 10,000 oscillators took "
            f"{target['wall_seconds']:.3f}s (>= 10s)"
        )

    # Save JSON
    out_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "oscillator_scaling",
        "task": "1.7",
        "target": "10000 oscillators, 1000 steps, < 10 seconds",
        "results": results,
    }
    out_file = out_dir / "benchmark_oscillator_scaling.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
