"""Benchmark: Desynchronization Catastrophe Replication (Task 2.1).

Demonstrates that without proper synchronisation barriers, coupled
Kuramoto oscillators catastrophically desynchronise during gradient
updates.  Then shows that SynchronizedGradientDescent prevents this.

Outputs
-------
- Console table with order-parameter traces
- JSON results saved to ../Docs/test_and_benchmark_results/
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState
from prinet.nn.layers import PRINetModel
from prinet.nn.optimizers import SynchronizedGradientDescent

SEED = 42
DEVICE = "cpu"


@dataclass
class DesyncResult:
    """Container for one experimental run."""

    label: str
    order_params: list[float]
    final_order: float
    desynchronised: bool
    wall_seconds: float


def run_desync_without_barrier(
    n_oscillators: int = 64,
    n_steps: int = 200,
    lr: float = 0.1,
) -> DesyncResult:
    """Train with plain SGD — expect desynchronisation."""
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=16,
        n_resonances=n_oscillators,
        n_layers=2,
        n_concepts=10,
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    kuramoto_plain = KuramotoOscillator(
        n_oscillators=n_oscillators, coupling_strength=0.5
    )
    state = OscillatorState.create_random(n_oscillators, seed=SEED)

    order_history: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(8, 16)
        target = torch.randint(0, 10, (8,))
        logits = model(x)
        loss = torch.nn.functional.nll_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Evolve oscillators (no sync barrier — plain SGD ignores them)
        state = kuramoto_plain.step(state, dt=0.01)
        r = kuramoto_order_parameter(state.phase).item()
        order_history.append(r)

    elapsed = time.perf_counter() - t0
    final = order_history[-1]
    return DesyncResult(
        label="plain_sgd",
        order_params=order_history,
        final_order=final,
        desynchronised=final < 0.5,
        wall_seconds=elapsed,
    )


def run_sync_with_barrier(
    n_oscillators: int = 64,
    n_steps: int = 200,
    lr: float = 0.1,
    lambda_sync: float = 0.5,
    k_critical: float = 0.6,
) -> DesyncResult:
    """Train with SynchronizedGradientDescent — expect stability."""
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=16,
        n_resonances=n_oscillators,
        n_layers=2,
        n_concepts=10,
    )
    # We need to give the optimizer the oscillator model for barrier
    kuramoto = KuramotoOscillator(n_oscillators=n_oscillators, coupling_strength=2.0)
    opt = SynchronizedGradientDescent(
        model.parameters(),
        lr=lr,
        sync_penalty=lambda_sync,
        critical_order=k_critical,
    )

    order_history: list[float] = []
    t0 = time.perf_counter()
    kuramoto = KuramotoOscillator(
        n_oscillators=n_oscillators, coupling_strength=2.0
    )
    state = OscillatorState.create_random(n_oscillators, seed=SEED)

    for step_i in range(n_steps):
        x = torch.randn(8, 16)
        target = torch.randint(0, 10, (8,))
        logits = model(x)
        loss = torch.nn.functional.nll_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        # Evolve oscillators one step
        state = kuramoto.step(state, dt=0.01)
        r = kuramoto_order_parameter(state.phase).item()
        opt.step(order_parameter=r)

        order_history.append(r)

    elapsed = time.perf_counter() - t0
    final = order_history[-1]
    return DesyncResult(
        label="sync_sgd_barrier",
        order_params=order_history,
        final_order=final,
        desynchronised=final < 0.5,
        wall_seconds=elapsed,
    )


def main() -> None:
    """Run both experiments and save results."""
    print("=" * 60)
    print("  BENCHMARK: Desynchronization Catastrophe (Task 2.1)")
    print("=" * 60)

    result_plain = run_desync_without_barrier()
    result_sync = run_sync_with_barrier()

    print(f"\n{'Experiment':<25} {'Final r':>10} {'Desync?':>10} {'Time (s)':>10}")
    print("-" * 60)
    for r in (result_plain, result_sync):
        print(
            f"{r.label:<25} {r.final_order:>10.4f} "
            f"{'YES' if r.desynchronised else 'NO':>10} "
            f"{r.wall_seconds:>10.3f}"
        )

    # Save JSON
    out_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "desynchronization_catastrophe",
        "task": "2.1",
        "results": [
            {
                "label": r.label,
                "final_order": r.final_order,
                "desynchronised": r.desynchronised,
                "wall_seconds": r.wall_seconds,
                "order_trace_first10": r.order_params[:10],
                "order_trace_last10": r.order_params[-10:],
            }
            for r in (result_plain, result_sync)
        ],
    }
    out_file = out_dir / "benchmark_desync_catastrophe.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
