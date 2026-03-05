"""Activation Function GPU Profiling.

Profiles ``dSiLU``, ``HolomorphicActivation``, and ``PhaseActivation``
on GPU at batch_size=256, N=1024. Saves timing results.

NN TODO Task: Activation profiling.

Usage::

    python -m benchmarks.activation_profile
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from prinet.nn.activations import (
    GatedPhaseActivation,
    HolomorphicActivation,
    PhaseActivation,
    dSiLU,
)

SEED = 42
BATCH_SIZE = 256
N = 1024
N_WARMUP = 10
N_ITERS = 50


def _profile_activation(
    activation_fn: Any,
    name: str,
    x: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    """Profile a single activation function forward + backward."""
    # Warmup
    for _ in range(N_WARMUP):
        x_in = x.clone().detach().requires_grad_(True)
        out = activation_fn(x_in)
        loss = out.sum()
        loss.backward()

    if device != "cpu":
        torch.cuda.synchronize()

    fwd_times: list[float] = []
    bwd_times: list[float] = []

    for _ in range(N_ITERS):
        x_in = x.clone().detach().requires_grad_(True)

        t0 = time.perf_counter()
        out = activation_fn(x_in)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = out.sum()
        loss.backward()
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)  # ms
        bwd_times.append((t2 - t1) * 1000)

    return {
        "name": name,
        "device": device,
        "batch_size": BATCH_SIZE,
        "n_features": N,
        "fwd_mean_ms": sum(fwd_times) / len(fwd_times),
        "fwd_min_ms": min(fwd_times),
        "fwd_max_ms": max(fwd_times),
        "bwd_mean_ms": sum(bwd_times) / len(bwd_times),
        "bwd_min_ms": min(bwd_times),
        "bwd_max_ms": max(bwd_times),
        "total_mean_ms": (sum(fwd_times) + sum(bwd_times)) / len(fwd_times),
    }


def main() -> None:
    """Run activation profiling suite."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    torch.manual_seed(SEED)

    activations: list[tuple[str, Any]] = [
        ("dSiLU", dSiLU()),
        ("HolomorphicActivation", HolomorphicActivation()),
        ("PhaseActivation", PhaseActivation()),
        ("GatedPhaseActivation", GatedPhaseActivation(n_dims=N)),
    ]

    all_results: list[dict[str, Any]] = []

    for dev in devices:
        print(f"\n--- Device: {dev} ---")
        x = torch.randn(BATCH_SIZE, N, device=dev)

        for name, fn in activations:
            fn_dev = fn.to(dev) if hasattr(fn, "to") else fn
            try:
                result = _profile_activation(fn_dev, name, x, dev)
                all_results.append(result)
                print(
                    f"  {name:30s} fwd={result['fwd_mean_ms']:.3f}ms  "
                    f"bwd={result['bwd_mean_ms']:.3f}ms  "
                    f"total={result['total_mean_ms']:.3f}ms"
                )
            except Exception as exc:
                print(f"  {name:30s} ERROR: {exc}")
                all_results.append(
                    {"name": name, "device": dev, "error": str(exc)}
                )

    # Save results
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "activation_profile.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "profile": "activation_functions",
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
