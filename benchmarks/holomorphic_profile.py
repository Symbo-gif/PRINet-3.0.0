"""Holomorphic Energy Profiling — Fashion-MNIST Scale.

Profiles ``HolomorphicEnergy`` forward/backward passes on a
784 → 64 → 10 architecture (Fashion-MNIST scale) to identify
bottlenecks in free/nudge phases and Jacobian computation.

Core TODO Task: Holomorphic dynamics refinement.

Usage::

    python -m benchmarks.holomorphic_profile
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from prinet.nn.hep import HolomorphicEnergy, HolomorphicEPTrainer
from prinet.nn.layers import PRINetModel

# Configuration
SEED = 42
BATCH_SIZE = 64
N_INPUT = 784  # Fashion-MNIST flat
N_RESONANCES = 64
N_CONCEPTS = 10
N_WARMUP = 5
N_ITERS = 20


def _profile_holomorphic_energy(device: str = "cpu") -> dict[str, Any]:
    """Profile HolomorphicEnergy forward + backward."""
    torch.manual_seed(SEED)

    energy_fn = HolomorphicEnergy(n_oscillators=N_RESONANCES).to(device)
    z = torch.randn(BATCH_SIZE, N_RESONANCES, dtype=torch.cfloat, device=device)
    z.requires_grad_(True)
    coupling = torch.randn(N_RESONANCES, N_RESONANCES, device=device)
    coupling = (coupling + coupling.T) / 2.0  # symmetric
    logits = torch.randn(BATCH_SIZE, N_CONCEPTS, device=device)
    labels = torch.randint(0, N_CONCEPTS, (BATCH_SIZE,), device=device)

    # Warmup
    for _ in range(N_WARMUP):
        if z.grad is not None:
            z.grad = None
        e = energy_fn(z, coupling, logits, labels, beta=1.0)
        e.backward()

    if device != "cpu":
        torch.cuda.synchronize()

    # Free phase timing
    free_fwd_times: list[float] = []
    free_bwd_times: list[float] = []
    for _ in range(N_ITERS):
        if z.grad is not None:
            z.grad = None

        t0 = time.perf_counter()
        e = energy_fn(z, coupling)  # free phase: no target
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e.backward()
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        free_fwd_times.append(t1 - t0)
        free_bwd_times.append(t2 - t1)

    # Nudge phase timing
    nudge_fwd_times: list[float] = []
    nudge_bwd_times: list[float] = []
    for _ in range(N_ITERS):
        if z.grad is not None:
            z.grad = None

        t0 = time.perf_counter()
        e = energy_fn(z, coupling, logits, labels, beta=1.0)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e.backward()
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        nudge_fwd_times.append(t1 - t0)
        nudge_bwd_times.append(t2 - t1)

    return {
        "device": device,
        "batch_size": BATCH_SIZE,
        "n_oscillators": N_RESONANCES,
        "n_concepts": N_CONCEPTS,
        "free_fwd_ms": _stats(free_fwd_times),
        "free_bwd_ms": _stats(free_bwd_times),
        "nudge_fwd_ms": _stats(nudge_fwd_times),
        "nudge_bwd_ms": _stats(nudge_bwd_times),
    }


def _profile_full_model(device: str = "cpu") -> dict[str, Any]:
    """Profile a full PRINetModel forward + backward at Fashion-MNIST scale."""
    torch.manual_seed(SEED)

    model = PRINetModel(
        n_resonances=N_RESONANCES,
        n_dims=N_INPUT,
        n_concepts=N_CONCEPTS,
    ).to(device)

    x = torch.randn(BATCH_SIZE, N_INPUT, device=device)
    labels = torch.randint(0, N_CONCEPTS, (BATCH_SIZE,), device=device)

    # Warmup
    for _ in range(N_WARMUP):
        model.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, labels)
        loss.backward()

    if device != "cpu":
        torch.cuda.synchronize()

    fwd_times: list[float] = []
    bwd_times: list[float] = []
    for _ in range(N_ITERS):
        model.zero_grad()

        t0 = time.perf_counter()
        out = model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = F.nll_loss(out, labels)
        loss.backward()
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    return {
        "device": device,
        "batch_size": BATCH_SIZE,
        "n_input": N_INPUT,
        "n_resonances": N_RESONANCES,
        "n_concepts": N_CONCEPTS,
        "param_count": sum(p.numel() for p in model.parameters()),
        "fwd_ms": _stats(fwd_times),
        "bwd_ms": _stats(bwd_times),
    }


def _stats(times: list[float]) -> dict[str, float]:
    """Compute timing statistics in milliseconds."""
    ms = [t * 1000 for t in times]
    return {
        "mean": sum(ms) / len(ms),
        "min": min(ms),
        "max": max(ms),
        "std": (sum((t - sum(ms) / len(ms)) ** 2 for t in ms) / len(ms)) ** 0.5,
    }


def main() -> None:
    """Run holomorphic profiling suite."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results: dict[str, Any] = {
        "profile": "holomorphic_energy",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    for dev in devices:
        print(f"\n--- Profiling on {dev} ---")

        print("  HolomorphicEnergy (784→64→10 scale)...")
        he_result = _profile_holomorphic_energy(dev)
        results[f"holomorphic_energy_{dev}"] = he_result
        print(
            f"    Free  fwd: {he_result['free_fwd_ms']['mean']:.2f}ms, "
            f"bwd: {he_result['free_bwd_ms']['mean']:.2f}ms"
        )
        print(
            f"    Nudge fwd: {he_result['nudge_fwd_ms']['mean']:.2f}ms, "
            f"bwd: {he_result['nudge_bwd_ms']['mean']:.2f}ms"
        )

        print("  PRINetModel (784→64→10 full pipeline)...")
        model_result = _profile_full_model(dev)
        results[f"full_model_{dev}"] = model_result
        print(
            f"    Fwd: {model_result['fwd_ms']['mean']:.2f}ms, "
            f"Bwd: {model_result['bwd_ms']['mean']:.2f}ms"
        )

    # Save
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "holomorphic_energy_profile.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
