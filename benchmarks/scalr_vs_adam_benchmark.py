"""SCALR vs Adam Optimizer Benchmark.

Trains PRINetModel on Fashion-MNIST proxy with SCALR, Adam, and
SGD+momentum. Tracks: convergence speed, final accuracy, r(t)
stability, lr history.

Benchmark TODO Tasks: 1.5 — SCALR vs Adam benchmark script
                      1.5a — SCALR oscillation-aware decay benchmark

Usage::

    python -m benchmarks.scalr_vs_adam_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prinet.core.measurement import kuramoto_order_parameter
from prinet.nn.layers import PRINetModel
from prinet.nn.optimizers import SCALROptimizer

SEED = 42
N_TRAIN = 3000
N_TEST = 500
N_DIMS = 784
N_CLASSES = 10
N_RES = 64
BATCH_SIZE = 64
N_EPOCHS = 50
TARGET_ACC = 0.90


def _make_fashion_proxy(
    n_samples: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Generate Fashion-MNIST-like synthetic data."""
    rng = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, N_CLASSES, (n_samples,), generator=rng)
    centers = torch.randn(N_CLASSES, N_DIMS, generator=rng) * 2.0
    data = centers[labels] + torch.randn(n_samples, N_DIMS, generator=rng) * 0.3
    return data, labels


def _train_with_optimizer(
    model: nn.Module,
    optimizer: Any,
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train model and collect per-epoch metrics."""
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    is_scalr = isinstance(optimizer, SCALROptimizer)
    last_r: float = 0.5  # initial order param estimate

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_acc": [],
        "r_t": [],
        "lr": [],
    }

    convergence_epoch: int | None = None
    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            out = model(xb)
            loss = F.nll_loss(out, yb)
            loss.backward()
            if is_scalr:
                optimizer.step(order_parameter=last_r)
            else:
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Record metrics
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        # Get current LR
        for pg in optimizer.param_groups:
            history["lr"].append(pg.get("lr", 0.0))
            break

        # Order parameter from model's resonance layer activations
        model.eval()
        with torch.no_grad():
            sample = X_test[:batch_size]
            # Run through input layer to get resonance activations
            h = model.input_layer(sample)
            h = model.layer_norms[0](h)
            # Treat activations as phase-like signal → compute coherence
            r = torch.abs(torch.exp(1j * h.double()).mean(dim=-1)).float().mean()
            last_r = r.item()
            history["r_t"].append(last_r)

            test_out = model(X_test)
            test_acc = (test_out.argmax(1) == y_test).float().mean().item()
            history["test_acc"].append(test_acc)

        if convergence_epoch is None and test_acc >= TARGET_ACC:
            convergence_epoch = epoch + 1

    wall_time = time.perf_counter() - t_start

    return {
        "final_test_acc": history["test_acc"][-1] if history["test_acc"] else 0.0,
        "final_train_loss": history["train_loss"][-1],
        "convergence_epoch": convergence_epoch,
        "wall_time_s": round(wall_time, 3),
        "param_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }


def run_scalr_vs_adam_benchmark(
    device: str = "cpu",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run SCALR vs Adam vs SGD optimizer comparison.

    Args:
        device: Device string.
        save_path: Optional JSON save path.

    Returns:
        Results dict.
    """
    torch.manual_seed(SEED)

    X_train, y_train = _make_fashion_proxy(N_TRAIN, SEED)
    X_test, y_test = _make_fashion_proxy(N_TEST, SEED + 1000)

    results: dict[str, Any] = {
        "benchmark": "scalr_vs_adam",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_epochs": N_EPOCHS,
        "target_acc": TARGET_ACC,
        "optimizers": {},
    }

    configs: list[tuple[str, Any]] = [
        ("Adam", lambda params: torch.optim.Adam(params, lr=1e-3)),
        ("SGD_momentum", lambda params: torch.optim.SGD(params, lr=1e-2, momentum=0.9)),
        (
            "SCALR",
            lambda params: SCALROptimizer(
                params, lr=1e-3, r_min=0.3, alpha=1.5
            ),
        ),
        (
            "SCALR_osc_decay",
            lambda params: SCALROptimizer(
                params,
                lr=1e-3,
                r_min=0.3,
                alpha=1.5,
                oscillation_window=20,
                oscillation_threshold=0.005,
                oscillation_decay=0.95,
            ),
        ),
    ]

    for opt_name, opt_factory in configs:
        print(f"\n  Training with {opt_name}...")
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=N_RES, n_dims=N_DIMS, n_concepts=N_CLASSES
        )
        optimizer = opt_factory(model.parameters())
        opt_results = _train_with_optimizer(
            model, optimizer, X_train, y_train, X_test, y_test, device=device
        )
        # Strip history for JSON (keep only summary)
        summary = {k: v for k, v in opt_results.items() if k != "history"}
        summary["r_final"] = (
            opt_results["history"]["r_t"][-1]
            if opt_results["history"]["r_t"]
            else None
        )
        # Compute r stability (windowed variance)
        r_hist = opt_results["history"]["r_t"]
        if len(r_hist) >= 10:
            window = r_hist[-10:]
            mu = sum(window) / len(window)
            var = sum((x - mu) ** 2 for x in window) / len(window)
            summary["r_stability_var"] = round(var, 6)
        else:
            summary["r_stability_var"] = None

        results["optimizers"][opt_name] = summary
        print(
            f"    final_acc={summary['final_test_acc']:.3f}, "
            f"convergence_epoch={summary['convergence_epoch']}, "
            f"r_final={summary.get('r_final', 'N/A')}"
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SCALR vs Adam Benchmark  (device={dev})")
    print("=" * 60)

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    run_scalr_vs_adam_benchmark(
        device=dev,
        save_path=output_dir / "benchmark_scalr_vs_adam.json",
    )
