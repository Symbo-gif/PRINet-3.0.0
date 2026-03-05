"""Phase-to-Rate Autoencoder Benchmark — Information Preservation Test.

Tests whether sparse rate codes from the Phase-to-Rate bottleneck
preserve sufficient information for:
1. Reconstruction (MSE loss)
2. Classification (accuracy with 10% sparsity)

Compares:
- ``PhaseToRateAutoencoder`` (10% sparse bottleneck)
- ``DenseAutoencoder`` (dense bottleneck baseline)

NN TODO Tasks: 3.4 — Phase-to-Rate autoencoder
              3.4a — Information preservation metric

Usage::

    python -m benchmarks.phase_to_rate_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prinet.nn.layers import DenseAutoencoder, PhaseToRateAutoencoder

SEED = 42
N_TRAIN = 5000
N_TEST = 1000
D_INPUT = 784  # MNIST-like
N_OSCILLATORS = 64
N_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 30
LR = 1e-3


def _make_synthetic_mnist(
    n_samples: int,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic MNIST-like data (random 784-d vectors with labels)."""
    rng = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, N_CLASSES, (n_samples,), generator=rng)

    # Create class-conditional clusters in 784-d space
    centers = torch.randn(N_CLASSES, D_INPUT, generator=rng) * 3.0
    data = centers[labels] + torch.randn(n_samples, D_INPUT, generator=rng) * 0.5

    return data, labels


def _train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader[Any],
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    device: str = "cpu",
    recon_weight: float = 1.0,
    class_weight: float = 0.5,
) -> dict[str, list[float]]:
    """Train autoencoder with reconstruction + classification loss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "recon_loss": [],
        "class_loss": [],
        "train_acc": [],
    }

    for _epoch in range(n_epochs):
        total_recon = 0.0
        total_class = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            recon, _rates = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)

            logits = model.classify(batch_x)
            class_loss = F.nll_loss(logits, batch_y)

            loss = recon_weight * recon_loss + class_weight * class_loss
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item() * batch_x.shape[0]
            total_class += class_loss.item() * batch_x.shape[0]
            correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total += batch_x.shape[0]

        history["recon_loss"].append(total_recon / total)
        history["class_loss"].append(total_class / total)
        history["train_acc"].append(correct / total)

    return history


@torch.no_grad()
def _eval_autoencoder(
    model: nn.Module,
    loader: DataLoader[Any],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate autoencoder on test data."""
    model.to(device)
    model.eval()

    total_recon = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        recon, rates = model(batch_x)
        total_recon += F.mse_loss(recon, batch_x, reduction="sum").item()

        logits = model.classify(batch_x)
        correct += (logits.argmax(dim=-1) == batch_y).sum().item()
        total += batch_x.shape[0]

    sparsity = 0.0
    # Compute actual sparsity of rate codes
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        _, rates = model(batch_x)
        # Count near-zero activations
        sparsity += (rates.abs() < 0.01).float().sum().item()
        total_elements = rates.numel()
    actual_sparsity = sparsity / max(total_elements, 1)

    return {
        "test_recon_mse": total_recon / total,
        "test_accuracy": correct / total,
        "actual_sparsity": actual_sparsity,
    }


def run_phase_to_rate_benchmark(
    device: str = "cpu",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Phase-to-Rate autoencoder information preservation benchmark.

    Args:
        device: Device string.
        save_path: Optional JSON path to save results.

    Returns:
        Result dict with metrics for both models.
    """
    torch.manual_seed(SEED)

    # Generate data
    train_x, train_y = _make_synthetic_mnist(N_TRAIN, seed=SEED)
    test_x, test_y = _make_synthetic_mnist(N_TEST, seed=SEED + 1000)

    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    train_loader: DataLoader[Any] = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader: DataLoader[Any] = DataLoader(test_ds, batch_size=BATCH_SIZE)

    results: dict[str, Any] = {
        "benchmark": "phase_to_rate_autoencoder",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_oscillators": N_OSCILLATORS,
        "n_epochs": N_EPOCHS,
        "target_sparsity": 0.1,
    }

    # 1. PhaseToRateAutoencoder (sparse bottleneck)
    print("  Training PhaseToRateAutoencoder (10% sparse)...")
    torch.manual_seed(SEED)
    sparse_ae = PhaseToRateAutoencoder(
        n_input=D_INPUT,
        n_oscillators=N_OSCILLATORS,
        sparsity=0.1,
        mode="soft",
    )

    t0 = time.perf_counter()
    sparse_history = _train_autoencoder(sparse_ae, train_loader, device=device)
    sparse_wall = time.perf_counter() - t0
    sparse_test = _eval_autoencoder(sparse_ae, test_loader, device=device)

    results["sparse_ae"] = {
        "final_train_acc": sparse_history["train_acc"][-1],
        "final_recon_loss": sparse_history["recon_loss"][-1],
        "test_accuracy": sparse_test["test_accuracy"],
        "test_recon_mse": sparse_test["test_recon_mse"],
        "actual_sparsity": sparse_test["actual_sparsity"],
        "wall_time_s": sparse_wall,
        "param_count": sum(p.numel() for p in sparse_ae.parameters()),
    }
    print(
        f"    test_acc={sparse_test['test_accuracy']:.3f}, "
        f"recon_mse={sparse_test['test_recon_mse']:.4f}, "
        f"sparsity={sparse_test['actual_sparsity']:.3f}"
    )

    # 2. DenseAutoencoder (baseline)
    print("  Training DenseAutoencoder (dense baseline)...")
    torch.manual_seed(SEED)
    dense_ae = DenseAutoencoder(
        n_input=D_INPUT,
        n_bottleneck=N_OSCILLATORS,
    )

    t0 = time.perf_counter()
    dense_history = _train_autoencoder(dense_ae, train_loader, device=device)
    dense_wall = time.perf_counter() - t0
    dense_test = _eval_autoencoder(dense_ae, test_loader, device=device)

    results["dense_ae"] = {
        "final_train_acc": dense_history["train_acc"][-1],
        "final_recon_loss": dense_history["recon_loss"][-1],
        "test_accuracy": dense_test["test_accuracy"],
        "test_recon_mse": dense_test["test_recon_mse"],
        "actual_sparsity": dense_test["actual_sparsity"],
        "wall_time_s": dense_wall,
        "param_count": sum(p.numel() for p in dense_ae.parameters()),
    }
    print(
        f"    test_acc={dense_test['test_accuracy']:.3f}, "
        f"recon_mse={dense_test['test_recon_mse']:.4f}"
    )

    # 3. Information preservation metric
    preservation = (
        sparse_test["test_accuracy"] / max(dense_test["test_accuracy"], 1e-8)
    )
    results["info_preservation_ratio"] = preservation
    results["status"] = "PASS" if preservation > 0.85 else "NEEDS_IMPROVEMENT"
    print(f"\n  Information preservation: {preservation:.1%}")
    print(f"  Status: {results['status']}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {save_path}")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase-to-Rate Autoencoder Benchmark  (device={device})")
    print("=" * 60)

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    run_phase_to_rate_benchmark(
        device=device,
        save_path=output_dir / "benchmark_phase_to_rate_ae.json",
    )
