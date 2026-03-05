"""Benchmark: MNIST subset baseline with order-parameter monitoring (Task 2.5).

Trains a small PRINetModel on a 1,000-sample MNIST subset and tracks the
Kuramoto order parameter r(t) throughout training.

Success criterion: r > 0.8 stability on the toy task within 20 epochs.

NOTE: This benchmark requires ``torchvision`` to load MNIST. If torchvision
is unavailable it falls back to synthetic random data so the benchmark
script can always execute.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState
from prinet.nn.layers import PRINetModel
from prinet.nn.optimizers import SynchronizedGradientDescent

SEED = 42
N_OSCILLATORS = 32
N_EPOCHS = 20
BATCH_SIZE = 64
SUBSET_SIZE = 1_000
LR = 0.05
LAMBDA_SYNC = 0.3
K_CRITICAL = 0.6


def _load_mnist_subset() -> tuple[torch.Tensor, torch.Tensor]:
    """Try loading real MNIST; fall back to synthetic data."""
    try:
        from torchvision import datasets, transforms

        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        ds = datasets.MNIST(
            root="/tmp/mnist", train=True, download=True, transform=tf
        )
        # Select subset
        torch.manual_seed(SEED)
        indices = torch.randperm(len(ds))[:SUBSET_SIZE]
        images = torch.stack([ds[i][0].flatten() for i in indices])
        labels = torch.tensor([ds[i][1] for i in indices])
        print("  Using real MNIST data.")
        return images, labels
    except ImportError:
        print("  torchvision not found — using synthetic random data.")
        torch.manual_seed(SEED)
        images = torch.randn(SUBSET_SIZE, 784)
        labels = torch.randint(0, 10, (SUBSET_SIZE,))
        return images, labels


def main() -> None:
    print("=" * 60)
    print("  BENCHMARK: MNIST Subset Baseline (Task 2.5)")
    print(f"  Target: r > 0.8 within {N_EPOCHS} epochs")
    print("=" * 60)
    print()

    images, labels = _load_mnist_subset()
    input_dim = images.shape[1]

    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=input_dim,
        n_resonances=N_OSCILLATORS,
        n_layers=2,
        n_concepts=10,
    )

    kuramoto = KuramotoOscillator(
        n_oscillators=N_OSCILLATORS, coupling_strength=2.0
    )
    opt = SynchronizedGradientDescent(
        model.parameters(),
        lr=LR,
        sync_penalty=LAMBDA_SYNC,
        critical_order=K_CRITICAL,
    )

    state = OscillatorState.create_random(N_OSCILLATORS, seed=SEED)
    n_batches = max(1, SUBSET_SIZE // BATCH_SIZE)
    epoch_records: list[dict] = []

    t0 = time.perf_counter()
    for epoch in range(1, N_EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        for bi in range(n_batches):
            start = bi * BATCH_SIZE
            end = start + BATCH_SIZE
            xb = images[start:end]
            yb = labels[start:end]

            logits = model(xb)
            loss = F.nll_loss(logits, yb)

            opt.zero_grad()
            loss.backward()

            # Evolve oscillators
            state = kuramoto.step(state, dt=0.01)
            r_step = kuramoto_order_parameter(state.phase).item()
            opt.step(order_parameter=r_step)

            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()

        r = kuramoto_order_parameter(state.phase).item()
        avg_loss = total_loss / n_batches
        accuracy = correct / min(SUBSET_SIZE, n_batches * BATCH_SIZE)
        rec = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 4),
            "order_param": round(r, 4),
        }
        epoch_records.append(rec)
        print(
            f"  Epoch {epoch:>2d}/{N_EPOCHS}  "
            f"loss={avg_loss:.4f}  acc={accuracy:.4f}  r={r:.4f}"
        )

    elapsed = time.perf_counter() - t0
    final_r = epoch_records[-1]["order_param"]
    passed = final_r > 0.8

    print()
    if passed:
        print(f"SUCCESS: Final r = {final_r:.4f} > 0.8")
    else:
        print(f"NOTE: Final r = {final_r:.4f} (threshold 0.8)")
    print(f"Total wall time: {elapsed:.2f}s")

    # Save JSON
    out_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "mnist_subset_baseline",
        "task": "2.5",
        "target": "r > 0.8 within 20 epochs",
        "n_epochs": N_EPOCHS,
        "subset_size": SUBSET_SIZE,
        "final_order_param": final_r,
        "passed": passed,
        "wall_seconds": round(elapsed, 3),
        "epoch_records": epoch_records,
    }
    out_file = out_dir / "benchmark_mnist_subset.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
