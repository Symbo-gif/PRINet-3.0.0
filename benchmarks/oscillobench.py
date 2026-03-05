"""OscilloBench v1.0 — PRINet Benchmark Suite for Year 1 Q2.

Implements the OscilloBench benchmark framework with three categories:

Category A — Capacity Benchmarks (Task 4.1)
    * XOR-n: Binary XOR parity up to n bits (capacity scaling)
    * Random Dichotomies: Random binary classification of N points in R^d

Category B — Convergence Benchmarks (Task 4.2)
    * MNIST-subset: 10-class digit classification (synthetic proxy)
    * Fashion-MNIST: 10-class fashion classification (synthetic proxy)

Baselines (Task 4.3)
    * LSTM / RNN: Recurrent neural network with matching parameter count
    * Modern Hopfield Network: Energy-based associative memory
    * Vanilla Transformer (encoder-only): Self-attention baseline

All models and datasets are self-contained — no external data downloads
are required. Classification tasks use synthetic random data shaped to
match the true datasets (784-d, 10 classes).

Metrics
-------
* ``train_loss_final``: Final training cross-entropy loss
* ``test_acc``: Classification accuracy on held-out split
* ``convergence_epoch``: Epoch where test_loss first drops below threshold
* ``wall_time_s``: Total training wall-clock time
* ``param_count``: Number of learnable parameters

Results saved to:
    Docs/test_and_benchmark_results/benchmark_oscillobench.json

References
----------
* PRINet_Year1_Q2_Plan.md — Tasks 4.1, 4.2, 4.3, M2.4
* Testing_Standards.md — seeded RNG, reproducible benchmarks
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.nn.layers import PRINetModel  # noqa: E402

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_CAPACITY = 100  # XOR / dichotomies — small data, more epochs
EPOCHS_CONVERGENCE = 20  # MNIST / FashionMNIST — larger data, fewer
LR = 1e-3

# Available task names
AVAILABLE_TASKS = ("xor_n", "random_dichotomies", "mnist", "fashion_mnist", "clevr_n")


# ===================================================================
# OscilloBench v1.0 — Clean Public API
# ===================================================================


class OscilloBench:
    """OscilloBench v1.0 — Unified benchmark suite for PRINet models.

    Provides a clean API for running benchmark tasks against any model
    factory. Supports per-task selection and standardised JSON output.

    Example::

        bench = OscilloBench(device="cuda", seed=42)
        results = bench.run(
            model_factory=lambda n_dims, n_classes: PRINetModel(64, n_dims, n_classes),
            model_name="PRINet-64",
            tasks=["xor_n", "mnist"],
        )
        bench.save(results, "results.json")
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: int = SEED,
        epochs_capacity: int = EPOCHS_CAPACITY,
        epochs_convergence: int = EPOCHS_CONVERGENCE,
    ) -> None:
        self.device = torch.device(device)
        self.seed = seed
        self.epochs_capacity = epochs_capacity
        self.epochs_convergence = epochs_convergence
        self._task_registry: dict[str, Any] = {
            "xor_n": self._run_xor_n,
            "random_dichotomies": self._run_random_dichotomies,
            "mnist": self._run_mnist,
            "fashion_mnist": self._run_fashion_mnist,
            "clevr_n": self._run_clevr_n,
        }

    @staticmethod
    def available_tasks() -> tuple[str, ...]:
        """Return tuple of available task names."""
        return AVAILABLE_TASKS

    def run(
        self,
        model_factory: Any = None,
        model_name: str = "PRINet",
        tasks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run benchmark tasks.

        Args:
            model_factory: Callable ``(n_dims, n_classes) → nn.Module``.
                If ``None``, uses default PRINetModel.
            model_name: Name for results labelling.
            tasks: List of task names to run. ``None`` = all tasks.

        Returns:
            Dict with suite metadata and per-task results.
        """
        if tasks is None:
            tasks = list(AVAILABLE_TASKS)

        if model_factory is None:
            model_factory = lambda n_dims, n_classes: PRINetModel(  # noqa: E731
                n_resonances=64, n_dims=n_dims, n_concepts=n_classes
            )

        results: dict[str, Any] = {
            "suite": "OscilloBench v1.0",
            "model_name": model_name,
            "device": str(self.device),
            "seed": self.seed,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "tasks": {},
        }

        for task_name in tasks:
            if task_name not in self._task_registry:
                results["tasks"][task_name] = {
                    "status": "UNKNOWN_TASK",
                    "error": f"Unknown task: {task_name}",
                }
                continue

            print(f"  [{task_name}] Running...")
            try:
                task_result = self._task_registry[task_name](
                    model_factory, model_name
                )
                results["tasks"][task_name] = task_result
            except Exception as exc:
                results["tasks"][task_name] = {
                    "status": "ERROR",
                    "error": str(exc),
                }
                print(f"    ERROR: {exc}")

        return results

    def save(
        self,
        results: dict[str, Any],
        path: str | Path,
    ) -> None:
        """Save results to JSON file.

        Args:
            results: Results dict from ``run()``.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")

    def _run_xor_n(
        self, model_factory: Any, model_name: str
    ) -> dict[str, Any]:
        """Run XOR-n benchmark with given model factory."""
        torch.manual_seed(self.seed)
        n_values = [2, 3, 4, 5, 6]
        per_n: list[dict[str, Any]] = []
        for n in n_values:
            X_train, y_train = make_xor_n(n, n_samples=256, seed=self.seed)
            X_test, y_test = make_xor_n(n, n_samples=128, seed=self.seed + 1)
            model = model_factory(n_dims=n, n_classes=2)
            metrics = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                epochs=self.epochs_capacity,
            )
            metrics["n"] = n
            metrics["model"] = model_name
            per_n.append(metrics)
            print(f"    n={n}: test_acc={metrics['test_acc']:.3f}")
        return {"status": "PASS", "results": per_n}

    def _run_random_dichotomies(
        self, model_factory: Any, model_name: str
    ) -> dict[str, Any]:
        """Run random dichotomies benchmark."""
        torch.manual_seed(self.seed)
        N_values = [16, 32, 64, 128]
        per_N: list[dict[str, Any]] = []
        for N in N_values:
            X_train, y_train = make_random_dichotomies(
                N, n_dims=32, seed=self.seed
            )
            X_test, y_test = make_random_dichotomies(
                64, n_dims=32, seed=self.seed + 1
            )
            model = model_factory(n_dims=32, n_classes=2)
            metrics = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                epochs=self.epochs_capacity,
            )
            metrics["N"] = N
            metrics["model"] = model_name
            per_N.append(metrics)
            print(f"    N={N}: test_acc={metrics['test_acc']:.3f}")
        return {"status": "PASS", "results": per_N}

    def _run_mnist(
        self, model_factory: Any, model_name: str
    ) -> dict[str, Any]:
        """Run synthetic MNIST benchmark."""
        torch.manual_seed(self.seed)
        X_train, y_train = make_synthetic_mnist(n_train=2000, seed=self.seed)
        X_test, y_test = make_synthetic_mnist(n_train=500, seed=self.seed + 1)
        model = model_factory(n_dims=784, n_classes=10)
        metrics = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            epochs=self.epochs_convergence,
        )
        metrics["model"] = model_name
        print(f"    test_acc={metrics['test_acc']:.3f}")
        return {"status": "PASS", **metrics}

    def _run_fashion_mnist(
        self, model_factory: Any, model_name: str
    ) -> dict[str, Any]:
        """Run synthetic Fashion-MNIST benchmark."""
        torch.manual_seed(self.seed)
        X_train, y_train = make_synthetic_fashion_mnist(
            n_train=2000, seed=self.seed
        )
        X_test, y_test = make_synthetic_fashion_mnist(
            n_train=500, seed=self.seed + 1
        )
        model = model_factory(n_dims=784, n_classes=10)
        metrics = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            epochs=self.epochs_convergence,
        )
        metrics["model"] = model_name
        print(f"    test_acc={metrics['test_acc']:.3f}")
        return {"status": "PASS", **metrics}

    def _run_clevr_n(
        self, model_factory: Any, model_name: str
    ) -> dict[str, Any]:
        """Run CLEVR-N binding capacity benchmark.

        Uses the CLEVR-N framework from ``benchmarks/clevr_n.py``
        to sweep N=2,4,6,8 items and measure binding accuracy.
        """
        from benchmarks.clevr_n import (
            make_clevr_n,
            encode_features_phase,
            D_PHASE,
            D_FEAT,
        )

        torch.manual_seed(self.seed)
        n_values = [2, 4, 6, 8]
        per_n: list[dict[str, Any]] = []
        for n_items in n_values:
            scenes, queries, labels = make_clevr_n(
                n_items=n_items, n_samples=200, seed=self.seed
            )
            n_train = int(len(labels) * 0.8)
            train_scenes, test_scenes = scenes[:n_train], scenes[n_train:]
            train_queries, test_queries = queries[:n_train], queries[n_train:]
            train_labels, test_labels = labels[:n_train], labels[n_train:]

            # Use a simple MLP classifier on flattened scene+query
            flat_dim = n_items * D_PHASE + D_FEAT * 2
            model = nn.Sequential(
                nn.Linear(flat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

            # Flatten and encode for training
            train_enc = encode_features_phase(train_scenes)
            test_enc = encode_features_phase(test_scenes)
            X_train = torch.cat([
                train_enc.view(n_train, -1),
                train_queries,
            ], dim=-1)
            X_test = torch.cat([
                test_enc.view(len(test_labels), -1),
                test_queries,
            ], dim=-1)

            metrics = train_and_evaluate(
                model, X_train, train_labels,
                X_test, test_labels,
                epochs=min(self.epochs_capacity, 30),
            )
            metrics["n_items"] = n_items
            metrics["model"] = model_name
            per_n.append(metrics)
            print(f"    N={n_items}: test_acc={metrics['test_acc']:.3f}")
        return {"status": "PASS", "results": per_n}

    def generate_comparison_dashboard(
        self,
        results_list: list[dict[str, Any]],
    ) -> str:
        """Generate a Markdown comparison table across multiple model runs.

        Args:
            results_list: List of results dicts from multiple ``run()`` calls.

        Returns:
            Markdown-formatted comparison table string.
        """
        lines: list[str] = [
            "# OscilloBench Comparison Dashboard\n",
            "| Model | Task | Metric | Value |",
            "|-------|------|--------|-------|",
        ]
        for results in results_list:
            model_name = results.get("model_name", "Unknown")
            for task_name, task_data in results.get("tasks", {}).items():
                if task_data.get("status") == "ERROR":
                    lines.append(
                        f"| {model_name} | {task_name} | status | ERROR |"
                    )
                    continue
                if "results" in task_data:
                    for entry in task_data["results"]:
                        key = entry.get("n") or entry.get("N") or entry.get("n_items", "?")
                        acc = entry.get("test_acc", 0.0)
                        lines.append(
                            f"| {model_name} | {task_name} (n={key}) | test_acc | {acc:.3f} |"
                        )
                else:
                    acc = task_data.get("test_acc", 0.0)
                    lines.append(
                        f"| {model_name} | {task_name} | test_acc | {acc:.3f} |"
                    )
        return "\n".join(lines)


# ===================================================================
# Data Generators
# ===================================================================


def make_xor_n(n: int, *, n_samples: int = 256, seed: int = SEED) -> tuple[Tensor, Tensor]:
    """Generate n-bit XOR parity dataset.

    Each sample is a binary vector in {0, 1}^n and the label is
    the parity (XOR of all bits).

    Args:
        n: Number of input bits.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        (X, y) with X of shape (n_samples, n) float32 and
        y of shape (n_samples,) int64 ∈ {0, 1}.
    """
    rng = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 2, (n_samples, n), generator=rng, dtype=torch.float32)
    y = X.sum(dim=1).long() % 2  # parity
    return X, y


def make_random_dichotomies(
    n_points: int,
    n_dims: int,
    *,
    seed: int = SEED,
) -> tuple[Tensor, Tensor]:
    """Generate random dichotomy: N Gaussian points, random binary labels.

    Capacity theory predicts that a model with C parameters can shatter
    approximately C / n_dims patterns (Cover's theorem).

    Args:
        n_points: Number of data points.
        n_dims: Feature dimension.
        seed: Random seed.

    Returns:
        (X, y) with shapes (n_points, n_dims) and (n_points,).
    """
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n_points, n_dims, generator=rng)
    y = torch.randint(0, 2, (n_points,), generator=rng)
    return X, y


def make_synthetic_mnist(
    n_samples: int = 2000,
    *,
    seed: int = SEED,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create synthetic MNIST-like data (784-d, 10 classes).

    Uses Gaussian clusters centered at class-specific random means
    to simulate class-separable image features without torchvision.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    rng = torch.Generator().manual_seed(seed)
    n_classes = 10
    n_train = int(n_samples * 0.8)

    # Class-specific centroids in 784-d
    centroids = torch.randn(n_classes, 784, generator=rng) * 3.0
    labels = torch.randint(0, n_classes, (n_samples,), generator=rng)
    X = centroids[labels] + torch.randn(n_samples, 784, generator=rng) * 0.5

    return X[:n_train], labels[:n_train], X[n_train:], labels[n_train:]


def make_synthetic_fashion_mnist(
    n_samples: int = 2000,
    *,
    seed: int = SEED,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create synthetic Fashion-MNIST-like data.

    Similar structure to MNIST proxy but with different seed offset
    to produce different cluster geometry.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    return make_synthetic_mnist(n_samples, seed=seed + 1000)


# ===================================================================
# Baseline Models
# ===================================================================


class LSTMBaseline(nn.Module):
    """LSTM baseline treating 784-d input as a 28×28 sequence.

    Processes each row (28 pixels) as one time step of length 28,
    then classifies from the final hidden state.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 64,
        n_classes: int = 10,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = 28
        self.feature_dim = input_dim // self.seq_len  # 28
        self.lstm = nn.LSTM(
            self.feature_dim, hidden_dim, n_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x_seq = x.view(B, self.seq_len, self.feature_dim)
        _, (h_n, _) = self.lstm(x_seq)
        logits = self.fc(h_n[-1])
        return F.log_softmax(logits, dim=-1)


class ModernHopfieldBaseline(nn.Module):
    """Modern Hopfield Network (Ramsauer et al., 2020) baseline.

    Uses the continuous Hopfield energy with softmax attention for
    pattern retrieval, followed by a linear classifier.

    E = -log Σ_i exp(β · xᵢ · ξ)

    where ξ are stored patterns. We parameterize the patterns as
    a learnable memory matrix.
    """

    def __init__(
        self,
        input_dim: int = 784,
        n_patterns: int = 64,
        n_classes: int = 10,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.patterns = nn.Parameter(torch.randn(n_patterns, input_dim) * 0.01)
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Modern Hopfield retrieval: softmax attention over stored patterns
        # x: (B, D), patterns: (M, D)
        scores = self.beta * (x @ self.patterns.T)  # (B, M)
        attn = F.softmax(scores, dim=-1)  # (B, M)
        retrieved = attn @ self.patterns  # (B, D)
        logits = self.fc(retrieved)
        return F.log_softmax(logits, dim=-1)


class TransformerBaseline(nn.Module):
    """Vanilla Transformer encoder baseline.

    Splits 784-d input into 28 patches of 28-d (like image rows),
    adds learnable positional encoding, runs through encoder layers,
    then pools and classifies.
    """

    def __init__(
        self,
        input_dim: int = 784,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.seq_len = 28
        self.patch_dim = input_dim // self.seq_len  # 28
        self.proj = nn.Linear(self.patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x_seq = x.view(B, self.seq_len, self.patch_dim)
        h = self.proj(x_seq) + self.pos_embed
        h = self.encoder(h)
        h_pool = h.mean(dim=1)  # global average pool
        logits = self.fc(h_pool)
        return F.log_softmax(logits, dim=-1)


# ===================================================================
# Training Loop
# ===================================================================


def train_and_evaluate(
    model: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    *,
    epochs: int,
    lr: float = LR,
    batch_size: int = 64,
    loss_threshold: float = 1.0,
) -> dict[str, Any]:
    """Train a model and return benchmark metrics.

    Args:
        model: The nn.Module to train.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        loss_threshold: Test loss threshold for convergence_epoch.

    Returns:
        Dict with train_loss_final, test_acc, convergence_epoch,
        wall_time_s, param_count keys.
    """
    model = model.to(DEVICE)
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = X_train.size(0)
    convergence_epoch: int | None = None

    t_start = time.perf_counter()
    train_loss = float("nan")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            opt.zero_grad()
            out = model(xb)
            loss = F.nll_loss(out, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Test evaluation
        if convergence_epoch is None:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                test_loss = F.nll_loss(test_out, y_test).item()
            if test_loss < loss_threshold:
                convergence_epoch = epoch + 1

    wall_time = time.perf_counter() - t_start

    # Final test accuracy
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        preds = test_out.argmax(dim=1)
        test_acc = (preds == y_test).float().mean().item()

    param_count = sum(p.numel() for p in model.parameters())

    return {
        "train_loss_final": round(train_loss, 6),
        "test_acc": round(test_acc, 4),
        "convergence_epoch": convergence_epoch,
        "wall_time_s": round(wall_time, 3),
        "param_count": param_count,
    }


# ===================================================================
# Category A: Capacity Benchmarks
# ===================================================================


def benchmark_xor_n() -> dict[str, Any]:
    """Category A — XOR-n capacity benchmark."""
    print("  XOR-n capacity benchmark")
    torch.manual_seed(SEED)
    results: list[dict] = []

    for n_bits in [2, 3, 4, 5, 6]:
        X, y = make_xor_n(n_bits, n_samples=256)
        n_train = 200
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        # PRINet
        prinet = PRINetModel(
            n_resonances=32,
            n_dims=n_bits,
            n_concepts=2,
            n_layers=2,
            n_steps=5,
        )
        prinet_metrics = train_and_evaluate(
            prinet, X_train, y_train, X_test, y_test,
            epochs=EPOCHS_CAPACITY, loss_threshold=0.5,
        )

        # MLP baseline (matched parameter count ~= PRINet)
        mlp = nn.Sequential(
            nn.Linear(n_bits, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=-1),
        )
        mlp_metrics = train_and_evaluate(
            mlp, X_train, y_train, X_test, y_test,
            epochs=EPOCHS_CAPACITY, loss_threshold=0.5,
        )

        entry = {
            "n_bits": n_bits,
            "prinet": prinet_metrics,
            "mlp_baseline": mlp_metrics,
        }
        results.append(entry)
        print(f"    XOR-{n_bits}: PRINet acc={prinet_metrics['test_acc']:.3f}"
              f"  MLP acc={mlp_metrics['test_acc']:.3f}")

    return {
        "name": "Category_A_XOR_n",
        "status": "PASS",
        "results": results,
    }


def benchmark_random_dichotomies() -> dict[str, Any]:
    """Category A — Random Dichotomies capacity benchmark."""
    print("  Random Dichotomies capacity benchmark")
    torch.manual_seed(SEED)
    results: list[dict] = []

    for n_points in [32, 64, 128]:
        X, y = make_random_dichotomies(n_points, n_dims=32)
        n_train = int(n_points * 0.8)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        prinet = PRINetModel(
            n_resonances=32,
            n_dims=32,
            n_concepts=2,
            n_layers=2,
            n_steps=5,
        )
        prinet_metrics = train_and_evaluate(
            prinet, X_train, y_train, X_test, y_test,
            epochs=EPOCHS_CAPACITY, loss_threshold=0.5,
        )

        mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=-1),
        )
        mlp_metrics = train_and_evaluate(
            mlp, X_train, y_train, X_test, y_test,
            epochs=EPOCHS_CAPACITY, loss_threshold=0.5,
        )

        results.append({
            "n_points": n_points,
            "n_dims": 32,
            "prinet": prinet_metrics,
            "mlp_baseline": mlp_metrics,
        })
        print(f"    N={n_points}: PRINet acc={prinet_metrics['test_acc']:.3f}"
              f"  MLP acc={mlp_metrics['test_acc']:.3f}")

    return {
        "name": "Category_A_Random_Dichotomies",
        "status": "PASS",
        "results": results,
    }


# ===================================================================
# Category B: Convergence Benchmarks
# ===================================================================


def benchmark_mnist_convergence() -> dict[str, Any]:
    """Category B — MNIST (synthetic proxy) convergence benchmark."""
    print("  MNIST convergence benchmark (synthetic proxy)")
    torch.manual_seed(SEED)

    X_train, y_train, X_test, y_test = make_synthetic_mnist(2000)

    models: dict[str, nn.Module] = {
        "PRINet": PRINetModel(
            n_resonances=32, n_dims=784, n_concepts=10,
            n_layers=2, n_steps=5,
        ),
        "LSTM": LSTMBaseline(hidden_dim=64, n_layers=1),
        "ModernHopfield": ModernHopfieldBaseline(n_patterns=64),
        "Transformer": TransformerBaseline(d_model=64, n_heads=4, n_encoder_layers=2),
        "MLP": nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=-1),
        ),
    }

    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        try:
            metrics = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                epochs=EPOCHS_CONVERGENCE, loss_threshold=1.5,
            )
            results[name] = metrics
            print(f"acc={metrics['test_acc']:.3f} "
                  f"loss={metrics['train_loss_final']:.4f} "
                  f"time={metrics['wall_time_s']:.1f}s")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"ERROR: {e}")

        # Free GPU memory between models
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "name": "Category_B_MNIST",
        "status": "PASS",
        "dataset": "synthetic_mnist_2000",
        "epochs": EPOCHS_CONVERGENCE,
        "results": results,
    }


def benchmark_fashion_mnist_convergence() -> dict[str, Any]:
    """Category B — Fashion-MNIST (synthetic proxy) convergence benchmark."""
    print("  Fashion-MNIST convergence benchmark (synthetic proxy)")
    torch.manual_seed(SEED)

    X_train, y_train, X_test, y_test = make_synthetic_fashion_mnist(2000)

    models: dict[str, nn.Module] = {
        "PRINet": PRINetModel(
            n_resonances=32, n_dims=784, n_concepts=10,
            n_layers=2, n_steps=5,
        ),
        "LSTM": LSTMBaseline(hidden_dim=64, n_layers=1),
        "ModernHopfield": ModernHopfieldBaseline(n_patterns=64),
        "Transformer": TransformerBaseline(d_model=64, n_heads=4, n_encoder_layers=2),
    }

    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        try:
            metrics = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                epochs=EPOCHS_CONVERGENCE, loss_threshold=1.5,
            )
            results[name] = metrics
            print(f"acc={metrics['test_acc']:.3f} "
                  f"loss={metrics['train_loss_final']:.4f} "
                  f"time={metrics['wall_time_s']:.1f}s")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"ERROR: {e}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "name": "Category_B_Fashion_MNIST",
        "status": "PASS",
        "dataset": "synthetic_fashion_mnist_2000",
        "epochs": EPOCHS_CONVERGENCE,
        "results": results,
    }


# ===================================================================
# Main Runner
# ===================================================================


def main() -> None:
    """Run all OscilloBench benchmarks and save results."""
    print("=" * 60)
    print("OscilloBench v1.0 — PRINet Benchmark Suite")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    benchmarks = [
        ("Category A: XOR-n", benchmark_xor_n),
        ("Category A: Random Dichotomies", benchmark_random_dichotomies),
        ("Category B: MNIST", benchmark_mnist_convergence),
        ("Category B: Fashion-MNIST", benchmark_fashion_mnist_convergence),
    ]

    all_results: list[dict] = []
    for name, fn in benchmarks:
        print(f"\n[{name}]")
        try:
            result = fn()
            all_results.append(result)
            print(f"  -> {result.get('status', 'DONE')}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            all_results.append({"name": name, "status": "ERROR", "error": str(e)})

    # Save results
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "test_and_benchmark_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_oscillobench.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "suite": "OscilloBench v1.0",
                "phase": "Year1_Q2",
                "device": str(DEVICE),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "benchmarks": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")

    # Summary
    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    failed = sum(1 for r in all_results if r.get("status") in ("FAIL", "ERROR"))
    print(f"Summary: {passed} passed, {failed} failed out of {len(all_results)}")


if __name__ == "__main__":
    main()
