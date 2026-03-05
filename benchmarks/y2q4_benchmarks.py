"""Year 2 Q4 Benchmarks — Consolidation.

Workstream J.3: Automated performance regression suite (goldilocks + CLEVR-6 + throughput).
Workstream K.1: Empirical capacity — max N on CLEVR-N where acc > 70%, 3+ architectures.
Workstream K.2: Phase diagram extension — K x Delta for 10/50/100-oscillator systems.
Workstream L.1: Archive all benchmark JSONs for Year 2 Comprehensive Report.

Usage:
    python benchmarks/y2q4_benchmarks.py --workstream J
    python benchmarks/y2q4_benchmarks.py --workstream K1
    python benchmarks/y2q4_benchmarks.py --workstream K2
    python benchmarks/y2q4_benchmarks.py --all
    python benchmarks/y2q4_benchmarks.py --all --device cuda --epochs 30
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# ---- Imports from PRINet ---------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.propagation import (
    DiscreteDeltaThetaGamma,
    KuramotoOscillator,
    OscillatorState,
)
from prinet.core.measurement import kuramoto_order_parameter
from prinet.nn.hybrid import (
    HybridPRINetV2,
    HybridPRINetV2CLEVRN,
    InterleavedHybridPRINet,
)
from prinet.nn.layers import (
    DiscreteDeltaThetaGammaLayer,
    PRINetModel,
)

# ---- CLEVR-N Helpers -------------------------------------------------------

# Inline CLEVR-N dataset to avoid circular imports
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink"]
SHAPES = ["circle", "square", "triangle", "diamond", "star", "hexagon", "cross", "oval"]
SIZES = ["small", "medium", "large"]
N_POSITIONS = 16


def _encode_feature(color_idx: int, shape_idx: int, size_idx: int) -> Tensor:
    """One-hot encode a single object's features."""
    vec = torch.zeros(len(COLORS) + len(SHAPES) + len(SIZES))
    vec[color_idx] = 1.0
    vec[len(COLORS) + shape_idx] = 1.0
    vec[len(COLORS) + len(SHAPES) + size_idx] = 1.0
    return vec


def generate_clevr_n(
    n_objects: int,
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Generate CLEVR-N: 'are all N object colors different?' task."""
    rng = torch.Generator().manual_seed(seed)
    feat_dim = len(COLORS) + len(SHAPES) + len(SIZES)
    X = torch.zeros(n_samples, n_objects, feat_dim)
    y = torch.zeros(n_samples, dtype=torch.long)

    max_unique = len(COLORS)  # 8 colours available
    for i in range(n_samples):
        colors = torch.randint(0, len(COLORS), (n_objects,), generator=rng).tolist()
        shapes = torch.randint(0, len(SHAPES), (n_objects,), generator=rng).tolist()
        sizes = torch.randint(0, len(SIZES), (n_objects,), generator=rng).tolist()

        # 50% positive (force unique colors when possible)
        if i % 2 == 0:
            if n_objects <= max_unique:
                perm = torch.randperm(max_unique, generator=rng)[:n_objects].tolist()
                colors = perm
            else:
                # More objects than colours: use all colours + random extras
                base = torch.randperm(max_unique, generator=rng).tolist()
                extras = torch.randint(0, max_unique, (n_objects - max_unique,), generator=rng).tolist()
                colors = base + extras
                # Label will be 0 (not all unique) which is correct

        for j in range(n_objects):
            X[i, j] = _encode_feature(colors[j], shapes[j], sizes[j])

        unique_colors = len(set(colors))
        y[i] = 1 if unique_colors == n_objects else 0

    return X, y


# ---- Model Wrappers -------------------------------------------------------


class TransformerCLEVRN(nn.Module):
    """Transformer baseline for CLEVR-N."""

    def __init__(self, input_dim: int, n_classes: int = 2, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return F.log_softmax(self.classifier(h), dim=-1)


class LSTMBaseline(nn.Module):
    """LSTM baseline for CLEVR-N."""

    def __init__(self, input_dim: int, n_classes: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _, (h, _) = self.lstm(x)
        return F.log_softmax(self.classifier(h[-1]), dim=-1)


class DiscreteDTGCLEVRN(nn.Module):
    """DiscreteDeltaThetaGamma CLEVR-N wrapper."""

    def __init__(self, input_dim: int, n_classes: int = 2) -> None:
        super().__init__()
        n_osc = 14  # 2+4+8
        self.dtg = DiscreteDeltaThetaGammaLayer(
            n_delta=2, n_theta=4, n_gamma=8,
            n_dims=input_dim, coupling_strength=1.5, dt=0.01,
        )
        self.classifier = nn.Linear(n_osc, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N_objects, feat_dim) -> flatten to (B, feat_dim) via mean
        if x.dim() == 3:
            x = x.mean(dim=1)
        h = self.dtg(x)
        return F.log_softmax(self.classifier(h), dim=-1)


class HybridV2Wrapper(nn.Module):
    """Simple HybridPRINetV2 wrapper for CLEVR-N capacity sweep.

    Aggregates object features via mean-pooling then runs V2.
    """

    def __init__(self, input_dim: int, n_classes: int = 2) -> None:
        super().__init__()
        n_osc = 14  # 2+4+8
        self.proj = nn.Linear(input_dim, n_osc)
        self.v2 = HybridPRINetV2(
            n_input=n_osc, n_classes=n_classes,
            d_model=64, n_heads=4, n_layers=2,
            n_delta=2, n_theta=4, n_gamma=8,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        h = self.proj(x)
        return self.v2(h)


# ---- J.3: Performance Regression Suite ------------------------------------


def run_j3_regression_suite(
    device: str = "cpu",
    epochs: int = 10,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """J.3 — Automated performance regression benchmarks.

    Runs three core benchmarks and checks against regression thresholds:
    1. Goldilocks: V2 CLEVR-6 trains to >40% acc in 10 epochs
    2. Throughput: V2 forward pass < 50ms per batch
    3. Stability: 100-step oscillator integration produces no NaN

    Args:
        device: Device to use ("cpu" or "cuda").
        epochs: Number of training epochs for goldilocks test.
        results_dir: Directory to save JSON results.

    Returns:
        Dictionary of benchmark results with pass/fail status.
    """
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "benchmarks": {},
    }

    # --- 1. Goldilocks: V2 CLEVR-6 convergence ---
    print("=== J.3.1: Goldilocks — V2 CLEVR-6 convergence ===")
    N = 6
    feat_dim = len(COLORS) + len(SHAPES) + len(SIZES)
    X_train, y_train = generate_clevr_n(N, n_samples=500, seed=42)
    X_test, y_test = generate_clevr_n(N, n_samples=200, seed=99)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = HybridV2Wrapper(feat_dim, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True,
    )

    t0 = time.perf_counter()
    final_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=-1)
            final_acc = (preds == y_test).float().mean().item() * 100.0
    train_time = time.perf_counter() - t0

    goldilocks_pass = final_acc > 40.0
    results["benchmarks"]["goldilocks"] = {
        "accuracy": round(final_acc, 1),
        "train_time_s": round(train_time, 2),
        "epochs": epochs,
        "threshold": 40.0,
        "pass": goldilocks_pass,
    }
    print(f"  Accuracy: {final_acc:.1f}% (threshold: 40%) — {'PASS' if goldilocks_pass else 'FAIL'}")

    # --- 2. Throughput: forward pass latency ---
    print("=== J.3.2: Throughput — V2 forward latency ===")
    model.eval()
    dummy = torch.randn(32, N, feat_dim, device=device)

    # Warmup
    for _ in range(5):
        model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters * 1000  # ms

    throughput_pass = elapsed < 50.0
    results["benchmarks"]["throughput"] = {
        "ms_per_batch": round(elapsed, 3),
        "batch_size": 32,
        "n_iters": n_iters,
        "threshold_ms": 50.0,
        "pass": throughput_pass,
    }
    print(f"  Latency: {elapsed:.3f} ms/batch (threshold: 50ms) — {'PASS' if throughput_pass else 'FAIL'}")

    # --- 3. Stability: 100-step oscillator integration ---
    print("=== J.3.3: Stability — 100-step oscillator integration ===")
    dtg = DiscreteDeltaThetaGamma(
        n_delta=2, n_theta=4, n_gamma=8,
        coupling_strength=1.5,
    )
    torch.manual_seed(42)
    n_total = dtg.n_total
    phase = torch.rand(8, n_total) * 2 * math.pi
    amplitude = torch.ones(8, n_total)
    all_finite = True
    for s in range(100):
        phase, amplitude = dtg.step(phase, amplitude, dt=0.01)
        if not torch.isfinite(phase).all() or not torch.isfinite(amplitude).all():
            all_finite = False
            break

    results["benchmarks"]["stability"] = {
        "steps": 100,
        "all_finite": all_finite,
        "pass": all_finite,
    }
    print(f"  100 steps, all finite: {all_finite} — {'PASS' if all_finite else 'FAIL'}")

    # --- Summary ---
    all_pass = all(b["pass"] for b in results["benchmarks"].values())
    results["all_pass"] = all_pass
    print(f"\n=== J.3 Regression Suite: {'ALL PASS' if all_pass else 'FAIL'} ===\n")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        fp = results_dir / "benchmark_y2q4_j3_regression.json"
        with open(fp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {fp}")

    return results


# ---- K.1: Empirical Capacity Measurement ----------------------------------


def run_k1_capacity_sweep(
    device: str = "cpu",
    epochs: int = 30,
    n_range: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50),
    acc_threshold: float = 70.0,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """K.1 — Empirical capacity: find max N where acc > threshold.

    Tests 4 architectures: DiscreteDTG, Transformer, LSTM, HybridPRINetV2.

    Args:
        device: Device for training.
        epochs: Training epochs per (model, N) pair.
        n_range: Object counts to sweep.
        acc_threshold: Accuracy threshold (default 70%).
        results_dir: Directory to save JSON results.

    Returns:
        Dictionary with per-model, per-N accuracy and max capacity.
    """
    print("=== K.1: Empirical Capacity Measurement ===")
    feat_dim = len(COLORS) + len(SHAPES) + len(SIZES)

    model_factories: dict[str, Any] = {
        "DiscreteDTG": lambda n: DiscreteDTGCLEVRN(feat_dim, n_classes=2).to(device),
        "Transformer": lambda n: TransformerCLEVRN(feat_dim, n_classes=2).to(device),
        "LSTM": lambda n: LSTMBaseline(feat_dim, n_classes=2).to(device),
        "HybridV2": lambda n: HybridV2Wrapper(feat_dim, n_classes=2).to(device),
    }

    all_results: dict[str, dict[str, Any]] = {}

    for model_name, factory in model_factories.items():
        print(f"\n--- {model_name} ---")
        model_results: dict[str, Any] = {"accuracies": {}, "max_n_above_threshold": 0}

        for N in n_range:
            if N > len(COLORS):
                # Cannot generate unique colors for N > 8, use wrapped approach
                pass

            torch.manual_seed(42)
            X_train, y_train = generate_clevr_n(N, n_samples=500, seed=42)
            X_test, y_test = generate_clevr_n(N, n_samples=200, seed=99)
            X_train, y_train = X_train.to(device), y_train.to(device)
            X_test, y_test = X_test.to(device), y_test.to(device)

            try:
                model = factory(N)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loader = DataLoader(
                    TensorDataset(X_train, y_train), batch_size=64, shuffle=True,
                )

                for epoch in range(epochs):
                    model.train()
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        loss = F.nll_loss(model(xb), yb)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    preds = model(X_test).argmax(dim=-1)
                    acc = (preds == y_test).float().mean().item() * 100.0

                model_results["accuracies"][str(N)] = round(acc, 1)
                if acc >= acc_threshold:
                    model_results["max_n_above_threshold"] = N
                print(f"  N={N:3d}: {acc:5.1f}%{'  <-- below threshold' if acc < acc_threshold else ''}")

            except Exception as e:
                model_results["accuracies"][str(N)] = f"ERROR: {e}"
                print(f"  N={N:3d}: ERROR — {e}")

        all_results[model_name] = model_results

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "epochs": epochs,
        "acc_threshold": acc_threshold,
        "n_range": list(n_range),
        "models": all_results,
    }

    # Summary
    print("\n=== K.1 Capacity Summary ===")
    for name, mr in all_results.items():
        max_n = mr["max_n_above_threshold"]
        print(f"  {name:15s}: max N with acc>{acc_threshold}% = {max_n}")

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        fp = results_dir / "benchmark_y2q4_k1_capacity.json"
        with open(fp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {fp}")

    return results


# ---- K.2: Phase Diagram Extension ------------------------------------------


def run_k2_phase_diagrams(
    n_oscillators_list: tuple[int, ...] = (10, 50, 100),
    K_range: tuple[float, float, int] = (0.0, 5.0, 25),
    delta_range: tuple[float, float, int] = (0.0, 3.0, 25),
    n_steps: int = 200,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """K.2 — Extended phase diagrams: K x Delta for various N.

    For each oscillator count, sweeps coupling strength (K) and frequency
    detuning (Delta=std of natural frequencies) to measure the final
    Kuramoto order parameter r, producing a 2D phase diagram.

    Args:
        n_oscillators_list: System sizes to evaluate.
        K_range: (min, max, n_points) for coupling strength.
        delta_range: (min, max, n_points) for frequency spread.
        n_steps: Integration steps per (K, Delta) point.
        results_dir: Directory to save JSON results.

    Returns:
        Dictionary with phase diagram data per system size.
    """
    print("=== K.2: Phase Diagram Extension (K x Delta) ===")
    K_values = np.linspace(*K_range).tolist()
    delta_values = np.linspace(*delta_range).tolist()

    all_diagrams: dict[str, Any] = {}

    for N in n_oscillators_list:
        print(f"\n--- N={N} oscillators ({K_range[2]}x{delta_range[2]} grid) ---")
        diagram = np.zeros((len(K_values), len(delta_values)))

        for i, K in enumerate(K_values):
            for j, delta in enumerate(delta_values):
                torch.manual_seed(42)
                # Natural frequencies with spread delta
                if delta == 0.0:
                    natural_freq = torch.ones(N) * 1.0
                else:
                    natural_freq = torch.normal(
                        mean=1.0, std=delta, size=(N,),
                        generator=torch.Generator().manual_seed(42),
                    )

                # Create Kuramoto system
                osc = KuramotoOscillator(
                    n_oscillators=N,
                    coupling_strength=K,
                    coupling_mode="mean_field" if N >= 50 else "full",
                )

                # Random initial phases
                state = OscillatorState.create_random(N, seed=42)
                # Override natural frequencies
                state = OscillatorState(
                    phase=state.phase,
                    amplitude=state.amplitude,
                    frequency=natural_freq,
                )

                # Integrate
                dt = 0.05
                for _ in range(n_steps):
                    state = osc.step(state, dt=dt)

                # Measure final order parameter
                r = kuramoto_order_parameter(state.phase).item()
                diagram[i, j] = r

            # Progress
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Row {i+1}/{len(K_values)} done (K={K:.2f})")

        all_diagrams[str(N)] = {
            "order_params": diagram.tolist(),
            "K_values": K_values,
            "delta_values": delta_values,
            "n_steps": n_steps,
        }

        # Summary statistics
        r_mean = np.mean(diagram)
        r_sync = np.mean(diagram > 0.8)
        print(f"  Mean r: {r_mean:.3f}, fraction sync (r>0.8): {r_sync:.1%}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_oscillators_list": list(n_oscillators_list),
        "K_range": list(K_range),
        "delta_range": list(delta_range),
        "n_steps": n_steps,
        "diagrams": all_diagrams,
    }

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        fp = results_dir / "benchmark_y2q4_k2_phase_diagrams.json"
        with open(fp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {fp}")

    return results


# ---- L.1: Archive All Benchmark JSONs -------------------------------------


def run_l1_archive(results_dir: Path | None = None) -> dict[str, Any]:
    """L.1 — Archive all benchmark JSON files for Year 2 report.

    Scans the results directory and Docs/test_and_benchmark_results for
    all .json benchmark files and produces a manifest.

    Args:
        results_dir: Directory to scan (defaults to benchmarks/).

    Returns:
        Manifest of all benchmark JSONs found.
    """
    print("=== L.1: Benchmark Archive Manifest ===")
    scan_dirs = [
        Path(__file__).parent,  # benchmarks/
        Path(__file__).parents[1] / "Docs" / "test_and_benchmark_results",
    ]

    manifest: list[dict[str, str]] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for fp in sorted(scan_dir.glob("*.json")):
            stat = fp.stat()
            manifest.append({
                "file": fp.name,
                "path": str(fp.relative_to(Path(__file__).parents[1])),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
            print(f"  {fp.name} ({stat.st_size / 1024:.1f} KB)")

    results = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(manifest),
        "files": manifest,
    }

    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        fp = results_dir / "benchmark_y2q4_l1_archive.json"
        with open(fp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {fp}")

    return results


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Y2 Q4 benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Year 2 Q4 Benchmarks")
    parser.add_argument(
        "--workstream", type=str, default="all",
        choices=["J", "K1", "K2", "L", "all"],
        help="Which workstream to run",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(__file__).parent

    if args.workstream in ("J", "all"):
        run_j3_regression_suite(args.device, args.epochs, results_dir)

    if args.workstream in ("K1", "all"):
        run_k1_capacity_sweep(
            args.device, args.epochs,
            n_range=(2, 3, 4, 5, 6, 7, 8),
            results_dir=results_dir,
        )

    if args.workstream in ("K2", "all"):
        run_k2_phase_diagrams(results_dir=results_dir)

    if args.workstream in ("L", "all"):
        run_l1_archive(results_dir)


if __name__ == "__main__":
    main()
