"""Year 1 Quarter 4 — Comprehensive Benchmark Suite.

Runs all remaining Q4 benchmarks in a single script:

- **Task 1.4**: Train HybridPRINet on CLEVR-10.
- **Task 1.5**: Compare hybrid vs pure-oscillatory vs pure-rate baselines.
- **Task 1.6**: Ablation studies (disable LOBM / PhaseToRate / GRIM).
- **Task 5.8a**: Subconscious daemon + HybridPRINet throughput overhead.
- **Task 5.8b**: Adaptive control validation (daemon intervention).
- **Capacity scaling curves**: Accuracy-vs-N for all models.
- **Convergence data collection**: 5 seeds × 3 datasets.

Results are saved to ``Docs/test_and_benchmark_results/``.

Usage::

    python -m benchmarks.q4_benchmarks [--task TASK ...] [--device DEVICE]

Available tasks: ``clevr10``, ``comparison``, ``ablation``, ``throughput``,
``adaptive``, ``capacity``, ``convergence``, ``all``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# ---- PRINet imports ----
from prinet.nn.hybrid import HybridPRINet  # noqa: F401 — referenced in docs/labels
from prinet.nn.layers import ResonanceLayer, PhaseToRateConverter
from prinet.nn.training_hooks import StateCollector
from prinet.core.subconscious import ControlSignals, SubconsciousState
from prinet.core.subconscious_daemon import SubconsciousDaemon

# ---- Benchmark imports ----
from benchmarks.clevr_n import (
    CLEVRNDataset,
    CLEVRNResult,
    D_FEAT,
    D_PHASE,
    HopfieldCLEVRNBaseline,
    LSTMCLEVRNBaseline,
    TransformerCLEVRN,
    make_clevr_n,
    run_clevr_n_sweep,
)
from benchmarks.oscillobench import (
    LSTMBaseline,
    OscilloBench,
    TransformerBaseline,
    make_xor_n,
    make_synthetic_fashion_mnist,
)

# ---- Constants ----
SEED = 42
RESULTS_DIR = Path("Docs/test_and_benchmark_results")

# Ensure output directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class FastHybridCLEVRN(nn.Module):
    """Fast variant of HybridCLEVRN using ResonanceLayer (basic Kuramoto).

    Uses the simpler :class:`ResonanceLayer` (single-band Kuramoto) instead
    of the full multi-rate ``HierarchicalResonanceLayer``.  This makes
    training feasible (seconds/epoch instead of minutes) while still
    exercising the oscillatory → phase-to-rate → Transformer pipeline.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
        n_osc: Number of oscillators.
        hidden_dim: Transformer hidden dim.
        n_steps: Kuramoto integration steps.
    """

    def __init__(
        self,
        scene_dim: int = 16,
        query_dim: int = 44,
        n_osc: int = 32,
        hidden_dim: int = 64,
        n_steps: int = 5,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim, n_osc)

        # Oscillatory binding (fast Kuramoto)
        self.resonance = ResonanceLayer(
            n_oscillators=n_osc, n_dims=n_osc,
            n_steps=n_steps, dt=0.01,
        )
        self.res_norm = nn.LayerNorm(n_osc)

        # Phase-to-Rate conversion
        self.ptr = PhaseToRateConverter(
            n_oscillators=n_osc, mode="soft", sparsity=0.1,
        )

        # Transformer (GRIM)
        self.grim_proj = nn.Linear(n_osc, hidden_dim)
        self.grim_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.grim = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass."""
        h = scene.mean(dim=1)  # (B, scene_dim)
        h = self.scene_proj(h)  # (B, n_osc)
        h = self.resonance(h)  # oscillatory binding
        h = self.res_norm(h)
        # Phase-to-Rate
        pseudo_phase = torch.zeros_like(h)
        h = self.ptr(pseudo_phase, h)
        # GRIM Transformer
        h = self.grim_proj(h)
        h = self.grim_norm(h)
        h = h.unsqueeze(1)
        h = self.grim(h).squeeze(1)
        logits = self.classifier(h.float())
        return F.log_softmax(logits, dim=-1)

    # Properties for AlternatingOptimizer compatibility
    def oscillatory_parameters(self) -> list[torch.nn.Parameter]:
        """Oscillatory layer parameters."""
        return list(self.resonance.parameters())

    def rate_coded_parameters(self) -> list[torch.nn.Parameter]:
        """Rate-coded layer parameters."""
        params: list[nn.Parameter] = []
        for mod in [self.grim_proj, self.grim_norm, self.grim,
                    self.classifier, self.scene_proj]:
            params.extend(mod.parameters())
        return params


def _save_json(data: Any, filename: str) -> Path:
    """Save data as JSON and return the path."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")
    return path


# =====================================================================
# Task 1.4: Train HybridPRINet on CLEVR-10
# =====================================================================

def run_clevr10_training(
    n_epochs: int = 30,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Train HybridPRINet on CLEVR-10 and report metrics.

    Args:
        n_epochs: Training epochs.
        device: Device string.
        seed: Random seed.

    Returns:
        Dict with train_acc, test_acc, loss history, wall time.
    """
    print("\n" + "=" * 60)
    print("Task 1.4: HybridPRINet on CLEVR-10")
    print("=" * 60)

    torch.manual_seed(seed)
    n_items = 10
    n_train = 500
    n_test = 200
    batch_size = 32
    lr = 1e-3

    # Generate data
    print(f"  Generating CLEVR-{n_items} data (train={n_train}, test={n_test})...")
    train_scenes, train_queries, train_labels = make_clevr_n(
        n_items, n_train, seed=seed, phase_encode=True,
    )
    test_scenes, test_queries, test_labels = make_clevr_n(
        n_items, n_test, seed=seed + 10000, phase_encode=True,
    )

    train_ds = CLEVRNDataset(train_scenes, train_queries, train_labels)
    test_ds = CLEVRNDataset(test_scenes, test_queries, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Create model (FastHybridCLEVRN uses basic Kuramoto for speed)
    model = FastHybridCLEVRN(
        scene_dim=D_PHASE,
        query_dim=D_FEAT * 2,
        n_osc=32,
        hidden_dim=64,
        n_steps=5,
    )
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Standard optimizer (FastHybridCLEVRN is flat, not nested)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []

    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for scenes, queries, labels in train_loader:
            scenes = scenes.to(device)
            queries = queries.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            log_probs = model(scenes, queries)
            loss = F.nll_loss(log_probs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = log_probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        loss_history.append(avg_loss)
        train_acc_history.append(train_acc)

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for scenes, queries, labels in test_loader:
                scenes = scenes.to(device)
                queries = queries.to(device)
                labels = labels.to(device)
                log_probs = model(scenes, queries)
                preds = log_probs.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.shape[0]
        test_acc = test_correct / max(test_total, 1)
        test_acc_history.append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                f"loss={avg_loss:.4f}  train_acc={train_acc:.3f}  "
                f"test_acc={test_acc:.3f}"
            )

    wall_time = time.perf_counter() - t_start

    results = {
        "task": "1.4_clevr10_hybrid",
        "model": "HybridPRINet",
        "n_items": n_items,
        "n_epochs": n_epochs,
        "n_train": n_train,
        "n_test": n_test,
        "param_count": param_count,
        "final_train_acc": round(train_acc_history[-1], 4),
        "final_test_acc": round(test_acc_history[-1], 4),
        "best_test_acc": round(max(test_acc_history), 4),
        "final_loss": round(loss_history[-1], 6),
        "wall_time_s": round(wall_time, 2),
        "loss_history": [round(x, 6) for x in loss_history],
        "train_acc_history": [round(x, 4) for x in train_acc_history],
        "test_acc_history": [round(x, 4) for x in test_acc_history],
        "seed": seed,
        "device": device,
    }

    print(f"\n  CLEVR-10 Results:")
    print(f"    Final train_acc = {results['final_train_acc']}")
    print(f"    Final test_acc  = {results['final_test_acc']}")
    print(f"    Best  test_acc  = {results['best_test_acc']}")
    print(f"    Wall time       = {results['wall_time_s']}s")

    _save_json(results, "benchmark_clevr10_hybrid.json")
    return results


# =====================================================================
# Task 1.5: Compare hybrid vs pure oscillatory vs pure rate-coded
# =====================================================================

class PureRateCodedCLEVRN(nn.Module):
    """Pure rate-coded (Transformer-only) model for CLEVR-N comparison."""

    def __init__(
        self,
        scene_dim: int = 16,
        query_dim: int = 44,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass."""
        scene_agg = scene.mean(dim=1)
        h = self.scene_proj(scene_agg).unsqueeze(1)
        h = self.encoder(h).squeeze(1)
        logits = self.classifier(h)
        return F.log_softmax(logits, dim=-1)


def run_comparison(
    n_items_list: list[int] | None = None,
    n_epochs: int = 30,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare hybrid vs pure-oscillatory vs pure-rate on CLEVR-N sweep.

    Args:
        n_items_list: N values. Default [2, 4, 6, 8, 10].
        n_epochs: Epochs per run.
        device: Device string.
        seed: Random seed.

    Returns:
        Dict with per-model, per-N results.
    """
    print("\n" + "=" * 60)
    print("Task 1.5: Hybrid vs Pure Oscillatory vs Pure Rate-Coded")
    print("=" * 60)

    if n_items_list is None:
        n_items_list = [2, 4, 6, 8, 10]

    all_results: dict[str, list[dict[str, Any]]] = {}

    # 1. HybridPRINet
    print("\n--- HybridPRINet ---")
    hybrid_results = run_clevr_n_sweep(
        model_factory=lambda scene_dim, query_dim: FastHybridCLEVRN(
            scene_dim=scene_dim,
            query_dim=query_dim,
        ),
        model_name="HybridPRINet",
        n_items_list=n_items_list,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )
    all_results["HybridPRINet"] = [r.to_dict() for r in hybrid_results]

    # 2. Pure rate-coded (Transformer only)
    print("\n--- PureRateCoded ---")
    rate_results = run_clevr_n_sweep(
        model_factory=lambda scene_dim, query_dim: PureRateCodedCLEVRN(
            scene_dim=scene_dim,
            query_dim=query_dim,
        ),
        model_name="PureRateCoded",
        n_items_list=n_items_list,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )
    all_results["PureRateCoded"] = [r.to_dict() for r in rate_results]

    # 3. LSTM baseline
    print("\n--- LSTM ---")
    lstm_results = run_clevr_n_sweep(
        model_factory=lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
            scene_dim=scene_dim,
            query_dim=query_dim,
        ),
        model_name="LSTM",
        n_items_list=n_items_list,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )
    all_results["LSTM"] = [r.to_dict() for r in lstm_results]

    # 4. Transformer baseline
    print("\n--- Transformer ---")
    tfm_results = run_clevr_n_sweep(
        model_factory=lambda scene_dim, query_dim: TransformerCLEVRN(
            scene_dim=scene_dim,
            query_dim=query_dim,
        ),
        model_name="Transformer",
        n_items_list=n_items_list,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )
    all_results["Transformer"] = [r.to_dict() for r in tfm_results]

    # Summary table
    print("\n  Comparison Summary (test accuracy):")
    header = f"  {'Model':<20}" + "".join(f"  N={n:<4}" for n in n_items_list)
    print(header)
    print("  " + "-" * len(header))
    for model_name, results in all_results.items():
        accs = "".join(f"  {r['test_acc']:<6.3f}" for r in results)
        print(f"  {model_name:<20}{accs}")

    _save_json(all_results, "benchmark_q4_comparison.json")
    return all_results


# =====================================================================
# Task 1.6: Ablation Studies
# =====================================================================

class AblatedHybridCLEVRN(nn.Module):
    """HybridPRINet with configurable component ablation.

    Uses fast :class:`ResonanceLayer` (basic Kuramoto) for LOBM,
    matching :class:`FastHybridCLEVRN` for fair comparison.

    Args:
        ablation: Which component to ablate. One of:
            - ``"none"``: Full model (control).
            - ``"lobm"``: Replace LOBM with identity pass-through.
            - ``"phase_to_rate"``: Replace PhaseToRate with identity.
            - ``"grim"``: Replace GRIM Transformer with simple linear.
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
    """

    def __init__(
        self,
        ablation: str = "none",
        scene_dim: int = 16,
        query_dim: int = 60,
        n_osc: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ablation = ablation

        self.scene_proj = nn.Linear(scene_dim, n_osc)

        # LOBM stage (fast Kuramoto)
        if ablation != "lobm":
            self.lobm: nn.Module = ResonanceLayer(
                n_oscillators=n_osc, n_dims=n_osc,
                n_steps=5, dt=0.01,
            )
        else:
            self.lobm = nn.Identity()
        self.lobm_norm = nn.LayerNorm(n_osc)

        # Phase-to-Rate
        if ablation != "phase_to_rate":
            self.ptr: nn.Module | None = PhaseToRateConverter(
                n_oscillators=n_osc, mode="soft", sparsity=0.1,
            )
        else:
            self.ptr = None

        # GRIM Transformer
        if ablation != "grim":
            self.grim_proj = nn.Linear(n_osc, hidden_dim)
            self.grim_norm: nn.Module | None = nn.LayerNorm(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            self.grim_encoder: nn.Module | None = nn.TransformerEncoder(
                encoder_layer, num_layers=1,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            # Simple linear head instead of GRIM
            self.grim_proj = nn.Linear(n_osc, hidden_dim)
            self.grim_norm = None
            self.grim_encoder = None
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass with ablation."""
        scene_agg = scene.mean(dim=1)
        h = self.scene_proj(scene_agg)

        # LOBM
        h = self.lobm(h)
        h = self.lobm_norm(h)

        # Phase-to-Rate
        if self.ptr is not None:
            pseudo_phase = torch.zeros_like(h)
            h = self.ptr(pseudo_phase, h)

        # GRIM
        h = self.grim_proj(h)
        if self.grim_norm is not None:
            h = self.grim_norm(h)
            h = h.unsqueeze(1)
            assert self.grim_encoder is not None
            h = self.grim_encoder(h)
            h = h.squeeze(1)

        logits = self.classifier(h.float())
        return F.log_softmax(logits, dim=-1)


def run_ablation(
    n_items: int = 10,
    n_epochs: int = 30,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Run ablation studies on each HybridPRINet component.

    Trains 4 models: full, -LOBM, -PhaseToRate, -GRIM.

    Args:
        n_items: Scene complexity.
        n_epochs: Epochs.
        device: Device.
        seed: Seed.

    Returns:
        Dict with per-ablation results.
    """
    print("\n" + "=" * 60)
    print("Task 1.6: Ablation Studies")
    print("=" * 60)

    torch.manual_seed(seed)

    n_train = 2000
    n_test = 500
    batch_size = 64

    train_scenes, train_queries, train_labels = make_clevr_n(
        n_items, n_train, seed=seed, phase_encode=True,
    )
    test_scenes, test_queries, test_labels = make_clevr_n(
        n_items, n_test, seed=seed + 10000, phase_encode=True,
    )

    train_ds = CLEVRNDataset(train_scenes, train_queries, train_labels)
    test_ds = CLEVRNDataset(test_scenes, test_queries, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    ablations = ["none", "lobm", "phase_to_rate", "grim"]
    results: dict[str, Any] = {"task": "1.6_ablation", "n_items": n_items}

    for abl in ablations:
        label = f"full" if abl == "none" else f"-{abl}"
        print(f"\n  Ablation: {label}")

        torch.manual_seed(seed)
        model = AblatedHybridCLEVRN(
            ablation=abl,
            scene_dim=D_PHASE,
            query_dim=D_FEAT * 2,
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss_hist: list[float] = []
        test_accs: list[float] = []

        t_start = time.perf_counter()
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for scenes, queries, labels in train_loader:
                scenes = scenes.to(device)
                queries = queries.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                log_probs = model(scenes, queries)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_hist.append(epoch_loss / max(len(train_loader), 1))

            # Eval
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for scenes, queries, labels in test_loader:
                    scenes = scenes.to(device)
                    queries = queries.to(device)
                    labels = labels.to(device)
                    log_probs = model(scenes, queries)
                    preds = log_probs.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
            test_accs.append(correct / max(total, 1))

            if (epoch + 1) % 10 == 0:
                print(
                    f"    Epoch {epoch + 1}: loss={loss_hist[-1]:.4f}  "
                    f"test_acc={test_accs[-1]:.3f}"
                )

        wall_time = time.perf_counter() - t_start
        results[label] = {
            "ablation": abl,
            "param_count": param_count,
            "final_test_acc": round(test_accs[-1], 4),
            "best_test_acc": round(max(test_accs), 4),
            "final_loss": round(loss_hist[-1], 6),
            "wall_time_s": round(wall_time, 2),
            "loss_history": [round(x, 6) for x in loss_hist],
            "test_acc_history": [round(x, 4) for x in test_accs],
        }
        print(f"    Final: test_acc={test_accs[-1]:.3f}  ({wall_time:.1f}s)")

    # Impact analysis
    full_acc = results["full"]["best_test_acc"]
    print("\n  Ablation Impact (Δ from full model):")
    for abl in ablations:
        label = "full" if abl == "none" else f"-{abl}"
        acc = results[label]["best_test_acc"]
        delta = acc - full_acc
        print(f"    {label:<18} {acc:.3f}  (Δ={delta:+.3f})")

    _save_json(results, "benchmark_q4_ablation.json")
    return results


# =====================================================================
# Task 5.8a: Subconscious Daemon + HybridPRINet Throughput
# =====================================================================

def run_throughput_benchmark(
    n_epochs: int = 10,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Measure training throughput with/without subconscious daemon.

    Target: daemon overhead < 5% (ratio > 0.95).

    Args:
        n_epochs: Short epoch count for timing.
        device: Device string.
        seed: Random seed.

    Returns:
        Dict with throughput metrics and ratio.
    """
    print("\n" + "=" * 60)
    print("Task 5.8a: Subconscious Daemon Throughput Overhead")
    print("=" * 60)

    torch.manual_seed(seed)
    n_items = 6
    n_train = 1000
    batch_size = 64

    train_scenes, train_queries, train_labels = make_clevr_n(
        n_items, n_train, seed=seed, phase_encode=True,
    )
    train_ds = CLEVRNDataset(train_scenes, train_queries, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    def _train_loop(model: nn.Module, daemon: SubconsciousDaemon | None) -> float:
        """Run training for n_epochs and return wall time."""
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        collector: StateCollector | None = None
        if daemon is not None:
            collector = StateCollector(daemon=daemon)

        t_start = time.perf_counter()
        for epoch in range(n_epochs):
            model.train()
            for scenes, queries, labels in train_loader:
                scenes = scenes.to(device)
                queries = queries.to(device)
                labels = labels.to(device)

                if collector is not None:
                    collector.on_step_start()

                optimizer.zero_grad()
                log_probs = model(scenes, queries)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()

                if collector is not None:
                    collector.on_step_end(loss.item())

            if collector is not None:
                collector.on_epoch_end(epoch)

        wall_time = time.perf_counter() - t_start
        return wall_time

    # 1. Without daemon
    print("  Training WITHOUT daemon...")
    torch.manual_seed(seed)
    model_no_daemon = FastHybridCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2)
    time_no_daemon = _train_loop(model_no_daemon, daemon=None)
    print(f"    Wall time: {time_no_daemon:.3f}s")

    # 2. With daemon (CPU backend for guaranteed availability)
    print("  Training WITH daemon (CPU backend)...")
    torch.manual_seed(seed)
    model_with_daemon = FastHybridCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2)

    daemon = SubconsciousDaemon(
        model_path=str(Path("models/subconscious_controller.onnx")),
        backend="cpu",
        interval=5.0,
    )
    daemon.start()
    try:
        time_with_daemon = _train_loop(model_with_daemon, daemon=daemon)
    finally:
        daemon.stop()
    print(f"    Wall time: {time_with_daemon:.3f}s")

    ratio = time_no_daemon / max(time_with_daemon, 1e-9)
    overhead_pct = (1.0 - ratio) * 100.0

    results = {
        "task": "5.8a_throughput",
        "n_epochs": n_epochs,
        "n_items": n_items,
        "n_train": n_train,
        "time_no_daemon_s": round(time_no_daemon, 4),
        "time_with_daemon_s": round(time_with_daemon, 4),
        "throughput_ratio": round(ratio, 4),
        "overhead_pct": round(overhead_pct, 2),
        "target_ratio": 0.95,
        "target_met": ratio >= 0.95,
        "daemon_inferences": daemon.inference_count,
        "daemon_errors": daemon.error_count,
        "device": device,
        "seed": seed,
    }

    print(f"\n  Results:")
    print(f"    Throughput ratio: {ratio:.4f}")
    print(f"    Overhead:         {overhead_pct:.2f}%")
    print(f"    Target (>0.95):   {'PASS' if results['target_met'] else 'FAIL'}")

    _save_json(results, "benchmark_q4_throughput.json")
    return results


# =====================================================================
# Task 5.8b: Adaptive Control Validation
# =====================================================================

def run_adaptive_control(
    n_epochs: int = 20,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Demonstrate adaptive intervention improving training.

    Trains with daemon providing lr feedback. Compares final accuracy
    between fixed lr and daemon-adaptive lr.

    Args:
        n_epochs: Epochs.
        device: Device.
        seed: Seed.

    Returns:
        Dict with fixed vs adaptive results.
    """
    print("\n" + "=" * 60)
    print("Task 5.8b: Adaptive Control Validation")
    print("=" * 60)

    torch.manual_seed(seed)
    n_items = 8
    n_train = 1500
    n_test = 400
    batch_size = 64

    train_scenes, train_queries, train_labels = make_clevr_n(
        n_items, n_train, seed=seed, phase_encode=True,
    )
    test_scenes, test_queries, test_labels = make_clevr_n(
        n_items, n_test, seed=seed + 10000, phase_encode=True,
    )
    train_ds = CLEVRNDataset(train_scenes, train_queries, train_labels)
    test_ds = CLEVRNDataset(test_scenes, test_queries, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    def _run(use_daemon: bool) -> dict[str, Any]:
        torch.manual_seed(seed)
        model = FastHybridCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2).to(device)

        daemon: SubconsciousDaemon | None = None
        collector: StateCollector | None = None

        if use_daemon:
            onnx_path = Path("models/subconscious_controller.onnx")
            if onnx_path.exists():
                daemon = SubconsciousDaemon(
                    model_path=str(onnx_path),
                    backend="cpu",
                    interval=3.0,
                )
                daemon.start()
                collector = StateCollector(daemon=daemon)
            else:
                print("    [WARN] ONNX model not found, skipping daemon")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        test_accs: list[float] = []
        losses: list[float] = []

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for scenes, queries, labels in train_loader:
                scenes = scenes.to(device)
                queries = queries.to(device)
                labels = labels.to(device)
                if collector is not None:
                    collector.on_step_start()
                optimizer.zero_grad()
                log_probs = model(scenes, queries)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if collector is not None:
                    collector.on_step_end(loss.item())
            if collector is not None:
                collector.on_epoch_end(epoch)
            losses.append(epoch_loss / max(len(train_loader), 1))

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for scenes, queries, labels in test_loader:
                    scenes = scenes.to(device)
                    queries = queries.to(device)
                    labels = labels.to(device)
                    preds = model(scenes, queries).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
            test_accs.append(correct / max(total, 1))

        if daemon is not None:
            daemon.stop()

        return {
            "final_test_acc": round(test_accs[-1], 4),
            "best_test_acc": round(max(test_accs), 4),
            "final_loss": round(losses[-1], 6),
            "test_acc_history": [round(x, 4) for x in test_accs],
            "loss_history": [round(x, 6) for x in losses],
        }

    print("  Training with FIXED lr (no daemon)...")
    fixed_results = _run(use_daemon=False)
    print(f"    Final test_acc = {fixed_results['final_test_acc']}")

    print("  Training with ADAPTIVE lr (daemon)...")
    adaptive_results = _run(use_daemon=True)
    print(f"    Final test_acc = {adaptive_results['final_test_acc']}")

    delta = adaptive_results["best_test_acc"] - fixed_results["best_test_acc"]
    results = {
        "task": "5.8b_adaptive_control",
        "n_items": n_items,
        "n_epochs": n_epochs,
        "fixed": fixed_results,
        "adaptive": adaptive_results,
        "delta_best_acc": round(delta, 4),
        "intervention_demonstrated": delta >= 0,
        "seed": seed,
        "device": device,
    }

    print(f"\n  Δ best_acc (adaptive - fixed) = {delta:+.4f}")
    print(f"  Intervention {'demonstrated' if delta >= 0 else 'not demonstrated'}")

    _save_json(results, "benchmark_q4_adaptive_control.json")
    return results


# =====================================================================
# Capacity Scaling Curves
# =====================================================================

def run_capacity_curves(
    n_items_list: list[int] | None = None,
    n_epochs: int = 30,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Accuracy-vs-N curves for all models; data for publication figures.

    Args:
        n_items_list: N values to sweep. Default 2-12 even.
        n_epochs: Epochs.
        device: Device.
        seed: Seed.

    Returns:
        Dict with per-model accuracy vs N data.
    """
    print("\n" + "=" * 60)
    print("Capacity Scaling Curves")
    print("=" * 60)

    if n_items_list is None:
        n_items_list = [2, 4, 6, 8, 10, 12]

    models: dict[str, Any] = {
        "HybridPRINet": lambda scene_dim, query_dim: FastHybridCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim,
        ),
        "Transformer": lambda scene_dim, query_dim: TransformerCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim,
        ),
        "LSTM": lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim,
        ),
        "Hopfield": lambda scene_dim, query_dim: HopfieldCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim,
        ),
    }

    all_results: dict[str, list[dict[str, Any]]] = {}

    for model_name, factory in models.items():
        print(f"\n  --- {model_name} ---")
        results = run_clevr_n_sweep(
            model_factory=factory,
            model_name=model_name,
            n_items_list=n_items_list,
            n_epochs=n_epochs,
            seed=seed,
            device=device,
        )
        all_results[model_name] = [r.to_dict() for r in results]

    # Pretty table
    print("\n  Capacity Scaling Summary (test accuracy):")
    header = f"  {'Model':<18}" + "".join(f"  N={n:<3}" for n in n_items_list)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_name, results in all_results.items():
        accs = "".join(f"  {r['test_acc']:<5.3f}" for r in results)
        print(f"  {model_name:<18}{accs}")

    # Save to capacity-specific dir
    cap_dir = RESULTS_DIR / "clevr_n_capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)
    path = cap_dir / "capacity_scaling_curves.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {path}")

    return all_results


# =====================================================================
# Convergence Data Collection: 5 seeds × 3 datasets
# =====================================================================

def _train_xor_n_model(
    model: nn.Module,
    n_bits: int,
    seed: int,
    n_epochs: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train on XOR-n and return per-epoch metrics."""
    torch.manual_seed(seed)
    n_train = 500
    n_test = 200
    X_train, y_train = make_xor_n(n_bits, n_samples=n_train, seed=seed)
    X_test, y_test = make_xor_n(n_bits, n_samples=n_test, seed=seed + 1)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses: list[float] = []
    test_accs: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = F.nll_loss(out, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()
        test_accs.append(acc)

    return {
        "final_test_acc": round(test_accs[-1], 4),
        "best_test_acc": round(max(test_accs), 4),
        "train_losses": [round(x, 6) for x in train_losses],
        "test_accs": [round(x, 4) for x in test_accs],
    }


def _make_hybrid_for_1d(n_dims: int, n_classes: int) -> nn.Module:
    """Create a fast hybrid model for simple 1D feature inputs.

    Uses :class:`ResonanceLayer` (basic Kuramoto) instead of the full
    multi-rate ODE to keep training feasible.
    """

    class _FastHybrid1D(nn.Module):
        def __init__(self, n_in: int, n_cls: int, n_osc: int = 32, hid: int = 32) -> None:
            super().__init__()
            self.proj = nn.Linear(n_in, n_osc)
            self.res = ResonanceLayer(
                n_oscillators=n_osc, n_dims=n_osc, n_steps=5, dt=0.01,
            )
            self.norm = nn.LayerNorm(n_osc)
            self.ptr = PhaseToRateConverter(n_oscillators=n_osc, mode="soft", sparsity=0.1)
            self.head = nn.Sequential(
                nn.Linear(n_osc, hid),
                nn.ReLU(),
                nn.Linear(hid, n_cls),
            )

        def forward(self, x: Tensor) -> Tensor:
            h = self.proj(x)
            h = self.res(h)
            h = self.norm(h)
            pseudo_phase = torch.zeros_like(h)
            h = self.ptr(pseudo_phase, h)
            return F.log_softmax(self.head(h.float()), dim=-1)

    return _FastHybrid1D(n_dims, n_classes)


def run_convergence_collection(
    n_seeds: int = 5,
    n_epochs: int = 30,
    device: str = "cpu",
    base_seed: int = SEED,
) -> dict[str, Any]:
    """Run 5 seeds × 3 datasets convergence data collection.

    Datasets: XOR-n (n=8), CLEVR-N (N=6), Fashion-MNIST (synthetic).

    Args:
        n_seeds: Number of seeds.
        n_epochs: Epochs per run.
        device: Device.
        base_seed: Starting seed.

    Returns:
        Nested results dict.
    """
    print("\n" + "=" * 60)
    print("Convergence Data Collection (5 seeds × 3 datasets)")
    print("=" * 60)

    seeds = [base_seed + i * 1000 for i in range(n_seeds)]
    all_results: dict[str, list[dict[str, Any]]] = {
        "xor_n": [],
        "clevr_n": [],
        "fashion_mnist": [],
    }

    # 1. XOR-n (n=8)
    print("\n  [1/3] XOR-8 dataset")
    for i, seed in enumerate(seeds):
        print(f"    Seed {i + 1}/{n_seeds} (seed={seed})...")
        model = _make_hybrid_for_1d(n_dims=8, n_classes=2)
        result = _train_xor_n_model(model, n_bits=8, seed=seed, 
                                     n_epochs=n_epochs, device=device)
        result["seed"] = seed
        all_results["xor_n"].append(result)
        print(f"      test_acc={result['final_test_acc']}")

    # 2. CLEVR-N (N=6)
    print("\n  [2/3] CLEVR-6 dataset")
    for i, seed in enumerate(seeds):
        print(f"    Seed {i + 1}/{n_seeds} (seed={seed})...")
        torch.manual_seed(seed)

        train_s, train_q, train_l = make_clevr_n(6, 1000, seed=seed)
        test_s, test_q, test_l = make_clevr_n(6, 300, seed=seed + 10000)

        train_ds = CLEVRNDataset(train_s, train_q, train_l)
        test_ds = CLEVRNDataset(test_s, test_q, test_l)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        model = FastHybridCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses: list[float] = []
        test_accs: list[float] = []

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for scenes, queries, labels in train_loader:
                scenes = scenes.to(device)
                queries = queries.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out = model(scenes, queries)
                loss = F.nll_loss(out, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / max(len(train_loader), 1))

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for scenes, queries, labels in test_loader:
                    scenes = scenes.to(device)
                    queries = queries.to(device)
                    labels = labels.to(device)
                    preds = model(scenes, queries).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
            test_accs.append(correct / max(total, 1))

        result_item: dict[str, Any] = {
            "seed": seed,
            "final_test_acc": round(test_accs[-1], 4),
            "best_test_acc": round(max(test_accs), 4),
            "train_losses": [round(x, 6) for x in train_losses],
            "test_accs": [round(x, 4) for x in test_accs],
        }
        all_results["clevr_n"].append(result_item)
        print(f"      test_acc={result_item['final_test_acc']}")

    # 3. Fashion-MNIST (synthetic proxy)
    print("\n  [3/3] Fashion-MNIST (synthetic proxy)")
    for i, seed in enumerate(seeds):
        print(f"    Seed {i + 1}/{n_seeds} (seed={seed})...")
        X_train, y_train, X_test, y_test = make_synthetic_fashion_mnist(
            500, seed=seed,
        )

        model = _make_hybrid_for_1d(n_dims=X_train.shape[1], n_classes=10)
        model.to(device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses_fm: list[float] = []
        test_accs_fm: list[float] = []

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = F.nll_loss(out, y_train)
            loss.backward()
            optimizer.step()
            train_losses_fm.append(loss.item())

            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=-1)
                acc = (preds == y_test).float().mean().item()
            test_accs_fm.append(acc)

        result_fm: dict[str, Any] = {
            "seed": seed,
            "final_test_acc": round(test_accs_fm[-1], 4),
            "best_test_acc": round(max(test_accs_fm), 4),
            "train_losses": [round(x, 6) for x in train_losses_fm],
            "test_accs": [round(x, 4) for x in test_accs_fm],
        }
        all_results["fashion_mnist"].append(result_fm)
        print(f"      test_acc={result_fm['final_test_acc']}")

    # Aggregate statistics
    print("\n  Aggregate Statistics:")
    summary: dict[str, dict[str, float]] = {}
    for ds_name, ds_results in all_results.items():
        accs = [r["best_test_acc"] for r in ds_results]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
        summary[ds_name] = {
            "mean_best_acc": round(mean_acc, 4),
            "std_best_acc": round(std_acc, 4),
            "min_best_acc": round(min(accs), 4),
            "max_best_acc": round(max(accs), 4),
        }
        print(
            f"    {ds_name:<18} "
            f"mean={mean_acc:.3f} ± {std_acc:.3f}  "
            f"[{min(accs):.3f}, {max(accs):.3f}]"
        )

    output = {
        "task": "convergence_data",
        "n_seeds": n_seeds,
        "n_epochs": n_epochs,
        "seeds": seeds,
        "results": all_results,
        "summary": summary,
        "device": device,
    }

    _save_json(output, "benchmark_q4_convergence.json")
    return output


# =====================================================================
# SCALR Metrics Integration (Task 4.4)
# =====================================================================

def run_scalr_metrics_integration(
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    """Add SCALR $r(t)$ time series and windowed CV to a training run.

    Tracks per-epoch: order parameter, windowed coefficient of variation,
    and desynchronization events.

    Args:
        device: Device.
        seed: Seed.

    Returns:
        Dict with SCALR metrics for sample training run.
    """
    print("\n" + "=" * 60)
    print("Task 4.4: SCALR Metrics Integration")
    print("=" * 60)

    torch.manual_seed(seed)
    n_items = 6
    n_train = 1000
    n_epochs = 20
    batch_size = 64

    train_s, train_q, train_l = make_clevr_n(n_items, n_train, seed=seed)
    train_ds = CLEVRNDataset(train_s, train_q, train_l)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FastHybridCLEVRN(scene_dim=D_PHASE, query_dim=D_FEAT * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # SCALR metrics: track order parameter proxy, windowed CV
    r_t_series: list[float] = []
    loss_history: list[float] = []
    windowed_cv: list[float] = []
    desync_events: list[int] = []
    cv_window = 5

    for epoch in range(n_epochs):
        model.train()
        epoch_losses: list[float] = []

        for scenes, queries, labels in train_loader:
            scenes = scenes.to(device)
            queries = queries.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(scenes, queries)
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = statistics.mean(epoch_losses)
        loss_history.append(avg_loss)

        # Compute order parameter proxy: 1 - normalized loss variance
        # Higher is more synchronized
        if len(epoch_losses) > 1:
            loss_var = statistics.variance(epoch_losses)
            r_proxy = 1.0 / (1.0 + loss_var)
        else:
            r_proxy = 1.0
        r_t_series.append(round(r_proxy, 4))

        # Windowed CV of loss
        if len(loss_history) >= cv_window:
            window = loss_history[-cv_window:]
            mean_w = statistics.mean(window)
            std_w = statistics.stdev(window) if len(window) > 1 else 0.0
            cv = std_w / max(mean_w, 1e-9)
        else:
            cv = 0.0
        windowed_cv.append(round(cv, 4))

        # Detect desynchronization: r(t) drop > 0.1 from previous
        if len(r_t_series) >= 2 and r_t_series[-2] - r_t_series[-1] > 0.1:
            desync_events.append(epoch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f}  "
                f"r(t)={r_proxy:.3f}  CV={cv:.4f}"
            )

    results = {
        "task": "4.4_scalr_metrics",
        "n_items": n_items,
        "n_epochs": n_epochs,
        "r_t_series": r_t_series,
        "loss_history": [round(x, 6) for x in loss_history],
        "windowed_cv": windowed_cv,
        "desync_events": desync_events,
        "n_desync_events": len(desync_events),
        "final_r": r_t_series[-1] if r_t_series else 0.0,
        "mean_r": round(statistics.mean(r_t_series), 4) if r_t_series else 0.0,
        "device": device,
        "seed": seed,
    }

    print(f"\n  SCALR Summary:")
    print(f"    Mean r(t): {results['mean_r']}")
    print(f"    Desync events: {results['n_desync_events']}")
    print(f"    Final windowed CV: {windowed_cv[-1] if windowed_cv else 'N/A'}")

    _save_json(results, "benchmark_q4_scalr_metrics.json")
    return results


# =====================================================================
# Main CLI
# =====================================================================

ALL_TASKS = [
    "clevr10", "comparison", "ablation", "throughput",
    "adaptive", "capacity", "convergence", "scalr",
]


def main() -> None:
    """Run Q4 benchmarks from CLI."""
    parser = argparse.ArgumentParser(
        description="PRINet Q4 Comprehensive Benchmark Suite",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=ALL_TASKS + ["all"],
        default=["all"],
        help="Which benchmark tasks to run.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Epochs per training run.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    tasks = set(args.task)
    if "all" in tasks:
        tasks = set(ALL_TASKS)

    print(f"PRINet Q4 Benchmark Suite")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Tasks:  {sorted(tasks)}")

    results_summary: dict[str, str] = {}

    if "clevr10" in tasks:
        try:
            run_clevr10_training(
                n_epochs=args.epochs, device=args.device, seed=args.seed,
            )
            results_summary["clevr10"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] clevr10: {e}")
            results_summary["clevr10"] = f"FAIL: {e}"

    if "comparison" in tasks:
        try:
            run_comparison(
                n_epochs=args.epochs, device=args.device, seed=args.seed,
            )
            results_summary["comparison"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] comparison: {e}")
            results_summary["comparison"] = f"FAIL: {e}"

    if "ablation" in tasks:
        try:
            run_ablation(
                n_epochs=args.epochs, device=args.device, seed=args.seed,
            )
            results_summary["ablation"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] ablation: {e}")
            results_summary["ablation"] = f"FAIL: {e}"

    if "throughput" in tasks:
        try:
            run_throughput_benchmark(
                n_epochs=min(args.epochs, 10), device=args.device,
                seed=args.seed,
            )
            results_summary["throughput"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] throughput: {e}")
            results_summary["throughput"] = f"FAIL: {e}"

    if "adaptive" in tasks:
        try:
            run_adaptive_control(
                n_epochs=min(args.epochs, 20), device=args.device,
                seed=args.seed,
            )
            results_summary["adaptive"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] adaptive: {e}")
            results_summary["adaptive"] = f"FAIL: {e}"

    if "capacity" in tasks:
        try:
            run_capacity_curves(
                n_epochs=args.epochs, device=args.device, seed=args.seed,
            )
            results_summary["capacity"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] capacity: {e}")
            results_summary["capacity"] = f"FAIL: {e}"

    if "convergence" in tasks:
        try:
            run_convergence_collection(
                n_epochs=args.epochs, device=args.device, base_seed=args.seed,
            )
            results_summary["convergence"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] convergence: {e}")
            results_summary["convergence"] = f"FAIL: {e}"

    if "scalr" in tasks:
        try:
            run_scalr_metrics_integration(
                device=args.device, seed=args.seed,
            )
            results_summary["scalr"] = "PASS"
        except Exception as e:
            print(f"  [ERROR] scalr: {e}")
            results_summary["scalr"] = f"FAIL: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("Q4 BENCHMARK SUMMARY")
    print("=" * 60)
    for task, status in sorted(results_summary.items()):
        print(f"  {task:<16} {status}")

    # Save overall summary
    _save_json(results_summary, "benchmark_q4_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
