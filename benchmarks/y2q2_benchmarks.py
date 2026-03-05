"""Year 2 Q2 Benchmarks — Oscillatory Advantage & Temporal Binding.

Workstream D: CLEVR-N Scaling Sweep (N=2–15, 4+ models, crossover detection).
Workstream E: Temporal CLEVR (5-frame sequences, phase propagation benchmark).
Workstream F: Active Subconscious Control A/B Test (10 active vs 10 passive).

Usage:
    python benchmarks/y2q2_benchmarks.py --workstream D
    python benchmarks/y2q2_benchmarks.py --workstream E
    python benchmarks/y2q2_benchmarks.py --workstream F
    python benchmarks/y2q2_benchmarks.py --all
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---- Imports from PRINet ------------------------------------------------

from benchmarks.clevr_n import (
    CLEVRNDataset,
    CLEVRNResult,
    D_FEAT,
    D_PHASE,
    SEED,
    LSTMCLEVRNBaseline,
    TransformerCLEVRN,
    make_clevr_n,
    run_clevr_n_sweep,
    _train_model,
    _eval_model,
)
from benchmarks.y2q1_benchmarks import (
    DiscreteDTGCLEVRN,
    InterleavedCLEVRN,
)

_RESULTS_DIR = Path("Docs/test_and_benchmark_results")

TWO_PI = 2.0 * math.pi


def _save_json(data: Any, filename: str) -> Path:
    """Save benchmark results to JSON."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")
    return path


# =====================================================================
# Workstream D: CLEVR-N Scaling Sweep
# =====================================================================


def run_d1_scaling_sweep(
    n_items_list: list[int] | None = None,
    n_epochs: int = 30,
    n_train: int = 2000,
    n_test: int = 500,
    seed: int = SEED,
    device: str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    """D.1: CLEVR-N sweep (N=2–15) with 4+ models.

    Runs the accuracy-vs-N sweep for LSTM, Transformer, DiscreteDTG,
    and InterleavedHybrid models. Reports per-N accuracy for each.

    Args:
        n_items_list: N values to sweep. Default: [2,4,6,8,10,12,15].
        n_epochs: Training epochs per run.
        n_train: Training samples per N.
        n_test: Test samples per N.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict mapping model name → list of per-N result dicts.
    """
    if n_items_list is None:
        n_items_list = [2, 4, 6, 8, 10, 12, 15]

    models: dict[str, Any] = {
        "LSTM": lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "Transformer": lambda scene_dim, query_dim: TransformerCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "DiscreteDTG": lambda scene_dim, query_dim: DiscreteDTGCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim,
        ),
        "InterleavedHybrid": lambda scene_dim, query_dim: InterleavedCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim,
            n_items=max(n_items_list),
        ),
    }

    all_results: dict[str, list[dict[str, Any]]] = {}

    for name, factory in models.items():
        print(f"\n=== {name} ===")
        results = run_clevr_n_sweep(
            factory,
            name,
            n_items_list=n_items_list,
            n_train=n_train,
            n_test=n_test,
            n_epochs=n_epochs,
            seed=seed,
            device=device,
        )
        all_results[name] = [r.to_dict() for r in results]

    _save_json(all_results, "y2q2_d1_scaling_sweep.json")
    return all_results


def find_crossover_point(
    sweep_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """D.2: Identify crossover point where oscillatory models exceed baselines.

    Scans the sweep results for the first N where DiscreteDTG or
    InterleavedHybrid test accuracy exceeds Transformer test accuracy.

    Args:
        sweep_results: Output from :func:`run_d1_scaling_sweep`.

    Returns:
        Dict with crossover analysis: ``crossover_n``, ``model``,
        ``osc_acc``, ``transformer_acc``, or ``no_crossover: True``.
    """
    transformer_results = {
        r["n_items"]: r["test_acc"]
        for r in sweep_results.get("Transformer", [])
    }

    analysis: dict[str, Any] = {"crossover_found": False}
    best_advantage = -1.0

    for osc_name in ["DiscreteDTG", "InterleavedHybrid"]:
        osc_results = sweep_results.get(osc_name, [])
        for r in osc_results:
            n = r["n_items"]
            osc_acc = r["test_acc"]
            tf_acc = transformer_results.get(n, 0.0)
            advantage = osc_acc - tf_acc

            if advantage > best_advantage:
                best_advantage = advantage
                if advantage > 0:
                    analysis["crossover_found"] = True
                    analysis["crossover_n"] = n
                    analysis["model"] = osc_name
                    analysis["osc_acc"] = osc_acc
                    analysis["transformer_acc"] = tf_acc
                    analysis["advantage_pct"] = round(advantage * 100, 2)

    if not analysis["crossover_found"]:
        analysis["best_advantage_pct"] = round(best_advantage * 100, 2)
        analysis["recommendation"] = (
            "No crossover found. Consider harder binding task (D.3)."
        )

    _save_json(analysis, "y2q2_d2_crossover_analysis.json")
    return analysis


# =====================================================================
# Workstream E: Temporal CLEVR & Phase Propagation Benchmark
# =====================================================================


def make_temporal_clevr(
    n_items: int = 6,
    n_frames: int = 5,
    n_samples: int = 1000,
    seed: int = SEED,
    movement_scale: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate Temporal CLEVR dataset: multi-frame sequences.

    Each sample is a sequence of T frames. Objects persist across frames
    with small position changes (simulating movement). The query asks
    whether two specific objects maintain their relative order across
    all frames (temporal identity tracking).

    Args:
        n_items: Objects per scene.
        n_frames: Frames per sequence.
        n_samples: Number of sequences.
        seed: Random seed.
        movement_scale: Max position shift per frame (in grid units).

    Returns:
        scenes: ``(n_samples, n_frames, n_items, D_PHASE)``
        queries: ``(n_samples, D_FEAT * 2)``
        labels: ``(n_samples,)`` — 1 if relative order is maintained.
    """
    from benchmarks.clevr_n import (
        COLORS,
        D_COLOR,
        D_SHAPE,
        N_POSITIONS,
        SHAPES,
        encode_features_phase,
        _sample_scene,
    )

    rng = torch.Generator().manual_seed(seed)

    all_scenes: list[Tensor] = []
    all_queries: list[Tensor] = []
    all_labels: list[int] = []

    for _ in range(n_samples):
        # Initial scene
        colors, shapes, positions = _sample_scene(n_items, rng)
        positions_float = positions.float()

        frame_encodings: list[Tensor] = []

        for t in range(n_frames):
            if t > 0:
                # Small random position perturbation
                delta = (
                    torch.rand(n_items, generator=rng) * 2 - 1
                ) * movement_scale
                positions_float = positions_float + delta
                # Clamp to valid range
                positions_float = torch.clamp(
                    positions_float, 0.0, float(N_POSITIONS - 1)
                )

            # Quantize positions for encoding
            pos_ids = positions_float.round().long().clamp(0, N_POSITIONS - 1)
            enc = encode_features_phase(colors, shapes, pos_ids)
            frame_encodings.append(enc)

        scene_seq = torch.stack(frame_encodings)  # (T, n_items, D_PHASE)

        # Query: pick two objects, check if relative order maintained
        idx = torch.randperm(n_items, generator=rng)[:2]
        i, j = idx[0].item(), idx[1].item()

        # One-hot query for the two objects
        def _one_hot(c: int, s: int, p: int) -> Tensor:
            vec = torch.zeros(D_FEAT)
            vec[c] = 1.0
            vec[D_COLOR + s] = 1.0
            vec[D_COLOR + D_SHAPE + p] = 1.0
            return vec

        initial_pos = positions.clone()
        q_i = _one_hot(
            colors[i].item(), shapes[i].item(), initial_pos[i].item()
        )
        q_j = _one_hot(
            colors[j].item(), shapes[j].item(), initial_pos[j].item()
        )
        query = torch.cat([q_i, q_j])

        # Label: 1 if relative order is maintained across all frames
        initial_order = initial_pos[i] < initial_pos[j]
        # Check final frame positions
        final_pos = positions_float.round().long().clamp(0, N_POSITIONS - 1)
        final_order = final_pos[i] < final_pos[j]
        label = 1 if (initial_order == final_order) else 0

        all_scenes.append(scene_seq)
        all_queries.append(query)
        all_labels.append(label)

    scenes = torch.stack(all_scenes)   # (N, T, n_items, D_PHASE)
    queries = torch.stack(all_queries)  # (N, D_FEAT*2)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return scenes, queries, labels


class TemporalCLEVRDataset(Dataset):
    """Dataset wrapping Temporal CLEVR data."""

    def __init__(
        self, scenes: Tensor, queries: Tensor, labels: Tensor
    ) -> None:
        self.scenes = scenes
        self.queries = queries
        self.labels = labels

    def __len__(self) -> int:
        return self.scenes.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.scenes[idx], self.queries[idx], self.labels[idx]


# ---- Temporal Models ----


class TemporalLSTMBaseline(nn.Module):
    """LSTM baseline for temporal CLEVR.

    Processes each frame through an LSTM, using the final hidden state
    for classification.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
        hidden_dim: LSTM hidden dimension.
        n_items: Maximum number of objects per scene.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        hidden_dim: int = 64,
        n_items: int = 6,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim * n_items, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True
        )
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene_seq: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene_seq: ``(B, T, n_items, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log-probabilities ``(B, 2)``.
        """
        B, T, N, D = scene_seq.shape
        flat_scenes = scene_seq.view(B, T, -1)  # (B, T, N*D)
        h = self.scene_proj(flat_scenes)  # (B, T, hidden)
        output, (hn, _) = self.lstm(h)
        last_h = hn.squeeze(0)  # (B, hidden)
        q = self.query_proj(query)
        combined = torch.cat([last_h, q], dim=-1)
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)


class TemporalTransformerBaseline(nn.Module):
    """Transformer baseline for temporal CLEVR.

    Flattens all frames into a token sequence, applies Transformer
    encoder, then classifies from pooled output.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        n_items: Maximum objects per scene.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_items: int = 6,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.query_proj = nn.Linear(query_dim, d_model)
        # Temporal position encoding
        self.temporal_pos = nn.Parameter(torch.randn(1, 20, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2),
        )

    def forward(self, scene_seq: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene_seq: ``(B, T, n_items, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log-probabilities ``(B, 2)``.
        """
        B, T, N, D = scene_seq.shape
        h = self.scene_proj(scene_seq)  # (B, T, N, d_model)
        # Add temporal position encoding
        h = h + self.temporal_pos[:, :T, :, :]
        # Flatten to token sequence: (B, T*N, d_model)
        h = h.view(B, T * N, -1)
        # Prepend query as CLS token
        q = self.query_proj(query).unsqueeze(1)  # (B, 1, d_model)
        tokens = torch.cat([q, h], dim=1)  # (B, 1+T*N, d_model)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]  # CLS token output
        logits = self.classifier(cls_out)
        return F.log_softmax(logits, dim=-1)


class TemporalHybridCLEVRN(nn.Module):
    """TemporalHybridPRINet wrapper for Temporal CLEVR benchmark.

    Uses the temporal phase propagation architecture to maintain
    persistent object binding across frames.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
        d_model: Model dimension.
        n_heads: Attention heads.
        n_layers: Interleaved blocks.
        n_items: Maximum objects per scene.
        carry_strength: Phase carry-over strength.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_items: int = 6,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        carry_strength: float = 0.8,
    ) -> None:
        super().__init__()
        from prinet.nn.hybrid import TemporalHybridPRINet

        n_osc = n_delta + n_theta + n_gamma
        input_dim = scene_dim * n_items + query_dim

        self.model = TemporalHybridPRINet(
            n_input=input_dim,
            n_classes=2,
            n_tokens=n_osc,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.1,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_discrete_steps=3,
            carry_strength=carry_strength,
        )
        self._scene_dim = scene_dim
        self._n_items = n_items

    def forward(self, scene_seq: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene_seq: ``(B, T, n_items, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log-probabilities ``(B, 2)``.
        """
        B, T, N, D = scene_seq.shape
        # Pad/truncate items
        if N < self._n_items:
            pad = torch.zeros(
                B, T, self._n_items - N, D,
                device=scene_seq.device, dtype=scene_seq.dtype,
            )
            scene_seq = torch.cat([scene_seq, pad], dim=2)
        elif N > self._n_items:
            scene_seq = scene_seq[:, :, :self._n_items, :]

        # Flatten per-frame scene + query → (B, T, input_dim)
        flat_scenes = scene_seq.view(B, T, -1)  # (B, T, n_items*D)
        query_expanded = query.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([flat_scenes, query_expanded], dim=-1)

        return self.model(x)  # (B, 2) — last-frame classification


def _train_temporal_model(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Train a temporal CLEVR model and return final training accuracy."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    correct = 0
    total = 0

    for epoch in range(n_epochs):
        correct = 0
        total = 0
        for scene_seq, queries, labels in train_loader:
            scene_seq = scene_seq.to(device)
            queries = queries.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            log_probs = model(scene_seq, queries)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()

            preds = log_probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

    return correct / max(total, 1)


@torch.no_grad()
def _eval_temporal_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Evaluate a temporal CLEVR model and return accuracy."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for scene_seq, queries, labels in loader:
        scene_seq = scene_seq.to(device)
        queries = queries.to(device)
        labels = labels.to(device)
        log_probs = model(scene_seq, queries)
        preds = log_probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / max(total, 1)


def run_e_temporal_benchmark(
    n_items: int = 6,
    n_frames: int = 5,
    n_train: int = 2000,
    n_test: int = 500,
    n_epochs: int = 30,
    seed: int = SEED,
    device: str = "cpu",
) -> dict[str, Any]:
    """E.3: Benchmark HybridPRINet vs LSTM vs Transformer on temporal CLEVR.

    Args:
        n_items: Objects per scene.
        n_frames: Frames per sequence.
        n_train: Training samples.
        n_test: Test samples.
        n_epochs: Training epochs.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict with per-model accuracy results and analysis.
    """
    print(f"\n{'='*60}")
    print(f"Temporal CLEVR Benchmark: N={n_items}, T={n_frames}")
    print(f"{'='*60}")

    # Generate data
    train_scenes, train_queries, train_labels = make_temporal_clevr(
        n_items, n_frames, n_train, seed=seed
    )
    test_scenes, test_queries, test_labels = make_temporal_clevr(
        n_items, n_frames, n_test, seed=seed + 10000
    )

    train_ds = TemporalCLEVRDataset(train_scenes, train_queries, train_labels)
    test_ds = TemporalCLEVRDataset(test_scenes, test_queries, test_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    results: dict[str, Any] = {
        "n_items": n_items,
        "n_frames": n_frames,
        "n_train": n_train,
        "n_test": n_test,
        "n_epochs": n_epochs,
        "models": {},
    }

    temporal_models: dict[str, nn.Module] = {
        "TemporalLSTM": TemporalLSTMBaseline(n_items=n_items),
        "TemporalTransformer": TemporalTransformerBaseline(n_items=n_items),
        "TemporalHybridPRINet": TemporalHybridCLEVRN(n_items=n_items),
    }

    for name, model in temporal_models.items():
        print(f"\n--- {name} ---")
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        train_acc = _train_temporal_model(
            model, train_loader, n_epochs, device=device
        )
        train_time = time.time() - t0
        test_acc = _eval_temporal_model(model, test_loader, device=device)

        results["models"][name] = {
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "n_params": n_params,
            "train_time_s": round(train_time, 2),
        }
        print(
            f"  {name}: train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
            f"params={n_params} time={train_time:.1f}s"
        )

    # Check for oscillatory advantage (≥5% over best attention baseline)
    hybrid_acc = results["models"].get("TemporalHybridPRINet", {}).get(
        "test_acc", 0.0
    )
    best_baseline = max(
        results["models"].get("TemporalLSTM", {}).get("test_acc", 0.0),
        results["models"].get("TemporalTransformer", {}).get("test_acc", 0.0),
    )
    advantage = hybrid_acc - best_baseline
    results["analysis"] = {
        "hybrid_test_acc": hybrid_acc,
        "best_baseline_acc": best_baseline,
        "advantage_pct": round(advantage * 100, 2),
        "oscillatory_advantage_5pct": advantage >= 0.05,
    }

    _save_json(results, "y2q2_e3_temporal_benchmark.json")
    return results


# =====================================================================
# Workstream F: Active Subconscious Control A/B Test
# =====================================================================


def run_f_ab_test(
    n_runs_per_group: int = 10,
    n_epochs: int = 30,
    n_items: int = 6,
    base_seed: int = SEED,
    device: str = "cpu",
    max_adjustment: float = 0.05,
) -> dict[str, Any]:
    """F.2: A/B test — active vs passive subconscious control.

    Runs ``n_runs_per_group`` training runs with active control and
    ``n_runs_per_group`` with passive (observation-only) control.
    Compares final accuracy using Welch's t-test.

    Args:
        n_runs_per_group: Runs per group (active/passive).
        n_epochs: Epochs per run.
        n_items: CLEVR-N object count.
        base_seed: Base seed (each run uses base_seed + run_idx).
        device: Device string.
        max_adjustment: Max per-signal adjustment for active mode.

    Returns:
        Dict with per-run accuracies, t-statistic, p-value, and analysis.
    """
    from prinet.nn.training_hooks import ActiveControlTrainer, TelemetryLogger

    print(f"\n{'='*60}")
    print(f"A/B Test: Active vs Passive Control (N={n_items})")
    print(f"{'='*60}")

    results: dict[str, Any] = {
        "n_runs_per_group": n_runs_per_group,
        "n_epochs": n_epochs,
        "n_items": n_items,
        "max_adjustment": max_adjustment,
        "active_runs": [],
        "passive_runs": [],
    }

    # Generate shared test set
    _, _, _ = make_clevr_n(n_items, 500, seed=base_seed + 99999)

    for mode in ["active", "passive"]:
        is_active = mode == "active"
        run_accs: list[float] = []

        for run_idx in range(n_runs_per_group):
            run_seed = base_seed + (1000 if is_active else 2000) + run_idx
            torch.manual_seed(run_seed)

            # Generate data
            train_scenes, train_queries, train_labels = make_clevr_n(
                n_items, 2000, seed=run_seed
            )
            test_scenes, test_queries, test_labels = make_clevr_n(
                n_items, 500, seed=run_seed + 50000
            )
            train_ds = CLEVRNDataset(
                train_scenes, train_queries, train_labels
            )
            test_ds = CLEVRNDataset(test_scenes, test_queries, test_labels)
            train_loader = DataLoader(
                train_ds, batch_size=64, shuffle=True
            )
            test_loader = DataLoader(test_ds, batch_size=64)

            # Create model
            model = DiscreteDTGCLEVRN(
                scene_dim=D_PHASE, query_dim=D_FEAT * 2
            )
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Create trainer (no real daemon — use None for lightweight test)
            trainer = ActiveControlTrainer(
                model=model,
                optimizer=optimizer,
                daemon=None,
                max_adjustment=max_adjustment,
                active=is_active,
            )

            # Training loop
            model.train()
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0
                for scenes, queries, labels in train_loader:
                    scenes = scenes.to(device)
                    queries = queries.to(device)
                    labels = labels.to(device)

                    trainer.on_step_start()
                    optimizer.zero_grad()
                    log_probs = model(scenes, queries)
                    loss = F.nll_loss(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    trainer.on_step_end(loss.item())

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                trainer.on_epoch_end(
                    epoch=epoch, loss=avg_loss,
                    lr_current=optimizer.param_groups[0]["lr"],
                )

            # Evaluate
            test_acc = _eval_model(model, test_loader, device=device)
            run_accs.append(test_acc)

            print(
                f"  {mode} run {run_idx+1}/{n_runs_per_group}: "
                f"test_acc={test_acc:.3f}"
            )

        results[f"{mode}_runs"] = run_accs

    # Statistical comparison using Welch's t-test
    active_accs = torch.tensor(results["active_runs"])
    passive_accs = torch.tensor(results["passive_runs"])

    n_a = len(results["active_runs"])
    n_p = len(results["passive_runs"])
    mean_a = active_accs.mean().item()
    mean_p = passive_accs.mean().item()
    var_a = active_accs.var(unbiased=True).item()
    var_p = passive_accs.var(unbiased=True).item()

    # Welch's t-test
    se = math.sqrt(var_a / max(n_a, 1) + var_p / max(n_p, 1)) + 1e-10
    t_stat = (mean_a - mean_p) / se

    # Degrees of freedom (Welch-Satterthwaite)
    num = (var_a / max(n_a, 1) + var_p / max(n_p, 1)) ** 2
    denom = (
        (var_a / max(n_a, 1)) ** 2 / max(n_a - 1, 1)
        + (var_p / max(n_p, 1)) ** 2 / max(n_p - 1, 1)
    ) + 1e-10
    df = num / denom

    # Two-tailed p-value approximation (using normal for df > 30)
    # For simplicity, use a rough approximation
    import math as _math
    z = abs(t_stat)
    # Normal CDF approximation for p-value
    p_value = 2.0 * (1.0 / (1.0 + _math.exp(1.7 * z))) if z < 10 else 0.0

    results["statistics"] = {
        "active_mean": round(mean_a, 4),
        "passive_mean": round(mean_p, 4),
        "active_std": round(math.sqrt(max(var_a, 0)), 4),
        "passive_std": round(math.sqrt(max(var_p, 0)), 4),
        "t_statistic": round(t_stat, 4),
        "degrees_of_freedom": round(df, 2),
        "p_value_approx": round(p_value, 4),
        "significant_at_005": p_value < 0.05,
        "max_adjustment_pct": max_adjustment * 100,
    }

    print(f"\n--- A/B Test Results ---")
    print(f"Active:  mean={mean_a:.4f} +/- {math.sqrt(max(var_a, 0)):.4f}")
    print(f"Passive: mean={mean_p:.4f} +/- {math.sqrt(max(var_p, 0)):.4f}")
    print(f"t={t_stat:.3f}, p~={p_value:.4f}")

    _save_json(results, "y2q2_f2_ab_test_results.json")
    return results


# =====================================================================
# CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Y2 Q2 Benchmarks")
    parser.add_argument(
        "--workstream",
        choices=["D", "E", "F"],
        help="Run a specific workstream",
    )
    parser.add_argument("--all", action="store_true", help="Run all workstreams")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.all or args.workstream == "D":
        print("\n" + "=" * 70)
        print("WORKSTREAM D: CLEVR-N Scaling Sweep")
        print("=" * 70)
        sweep = run_d1_scaling_sweep(
            n_epochs=args.epochs, seed=args.seed, device=args.device
        )
        find_crossover_point(sweep)

    if args.all or args.workstream == "E":
        print("\n" + "=" * 70)
        print("WORKSTREAM E: Temporal CLEVR Benchmark")
        print("=" * 70)
        run_e_temporal_benchmark(
            n_epochs=args.epochs, seed=args.seed, device=args.device
        )

    if args.all or args.workstream == "F":
        print("\n" + "=" * 70)
        print("WORKSTREAM F: Active Control A/B Test")
        print("=" * 70)
        run_f_ab_test(
            n_epochs=args.epochs, base_seed=args.seed, device=args.device
        )
