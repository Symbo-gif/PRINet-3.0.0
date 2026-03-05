"""CLEVR-N Binding Capacity Benchmark.

Generates synthetic CLEVR-style scenes with N bindable items.
Each item = (color, shape, position) feature tuple encoded as phase offsets.
Queries test relational reasoning: "Is object A left of object B?"

This module provides:
- ``make_clevr_n``: Deterministic scene + query + label generator.
- ``encode_features_phase``: Maps discrete features → continuous phase vectors.
- ``CLEVRNDataset``: PyTorch Dataset wrapper.
- ``TransformerCLEVRN``: Small Transformer baseline for CLEVR-N.
- ``LSTMCLEVRNBaseline``: LSTM baseline for CLEVR-N.
- ``HopfieldCLEVRNBaseline``: Modern Hopfield baseline for CLEVR-N.
- ``run_clevr_n_sweep``: Run accuracy-vs-N sweep for a model.
- ``run_all_baselines``: Run all baselines and collect results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---- Constants ---------------------------------------------------------

SEED = 42

# Discrete feature vocabularies
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "gray"]
SHAPES = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon"]
N_POSITIONS = 16  # discrete x-positions (left → right)

# Feature dimensions
D_COLOR = len(COLORS)
D_SHAPE = len(SHAPES)
D_POS = N_POSITIONS
D_FEAT = D_COLOR + D_SHAPE + D_POS  # one-hot total
D_PHASE = 16  # phase encoding dimension

TWO_PI = 2.0 * math.pi


# ---- Data Generation ---------------------------------------------------


def _sample_scene(
    n_items: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a single CLEVR-N scene.

    Returns:
        color_ids ``(N,)``, shape_ids ``(N,)``, pos_ids ``(N,)``
    """
    colors = torch.randint(0, D_COLOR, (n_items,), generator=rng)
    shapes = torch.randint(0, D_SHAPE, (n_items,), generator=rng)
    # Positions are unique per scene (no two objects in same slot)
    positions = torch.randperm(N_POSITIONS, generator=rng)[:n_items]
    return colors, shapes, positions


def _make_query(
    pos_ids: Tensor,
    color_ids: Tensor,
    shape_ids: Tensor,
    rng: torch.Generator,
) -> tuple[Tensor, int]:
    """Generate a relational query for a scene.

    Query: "Is object i to the left of object j?" (pos_i < pos_j)

    Returns:
        query_vec ``(D_FEAT * 2,)`` — concatenated one-hot for both objects,
        label: 1 if True, 0 if False.
    """
    n = pos_ids.shape[0]
    # Pick two distinct objects
    idx = torch.randperm(n, generator=rng)[:2]
    i, j = idx[0].item(), idx[1].item()

    # One-hot encode both query objects
    def _one_hot(c: int, s: int, p: int) -> Tensor:
        vec = torch.zeros(D_FEAT)
        vec[c] = 1.0
        vec[D_COLOR + s] = 1.0
        vec[D_COLOR + D_SHAPE + p] = 1.0
        return vec

    q_i = _one_hot(color_ids[i].item(), shape_ids[i].item(), pos_ids[i].item())
    q_j = _one_hot(color_ids[j].item(), shape_ids[j].item(), pos_ids[j].item())

    query_vec = torch.cat([q_i, q_j])
    label = 1 if pos_ids[i] < pos_ids[j] else 0
    return query_vec, label


def encode_features_phase(
    color_ids: Tensor,
    shape_ids: Tensor,
    pos_ids: Tensor,
    d_phase: int = D_PHASE,
) -> Tensor:
    """Encode discrete features as phase vectors.

    Each feature dimension maps to a unique angular offset in [0, 2π).
    The encoding concatenates phase sin/cos pairs for each feature type.

    Args:
        color_ids: ``(N,)`` integer color indices.
        shape_ids: ``(N,)`` integer shape indices.
        pos_ids: ``(N,)`` integer position indices.
        d_phase: Output phase dimension (must be even).

    Returns:
        Phase encoding ``(N, d_phase)``.
    """
    N = color_ids.shape[0]
    d = d_phase // 6  # allocate dims per feature type (sin + cos pairs)
    if d < 1:
        d = 1
    enc = torch.zeros(N, d_phase)

    # Color phase: each color → unique angle
    color_angle = (color_ids.float() / D_COLOR) * TWO_PI  # (N,)
    for k in range(d):
        freq = float(k + 1)
        enc[:, 2 * k] = torch.sin(freq * color_angle)
        enc[:, 2 * k + 1] = torch.cos(freq * color_angle)

    # Shape phase
    shape_angle = (shape_ids.float() / D_SHAPE) * TWO_PI
    offset = 2 * d
    for k in range(d):
        freq = float(k + 1)
        enc[:, offset + 2 * k] = torch.sin(freq * shape_angle)
        enc[:, offset + 2 * k + 1] = torch.cos(freq * shape_angle)

    # Position phase
    pos_angle = (pos_ids.float() / N_POSITIONS) * TWO_PI
    offset = 4 * d
    for k in range(d):
        freq = float(k + 1)
        enc[:, offset + 2 * k] = torch.sin(freq * pos_angle)
        enc[:, offset + 2 * k + 1] = torch.cos(freq * pos_angle)

    return enc


def make_clevr_n(
    n_items: int,
    n_samples: int = 1000,
    seed: int = SEED,
    phase_encode: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate CLEVR-N benchmark dataset.

    Args:
        n_items: Number of objects per scene (2–15).
        n_samples: Number of (scene, query, label) samples.
        seed: Random seed for reproducibility.
        phase_encode: If True, return phase-encoded scenes; else one-hot.

    Returns:
        scenes: ``(n_samples, n_items, D)`` scene tensors.
        queries: ``(n_samples, D_FEAT * 2)`` query tensors.
        labels: ``(n_samples,)`` binary labels.
    """
    assert 2 <= n_items <= 15, f"n_items must be in [2, 15], got {n_items}"
    assert n_items <= N_POSITIONS, f"n_items ({n_items}) > N_POSITIONS ({N_POSITIONS})"

    rng = torch.Generator().manual_seed(seed)

    all_scenes = []
    all_queries = []
    all_labels = []

    for _ in range(n_samples):
        colors, shapes, positions = _sample_scene(n_items, rng)

        if phase_encode:
            scene_enc = encode_features_phase(colors, shapes, positions)
        else:
            # One-hot encoding
            scene_enc = torch.zeros(n_items, D_FEAT)
            for idx in range(n_items):
                scene_enc[idx, colors[idx]] = 1.0
                scene_enc[idx, D_COLOR + shapes[idx]] = 1.0
                scene_enc[idx, D_COLOR + D_SHAPE + positions[idx]] = 1.0

        query_vec, label = _make_query(positions, colors, shapes, rng)

        all_scenes.append(scene_enc)
        all_queries.append(query_vec)
        all_labels.append(label)

    scenes = torch.stack(all_scenes)  # (n_samples, n_items, D)
    queries = torch.stack(all_queries)  # (n_samples, D_FEAT * 2)
    labels = torch.tensor(all_labels, dtype=torch.long)  # (n_samples,)

    return scenes, queries, labels


class CLEVRNDataset(Dataset):
    """PyTorch Dataset wrapping CLEVR-N data."""

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


# ---- Baseline Models ---------------------------------------------------


class LSTMCLEVRNBaseline(nn.Module):
    """LSTM baseline for CLEVR-N relational queries.

    Processes scene items as a sequence, concatenates final hidden
    state with the query vector, and classifies.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        hidden_dim: int = 64,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(scene_dim, hidden_dim, n_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """
        Args:
            scene: ``(B, N, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log probabilities ``(B, 2)``.
        """
        _, (h_n, _) = self.lstm(scene)
        h = h_n[-1]  # (B, hidden)
        combined = torch.cat([h, query], dim=-1)
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)


class TransformerCLEVRN(nn.Module):
    """Small Transformer encoder baseline for CLEVR-N.

    Each scene object is a token. Query is prepended as a special token.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_items: int = 16,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.query_proj = nn.Linear(query_dim, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_items + 1, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        B, N, _ = scene.shape
        scene_tokens = self.scene_proj(scene)  # (B, N, d_model)
        query_token = self.query_proj(query).unsqueeze(1)  # (B, 1, d_model)
        tokens = torch.cat([query_token, scene_tokens], dim=1)  # (B, N+1, d)
        tokens = tokens + self.pos_embed[:, : N + 1, :]
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]  # query token output
        logits = self.classifier(cls_out)
        return F.log_softmax(logits, dim=-1)


class HopfieldCLEVRNBaseline(nn.Module):
    """Modern Hopfield Network baseline for CLEVR-N.

    Uses the query as a retrieval key over stored scene patterns,
    then classifies from the retrieved representation.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        d_model: int = 64,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.query_proj = nn.Linear(query_dim, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        B, N, _ = scene.shape
        keys = self.scene_proj(scene)  # (B, N, d_model)
        q = self.query_proj(query)  # (B, d_model)

        # Hopfield retrieval: softmax attention
        scores = self.beta * torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1)
        attn = F.softmax(scores, dim=-1)  # (B, N)
        retrieved = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)  # (B, d)

        logits = self.classifier(retrieved)
        return F.log_softmax(logits, dim=-1)


# ---- Hierarchical Oscillator Models -----------------------------------


class ThetaGammaCLEVRN(nn.Module):
    """2-frequency (Theta-Gamma) model for CLEVR-N binding.

    Encodes scene items as Gamma oscillators phase-locked under
    Theta envelope. Binding capacity expected ~7 items.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        n_theta: int = 16,
        n_gamma: int = 64,
        hidden_dim: int = 64,
        n_integration_steps: int = 20,
    ) -> None:
        super().__init__()
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        self.n_steps = n_integration_steps

        # Project scene items → gamma oscillator init phases
        self.scene_proj = nn.Linear(scene_dim, n_gamma)
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Readout from final oscillator state
        self.readout = nn.Sequential(
            nn.Linear(n_theta + n_gamma + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene: ``(B, N, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log probabilities ``(B, 2)``.
        """
        from prinet.core.measurement import kuramoto_order_parameter
        from prinet.core.propagation import ThetaGammaNetwork

        B = scene.shape[0]
        device = scene.device

        # Aggregate scene into gamma-dim init signal
        scene_flat = scene.mean(dim=1)  # (B, D_scene)
        gamma_init = torch.sigmoid(self.scene_proj(scene_flat))  # (B, n_gamma)

        # Run ThetaGamma network per batch
        order_feats: list[Tensor] = []
        for b in range(B):
            net = ThetaGammaNetwork(
                n_theta=self.n_theta,
                n_gamma=self.n_gamma,
                device=device,
            )
            state = net.create_initial_state(seed=42 + b)
            # Inject scene info into gamma amplitudes
            gamma_s = state[1]
            from prinet.core.propagation import OscillatorState
            state = (
                state[0],
                OscillatorState(
                    phase=gamma_s.phase,
                    amplitude=gamma_init[b].detach(),
                    frequency=gamma_s.frequency,
                ),
            )
            final, _ = net.integrate(state, self.n_steps, dt=0.01)
            r_theta = kuramoto_order_parameter(final[0].phase)
            r_gamma = kuramoto_order_parameter(final[1].phase)
            feat = torch.cat([
                final[0].amplitude,
                final[1].amplitude,
            ])
            order_feats.append(feat)

        osc_feat = torch.stack(order_feats, dim=0).to(device)  # (B, n_theta+n_gamma)
        q_feat = self.query_proj(query)  # (B, hidden)
        combined = torch.cat([osc_feat, q_feat], dim=-1)
        logits = self.readout(combined)
        return F.log_softmax(logits, dim=-1)


class DeltaThetaGammaCLEVRN(nn.Module):
    """3-frequency (Delta-Theta-Gamma) model for CLEVR-N binding.

    Encodes scene items using a 3-level oscillatory hierarchy.
    Binding capacity expected > 7 items.
    """

    def __init__(
        self,
        scene_dim: int = D_PHASE,
        query_dim: int = D_FEAT * 2,
        n_delta: int = 8,
        n_theta: int = 16,
        n_gamma: int = 64,
        hidden_dim: int = 64,
        n_integration_steps: int = 20,
    ) -> None:
        super().__init__()
        self.n_delta = n_delta
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        self.n_steps = n_integration_steps

        self.scene_proj = nn.Linear(scene_dim, n_gamma)
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        total_osc = n_delta + n_theta + n_gamma
        self.readout = nn.Sequential(
            nn.Linear(total_osc + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene: ``(B, N, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log probabilities ``(B, 2)``.
        """
        from prinet.core.measurement import kuramoto_order_parameter
        from prinet.core.propagation import (
            DeltaThetaGammaNetwork,
            OscillatorState,
        )

        B = scene.shape[0]
        device = scene.device

        scene_flat = scene.mean(dim=1)
        gamma_init = torch.sigmoid(self.scene_proj(scene_flat))

        order_feats: list[Tensor] = []
        for b in range(B):
            net = DeltaThetaGammaNetwork(
                n_delta=self.n_delta,
                n_theta=self.n_theta,
                n_gamma=self.n_gamma,
                device=device,
            )
            state = net.create_initial_state(seed=42 + b)
            gamma_s = state[2]
            state = (
                state[0],
                state[1],
                OscillatorState(
                    phase=gamma_s.phase,
                    amplitude=gamma_init[b].detach(),
                    frequency=gamma_s.frequency,
                ),
            )
            final, _ = net.integrate(state, self.n_steps, dt=0.01)
            feat = torch.cat([
                final[0].amplitude,
                final[1].amplitude,
                final[2].amplitude,
            ])
            order_feats.append(feat)

        osc_feat = torch.stack(order_feats, dim=0).to(device)
        q_feat = self.query_proj(query)
        combined = torch.cat([osc_feat, q_feat], dim=-1)
        logits = self.readout(combined)
        return F.log_softmax(logits, dim=-1)


# ---- Training & Evaluation Utilities -----------------------------------


@dataclass
class CLEVRNResult:
    """Results for a single (model, n_items) run."""

    model_name: str
    n_items: int
    train_acc: float
    test_acc: float
    n_epochs: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_items": self.n_items,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
        }


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Train a CLEVR-N model and return final training accuracy."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    correct = 0
    total = 0

    for epoch in range(n_epochs):
        correct = 0
        total = 0
        for scenes, queries, labels in train_loader:
            scenes, queries, labels = (
                scenes.to(device),
                queries.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            log_probs = model(scenes, queries)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()

            preds = log_probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

    return correct / max(total, 1)


@torch.no_grad()
def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Evaluate a CLEVR-N model and return accuracy."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for scenes, queries, labels in loader:
        scenes, queries, labels = (
            scenes.to(device),
            queries.to(device),
            labels.to(device),
        )
        log_probs = model(scenes, queries)
        preds = log_probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / max(total, 1)


def run_clevr_n_sweep(
    model_factory: Any,
    model_name: str,
    n_items_list: list[int] | None = None,
    n_train: int = 2000,
    n_test: int = 500,
    n_epochs: int = 30,
    batch_size: int = 64,
    seed: int = SEED,
    device: str = "cpu",
    phase_encode: bool = True,
) -> list[CLEVRNResult]:
    """Run accuracy-vs-N sweep for a model factory.

    Args:
        model_factory: Callable that takes ``(scene_dim, query_dim)``
            keyword args and returns an ``nn.Module``.
        model_name: Name for results.
        n_items_list: List of N values to sweep. Default: [2,4,6,8,10,12].
        n_train: Training samples per N.
        n_test: Test samples per N.
        n_epochs: Training epochs per run.
        batch_size: Batch size.
        seed: Global seed.
        device: Device string.
        phase_encode: Use phase encoding.

    Returns:
        List of ``CLEVRNResult`` for each N.
    """
    if n_items_list is None:
        n_items_list = [2, 4, 6, 8, 10, 12]

    scene_dim = D_PHASE if phase_encode else D_FEAT
    results = []

    for n_items in n_items_list:
        # Generate data
        train_scenes, train_queries, train_labels = make_clevr_n(
            n_items, n_train, seed=seed, phase_encode=phase_encode
        )
        test_scenes, test_queries, test_labels = make_clevr_n(
            n_items, n_test, seed=seed + 10000, phase_encode=phase_encode
        )

        train_ds = CLEVRNDataset(train_scenes, train_queries, train_labels)
        test_ds = CLEVRNDataset(test_scenes, test_queries, test_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Create model
        model = model_factory(scene_dim=scene_dim, query_dim=D_FEAT * 2)

        # Train and evaluate
        train_acc = _train_model(
            model, train_loader, n_epochs, device=device
        )
        test_acc = _eval_model(model, test_loader, device=device)

        results.append(
            CLEVRNResult(
                model_name=model_name,
                n_items=n_items,
                train_acc=train_acc,
                test_acc=test_acc,
                n_epochs=n_epochs,
                seed=seed,
            )
        )
        print(
            f"  {model_name} N={n_items}: "
            f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}"
        )

    return results


def run_all_baselines(
    n_items_list: list[int] | None = None,
    n_epochs: int = 30,
    seed: int = SEED,
    device: str = "cpu",
    save_path: str | Path | None = None,
) -> dict[str, list[CLEVRNResult]]:
    """Run all baselines on CLEVR-N sweep and optionally save results.

    Args:
        n_items_list: N values to sweep.
        n_epochs: Epochs per run.
        seed: Global seed.
        device: Device string.
        save_path: Optional JSON path to save results.

    Returns:
        Dict mapping model name → list of results.
    """
    baselines: dict[str, Any] = {
        "LSTM": lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "Transformer": lambda scene_dim, query_dim: TransformerCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "Hopfield": lambda scene_dim, query_dim: HopfieldCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "ThetaGamma": lambda scene_dim, query_dim: ThetaGammaCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim, n_integration_steps=10
        ),
        "DeltaThetaGamma": lambda scene_dim, query_dim: DeltaThetaGammaCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim, n_integration_steps=10
        ),
    }

    all_results: dict[str, list[CLEVRNResult]] = {}

    for name, factory in baselines.items():
        print(f"\n--- {name} Baseline ---")
        results = run_clevr_n_sweep(
            factory,
            name,
            n_items_list=n_items_list,
            n_epochs=n_epochs,
            seed=seed,
            device=device,
        )
        all_results[name] = results

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            name: [r.to_dict() for r in results]
            for name, results in all_results.items()
        }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return all_results


# ---- M.3: Extended 24-colour palette -----------------------------------

#: 24-colour vocabulary for extended CLEVR-N (M.3).
#: Includes the original 8 colours plus 16 additional perceptually
#: distinct colours for capacity curves up to N=16.
COLORS_24 = [
    # Original 8
    "red", "blue", "green", "yellow", "purple", "cyan", "orange", "gray",
    # Extended 16
    "pink", "lime", "maroon", "navy", "teal", "gold", "violet", "brown",
    "salmon", "turquoise", "indigo", "coral", "olive", "azure", "magenta",
    "silver",
]

#: sRGB values (0–255) for each colour in COLORS_24 (for rendering).
COLORS_24_RGB: dict[str, tuple[int, int, int]] = {
    "red":       (220,  50,  50),
    "blue":      ( 50,  50, 220),
    "green":     ( 50, 180,  50),
    "yellow":    (220, 220,  50),
    "purple":    (128,   0, 128),
    "cyan":      ( 50, 200, 200),
    "orange":    (230, 130,  50),
    "gray":      (128, 128, 128),
    "pink":      (255, 150, 180),
    "lime":      (150, 255,  50),
    "maroon":    (128,   0,   0),
    "navy":      (  0,   0, 128),
    "teal":      (  0, 128, 128),
    "gold":      (212, 175,  55),
    "violet":    (238, 130, 238),
    "brown":     (139,  69,  19),
    "salmon":    (250, 128, 114),
    "turquoise": ( 64, 224, 208),
    "indigo":    ( 75,   0, 130),
    "coral":     (255, 127,  80),
    "olive":     (128, 128,   0),
    "azure":     (240, 255, 255),
    "magenta":   (255,   0, 255),
    "silver":    (192, 192, 192),
}

D_COLOR_24 = len(COLORS_24)


def _make_query_ext(
    pos_ids: torch.Tensor,
    color_ids: torch.Tensor,
    shape_ids: torch.Tensor,
    rng: torch.Generator,
    n_colors: int = D_COLOR_24,
) -> tuple[torch.Tensor, int]:
    """Like _make_query but supports an arbitrary n_colors vocabulary.

    Returns:
        query_vec ``(feat_dim * 2,)`` with feat_dim = n_colors + D_SHAPE + D_POS
        label: 1 if obj_i left of obj_j, else 0.
    """
    feat_dim = n_colors + D_SHAPE + D_POS
    n = pos_ids.shape[0]
    idx = torch.randperm(n, generator=rng)[:2]
    i, j = int(idx[0].item()), int(idx[1].item())

    def _oh(c: int, s: int, p: int) -> torch.Tensor:
        v = torch.zeros(feat_dim)
        v[c] = 1.0
        v[n_colors + s] = 1.0
        v[n_colors + D_SHAPE + p] = 1.0
        return v

    q_i = _oh(int(color_ids[i].item()), int(shape_ids[i].item()), int(pos_ids[i].item()))
    q_j = _oh(int(color_ids[j].item()), int(shape_ids[j].item()), int(pos_ids[j].item()))
    label = 1 if pos_ids[i] < pos_ids[j] else 0
    return torch.cat([q_i, q_j]), label


def make_clevr_n_extended(
    n_items: int,
    n_samples: int = 1000,
    seed: int = SEED,
    device: str | torch.device = "cpu",
) -> TensorDataset:
    """Generate a CLEVR-N dataset using the 24-colour extended palette.

    Identical to :func:`make_clevr_n` but draws colours from
    :data:`COLORS_24` (24 entries) instead of the original 8.

    Args:
        n_items: Number of objects per scene.
        n_samples: Number of scene samples.
        seed: Random seed.
        device: Target device.

    Returns:
        :class:`~torch.utils.data.TensorDataset` of
        ``(scene_enc, query_vec, label)`` triples.

    Example:
        >>> ds = make_clevr_n_extended(n_items=8, n_samples=200, seed=0)
        >>> len(ds)
        200
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    d_feat_24 = D_COLOR_24 + D_SHAPE + D_POS
    scene_encs = torch.zeros(n_samples, n_items, d_feat_24)
    # Query = concatenated one-hot for two objects → 2 × d_feat_24
    query_vecs = torch.zeros(n_samples, d_feat_24 * 2)
    labels = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        colors = torch.randint(0, D_COLOR_24, (n_items,), generator=rng)
        shapes = torch.randint(0, D_SHAPE, (n_items,), generator=rng)
        positions = torch.randperm(N_POSITIONS, generator=rng)[:n_items]

        scene_enc = torch.zeros(n_items, d_feat_24)
        for idx in range(n_items):
            scene_enc[idx, colors[idx]] = 1.0
            scene_enc[idx, D_COLOR_24 + shapes[idx]] = 1.0
            scene_enc[idx, D_COLOR_24 + D_SHAPE + positions[idx]] = 1.0

        query_vec, label = _make_query_ext(positions, colors, shapes, rng, n_colors=D_COLOR_24)

        scene_encs[i] = scene_enc
        query_vecs[i] = query_vec
        labels[i] = label

    dev = torch.device(device)
    return TensorDataset(
        scene_encs.to(dev), query_vecs.to(dev), labels.to(dev)
    )


# ---- M.4: Adversarial similar-colour distractors -----------------------


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB (0-255) to approximate CIELAB (D65 illuminant).

    Uses the standard linearisation + Bradford matrix approximation.
    Accurate enough for perceptual proximity comparisons.
    """
    # Linearise
    r, g, b = (c / 255.0 for c in rgb)

    def _lin(v: float) -> float:
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _lin(r), _lin(g), _lin(b)

    # sRGB → XYZ (D65)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # XYZ → Lab
    xn, yn, zn = 0.9505, 1.0000, 1.0890  # D65 white point

    def _f(t: float) -> float:
        delta = 6.0 / 29.0
        return t ** (1 / 3) if t > delta ** 3 else t / (3 * delta ** 2) + 4.0 / 29.0

    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return L, a, b_


def _delta_e(
    rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]
) -> float:
    """CIE76 colour difference (ΔE) between two sRGB colours."""
    L1, a1, b1 = _rgb_to_lab(rgb1)
    L2, a2, b2 = _rgb_to_lab(rgb2)
    return math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def build_adversarial_colour_pairs(
    palette: dict[str, tuple[int, int, int]] | None = None,
    max_delta_e: float = 25.0,
) -> list[tuple[str, str, float]]:
    """Find visually similar colour pairs (potential distractors).

    Pairs colours whose CIE76 ΔE (perceptual distance) is below
    *max_delta_e*, ordering by proximity (most similar first).

    Args:
        palette: Dict of ``{colour_name: (R, G, B)}``.
            Defaults to :data:`COLORS_24_RGB`.
        max_delta_e: Maximum ΔE to include in output.

    Returns:
        List of ``(colour_a, colour_b, delta_e)`` tuples sorted
        by *delta_e* ascending.

    Example:
        >>> pairs = build_adversarial_colour_pairs(max_delta_e=30.0)
        >>> assert all(de <= 30.0 for _, _, de in pairs)
    """
    if palette is None:
        palette = COLORS_24_RGB
    names = list(palette.keys())
    pairs: list[tuple[str, str, float]] = []
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            de = _delta_e(palette[na], palette[nb])
            if de <= max_delta_e:
                pairs.append((na, nb, de))
    pairs.sort(key=lambda t: t[2])
    return pairs


def make_adversarial_clevr(
    n_items: int,
    n_samples: int = 1000,
    seed: int = SEED,
    max_distractor_delta_e: float = 25.0,
    device: str | torch.device = "cpu",
) -> TensorDataset:
    """Generate adversarial CLEVR-N with similar-colour distractors.

    Each scene contains at least one *adversarial pair*: two objects
    with perceptually similar colours (ΔE ≤ *max_distractor_delta_e*).
    This tests whether oscillatory binding can disambiguate items that
    a pure feature detector would confuse.

    Args:
        n_items: Objects per scene (must be ≥ 2 for adversarial pairing).
        n_samples: Number of scenes.
        seed: Random seed.
        max_distractor_delta_e: Maximum CIE76 ΔE for adversarial pair.
        device: Target device.

    Returns:
        :class:`~torch.utils.data.TensorDataset` of
        ``(scene_enc, query_vec, label)`` triples.  The encoding uses
        the 24-colour palette features (dimension = D_COLOR_24 + D_SHAPE
        + D_POS).

    Example:
        >>> ds = make_adversarial_clevr(n_items=4, n_samples=100, seed=0)
        >>> len(ds)
        100
    """
    adv_pairs = build_adversarial_colour_pairs(max_delta_e=max_distractor_delta_e)
    if not adv_pairs:
        raise ValueError(
            f"No adversarial pairs found at max_delta_e={max_distractor_delta_e}. "
            "Try increasing max_distractor_delta_e."
        )

    rng = torch.Generator()
    rng.manual_seed(seed)
    import random as _random

    _random.seed(seed)

    d_feat_24 = D_COLOR_24 + D_SHAPE + D_POS
    scene_encs = torch.zeros(n_samples, n_items, d_feat_24)
    query_vecs = torch.zeros(n_samples, d_feat_24 * 2)
    labels = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        # Pick an adversarial colour pair for the first two objects
        pair_idx = int(torch.randint(0, min(len(adv_pairs), 10), (1,), generator=rng).item())
        ca_name, cb_name, _ = adv_pairs[pair_idx]
        ca = COLORS_24.index(ca_name)
        cb = COLORS_24.index(cb_name)

        colors = torch.randint(0, D_COLOR_24, (n_items,), generator=rng)
        colors[0] = ca
        colors[1] = cb

        shapes = torch.randint(0, D_SHAPE, (n_items,), generator=rng)
        positions = torch.randperm(N_POSITIONS, generator=rng)[:n_items]

        scene_enc = torch.zeros(n_items, d_feat_24)
        for idx in range(n_items):
            scene_enc[idx, colors[idx]] = 1.0
            scene_enc[idx, D_COLOR_24 + shapes[idx]] = 1.0
            scene_enc[idx, D_COLOR_24 + D_SHAPE + positions[idx]] = 1.0

        query_vec, label = _make_query_ext(positions, colors, shapes, rng, n_colors=D_COLOR_24)

        scene_encs[i] = scene_enc
        query_vecs[i] = query_vec
        labels[i] = label

    dev = torch.device(device)
    return TensorDataset(
        scene_encs.to(dev), query_vecs.to(dev), labels.to(dev)
    )


# ---- CLI Entry Point ---------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLEVR-N Benchmark")
    parser.add_argument(
        "--n-items",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="N values to sweep",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    run_all_baselines(
        n_items_list=args.n_items,
        n_epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        save_path=args.save,
    )
