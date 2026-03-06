"""Unified temporal training framework for fair PT vs SA comparison.

Year 4 Q1.7: Implements differentiable training loops for both
:class:`~prinet.nn.hybrid.PhaseTracker` and
:class:`~prinet.nn.slot_attention.TemporalSlotAttentionMOT`.

Key fairness constraints:
    - Identical loss function (similarity-based matching)
    - Identical optimizer (Adam) and LR schedule
    - Identical data augmentation pipeline
    - Parameter budget matching (complex params counted as 2× real)
    - Validation-based early stopping with oscillation-aware smoothing
    - 5-seed protocol for statistical reliability

This module also provides the CLEVR-N temporal sequence generator
for producing training/validation/test data with controllable
perturbations (occlusion, swaps, reversals, noise).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prinet.utils.temporal_metrics import (
    TemporalMetrics,
    compute_full_temporal_metrics,
    identity_switches,
    track_fragmentation_rate,
)

# =========================================================================
# 1. Dataset Generator
# =========================================================================


@dataclass
class SequenceData:
    """Container for a generated temporal sequence.

    Attributes:
        frames: List of T tensors, each ``(N, D)`` — per-frame detections.
        positions: ``(T, N, 2)`` — 2D positions for each object/frame.
        velocities: ``(T, N, 2)`` — per-frame velocities.
        identities: ``(T, N)`` — ground-truth identity labels (LongTensor).
        occlusion_mask: ``(T, N)`` — 1=visible, 0=occluded.
        n_objects: Number of ground-truth objects.
        n_frames: Number of frames.
    """

    frames: list[Tensor]
    positions: Tensor
    velocities: Tensor
    identities: Tensor
    occlusion_mask: Tensor
    n_objects: int
    n_frames: int


def generate_temporal_clevr_n(
    n_objects: int = 4,
    n_frames: int = 20,
    det_dim: int = 4,
    velocity_range: tuple[float, float] = (0.5, 2.0),
    occlusion_rate: float = 0.0,
    swap_rate: float = 0.0,
    reversal_count: int = 0,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> SequenceData:
    """Generate a CLEVR-N–style temporal sequence with ground-truth labels.

    Objects move with constant velocity (plus optional perturbations) in
    a 2D feature space. The first 2 dims of each detection encode (x, y)
    position; remaining dims encode object appearance features.

    Args:
        n_objects: Number of objects in the scene.
        n_frames: Number of frames in the sequence.
        det_dim: Per-detection feature dimension (>= 4).
        velocity_range: (min, max) speed for initial velocities.
        occlusion_rate: Fraction of frames with random occlusion per object.
        swap_rate: Fraction of frames where two objects swap appearance.
        reversal_count: Number of velocity reversals injected.
        noise_sigma: Gaussian noise scale on positions (fraction of scale).
        seed: Random seed.

    Returns:
        :class:`SequenceData` with frames, positions, velocities, etc.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Object identities (constant across frames)
    identities = torch.arange(n_objects).unsqueeze(0).expand(n_frames, -1)

    # Initial positions in [0, 10] × [0, 10]
    pos = torch.rand(n_objects, 2, generator=gen) * 10.0

    # Initial velocities
    speed = (
        torch.rand(n_objects, 1, generator=gen)
        * (velocity_range[1] - velocity_range[0])
        + velocity_range[0]
    )
    angle = torch.rand(n_objects, 1, generator=gen) * 2.0 * math.pi
    vel = speed * torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)

    # Appearance features (fixed per object, dims 2..det_dim)
    appearance = torch.randn(n_objects, max(det_dim - 2, 2), generator=gen) * 0.5

    # Pre-compute velocity reversal frames
    reversal_frames: set[int] = set()
    if reversal_count > 0 and n_frames > 2:
        rev_gen = torch.Generator()
        rev_gen.manual_seed(seed + 999)
        rev_idx = torch.randint(1, n_frames - 1, (reversal_count,), generator=rev_gen)
        reversal_frames = set(rev_idx.tolist())

    # Pre-compute occlusion mask (1=visible, 0=occluded)
    occ_mask = torch.ones(n_frames, n_objects)
    if occlusion_rate > 0:
        occ_gen = torch.Generator()
        occ_gen.manual_seed(seed + 1000)
        occ_rand = torch.rand(n_frames, n_objects, generator=occ_gen)
        occ_mask = (occ_rand > occlusion_rate).float()
        # First frame always visible for initialization
        occ_mask[0] = 1.0

    # Pre-compute swap frames
    swap_frames: set[int] = set()
    if swap_rate > 0 and n_frames > 1 and n_objects >= 2:
        swap_gen = torch.Generator()
        swap_gen.manual_seed(seed + 2000)
        n_swaps = max(1, int(n_frames * swap_rate))
        swap_idx = torch.randint(1, n_frames, (n_swaps,), generator=swap_gen)
        swap_frames = set(swap_idx.tolist())

    positions_list: list[Tensor] = []
    velocities_list: list[Tensor] = []
    frames: list[Tensor] = []

    for t in range(n_frames):
        # Apply velocity reversals
        if t in reversal_frames:
            vel = -vel

        # Update position
        if t > 0:
            pos = pos + vel * 0.1  # dt = 0.1 per frame

        # Bounce off boundaries [0, 10]
        for dim_i in range(2):
            low_mask = pos[:, dim_i] < 0
            high_mask = pos[:, dim_i] > 10
            vel[low_mask, dim_i] = vel[low_mask, dim_i].abs()
            vel[high_mask, dim_i] = -vel[high_mask, dim_i].abs()
            pos[:, dim_i] = pos[:, dim_i].clamp(0, 10)

        positions_list.append(pos.clone())
        velocities_list.append(vel.clone())

        # Build detection features: [pos_x, pos_y, appearance...]
        det = torch.cat([pos, appearance[:, : det_dim - 2]], dim=-1)

        # Apply noise
        if noise_sigma > 0:
            noise_gen = torch.Generator()
            noise_gen.manual_seed(seed + 3000 + t)
            noise = torch.randn(n_objects, det_dim, generator=noise_gen) * noise_sigma
            det = det + noise

        # Apply appearance swaps
        if t in swap_frames:
            # Swap appearance features between first two objects
            swap_gen2 = torch.Generator()
            swap_gen2.manual_seed(seed + 4000 + t)
            i, j = 0, 1
            if n_objects > 2:
                perm = torch.randperm(n_objects, generator=swap_gen2)
                i, j = int(perm[0].item()), int(perm[1].item())
            det_copy = det.clone()
            det[i, 2:] = det_copy[j, 2:]
            det[j, 2:] = det_copy[i, 2:]

        # Apply occlusion (zero out features for occluded objects)
        vis = occ_mask[t].unsqueeze(-1)  # (N, 1)
        det = det * vis

        frames.append(det.clone())

    positions = torch.stack(positions_list, dim=0)  # (T, N, 2)
    velocities = torch.stack(velocities_list, dim=0)  # (T, N, 2)

    return SequenceData(
        frames=frames,
        positions=positions,
        velocities=velocities,
        identities=identities.clone(),
        occlusion_mask=occ_mask,
        n_objects=n_objects,
        n_frames=n_frames,
    )


def generate_dataset(
    n_sequences: int,
    n_objects: int = 4,
    n_frames: int = 20,
    det_dim: int = 4,
    occlusion_rate: float = 0.0,
    swap_rate: float = 0.0,
    reversal_count: int = 0,
    noise_sigma: float = 0.0,
    base_seed: int = 42,
) -> list[SequenceData]:
    """Generate a dataset of multiple temporal sequences.

    Args:
        n_sequences: Number of sequences to generate.
        n_objects: Objects per sequence.
        n_frames: Frames per sequence.
        det_dim: Detection feature dimension.
        occlusion_rate: Per-frame occlusion probability.
        swap_rate: Per-frame appearance swap probability.
        reversal_count: Velocity reversals per sequence.
        noise_sigma: Position noise scale.
        base_seed: Base random seed (incremented per sequence).

    Returns:
        List of :class:`SequenceData`.
    """
    return [
        generate_temporal_clevr_n(
            n_objects=n_objects,
            n_frames=n_frames,
            det_dim=det_dim,
            occlusion_rate=occlusion_rate,
            swap_rate=swap_rate,
            reversal_count=reversal_count,
            noise_sigma=noise_sigma,
            seed=base_seed + i,
        )
        for i in range(n_sequences)
    ]


# =========================================================================
# 2. Loss Function
# =========================================================================


def hungarian_similarity_loss(
    similarity: Tensor,
    n_objects: int,
) -> Tensor:
    """Compute assignment loss on the similarity matrix.

    For a correctly-trained tracker, the diagonal of the similarity
    matrix (if objects maintain identity) should be maximized.
    This loss encourages the similarity between ground-truth
    correspondences to be high while suppressing off-diagonal entries.

    Uses a soft cross-entropy approach: treat each row as a
    classification problem where the correct class is the diagonal.

    .. math::

        L = -\\frac{1}{N} \\sum_{i} \\log \\frac{\\exp(s_{i,i})}{\\sum_j \\exp(s_{i,j})}

    where s is the similarity matrix. This is equivalent to
    cross-entropy with identity permutation as the target.

    Args:
        similarity: Similarity matrix ``(N, M)`` from tracker forward pass.
        n_objects: Number of ground-truth objects.

    Returns:
        Scalar loss tensor.
    """
    N = min(similarity.shape[0], similarity.shape[1], n_objects)
    if N == 0:
        return similarity.new_tensor(0.0)

    # Use the top-left (N, N) block
    sim_block = similarity[:N, :N]

    # Scale similarity for better gradient flow (temperature)
    temperature = 0.1
    logits = sim_block / temperature

    # Target: identity permutation (object i should match to object i)
    target = torch.arange(N, device=similarity.device)

    return F.cross_entropy(logits, target)


def temporal_smoothness_loss(
    similarity_sequence: list[Tensor],
) -> Tensor:
    """Penalize jittery similarity patterns across frames.

    Encourages smooth evolution of the similarity matrix over time.

    Args:
        similarity_sequence: List of T-1 similarity matrices.

    Returns:
        Scalar loss tensor.
    """
    if len(similarity_sequence) < 2:
        return (
            similarity_sequence[0].new_tensor(0.0)
            if similarity_sequence
            else torch.tensor(0.0)
        )

    diffs = []
    for t in range(1, len(similarity_sequence)):
        prev = similarity_sequence[t - 1]
        curr = similarity_sequence[t]
        # Compare same-size blocks
        n = min(prev.shape[0], curr.shape[0])
        m = min(prev.shape[1], curr.shape[1])
        diff = (prev[:n, :m] - curr[:n, :m]).pow(2).mean()
        diffs.append(diff)

    return torch.stack(diffs).mean()


# =========================================================================
# 3. Parameter Counting
# =========================================================================


def count_parameters(
    model: nn.Module,
    count_complex_as_double: bool = True,
) -> dict[str, int]:
    """Count model parameters with complex-aware counting.

    Complex-valued parameters count as 2× real parameters for fair
    comparison between PhaseTracker (Kuramoto coupling uses complex
    math) and SlotAttention (all real-valued).

    Args:
        model: PyTorch model.
        count_complex_as_double: If True, complex params count as 2×.

    Returns:
        Dict with ``total``, ``trainable``, ``frozen``, ``complex_adjusted``.
    """
    total = 0
    trainable = 0
    frozen = 0
    complex_adjusted = 0

    for name, p in model.named_parameters():
        numel = p.numel()
        is_complex = p.is_complex()
        real_count = numel * 2 if (is_complex and count_complex_as_double) else numel

        total += numel
        complex_adjusted += real_count
        if p.requires_grad:
            trainable += numel
        else:
            frozen += numel

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "complex_adjusted": complex_adjusted,
    }


# =========================================================================
# 4. Training Dynamics Snapshot
# =========================================================================


@dataclass
class TrainingSnapshot:
    """Captured training state at a specific epoch.

    Attributes:
        epoch: Epoch number.
        train_loss: Training loss.
        val_loss: Validation loss.
        val_ip: Validation identity preservation.
        val_idsw: Validation identity switches.
        gradient_norm: Mean gradient L2 norm.
        param_norm: Mean parameter L2 norm.
        phase_coherence: Mean phase coherence (PT only).
        slot_entropy: Mean slot attention entropy (SA only).
    """

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_ip: float = 0.0
    val_idsw: int = 0
    gradient_norm: float = 0.0
    param_norm: float = 0.0
    phase_coherence: float = 0.0
    slot_entropy: float = 0.0


@dataclass
class TrainingResult:
    """Complete training result with dynamics.

    Attributes:
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
        final_val_ip: Final validation IP.
        best_val_loss: Best validation loss.
        best_epoch: Epoch with best validation loss.
        total_epochs: Total epochs trained.
        wall_time_s: Total wall time in seconds.
        snapshots: List of training snapshots.
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        val_ips: Per-epoch validation IPs.
    """

    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_val_ip: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    wall_time_s: float = 0.0
    snapshots: list[TrainingSnapshot] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_ips: list[float] = field(default_factory=list)


# =========================================================================
# 5. Temporal Trainer
# =========================================================================


class TemporalTrainer:
    """Fair comparison training for PhaseTracker vs SlotAttention.

    Manages the complete training loop with:
        - Identical loss, optimizer, and schedule for both architectures.
        - Validation-based early stopping with moving average smoothing.
        - Training dynamics snapshots at configurable epochs.
        - Gradient clipping for stability.

    Args:
        model: Tracker model (PhaseTracker or TemporalSlotAttentionMOT).
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs).
        smoothing_window: Moving average window for loss smoothing.
        warmup_epochs: Linear LR warmup epochs.
        grad_clip: Maximum gradient norm for clipping.
        snapshot_epochs: Epochs at which to capture training snapshots.
        device: Device string.
        seed: Random seed.

    Example:
        >>> from prinet.nn.hybrid import PhaseTracker
        >>> model = PhaseTracker(detection_dim=4)
        >>> trainer = TemporalTrainer(model, lr=3e-4)
        >>> train_data = generate_dataset(100, n_objects=4, n_frames=20)
        >>> val_data = generate_dataset(20, n_objects=4, n_frames=20, base_seed=9999)
        >>> result = trainer.train(train_data, val_data)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        patience: int = 10,
        smoothing_window: int = 5,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        snapshot_epochs: tuple[int, ...] = (0, 10, 25, 50, 100),
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.smoothing_window = smoothing_window
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.snapshot_epochs = set(snapshot_epochs)
        self.seed = seed
        self.lr = lr

        torch.manual_seed(seed)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

    def _warmup_lr(self, epoch: int) -> None:
        """Apply linear warmup to learning rate."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def _compute_param_norm(self) -> float:
        """Compute total parameter L2 norm."""
        total = 0.0
        for p in self.model.parameters():
            total += p.data.norm(2).item() ** 2
        return math.sqrt(total)

    def _train_step_pt(
        self,
        seq: SequenceData,
    ) -> tuple[float, Tensor]:
        """Training step for PhaseTracker using forward() similarity.

        Uses the differentiable similarity matrix from ``forward()``
        to compute the Hungarian similarity loss. Does NOT use
        ``track_sequence()`` which runs under ``torch.no_grad()``.

        Args:
            seq: Single training sequence.

        Returns:
            Tuple of loss value (float) and loss tensor.
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_transitions = 0
        sim_history: list[Tensor] = []

        for t in range(1, seq.n_frames):
            dets_prev = seq.frames[t - 1].to(self.device)
            dets_curr = seq.frames[t].to(self.device)

            # Skip if either frame has occluded objects producing zero-vectors
            if dets_prev.abs().sum() < 1e-8 or dets_curr.abs().sum() < 1e-8:
                continue

            _, sim = self.model(dets_prev, dets_curr)
            loss = hungarian_similarity_loss(sim, seq.n_objects)
            total_loss = total_loss + loss
            sim_history.append(sim.detach())
            n_transitions += 1

        if n_transitions > 0:
            total_loss = total_loss / n_transitions
            # Add temporal smoothness regularization
            if len(sim_history) >= 2:
                ts_loss = temporal_smoothness_loss(sim_history)
                total_loss = total_loss + 0.1 * ts_loss

        return float(total_loss.item()) if n_transitions > 0 else 0.0, total_loss

    def _train_step_sa(
        self,
        seq: SequenceData,
    ) -> tuple[float, Tensor]:
        """Training step for SlotAttention using process_frame().

        Uses the differentiable slot states from ``process_frame()``
        and ``slot_similarity()`` to compute loss.

        Args:
            seq: Single training sequence.

        Returns:
            Tuple of loss value (float) and loss tensor.
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_transitions = 0
        sim_history: list[Tensor] = []
        prev_slots = None
        dyn_model: Any = self.model

        for t in range(seq.n_frames):
            dets = seq.frames[t].to(self.device)
            slots = dyn_model.process_frame(dets, prev_slots)

            if prev_slots is not None:
                sim = dyn_model.slot_similarity(prev_slots, slots)
                n = min(sim.shape[0], sim.shape[1], seq.n_objects)
                if n > 0:
                    loss = hungarian_similarity_loss(sim, seq.n_objects)
                    total_loss = total_loss + loss
                    sim_history.append(sim.detach())
                    n_transitions += 1

            prev_slots = slots

        if n_transitions > 0:
            total_loss = total_loss / n_transitions
            if len(sim_history) >= 2:
                ts_loss = temporal_smoothness_loss(sim_history)
                total_loss = total_loss + 0.1 * ts_loss

        return float(total_loss.item()) if n_transitions > 0 else 0.0, total_loss

    def _is_phase_tracker(self) -> bool:
        """Check if model is a PhaseTracker or a wrapped PhaseTracker variant.

        Recognizes PhaseTracker directly (has ``det_to_phase`` and
        ``dynamics``) as well as ablation wrappers like
        PhaseTrackerFrozen and PhaseTrackerStatic that delegate via
        ``_inner`` or provide a ``forward()`` returning ``(matches, sim)``.
        """
        m = self.model
        # Direct PhaseTracker
        if hasattr(m, "det_to_phase") and hasattr(m, "dynamics"):
            return True
        # Wrapped variant with _inner PhaseTracker
        if hasattr(m, "_inner") and hasattr(m._inner, "det_to_phase"):
            return True
        # PhaseTrackerStatic (has det_to_phase but no dynamics)
        if hasattr(m, "det_to_phase") and hasattr(m, "frequencies"):
            return True
        return False

    def train_epoch(self, dataset: list[SequenceData]) -> float:
        """Run one training epoch over the dataset.

        Args:
            dataset: List of training sequences.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        is_pt = self._is_phase_tracker()

        for seq in dataset:
            self.optimizer.zero_grad()
            if is_pt:
                loss_val, loss_tensor = self._train_step_pt(seq)
            else:
                loss_val, loss_tensor = self._train_step_sa(seq)

            if loss_val > 0:
                # Only backpropagate if there are trainable parameters
                has_trainable = any(p.requires_grad for p in self.model.parameters())
                if has_trainable and loss_tensor.requires_grad:
                    loss_tensor.backward()  # type: ignore[no-untyped-call]
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

            total_loss += loss_val

        return total_loss / max(len(dataset), 1)

    @torch.no_grad()
    def evaluate(self, dataset: list[SequenceData]) -> dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            dataset: List of evaluation sequences.

        Returns:
            Dict with ``loss``, ``ip``, ``idsw``, ``tfr``.
        """
        self.model.eval()
        is_pt = self._is_phase_tracker()
        total_loss = 0.0
        total_ip = 0.0
        total_idsw = 0
        total_tfr = 0.0
        n_seqs = 0
        dyn_model: Any = self.model

        for seq in dataset:
            frames = [f.to(self.device) for f in seq.frames]

            if is_pt:
                result = dyn_model.track_sequence(frames)
                matches = result["identity_matches"]
                ip = result["identity_preservation"]
            else:
                result = dyn_model.track_sequence(frames)
                matches = result["identity_matches"]
                ip = result["identity_preservation"]

            total_ip += ip
            total_idsw += identity_switches(matches, seq.n_objects)
            total_tfr += track_fragmentation_rate(matches, seq.n_objects)
            n_seqs += 1

            # Compute loss for reporting
            loss = 0.0
            n_trans = 0
            for t in range(1, seq.n_frames):
                dets_prev = frames[t - 1]
                dets_curr = frames[t]
                if dets_prev.abs().sum() < 1e-8 or dets_curr.abs().sum() < 1e-8:
                    continue
                if is_pt:
                    _, sim = dyn_model(dets_prev, dets_curr)
                else:
                    # Re-run to get similarity
                    prev_slots_eval = dyn_model.process_frame(dets_prev)
                    curr_slots_eval = dyn_model.process_frame(
                        dets_curr, prev_slots_eval
                    )
                    sim = dyn_model.slot_similarity(prev_slots_eval, curr_slots_eval)
                loss += float(hungarian_similarity_loss(sim, seq.n_objects).item())
                n_trans += 1
            total_loss += loss / max(n_trans, 1)

        n = max(n_seqs, 1)
        return {
            "loss": total_loss / n,
            "ip": total_ip / n,
            "idsw": total_idsw / n,
            "tfr": total_tfr / n,
        }

    def _capture_snapshot(
        self, epoch: int, train_loss: float, val_metrics: dict[str, float]
    ) -> TrainingSnapshot:
        """Capture a training dynamics snapshot."""
        snap = TrainingSnapshot(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics.get("loss", 0.0),
            val_ip=val_metrics.get("ip", 0.0),
            val_idsw=int(val_metrics.get("idsw", 0)),
            gradient_norm=self._compute_gradient_norm(),
            param_norm=self._compute_param_norm(),
        )

        # Phase-specific: compute coherence
        if self._is_phase_tracker() and hasattr(self.model, "dynamics"):
            try:
                dyn_model: Any = self.model
                test_phase = (
                    torch.rand(1, dyn_model.n_osc, device=self.device) * 2 * math.pi
                )
                test_amp = torch.ones(1, dyn_model.n_osc, device=self.device)
                evolved_phase, _ = dyn_model.evolve(test_phase, test_amp)
                z = torch.exp(1j * evolved_phase.to(torch.complex64))
                coherence = float(z.mean(dim=-1).abs().mean().item())
                snap.phase_coherence = coherence
            except Exception:
                snap.phase_coherence = 0.0

        return snap

    def train(
        self,
        train_data: list[SequenceData],
        val_data: list[SequenceData],
    ) -> TrainingResult:
        """Train the model with early stopping and dynamics capture.

        Args:
            train_data: Training sequences.
            val_data: Validation sequences.

        Returns:
            :class:`TrainingResult` with full training history.
        """
        result = TrainingResult()
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        val_loss_history: list[float] = []

        t0 = time.perf_counter()

        for epoch in range(self.max_epochs):
            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_loss = self.train_epoch(train_data)
            result.train_losses.append(train_loss)

            # Step scheduler (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Validate
            val_metrics = self.evaluate(val_data)
            val_loss = val_metrics["loss"]
            result.val_losses.append(val_loss)
            result.val_ips.append(val_metrics["ip"])
            val_loss_history.append(val_loss)

            # Snapshot
            if epoch in self.snapshot_epochs:
                snap = self._capture_snapshot(epoch, train_loss, val_metrics)
                result.snapshots.append(snap)

            # Early stopping with smoothed loss
            if len(val_loss_history) >= self.smoothing_window:
                smoothed = (
                    sum(val_loss_history[-self.smoothing_window :])
                    / self.smoothing_window
                )
            else:
                smoothed = val_loss

            if smoothed < best_val_loss - 1e-6:
                best_val_loss = smoothed
                best_state = copy.deepcopy(self.model.state_dict())
                result.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        wall_time = time.perf_counter() - t0
        final_val = self.evaluate(val_data)

        result.final_train_loss = (
            result.train_losses[-1] if result.train_losses else 0.0
        )
        result.final_val_loss = final_val["loss"]
        result.final_val_ip = final_val["ip"]
        result.best_val_loss = best_val_loss
        result.total_epochs = len(result.train_losses)
        result.wall_time_s = wall_time

        # Capture final snapshot if not already done
        if result.total_epochs - 1 not in self.snapshot_epochs:
            snap = self._capture_snapshot(
                result.total_epochs - 1, result.final_train_loss, final_val
            )
            result.snapshots.append(snap)

        return result


# =========================================================================
# 6. Multi-Seed Experiment Runner
# =========================================================================


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds.

    Attributes:
        model_name: Name of the model.
        seeds: List of seeds used.
        per_seed: List of TrainingResult per seed.
        mean_ip: Mean IP across seeds.
        std_ip: Std IP across seeds.
        mean_idsw: Mean IDSW across seeds.
        mean_tfr: Mean TFR across seeds.
        mean_epochs: Mean epochs to convergence.
        mean_wall_time: Mean wall time.
    """

    model_name: str = ""
    seeds: list[int] = field(default_factory=list)
    per_seed: list[TrainingResult] = field(default_factory=list)
    mean_ip: float = 0.0
    std_ip: float = 0.0
    mean_idsw: float = 0.0
    mean_tfr: float = 0.0
    mean_epochs: float = 0.0
    mean_wall_time: float = 0.0


def train_multi_seed(
    model_factory: Any,
    model_name: str,
    train_data: list[SequenceData],
    val_data: list[SequenceData],
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
    device: str = "cpu",
    **trainer_kwargs: Any,
) -> MultiSeedResult:
    """Train a model across multiple seeds for statistical reliability.

    Args:
        model_factory: Callable returning a fresh model instance.
        model_name: Name for logging.
        train_data: Training sequences.
        val_data: Validation sequences.
        seeds: Random seeds for independent runs.
        device: Device string.
        **trainer_kwargs: Passed to :class:`TemporalTrainer`.

    Returns:
        :class:`MultiSeedResult` with aggregated statistics.
    """
    result = MultiSeedResult(model_name=model_name, seeds=list(seeds))

    for seed in seeds:
        torch.manual_seed(seed)
        model = model_factory()
        trainer = TemporalTrainer(model, device=device, seed=seed, **trainer_kwargs)
        tr = trainer.train(train_data, val_data)
        result.per_seed.append(tr)

    ips = [r.final_val_ip for r in result.per_seed]
    result.mean_ip = sum(ips) / len(ips)
    result.std_ip = (
        sum((x - result.mean_ip) ** 2 for x in ips) / max(len(ips) - 1, 1)
    ) ** 0.5

    epochs = [r.total_epochs for r in result.per_seed]
    result.mean_epochs = sum(epochs) / len(epochs)

    times = [r.wall_time_s for r in result.per_seed]
    result.mean_wall_time = sum(times) / len(times)

    return result
