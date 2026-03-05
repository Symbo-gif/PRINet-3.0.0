"""Ablation model variants for Q1.7 temporal advantage benchmarks.

Provides structurally ablated versions of PhaseTracker and
TemporalSlotAttentionMOT to isolate which components drive
temporal binding. These preserve the same API as the full models
so they can use the same training infrastructure.

Variants:
    - **PT-frozen**: PhaseTracker with Kuramoto coupling weights frozen.
    - **PT-static**: PhaseTracker with no coupling (independent oscillators).
    - **SA-no-GRU**: TemporalSlotAttentionMOT without GRU carry-over.
    - **SA-frozen**: TemporalSlotAttentionMOT with all weights frozen.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================================================================
# PhaseTracker Ablation Variants
# =========================================================================


class PhaseTrackerFrozen(nn.Module):
    """PhaseTracker with frozen Kuramoto coupling weights.

    Identical to :class:`~prinet.nn.hybrid.PhaseTracker` except the
    :class:`~prinet.core.propagation.DiscreteDeltaThetaGamma` dynamics
    module has ``requires_grad=False``. The detection MLP is still
    trainable.

    Purpose: Tests whether training helps PT exploit oscillatory dynamics,
    or if the untrained dynamics already provide sufficient structure.

    Args:
        detection_dim: Per-detection dimension.
        n_delta: Delta oscillators.
        n_theta: Theta oscillators.
        n_gamma: Gamma oscillators.
        n_discrete_steps: Dynamics steps per frame.
        match_threshold: Matching threshold.
    """

    def __init__(
        self,
        detection_dim: int = 4,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 16,
        n_discrete_steps: int = 5,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        from prinet.nn.hybrid import PhaseTracker

        self._inner = PhaseTracker(
            detection_dim=detection_dim,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_discrete_steps=n_discrete_steps,
            match_threshold=match_threshold,
        )
        # Freeze dynamics (coupling, frequencies, PAC, Stuart-Landau)
        for p in self._inner.dynamics.parameters():
            p.requires_grad = False

    @property
    def n_osc(self) -> int:
        return self._inner.n_osc

    @property
    def match_threshold(self) -> float:
        return self._inner.match_threshold

    def encode(self, detections: Tensor) -> tuple[Tensor, Tensor]:
        return self._inner.encode(detections)

    def evolve(self, phase: Tensor, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        return self._inner.evolve(phase, amplitude)

    def phase_similarity(self, phase_a: Tensor, phase_b: Tensor) -> Tensor:
        return self._inner.phase_similarity(phase_a, phase_b)

    def forward(self, detections_t: Tensor, detections_t1: Tensor) -> tuple[Tensor, Tensor]:
        return self._inner(detections_t, detections_t1)

    def track_sequence(self, frame_detections: list[Tensor]) -> dict:
        return self._inner.track_sequence(frame_detections)


class PhaseTrackerStatic(nn.Module):
    """PhaseTracker with no coupling (independent oscillators).

    Replaces the DiscreteDeltaThetaGamma dynamics with a simple
    phase advance using fixed frequencies — no Kuramoto coupling
    between oscillators. This tests whether coupling matters or
    if independent phase evolution is sufficient.

    Args:
        detection_dim: Per-detection dimension.
        n_delta: Delta oscillators.
        n_theta: Theta oscillators.
        n_gamma: Gamma oscillators.
        n_discrete_steps: Steps per frame.
        match_threshold: Matching threshold.
    """

    _EPS = 1e-6

    def __init__(
        self,
        detection_dim: int = 4,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 16,
        n_discrete_steps: int = 5,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_osc = n_delta + n_theta + n_gamma
        self._n_discrete_steps = n_discrete_steps
        self.match_threshold = match_threshold
        self._n_delta = n_delta
        self._n_theta = n_theta
        self._n_gamma = n_gamma

        # Same detection encoder as PhaseTracker
        self.det_to_phase = nn.Sequential(
            nn.Linear(detection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_osc),
        )
        self.det_to_amp = nn.Sequential(
            nn.Linear(detection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_osc),
            nn.Softplus(),
        )

        # Fixed frequencies (no coupling) — not learnable
        freqs = []
        freqs.extend([2.0] * n_delta)
        freqs.extend([6.0] * n_theta)
        freqs.extend([40.0] * n_gamma)
        self.register_buffer(
            "frequencies", torch.tensor(freqs)
        )

    def encode(self, detections: Tensor) -> tuple[Tensor, Tensor]:
        phase_raw = self.det_to_phase(detections)
        phase = phase_raw % (2.0 * math.pi)
        amp = self.det_to_amp(detections)
        return phase, amp

    def evolve(self, phase: Tensor, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        """Simple phase advance with no coupling."""
        dt = 0.01
        for _ in range(self._n_discrete_steps):
            phase = phase + 2.0 * math.pi * self.frequencies * dt
            phase = phase % (2.0 * math.pi)
        return phase, amplitude

    def phase_similarity(self, phase_a: Tensor, phase_b: Tensor) -> Tensor:
        z_a = torch.exp(1j * phase_a.to(torch.complex64))
        z_b = torch.exp(1j * phase_b.to(torch.complex64))
        z_a_norm = z_a / (z_a.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + self._EPS)
        z_b_norm = z_b / (z_b.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + self._EPS)
        sim = (z_a_norm.unsqueeze(1) * z_b_norm.conj().unsqueeze(0)).sum(dim=-1).real.float()
        return sim

    def forward(self, detections_t: Tensor, detections_t1: Tensor) -> tuple[Tensor, Tensor]:
        phase_t, amp_t = self.encode(detections_t)
        phase_t1, amp_t1 = self.encode(detections_t1)
        phase_t_evolved, _ = self.evolve(phase_t, amp_t)
        sim = self.phase_similarity(phase_t_evolved, phase_t1)

        N_t = detections_t.shape[0]
        matches = torch.full((N_t,), -1, dtype=torch.long, device=detections_t.device)
        used = torch.zeros(detections_t1.shape[0], dtype=torch.bool, device=detections_t.device)
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)
        for idx in order:
            best_j = max_idxs[idx].item()
            if not used[best_j] and max_sims[idx] > self.match_threshold:
                matches[idx] = best_j
                used[best_j] = True
        return matches, sim

    def track_sequence(self, frame_detections: list[Tensor]) -> dict:
        """Track via independent phase evolution (no coupling)."""
        T = len(frame_detections)
        phase_history: list[Tensor] = []
        identity_matches: list[Tensor] = []
        per_frame_sim: list[float] = []
        total_matches = 0
        total_possible = 0

        with torch.no_grad():
            for t in range(T):
                dets = frame_detections[t]
                phase_t, amp_t = self.encode(dets)

                if t == 0:
                    phase_history.append(phase_t.detach().cpu())
                    continue

                prev_phase = phase_history[-1].to(dets.device)
                prev_amp = torch.ones_like(prev_phase)
                evolved_phase, _ = self.evolve(prev_phase, prev_amp)
                sim = self.phase_similarity(evolved_phase, phase_t)

                N_prev = evolved_phase.shape[0]
                N_curr = phase_t.shape[0]
                N_match = min(N_prev, N_curr)
                matches = torch.full((N_prev,), -1, dtype=torch.long, device=dets.device)
                used = torch.zeros(N_curr, dtype=torch.bool, device=dets.device)
                max_sims, max_idxs = sim.max(dim=1)
                order = max_sims.argsort(descending=True)
                for idx in order:
                    best_j = max_idxs[idx].item()
                    if best_j < N_curr and not used[best_j] and max_sims[idx] > self.match_threshold:
                        matches[idx] = best_j
                        used[best_j] = True

                n_matched = int((matches >= 0).sum().item())
                identity_matches.append(matches.cpu())
                per_frame_sim.append(float(max_sims.mean().item()))
                total_matches += n_matched
                total_possible += N_match
                phase_history.append(phase_t.detach().cpu())

        preservation = total_matches / max(total_possible, 1)
        return {
            "phase_history": phase_history,
            "identity_matches": identity_matches,
            "identity_preservation": preservation,
            "per_frame_similarity": per_frame_sim,
            "per_frame_phase_correlation": [],
        }


# =========================================================================
# SlotAttention Ablation Variants
# =========================================================================


class SlotAttentionNoGRU(nn.Module):
    """TemporalSlotAttentionMOT without GRU carry-over.

    Removes the temporal GRU so slots are re-initialized from scratch
    each frame. Tests whether temporal recurrence is necessary for
    identity preservation or if per-frame slot attention is sufficient.

    Args:
        detection_dim: Per-detection dimension.
        num_slots: Number of object slots.
        slot_dim: Slot dimension.
        num_iterations: Slot attention iterations.
        match_threshold: Matching threshold.
    """

    def __init__(
        self,
        detection_dim: int = 4,
        num_slots: int = 8,
        slot_dim: int = 64,
        num_iterations: int = 3,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        from prinet.nn.slot_attention import SlotAttentionModule

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.match_threshold = match_threshold

        self.det_encoder = nn.Sequential(
            nn.Linear(detection_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )

        self.slot_attention = SlotAttentionModule(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            num_iterations=num_iterations,
        )

        # NO GRU, NO temporal_norm

    def process_frame(
        self, detections: Tensor, prev_slots: Optional[Tensor] = None
    ) -> Tensor:
        """Process frame without temporal carry-over."""
        if detections.dim() == 2:
            detections = detections.unsqueeze(0)
        features = self.det_encoder(detections)
        # Ignore prev_slots — always fresh
        new_slots = self.slot_attention(features)
        return new_slots

    def slot_similarity(self, slots_a: Tensor, slots_b: Tensor) -> Tensor:
        if slots_a.dim() == 3:
            slots_a = slots_a.squeeze(0)
        if slots_b.dim() == 3:
            slots_b = slots_b.squeeze(0)
        a_norm = F.normalize(slots_a, dim=-1)
        b_norm = F.normalize(slots_b, dim=-1)
        return a_norm @ b_norm.T

    def track_sequence(self, frame_detections: list[Tensor]) -> dict:
        T = len(frame_detections)
        slot_history: list[Tensor] = []
        identity_matches: list[Tensor] = []
        per_frame_sim: list[float] = []
        total_matches = 0
        total_possible = 0

        with torch.no_grad():
            prev_slots = None
            for t in range(T):
                dets = frame_detections[t]
                slots = self.process_frame(dets, None)  # No carry-over
                slot_history.append(slots.detach().cpu())

                if prev_slots is not None:
                    sim = self.slot_similarity(prev_slots, slots)
                    K = self.num_slots
                    matches = torch.full((K,), -1, dtype=torch.long)
                    used = torch.zeros(K, dtype=torch.bool)
                    max_sims, max_idxs = sim.max(dim=1)
                    order = max_sims.argsort(descending=True)
                    for idx in order:
                        j = max_idxs[idx].item()
                        if not used[j] and max_sims[idx] > self.match_threshold:
                            matches[idx] = j
                            used[j] = True
                    n_matched = (matches >= 0).sum().item()
                    identity_matches.append(matches)
                    per_frame_sim.append(float(max_sims.mean().item()))
                    total_matches += n_matched
                    total_possible += K
                prev_slots = slots

        preservation = total_matches / max(total_possible, 1)
        return {
            "slot_history": slot_history,
            "identity_matches": identity_matches,
            "identity_preservation": preservation,
            "per_frame_similarity": per_frame_sim,
        }

    def forward(self, detections_t: Tensor, detections_t1: Tensor) -> tuple[Tensor, Tensor]:
        """Process two consecutive frames for training."""
        slots_t = self.process_frame(detections_t)
        slots_t1 = self.process_frame(detections_t1, None)
        sim = self.slot_similarity(slots_t, slots_t1)
        K = self.num_slots
        matches = torch.full((K,), -1, dtype=torch.long, device=detections_t.device)
        used = torch.zeros(K, dtype=torch.bool, device=detections_t.device)
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)
        for idx in order:
            j = max_idxs[idx].item()
            if not used[j] and max_sims[idx] > self.match_threshold:
                matches[idx] = j
                used[j] = True
        return matches, sim


class SlotAttentionFrozen(nn.Module):
    """TemporalSlotAttentionMOT with all parameters frozen.

    Exactly mirrors the full TemporalSlotAttentionMOT but with
    ``requires_grad=False`` on all parameters. No training occurs.

    Purpose: Provides the untrained SA baseline for comparison with
    PT-frozen, ensuring symmetry in ablation design.

    Args:
        detection_dim: Per-detection dimension.
        num_slots: Number of object slots.
        slot_dim: Slot dimension.
        num_iterations: Slot attention iterations.
        match_threshold: Matching threshold.
    """

    def __init__(
        self,
        detection_dim: int = 4,
        num_slots: int = 8,
        slot_dim: int = 64,
        num_iterations: int = 3,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        from prinet.nn.slot_attention import TemporalSlotAttentionMOT

        self._inner = TemporalSlotAttentionMOT(
            detection_dim=detection_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            match_threshold=match_threshold,
        )
        # Freeze everything
        for p in self._inner.parameters():
            p.requires_grad = False

    @property
    def num_slots(self) -> int:
        return self._inner.num_slots

    @property
    def slot_dim(self) -> int:
        return self._inner.slot_dim

    @property
    def match_threshold(self) -> float:
        return self._inner.match_threshold

    def process_frame(self, detections: Tensor, prev_slots: Optional[Tensor] = None) -> Tensor:
        return self._inner.process_frame(detections, prev_slots)

    def slot_similarity(self, slots_a: Tensor, slots_b: Tensor) -> Tensor:
        return self._inner.slot_similarity(slots_a, slots_b)

    def forward(self, detections_t: Tensor, detections_t1: Tensor) -> tuple[Tensor, Tensor]:
        """Process two consecutive frames."""
        prev_slots = self._inner.process_frame(detections_t)
        curr_slots = self._inner.process_frame(detections_t1, prev_slots)
        sim = self._inner.slot_similarity(prev_slots, curr_slots)
        K = self._inner.num_slots
        matches = torch.full((K,), -1, dtype=torch.long, device=detections_t.device)
        used = torch.zeros(K, dtype=torch.bool, device=detections_t.device)
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)
        for idx in order:
            j = max_idxs[idx].item()
            if not used[j] and max_sims[idx] > self.match_threshold:
                matches[idx] = j
                used[j] = True
        return matches, sim

    def track_sequence(self, frame_detections: list[Tensor]) -> dict:
        return self._inner.track_sequence(frame_detections)


# =========================================================================
# Factory Functions
# =========================================================================


def create_ablation_tracker(
    variant: str,
    detection_dim: int = 4,
    **kwargs: Any,
) -> nn.Module:
    """Create an ablation tracker variant by name.

    Args:
        variant: One of ``"pt_full"``, ``"pt_frozen"``, ``"pt_static"``,
            ``"sa_full"``, ``"sa_no_gru"``, ``"sa_frozen"``.
        detection_dim: Per-detection dimension.
        **kwargs: Additional constructor kwargs.

    Returns:
        Tracker module.

    Raises:
        ValueError: If variant is unknown.
    """
    from typing import Any as _Any

    if variant == "pt_full":
        from prinet.nn.hybrid import PhaseTracker
        return PhaseTracker(detection_dim=detection_dim, **kwargs)
    elif variant == "pt_frozen":
        return PhaseTrackerFrozen(detection_dim=detection_dim, **kwargs)
    elif variant == "pt_static":
        return PhaseTrackerStatic(detection_dim=detection_dim, **kwargs)
    elif variant == "sa_full":
        from prinet.nn.slot_attention import TemporalSlotAttentionMOT
        return TemporalSlotAttentionMOT(detection_dim=detection_dim, **kwargs)
    elif variant == "sa_no_gru":
        return SlotAttentionNoGRU(detection_dim=detection_dim, **kwargs)
    elif variant == "sa_frozen":
        return SlotAttentionFrozen(detection_dim=detection_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Choose from: pt_full, pt_frozen, pt_static, sa_full, sa_no_gru, sa_frozen"
        )
