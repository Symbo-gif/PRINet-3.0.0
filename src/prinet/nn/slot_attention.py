"""Slot Attention baseline for object-centric learning comparison.

Implements the Slot Attention mechanism from Locatello et al. (2020)
adapted for CLEVR-N binding benchmarks and temporal MOT evaluation.
Provides a non-oscillatory baseline for direct comparison with PRINet's
phase-based object individuation.

Year 4 Q1 (T.3): Added :class:`TemporalSlotAttentionMOT` for multi-frame
temporal object tracking, enabling head-to-head comparison with PRINet's
phase-based tracker on identity preservation over 20+ frame sequences.

Reference:
    Locatello, F. et al. (2020). "Object-Centric Learning with Slot Attention."
    NeurIPS 2020. arXiv:2006.15055.

Public API:
    - :class:`SlotAttentionModule` — Core iterative competitive attention.
    - :class:`SlotAttentionCLEVRN` — CLEVR-N adapter (scene + query → logits).
    - :class:`TemporalSlotAttentionMOT` — Multi-frame temporal MOT tracker.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlotAttentionModule(nn.Module):
    """Slot Attention mechanism with iterative competitive binding.

    Implements the core algorithm from Locatello et al. (2020):
    slots compete for input features via softmax-normalized attention,
    with GRU-based iterative refinement.

    Args:
        num_slots: Number of slots (object representations).
        slot_dim: Dimensionality of each slot vector.
        input_dim: Dimensionality of input features.
        num_iterations: Number of iterative refinement steps.
        hidden_dim: Hidden dimension for the slot update MLP.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        num_iterations: int = 3,
        hidden_dim: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.eps = eps

        if hidden_dim is None:
            hidden_dim = max(slot_dim, 128)

        # Learnable slot initialisation (mean + log-variance for sampling)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Layer norms
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Linear projections for attention
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )

        self._scale = slot_dim**-0.5

    def forward(self, inputs: Tensor) -> Tensor:
        """Run Slot Attention on input features.

        Args:
            inputs: ``(B, N, D_in)`` input feature vectors.

        Returns:
            slots: ``(B, K, D_slot)`` refined slot representations.
        """
        B, N, _ = inputs.shape
        K = self.num_slots

        # Normalise inputs
        inputs = self.norm_inputs(inputs)

        # Compute keys and values from inputs (shared across iterations)
        k = self.project_k(inputs)  # (B, N, D_slot)
        v = self.project_v(inputs)  # (B, N, D_slot)

        # Initialise slots via learned Gaussian
        mu = self.slot_mu.expand(B, K, -1)
        sigma = self.slot_log_sigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.num_iterations):
            slots_prev = slots

            # Normalise slots
            slots = self.norm_slots(slots)

            # Compute queries from slots
            q = self.project_q(slots)  # (B, K, D_slot)

            # Attention: (B, N, K)
            attn_logits = torch.einsum("bnd,bkd->bnk", k, q) * self._scale
            # Softmax over slots (competition)
            attn = F.softmax(attn_logits, dim=-1)  # (B, N, K)

            # Weighted mean with normalisation (prevents slot collapse)
            attn_weights = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
            updates = torch.einsum("bnk,bnd->bkd", attn_weights, v)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K, self.slot_dim),
                slots_prev.reshape(B * K, self.slot_dim),
            ).reshape(B, K, self.slot_dim)

            # MLP residual refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SlotAttentionCLEVRN(nn.Module):
    """Slot Attention model adapted for CLEVR-N binding benchmarks.

    Provides a direct comparison baseline against
    :class:`~prinet.nn.hybrid.HybridPRINetV2CLEVRN` on the same
    scene + query → binary-classification task.

    Architecture:
        1. Project scene items → slot input features.
        2. Run SlotAttention to produce K slot representations.
        3. Pool slots, concatenate with query, classify.

    Args:
        scene_dim: Per-item scene feature dimension (default 16 for phase).
        query_dim: Query vector dimension (default 60 = D_FEAT * 2).
        num_slots: Number of attention slots.
        slot_dim: Slot representation dimension.
        d_model: Internal model dimension.
        num_iterations: Slot Attention iterations.
    """

    def __init__(
        self,
        scene_dim: int = 16,
        query_dim: int = 60,
        num_slots: int = 8,
        slot_dim: int = 64,
        d_model: int = 64,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.slot_attention = SlotAttentionModule(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=d_model,
            num_iterations=num_iterations,
        )
        self.query_proj = nn.Linear(query_dim, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(slot_dim + d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass for CLEVR-N classification.

        Args:
            scene: ``(B, D_scene)`` or ``(B, N, D_scene)`` scene features.
            query: ``(B, D_query)`` relational query vector.

        Returns:
            Log probabilities ``(B, 2)``.
        """
        # Encode scene items — handle both 2D and 3D input
        if scene.dim() == 2:
            scene = scene.unsqueeze(1)  # (B, 1, D_scene)
        features = self.scene_proj(scene)  # (B, N, d_model)

        # Slot Attention
        slots = self.slot_attention(features)  # (B, K, slot_dim)

        # Pool slots (mean)
        slot_pool = slots.mean(dim=1)  # (B, slot_dim)

        # Combine with query
        query_enc = self.query_proj(query)  # (B, d_model)
        combined = torch.cat([slot_pool, query_enc], dim=-1)  # (B, slot_dim + d_model)

        # Classify
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)


# =========================================================================
# Year 4 Q1 — T.3: Temporal Slot Attention for Multi-Object Tracking
# =========================================================================


class TemporalSlotAttentionMOT(nn.Module):
    """Temporal Slot Attention for multi-frame object tracking.

    Extends Slot Attention with temporal slot carry-over: slots from
    frame t are used to initialise slots at frame t+1, enabling
    identity preservation across frames. This provides a direct
    baseline against PRINet's :class:`~prinet.nn.hybrid.PhaseTracker`
    for evaluating temporal coherence claims.

    Architecture:
        1. Per-frame: encode detections → SlotAttention features.
        2. Temporal: carry forward slot state via GRU update.
        3. Matching: cosine similarity between consecutive slot states.
        4. Identity assignment via greedy matching.

    Args:
        detection_dim: Per-detection feature dimension.
        num_slots: Number of object slots (max tracked objects).
        slot_dim: Slot representation dimension.
        num_iterations: Slot Attention iterations per frame.
        match_threshold: Minimum similarity for valid identity match.

    Example:
        >>> tracker = TemporalSlotAttentionMOT(detection_dim=4)
        >>> frames = [torch.randn(5, 4) for _ in range(20)]
        >>> result = tracker.track_sequence(frames)
        >>> print(result["identity_preservation"])
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
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.match_threshold = match_threshold

        # Detection encoder
        self.det_encoder = nn.Sequential(
            nn.Linear(detection_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )

        # Slot Attention for per-frame binding
        self.slot_attention = SlotAttentionModule(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            num_iterations=num_iterations,
        )

        # Temporal GRU for slot carry-over between frames
        self.temporal_gru = nn.GRUCell(slot_dim, slot_dim)

        # Layer norm for temporal slot refinement
        self.temporal_norm = nn.LayerNorm(slot_dim)

    def process_frame(
        self,
        detections: Tensor,
        prev_slots: Optional[Tensor] = None,
    ) -> Tensor:
        """Process a single frame and return updated slot states.

        Args:
            detections: Detection features ``(N_det, D)`` or
                ``(1, N_det, D)``.
            prev_slots: Previous frame's slots ``(1, K, slot_dim)``
                or ``None`` for the first frame.

        Returns:
            Updated slots ``(1, K, slot_dim)``.
        """
        if detections.dim() == 2:
            detections = detections.unsqueeze(0)  # (1, N_det, D)

        # Encode detections
        features = self.det_encoder(detections)  # (1, N_det, slot_dim)

        # Run Slot Attention
        new_slots = self.slot_attention(features)  # (1, K, slot_dim)

        # Temporal carry-over via GRU if we have previous slots
        if prev_slots is not None:
            K = self.num_slots
            updated = self.temporal_gru(
                new_slots.reshape(-1, self.slot_dim),
                prev_slots.reshape(-1, self.slot_dim),
            ).reshape(1, K, self.slot_dim)
            new_slots = self.temporal_norm(updated)

        return new_slots  # type: ignore[no-any-return]

    def slot_similarity(
        self,
        slots_a: Tensor,
        slots_b: Tensor,
    ) -> Tensor:
        """Compute cosine similarity between two sets of slots.

        Args:
            slots_a: ``(K, D)`` or ``(1, K, D)``.
            slots_b: ``(K, D)`` or ``(1, K, D)``.

        Returns:
            Similarity matrix ``(K, K)`` in [-1, 1].
        """
        if slots_a.dim() == 3:
            slots_a = slots_a.squeeze(0)
        if slots_b.dim() == 3:
            slots_b = slots_b.squeeze(0)

        a_norm = F.normalize(slots_a, dim=-1)
        b_norm = F.normalize(slots_b, dim=-1)
        return a_norm @ b_norm.T

    def forward(
        self,
        detections_t: Tensor,
        detections_t1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Match detections across two consecutive frames.

        Provides the same interface as
        :meth:`PhaseTracker.forward` so that SA can be used as a
        drop-in comparison model for adversarial and benchmark tools.

        Args:
            detections_t: Frame t detections ``(N_t, D)``.
            detections_t1: Frame t+1 detections ``(N_t1, D)``.

        Returns:
            Tuple of:
                - ``matches``: ``(N_t,)`` assignment indices (``-1`` if
                  unmatched).
                - ``similarity``: Cosine similarity ``(K, K)``.
        """
        slots_t = self.process_frame(detections_t, prev_slots=None)
        slots_t1 = self.process_frame(detections_t1, prev_slots=slots_t)

        sim = self.slot_similarity(slots_t, slots_t1)  # (K, K)
        K = self.num_slots
        matches = torch.full((K,), -1, dtype=torch.long, device=detections_t.device)
        used = torch.zeros(K, dtype=torch.bool, device=detections_t.device)
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)

        for idx in order:
            j = int(max_idxs[idx].item())
            if not used[j] and max_sims[idx] > self.match_threshold:
                matches[idx] = j
                used[j] = True

        return matches, sim

    def track_sequence(
        self,
        frame_detections: list[Tensor],
    ) -> dict[str, Any]:
        """Track objects across a sequence of frames.

        Args:
            frame_detections: List of T tensors, each ``(N_det, D)``
                containing per-frame detection features.

        Returns:
            Dict with:
                - ``"slot_history"``: List of T slot tensors.
                - ``"identity_matches"``: List of T-1 match tensors.
                - ``"identity_preservation"``: Float in [0, 1].
                - ``"per_frame_similarity"``: List of T-1 mean similarities.
        """
        T = len(frame_detections)
        slot_history: list[Tensor] = []
        identity_matches: list[Tensor] = []
        per_frame_sim: list[float] = []

        prev_slots: Optional[Tensor] = None
        total_matches = 0
        total_possible = 0

        with torch.no_grad():
            for t in range(T):
                dets = frame_detections[t]
                slots = self.process_frame(dets, prev_slots)
                slot_history.append(slots.detach().cpu())

                if prev_slots is not None:
                    sim = self.slot_similarity(prev_slots, slots)
                    # Greedy matching
                    K = self.num_slots
                    matches = torch.full((K,), -1, dtype=torch.long)
                    used = torch.zeros(K, dtype=torch.bool)
                    max_sims, max_idxs = sim.max(dim=1)
                    order = max_sims.argsort(descending=True)

                    for idx in order:
                        j = int(max_idxs[idx].item())
                        if not used[j] and max_sims[idx] > self.match_threshold:
                            matches[idx] = j
                            used[j] = True

                    n_matched = int((matches >= 0).sum().item())
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
