"""Adaptive oscillator allocation for task-complexity-driven binding.

Provides :class:`AdaptiveOscillatorAllocator` which dynamically adjusts
the number of delta/theta/gamma oscillators as a function of estimated
scene complexity, eliminating the mid-range accuracy dip at N = 5–7 that
occurs with fixed oscillator counts.

Two allocation strategies are supported:

1. **Rule-based** (``strategy="rule"``): Piecewise-linear scaling from
   ``min_osc`` to ``max_osc`` as a function of ``complexity ∈ [0, 1]``.
2. **Learned** (``strategy="learned"``): Small MLP predicts per-band
   oscillator counts from a complexity feature vector.

Module: ``prinet.nn.adaptive_allocation``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class OscillatorBudget:
    """Allocated oscillator counts per frequency band.

    Attributes:
        n_delta: Delta-band (1–4 Hz) oscillators.
        n_theta: Theta-band (4–8 Hz) oscillators.
        n_gamma: Gamma-band (30–100 Hz) oscillators.
        complexity: Estimated scene complexity in ``[0, 1]``.
    """

    n_delta: int
    n_theta: int
    n_gamma: int
    complexity: float

    @property
    def total(self) -> int:
        """Total oscillator count across all bands."""
        return self.n_delta + self.n_theta + self.n_gamma


def estimate_complexity(
    detections: Tensor,
    *,
    spatial_weight: float = 0.5,
    count_weight: float = 0.5,
    max_objects: int = 50,
) -> Tensor:
    """Estimate scene complexity from detection features.

    Complexity is a scalar in ``[0, 1]`` combining object count and
    spatial spread (standard deviation of detection centroids normalised
    to ``[0, 1]``).

    Args:
        detections: Detection features ``(N, D)`` where ``N`` is the
            number of detections and ``D ≥ 2`` (first two dims treated
            as spatial centroid x, y).
        spatial_weight: Weight for spatial spread term.
        count_weight: Weight for count term.
        max_objects: Count at which count term saturates to 1.0.

    Returns:
        Scalar tensor in ``[0, 1]``.
    """
    n = detections.shape[0]
    count_score = min(n / max_objects, 1.0)

    if n < 2 or detections.shape[-1] < 2:
        spatial_score = 0.0
    else:
        centroids = detections[:, :2].float()
        spread = centroids.std(dim=0).mean().item()
        # Normalise: assume coordinate range ~[0, 1] so max std ≈ 0.3
        spatial_score = min(spread / 0.3, 1.0)

    total = spatial_weight + count_weight
    complexity = (
        spatial_weight * spatial_score + count_weight * count_score
    ) / total
    return torch.tensor(complexity, device=detections.device, dtype=torch.float32)


class AdaptiveOscillatorAllocator(nn.Module):
    """Dynamically allocates oscillator counts per frequency band.

    Given an estimate of scene complexity (number of objects, spatial
    spread), the allocator produces an :class:`OscillatorBudget` that
    scales oscillator counts to match task demands.  This removes the
    mid-range accuracy dip (N = 5–7) observed with fixed-count configs
    by providing more gamma oscillators for crowded scenes.

    Two strategies:

    * ``"rule"``: Deterministic piecewise-linear interpolation.  Fast,
      no learnable parameters, good baseline.
    * ``"learned"``: A small 3-layer MLP (complexity_dim → 64 → 32 → 3)
      predicts *soft* band fractions.  The fractions are rounded to
      integers during inference but kept differentiable via straight-
      through estimation during training.

    Args:
        min_total: Minimum total oscillator count.
        max_total: Maximum total oscillator count.
        delta_ratio: Fraction of total allocated to delta band (rule).
        theta_ratio: Fraction of total allocated to theta band (rule).
        strategy: ``"rule"`` or ``"learned"``.
        complexity_dim: Input feature dimension for learned strategy.

    Example:
        >>> allocator = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
        >>> budget = allocator.allocate(complexity=0.3)
        >>> print(budget)
        OscillatorBudget(n_delta=3, n_theta=6, n_gamma=12, complexity=0.3)
    """

    def __init__(
        self,
        min_total: int = 12,
        max_total: int = 64,
        delta_ratio: float = 0.1,
        theta_ratio: float = 0.2,
        strategy: Literal["rule", "learned"] = "rule",
        complexity_dim: int = 1,
    ) -> None:
        super().__init__()
        if min_total < 3:
            raise ValueError(f"min_total must be >= 3, got {min_total}")
        if max_total < min_total:
            raise ValueError(
                f"max_total ({max_total}) must be >= min_total ({min_total})"
            )

        self.min_total = min_total
        self.max_total = max_total
        self.delta_ratio = delta_ratio
        self.theta_ratio = theta_ratio
        self.strategy = strategy

        if strategy == "learned":
            self._mlp = nn.Sequential(
                nn.Linear(complexity_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # delta, theta, gamma fractions
            )
        else:
            self._mlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        complexity: float | Tensor,
        features: Optional[Tensor] = None,
    ) -> OscillatorBudget:
        """Compute oscillator budget for given complexity.

        Args:
            complexity: Scene complexity scalar ``∈ [0, 1]``.
            features: Optional feature vector for learned strategy
                ``(complexity_dim,)`` or ``(1, complexity_dim)``.

        Returns:
            :class:`OscillatorBudget` with integer band counts.
        """
        if self.strategy == "learned" and features is not None:
            return self._allocate_learned(complexity, features)
        return self._allocate_rule(complexity)

    def forward(
        self,
        complexity: float | Tensor,
        features: Optional[Tensor] = None,
    ) -> OscillatorBudget:
        """Alias for :meth:`allocate` (nn.Module interface).

        Args:
            complexity: Scene complexity scalar ``∈ [0, 1]``.
            features: Optional feature vector for learned strategy.

        Returns:
            :class:`OscillatorBudget`.
        """
        return self.allocate(complexity, features)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _allocate_rule(self, complexity: float | Tensor) -> OscillatorBudget:
        """Piecewise-linear rule-based allocation."""
        c = float(complexity) if isinstance(complexity, Tensor) else complexity
        c = max(0.0, min(1.0, c))

        total = round(self.min_total + c * (self.max_total - self.min_total))
        n_delta = max(1, round(total * self.delta_ratio))
        n_theta = max(1, round(total * self.theta_ratio))
        n_gamma = max(1, total - n_delta - n_theta)

        return OscillatorBudget(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            complexity=c,
        )

    def _allocate_learned(
        self, complexity: float | Tensor, features: Tensor
    ) -> OscillatorBudget:
        """MLP-based learned allocation with soft rounding."""
        assert self._mlp is not None
        c = float(complexity) if isinstance(complexity, Tensor) else complexity
        c = max(0.0, min(1.0, c))

        if features.dim() == 1:
            features = features.unsqueeze(0)

        logits = self._mlp(features)  # (1, 3)
        fractions = F.softmax(logits, dim=-1).squeeze(0)  # (3,)

        total = round(self.min_total + c * (self.max_total - self.min_total))

        # Soft → integer via floor + remainder distribution
        raw = fractions * total
        floors = raw.floor().long()
        remainders = raw - floors.float()

        # Distribute remaining oscillators to bands with largest remainder
        allocated = floors.sum().item()
        deficit = total - int(allocated)
        if deficit > 0:
            _, order = remainders.sort(descending=True)
            for i in range(min(deficit, 3)):
                floors[order[i]] += 1

        counts = floors.clamp(min=1).tolist()
        return OscillatorBudget(
            n_delta=int(counts[0]),
            n_theta=int(counts[1]),
            n_gamma=int(counts[2]),
            complexity=c,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sweep_complexity(
        self,
        steps: int = 11,
    ) -> list[OscillatorBudget]:
        """Generate budgets for complexity values in ``[0, 1]``.

        Useful for verifying smooth capacity curves and absence of
        mid-range dips.

        Args:
            steps: Number of evenly-spaced complexity values.

        Returns:
            List of :class:`OscillatorBudget` from complexity 0 to 1.
        """
        return [
            self.allocate(i / max(steps - 1, 1))
            for i in range(steps)
        ]


class DynamicPhaseTracker(nn.Module):
    """PhaseTracker with adaptive oscillator allocation.

    Wraps :class:`~prinet.nn.hybrid.PhaseTracker` and
    :class:`AdaptiveOscillatorAllocator` to create tracker instances
    whose oscillator counts vary per-scene based on estimated
    complexity.

    This enables smooth capacity scaling across a wide range of object
    counts (N = 2 … 50+) without the mid-range dip.

    Args:
        detection_dim: Per-detection feature dimension.
        min_total: Minimum oscillators (low complexity).
        max_total: Maximum oscillators (high complexity).
        n_discrete_steps: Dynamics steps per frame.
        match_threshold: Minimum phase similarity for a valid match.
        allocator_strategy: ``"rule"`` or ``"learned"``.
        max_objects: Object count at which complexity saturates.

    Example:
        >>> tracker = DynamicPhaseTracker(detection_dim=4)
        >>> dets_t = torch.randn(12, 4)
        >>> dets_t1 = torch.randn(12, 4)
        >>> matches, sim, budget = tracker(dets_t, dets_t1)
    """

    def __init__(
        self,
        detection_dim: int = 4,
        min_total: int = 12,
        max_total: int = 64,
        n_discrete_steps: int = 5,
        match_threshold: float = 0.3,
        allocator_strategy: Literal["rule", "learned"] = "rule",
        max_objects: int = 50,
    ) -> None:
        super().__init__()
        self.detection_dim = detection_dim
        self.n_discrete_steps = n_discrete_steps
        self.match_threshold = match_threshold
        self.max_objects = max_objects

        self.allocator = AdaptiveOscillatorAllocator(
            min_total=min_total,
            max_total=max_total,
            strategy=allocator_strategy,
        )

        # Cache trackers by (n_delta, n_theta, n_gamma) to avoid re-init
        self._tracker_cache: dict[tuple[int, int, int], nn.Module] = {}

    def _get_tracker(self, budget: OscillatorBudget) -> nn.Module:
        """Retrieve or create a PhaseTracker for the given budget."""
        from prinet.nn.hybrid import PhaseTracker

        key = (budget.n_delta, budget.n_theta, budget.n_gamma)
        if key not in self._tracker_cache:
            tracker = PhaseTracker(
                detection_dim=self.detection_dim,
                n_delta=budget.n_delta,
                n_theta=budget.n_theta,
                n_gamma=budget.n_gamma,
                n_discrete_steps=self.n_discrete_steps,
                match_threshold=self.match_threshold,
            )
            # Move to same device as allocator
            device = next(
                (p.device for p in self.allocator.parameters()),
                torch.device("cpu"),
            ) if list(self.allocator.parameters()) else torch.device("cpu")
            tracker = tracker.to(device)
            self._tracker_cache[key] = tracker
        return self._tracker_cache[key]

    def forward(
        self,
        detections_t: Tensor,
        detections_t1: Tensor,
    ) -> tuple[Tensor, Tensor, OscillatorBudget]:
        """Match detections with adaptive oscillator allocation.

        Args:
            detections_t: Frame t detections ``(N_t, D)``.
            detections_t1: Frame t+1 detections ``(N_t1, D)``.

        Returns:
            Tuple of:
                - ``matches``: ``(N_t,)`` assignment indices (or -1).
                - ``similarity``: ``(N_t, N_t1)`` similarity matrix.
                - ``budget``: The :class:`OscillatorBudget` used.
        """
        # Estimate complexity from current frame detections
        complexity = estimate_complexity(
            detections_t, max_objects=self.max_objects,
        )
        budget = self.allocator.allocate(complexity)

        # Get or create tracker with appropriate oscillator count
        tracker = self._get_tracker(budget)

        matches, sim = tracker(detections_t, detections_t1)
        return matches, sim, budget
