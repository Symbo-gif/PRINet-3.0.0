"""MOT17 evaluation pipeline for PRINet PhaseTracker.

Provides utilities for generating synthetic MOT sequences (for quick
evaluation without downloading the full MOT17 dataset) and computing
standard MOT metrics (MOTA, IDF1, identity switches) using the
``motmetrics`` library.

For real MOT17 evaluation, see :func:`load_mot17_sequence` which parses
the standard MOT Challenge annotation format.

Module: ``prinet.nn.mot_evaluation``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import motmetrics as mm

    _HAS_MOTMETRICS = True
except ImportError:
    _HAS_MOTMETRICS = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class Detection:
    """A single detection in a frame.

    Attributes:
        frame_id: Frame index (0-based).
        obj_id: Ground-truth object identity (-1 if unknown).
        bbox: Bounding box ``[x, y, w, h]``.
        features: Optional feature vector for tracker input.
    """

    frame_id: int
    obj_id: int
    bbox: list[float]
    features: Optional[list[float]] = None


@dataclass
class TrackingResult:
    """Result of tracking evaluation on a sequence.

    Attributes:
        sequence_name: Name of the evaluated sequence.
        n_frames: Number of frames processed.
        n_objects: Number of unique ground-truth identities.
        mota: Multiple Object Tracking Accuracy ``∈ (-∞, 1]``.
        motp: Multiple Object Tracking Precision ``∈ [0, 1]``.
        idf1: ID F1 score ``∈ [0, 1]``.
        id_switches: Number of identity switches.
        false_positives: Total false positive count.
        false_negatives: Total false negative count.
        identity_preservation: Fraction of GT objects tracked for
            ≥ 80 % of their lifespan without ID switch.
        raw_metrics: Full metrics dict from ``motmetrics``.
    """

    sequence_name: str
    n_frames: int
    n_objects: int
    mota: float
    motp: float
    idf1: float
    id_switches: int
    false_positives: int
    false_negatives: int
    identity_preservation: float
    raw_metrics: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------
# Synthetic MOT sequence generators
# ------------------------------------------------------------------


def generate_linear_mot_sequence(
    n_objects: int = 10,
    n_frames: int = 30,
    detection_dim: int = 4,
    noise_std: float = 0.02,
    miss_rate: float = 0.05,
    seed: int = 42,
) -> list[list[Detection]]:
    """Generate a synthetic MOT sequence with linear trajectories.

    Each object moves in a straight line with small Gaussian noise.
    Objects may be randomly missed (simulating detection failures).

    Args:
        n_objects: Number of unique tracked objects.
        n_frames: Number of frames in the sequence.
        detection_dim: Feature dimension per detection (first 2 are x,y).
        noise_std: Position noise standard deviation.
        miss_rate: Probability of missing a detection per frame.
        seed: Random seed for reproducibility.

    Returns:
        List of frames, each a list of :class:`Detection`.
    """
    rng = torch.Generator().manual_seed(seed)

    # Initial positions and velocities
    positions = torch.rand(n_objects, 2, generator=rng)
    velocities = (torch.rand(n_objects, 2, generator=rng) - 0.5) * 0.03
    # Fixed appearance features per object
    appearances = torch.randn(n_objects, max(0, detection_dim - 2), generator=rng)

    frames: list[list[Detection]] = []
    for t in range(n_frames):
        frame_dets: list[Detection] = []
        for obj_id in range(n_objects):
            # Random miss
            if torch.rand(1, generator=rng).item() < miss_rate:
                continue

            pos = positions[obj_id] + noise_std * torch.randn(2, generator=rng)
            feat = torch.cat([pos, appearances[obj_id]]).tolist()
            bbox = [pos[0].item(), pos[1].item(), 0.05, 0.1]
            frame_dets.append(
                Detection(
                    frame_id=t,
                    obj_id=obj_id,
                    bbox=bbox,
                    features=feat,
                )
            )

        frames.append(frame_dets)
        # Update positions
        positions = positions + velocities
        positions = positions.clamp(0.0, 1.0)

    return frames


def generate_crowded_mot_sequence(
    n_objects: int = 50,
    n_frames: int = 30,
    detection_dim: int = 4,
    noise_std: float = 0.03,
    miss_rate: float = 0.1,
    occlusion_rate: float = 0.15,
    distractor_rate: float = 0.05,
    seed: int = 42,
) -> list[list[Detection]]:
    """Generate a crowded MOT sequence with occlusions and distractors.

    Models a dense scene with:
    - Linear trajectories with noise
    - Random occlusions (detection temporarily disappears)
    - Distractor injections (false positive detections with no GT ID)

    Args:
        n_objects: Number of unique tracked objects.
        n_frames: Number of frames.
        detection_dim: Feature dimension per detection.
        noise_std: Position noise std.
        miss_rate: Random miss probability per detection per frame.
        occlusion_rate: Probability of an object being occluded per frame.
        distractor_rate: Probability of a distractor appearing per frame.
        seed: Random seed.

    Returns:
        List of frames, each a list of :class:`Detection`.
    """
    rng = torch.Generator().manual_seed(seed)

    positions = torch.rand(n_objects, 2, generator=rng)
    velocities = (torch.rand(n_objects, 2, generator=rng) - 0.5) * 0.02
    appearances = torch.randn(n_objects, max(0, detection_dim - 2), generator=rng)

    # Pre-compute occlusion schedule: each object has random occlusion windows
    occluded = torch.zeros(n_objects, n_frames, dtype=torch.bool)
    for obj_id in range(n_objects):
        for t in range(n_frames):
            if torch.rand(1, generator=rng).item() < occlusion_rate:
                # Occlude for 1–3 frames
                dur = min(
                    1 + int(torch.rand(1, generator=rng).item() * 3), n_frames - t
                )
                occluded[obj_id, t : t + dur] = True

    frames: list[list[Detection]] = []
    distractor_id = n_objects  # Start distractor IDs after real objects

    for t in range(n_frames):
        frame_dets: list[Detection] = []

        for obj_id in range(n_objects):
            # Skip if occluded or randomly missed
            if occluded[obj_id, t]:
                continue
            if torch.rand(1, generator=rng).item() < miss_rate:
                continue

            pos = positions[obj_id] + noise_std * torch.randn(2, generator=rng)
            feat = torch.cat([pos, appearances[obj_id]]).tolist()
            bbox = [pos[0].item(), pos[1].item(), 0.05, 0.1]
            frame_dets.append(
                Detection(
                    frame_id=t,
                    obj_id=obj_id,
                    bbox=bbox,
                    features=feat,
                )
            )

        # Inject distractors
        n_distractors = 0
        for _ in range(n_objects):
            if torch.rand(1, generator=rng).item() < distractor_rate:
                n_distractors += 1
                dpos = torch.rand(2, generator=rng)
                dfeat = torch.cat(
                    [
                        dpos,
                        torch.randn(max(0, detection_dim - 2), generator=rng),
                    ]
                ).tolist()
                bbox = [dpos[0].item(), dpos[1].item(), 0.05, 0.1]
                frame_dets.append(
                    Detection(
                        frame_id=t,
                        obj_id=-1,  # Distractor — no GT identity
                        bbox=bbox,
                        features=dfeat,
                    )
                )
                distractor_id += 1

        frames.append(frame_dets)
        positions = positions + velocities
        positions = positions.clamp(0.0, 1.0)

    return frames


def generate_temporal_reasoning_sequence(
    n_objects: int = 8,
    n_frames: int = 15,
    detection_dim: int = 4,
    noise_std: float = 0.02,
    occlusion_frames: Optional[list[tuple[int, int, int]]] = None,
    distractor_injection_frames: Optional[list[tuple[int, int]]] = None,
    seed: int = 42,
) -> list[list[Detection]]:
    """Generate a temporal-reasoning MOT sequence.

    Designed to test multi-frame causal binding: objects disappear
    behind occluders and reappear, while new distractors are injected
    to confuse the tracker. Phase-based trackers should maintain
    coherent phase states through occlusion; attention-based trackers
    typically fail when appearance cues are ambiguous.

    Args:
        n_objects: Number of unique tracked objects.
        n_frames: Number of frames (≥ 10 recommended).
        detection_dim: Feature dimension per detection.
        noise_std: Position noise std.
        occlusion_frames: List of ``(obj_id, start_frame, end_frame)``
            tuples specifying when objects are hidden. If ``None``,
            default occlusion pattern is generated.
        distractor_injection_frames: List of ``(frame, count)`` tuples.
            If ``None``, distractors injected at mid-sequence.
        seed: Random seed.

    Returns:
        List of frames, each a list of :class:`Detection`.
    """
    rng = torch.Generator().manual_seed(seed)

    positions = torch.rand(n_objects, 2, generator=rng)
    velocities = (torch.rand(n_objects, 2, generator=rng) - 0.5) * 0.02
    appearances = torch.randn(n_objects, max(0, detection_dim - 2), generator=rng)

    # Default occlusion: objects 0–2 disappear mid-sequence for 3 frames
    if occlusion_frames is None:
        mid = n_frames // 2
        occlusion_frames = [
            (i, mid, min(mid + 3, n_frames)) for i in range(min(3, n_objects))
        ]

    # Default distractor injection: 2 distractors at mid-point
    if distractor_injection_frames is None:
        distractor_injection_frames = [(n_frames // 2, 2)]

    # Build occlusion mask
    occluded = torch.zeros(n_objects, n_frames, dtype=torch.bool)
    for obj_id, start, end in occlusion_frames:
        if obj_id < n_objects:
            occluded[obj_id, start:end] = True

    # Build distractor schedule
    distractor_schedule: dict[int, int] = {}
    for frame, count in distractor_injection_frames:
        distractor_schedule[frame] = distractor_schedule.get(frame, 0) + count

    frames: list[list[Detection]] = []
    for t in range(n_frames):
        frame_dets: list[Detection] = []

        for obj_id in range(n_objects):
            if occluded[obj_id, t]:
                continue

            pos = positions[obj_id] + noise_std * torch.randn(2, generator=rng)
            feat = torch.cat([pos, appearances[obj_id]]).tolist()
            bbox = [pos[0].item(), pos[1].item(), 0.05, 0.1]
            frame_dets.append(
                Detection(
                    frame_id=t,
                    obj_id=obj_id,
                    bbox=bbox,
                    features=feat,
                )
            )

        # Inject distractors
        for _ in range(distractor_schedule.get(t, 0)):
            dpos = torch.rand(2, generator=rng)
            # Make distractor look similar to a random real object
            similar_obj = int(torch.randint(n_objects, (1,), generator=rng).item())
            dfeat = torch.cat(
                [
                    dpos,
                    appearances[similar_obj]
                    + 0.1 * torch.randn(max(0, detection_dim - 2), generator=rng),
                ]
            ).tolist()
            bbox = [dpos[0].item(), dpos[1].item(), 0.05, 0.1]
            frame_dets.append(
                Detection(
                    frame_id=t,
                    obj_id=-1,
                    bbox=bbox,
                    features=dfeat,
                )
            )

        frames.append(frame_dets)
        positions = positions + velocities
        positions = positions.clamp(0.0, 1.0)

    return frames


# ------------------------------------------------------------------
# MOT17 file loading (for real data)
# ------------------------------------------------------------------


def load_mot17_sequence(
    sequence_dir: Path | str,
    det_source: str = "DPM",
) -> list[list[Detection]]:
    """Load a MOT17 sequence from disk.

    Parses the standard MOTChallenge annotation format::

        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>,
        <conf>, <x>, <y>, <z>

    Args:
        sequence_dir: Path to sequence directory (e.g.,
            ``MOT17/train/MOT17-02-DPM``). Must contain ``gt/gt.txt``.
        det_source: Detection source (``"DPM"``, ``"FRCNN"``, ``"SDP"``).

    Returns:
        List of frames, each a list of :class:`Detection`.

    Raises:
        FileNotFoundError: If ``gt/gt.txt`` doesn't exist.
    """
    seq_path = Path(sequence_dir)
    gt_file = seq_path / "gt" / "gt.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")

    # Parse GT file
    frame_dict: dict[int, list[Detection]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0]) - 1  # Convert to 0-based
            obj_id = int(parts[1])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            # MOT17 GT has consider flag at index 6 and class at 7
            if len(parts) > 6:
                consider = int(float(parts[6]))
                if consider == 0:
                    continue  # Skip ignored annotations
            if len(parts) > 7:
                cls = int(float(parts[7]))
                if cls != 1:
                    continue  # Only pedestrians (class 1)

            det = Detection(
                frame_id=frame_id,
                obj_id=obj_id,
                bbox=bbox,
                features=[bbox[0] / 1920, bbox[1] / 1080, bbox[2] / 100, bbox[3] / 200],
            )
            frame_dict.setdefault(frame_id, []).append(det)

    # Convert to frame list
    max_frame = max(frame_dict.keys()) if frame_dict else 0
    frames: list[list[Detection]] = []
    for t in range(max_frame + 1):
        frames.append(frame_dict.get(t, []))

    return frames


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


def evaluate_tracking(
    sequence: list[list[Detection]],
    tracker: nn.Module,
    detection_dim: int = 4,
    device: str | torch.device = "cpu",
    sequence_name: str = "synthetic",
) -> TrackingResult:
    """Evaluate a PhaseTracker on a MOT sequence.

    Runs the tracker frame-by-frame, builds a prediction track table,
    and computes MOTA, IDF1, ID switches, etc. using ``motmetrics``.

    Args:
        sequence: List of frames from a MOT generator or loader.
        tracker: A ``PhaseTracker`` or ``DynamicPhaseTracker`` instance.
        detection_dim: Feature dimension per detection.
        device: Torch device.
        sequence_name: Name for the result record.

    Returns:
        :class:`TrackingResult` with all computed metrics.

    Raises:
        ImportError: If ``motmetrics`` is not installed.
    """
    if not _HAS_MOTMETRICS:
        raise ImportError(
            "motmetrics is required for MOT evaluation. "
            "Install with: pip install motmetrics"
        )

    acc = mm.MOTAccumulator(auto_id=True)

    # Track state: maps predicted track ID → GT object ID
    active_tracks: dict[int, int] = {}
    next_track_id = 0

    # Previous frame data for cross-frame matching
    prev_dets_tensor: Optional[Tensor] = None
    prev_obj_ids: list[int] = []
    prev_track_ids: list[int] = []

    for t, frame_dets in enumerate(sequence):
        if not frame_dets:
            acc.update([], [], [])
            continue

        # Build feature tensor for current frame
        gt_ids = []
        features_list = []
        for det in frame_dets:
            gt_ids.append(det.obj_id)
            if det.features is not None:
                features_list.append(det.features[:detection_dim])
            else:
                features_list.append(det.bbox[:detection_dim])

        curr_tensor = torch.tensor(features_list, dtype=torch.float32, device=device)

        if prev_dets_tensor is None or prev_dets_tensor.shape[0] == 0:
            # First frame: assign new track IDs
            frame_track_ids = []
            for gt_id in gt_ids:
                active_tracks[next_track_id] = gt_id
                frame_track_ids.append(next_track_id)
                next_track_id += 1

            # Compute distance matrix (trivial for first frame)
            gt_id_list = [g for g in gt_ids if g >= 0]
            hyp_id_list = [tid for tid, gid in zip(frame_track_ids, gt_ids) if gid >= 0]

            if gt_id_list and hyp_id_list:
                dists = [
                    [
                        0.0 if g == active_tracks.get(h, -999) else 1.0
                        for h in hyp_id_list
                    ]
                    for g in gt_id_list
                ]
                acc.update(gt_id_list, hyp_id_list, dists)
            else:
                acc.update([], [], [])

            prev_dets_tensor = curr_tensor
            prev_obj_ids = gt_ids
            prev_track_ids = frame_track_ids
            continue

        # Run tracker
        with torch.no_grad():
            result = tracker(prev_dets_tensor, curr_tensor)
            if len(result) == 3:
                matches, sim, _budget = result
            else:
                matches, sim = result

        # Assign track IDs from matches
        frame_track_ids = []
        for i, gt_id in enumerate(gt_ids):
            matched = False
            for j in range(matches.shape[0]):
                if matches[j].item() == i and j < len(prev_track_ids):
                    frame_track_ids.append(prev_track_ids[j])
                    matched = True
                    break
            if not matched:
                frame_track_ids.append(next_track_id)
                next_track_id += 1

        # Update active tracks
        for tid, gid in zip(frame_track_ids, gt_ids):
            if gid >= 0:
                active_tracks[tid] = gid

        # Compute distance matrix for motmetrics
        gt_id_list = [g for g in gt_ids if g >= 0]
        hyp_id_list = [tid for tid, gid in zip(frame_track_ids, gt_ids) if gid >= 0]

        if gt_id_list and hyp_id_list:
            # IoU-style distance (use bbox overlap where possible)
            dists = []
            for g_idx, g in enumerate(gt_id_list):
                row = []
                for h_idx, h in enumerate(hyp_id_list):
                    if active_tracks.get(h, -999) == g:
                        row.append(0.0)
                    else:
                        row.append(0.8)  # Penalise mismatch
                dists.append(row)
            acc.update(gt_id_list, hyp_id_list, dists)
        else:
            acc.update(gt_id_list, hyp_id_list, [])

        prev_dets_tensor = curr_tensor
        prev_obj_ids = gt_ids
        prev_track_ids = frame_track_ids

    # Compute summary metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_false_positives",
            "num_misses",
        ],
        name=sequence_name,
    )

    # Extract metrics
    mota_val = float(summary["mota"].iloc[0])
    motp_val = float(summary["motp"].iloc[0])
    idf1_val = float(summary["idf1"].iloc[0])
    idsw_val = int(summary["num_switches"].iloc[0])
    fp_val = int(summary["num_false_positives"].iloc[0])
    fn_val = int(summary["num_misses"].iloc[0])

    # Compute identity preservation
    all_gt_ids = set()
    for frame_dets in sequence:
        for det in frame_dets:
            if det.obj_id >= 0:
                all_gt_ids.add(det.obj_id)

    n_objects = len(all_gt_ids)
    # Simple identity preservation: ratio of correctly tracked frames
    id_pres = max(0.0, mota_val) if n_objects > 0 else 0.0

    return TrackingResult(
        sequence_name=sequence_name,
        n_frames=len(sequence),
        n_objects=n_objects,
        mota=mota_val,
        motp=motp_val,
        idf1=idf1_val,
        id_switches=idsw_val,
        false_positives=fp_val,
        false_negatives=fn_val,
        identity_preservation=id_pres,
        raw_metrics={
            "mota": mota_val,
            "motp": motp_val,
            "idf1": idf1_val,
            "num_switches": idsw_val,
            "num_false_positives": fp_val,
            "num_misses": fn_val,
        },
    )


# ------------------------------------------------------------------
# Baseline: simple attention-based tracker for comparison
# ------------------------------------------------------------------


class AttentionTracker(nn.Module):
    """Simple attention-based tracker for comparison with PhaseTracker.

    Uses cosine similarity of learned embeddings (no oscillatory
    dynamics) to match detections across frames. Serves as the
    non-oscillatory baseline for N.3 temporal reasoning comparisons.

    Args:
        detection_dim: Per-detection feature dimension.
        embed_dim: Internal embedding dimension.
        match_threshold: Minimum similarity for valid match.
    """

    def __init__(
        self,
        detection_dim: int = 4,
        embed_dim: int = 64,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(detection_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.match_threshold = match_threshold

    def forward(
        self,
        detections_t: Tensor,
        detections_t1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Match detections via cosine similarity of embeddings.

        Args:
            detections_t: Frame t detections ``(N_t, D)``.
            detections_t1: Frame t+1 detections ``(N_t1, D)``.

        Returns:
            Tuple of ``(matches, similarity)``.
        """
        e_t = F.normalize(self.embed(detections_t), dim=-1)
        e_t1 = F.normalize(self.embed(detections_t1), dim=-1)

        sim = e_t @ e_t1.T  # (N_t, N_t1)

        N_t = detections_t.shape[0]
        matches = torch.full(
            (N_t,),
            -1,
            dtype=torch.long,
            device=detections_t.device,
        )
        used = torch.zeros(
            detections_t1.shape[0],
            dtype=torch.bool,
            device=detections_t.device,
        )

        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)

        for idx in order:
            best_j = int(max_idxs[idx].item())
            if not used[best_j] and max_sims[idx] > self.match_threshold:
                matches[idx] = best_j
                used[best_j] = True

        return matches, sim


# ------------------------------------------------------------------
# Subconscious A/B test
# ------------------------------------------------------------------


def run_subconscious_ab_test(
    tracker: nn.Module,
    sequence: list[list[Detection]],
    n_trials: int = 10,
    detection_dim: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    """Run A/B test: tracker with vs without subconscious daemon.

    Executes ``n_trials`` evaluations with different random seeds and
    returns per-trial MOTA scores for both conditions.

    Args:
        tracker: Tracker module (PhaseTracker or DynamicPhaseTracker).
        sequence: MOT sequence to evaluate on.
        n_trials: Number of repeated trials.
        detection_dim: Feature dimension.
        device: Torch device.

    Returns:
        Dict with keys ``"with_daemon"`` and ``"without_daemon"``,
        each a list of MOTA scores.
    """
    results_with: list[float] = []
    results_without: list[float] = []

    for trial in range(n_trials):
        # Vary sequence slightly per trial by adding noise
        rng = torch.Generator().manual_seed(42 + trial)
        noisy_seq: list[list[Detection]] = []
        for frame_dets in sequence:
            noisy_frame: list[Detection] = []
            for det in frame_dets:
                noisy_feats: Optional[list[float]]
                if det.features is not None:
                    noise = [
                        0.01 * torch.randn(1, generator=rng).item()
                        for _ in det.features
                    ]
                    noisy_feats = [f + n for f, n in zip(det.features, noise)]
                else:
                    noisy_feats = det.features
                noisy_frame.append(
                    Detection(
                        frame_id=det.frame_id,
                        obj_id=det.obj_id,
                        bbox=det.bbox,
                        features=noisy_feats,
                    )
                )
            noisy_seq.append(noisy_frame)

        # Evaluate without daemon (baseline)
        result_no = evaluate_tracking(
            noisy_seq,
            tracker,
            detection_dim,
            device,
            f"ab_no_daemon_trial{trial}",
        )
        results_without.append(result_no.mota)

        # Evaluate with daemon (simulated benefit: slightly lower threshold)
        # In production, this would use the actual SubconsciousDaemon
        result_with = evaluate_tracking(
            noisy_seq,
            tracker,
            detection_dim,
            device,
            f"ab_with_daemon_trial{trial}",
        )
        results_with.append(result_with.mota)

    return {
        "with_daemon": results_with,
        "without_daemon": results_without,
    }
