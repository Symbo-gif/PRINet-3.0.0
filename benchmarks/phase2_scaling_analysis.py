#!/usr/bin/env python
"""Phase 2: Deepening Existing Experiments — Scaling Analysis.

Implements key scaling experiments to identify distinct advantage
regimes for PhaseTracker vs SlotAttention:

    2.1  Object count scaling (N=5,8,10,15,20)
    2.2  Sequence length scaling (T=50,100,200,500,1000)
    2.3  Occlusion recovery dynamics (clear -> occluded -> clear)
    2.4  Velocity stress test (1x,2x,5x,10x speed multipliers)
    2.5  Fine-grained noise sweep (13 sigma levels + exponential fit)
    2.6  Chimera N-sweep (128,256,512,1024 oscillators)

Each experiment uses 7 seeds per condition and produces JSON artefacts
with bootstrap 95% CIs, Cliff's delta, and Welch's t-test statistics.

Usage:
    python benchmarks/phase2_scaling_analysis.py --all
    python benchmarks/phase2_scaling_analysis.py --object-scaling
    python benchmarks/phase2_scaling_analysis.py --sequence-scaling
    python benchmarks/phase2_scaling_analysis.py --occlusion-recovery
    python benchmarks/phase2_scaling_analysis.py --velocity-stress
    python benchmarks/phase2_scaling_analysis.py --noise-sweep
    python benchmarks/phase2_scaling_analysis.py --chimera-nsweep

Hardware: RTX 4060 8GB VRAM. Uses gradient checkpointing for T>=500.

Reference:
    PRINet Paper Roadmap, Phase 2 (Deepening Existing Experiments).

Authors:
    Michael Maillet
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

FORCE_RERUN = os.environ.get("FORCE_RERUN", "").lower() in ("1", "true", "yes")

# 7-seed protocol (aligned with roadmap)
SEEDS_7 = (42, 123, 456, 789, 1024, 2048, 3072)

# Training defaults
MAX_EPOCHS = 10
PATIENCE = 4
WARMUP = 1
LR = 3e-4
DET_DIM = 4
TRAIN_SEQS = 30
VAL_SEQS = 10
TEST_SEQS = 20

# Default model configs
PT_KWARGS: dict[str, Any] = dict(
    n_delta=4, n_theta=8, n_gamma=16,
    n_discrete_steps=5, match_threshold=0.1,
)
SA_KWARGS: dict[str, Any] = dict(
    num_slots=6, slot_dim=64, num_iterations=3,
    match_threshold=0.1,
)


# ---------------------------------------------------------------------------
# Utilities (mirrors y4q1_9_benchmarks.py patterns)
# ---------------------------------------------------------------------------


def _p(*args: Any, **kwargs: Any) -> None:
    """ASCII-safe print with flush."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict[str, Any]) -> bool:
    """Save JSON artefact. Returns False if skipped.

    Args:
        name: Artefact name (without prefix/extension).
        data: Dictionary to serialise.

    Returns:
        True if written, False if skipped.
    """
    path = RESULTS_DIR / f"phase2_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (exists, set FORCE_RERUN=1 to overwrite)")
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _p(f"  -> {path.name}")
    return True


def _cleanup() -> None:
    """Aggressive memory cleanup between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_pt(seed: int = 42) -> nn.Module:
    """Build PhaseTracker with given seed.

    Args:
        seed: Random seed for initialization.

    Returns:
        PhaseTracker module.
    """
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(detection_dim=DET_DIM, **PT_KWARGS)


def _build_sa(seed: int = 42) -> nn.Module:
    """Build TemporalSlotAttentionMOT with given seed.

    Args:
        seed: Random seed for initialization.

    Returns:
        TemporalSlotAttentionMOT module.
    """
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(detection_dim=DET_DIM, **SA_KWARGS)


def _gen(
    n: int,
    n_objects: int = 4,
    n_frames: int = 20,
    base_seed: int = 42,
    velocity_range: tuple[float, float] | None = None,
    **kw: Any,
) -> list:
    """Generate temporal CLEVR-N sequences.

    Args:
        n: Number of sequences.
        n_objects: Objects per sequence.
        n_frames: Frames per sequence.
        base_seed: Base random seed.
        velocity_range: Optional (min, max) speed override.
        **kw: Additional kwargs for generate_temporal_clevr_n
            (noise_sigma, occlusion_rate, swap_rate, reversal_count).

    Returns:
        List of SequenceData.
    """
    from prinet.utils.temporal_training import generate_temporal_clevr_n
    seqs = []
    for i in range(n):
        gen_kw: dict[str, Any] = dict(
            n_objects=n_objects, n_frames=n_frames,
            det_dim=DET_DIM, seed=base_seed + i,
        )
        if velocity_range is not None:
            gen_kw["velocity_range"] = velocity_range
        gen_kw.update(kw)
        seqs.append(generate_temporal_clevr_n(**gen_kw))
    return seqs


def _eval_ip(
    model: nn.Module, dataset: list, device: str = DEVICE
) -> list[float]:
    """Evaluate identity preservation per sequence.

    Args:
        model: Tracker model (PT or SA).
        dataset: List of SequenceData.
        device: Torch device.

    Returns:
        List of IP values, one per sequence.
    """
    model.eval()
    model = model.to(device)
    ips: list[float] = []
    with torch.no_grad():
        for seq in dataset:
            frames = [f.to(device) for f in seq.frames]
            res = model.track_sequence(frames)
            ips.append(res["identity_preservation"])
    return ips


def _bootstrap_ci(
    values: list[float], n_boot: int = 5000, alpha: float = 0.05
) -> dict[str, float]:
    """Bootstrap confidence interval.

    Args:
        values: Observed values.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level for CI.

    Returns:
        Dict with mean, std, ci_low, ci_high.
    """
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return {
            "mean": float(arr.mean()), "ci_low": float(arr[0]),
            "ci_high": float(arr[0]), "std": 0.0,
        }
    rng = np.random.default_rng(42)
    boot_means = sorted([
        float(rng.choice(arr, size=n, replace=True).mean())
        for _ in range(n_boot)
    ])
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1.0 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_low": boot_means[lo_idx],
        "ci_high": boot_means[min(hi_idx, n_boot - 1)],
    }


def _cliffs_delta(group_a: list[float], group_b: list[float]) -> float:
    """Compute Cliff's delta between two groups.

    Args:
        group_a: First group.
        group_b: Second group.

    Returns:
        Cliff's delta in [-1, +1].
    """
    if not group_a or not group_b:
        return 0.0
    n_a, n_b = len(group_a), len(group_b)
    more = sum(1 for a in group_a for b in group_b if a > b)
    less = sum(1 for a in group_a for b in group_b if a < b)
    return (more - less) / (n_a * n_b)


def _welch_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's t-test between two groups.

    Args:
        a: First group.
        b: Second group.

    Returns:
        Dict with t_stat, p_value, cohens_d, means, and sample sizes.
    """
    from scipy import stats
    a_arr, b_arr = np.array(a), np.array(b)
    n_a, n_b = len(a_arr), len(b_arr)
    mean_a, mean_b = float(a_arr.mean()), float(b_arr.mean())
    var_a = float(a_arr.var(ddof=1)) if n_a > 1 else 0.0
    var_b = float(b_arr.var(ddof=1)) if n_b > 1 else 0.0
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))
    if se < 1e-15:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0,
                "mean_a": mean_a, "mean_b": mean_b, "n_a": n_a, "n_b": n_b}
    t_stat = float((mean_a - mean_b) / se)
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2 / max(n_a - 1, 1)
              + (var_b / n_b) ** 2 / max(n_b - 1, 1))
    df = df_num / max(df_den, 1e-15)
    p_val = float(2 * stats.t.sf(abs(t_stat), df=df))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    cohens_d = float((mean_a - mean_b) / math.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    return {"t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
            "mean_a": mean_a, "mean_b": mean_b, "n_a": n_a, "n_b": n_b}


def _train_model(
    model: nn.Module,
    seed: int,
    n_objects: int = 4,
    n_frames: int = 20,
    **gen_kw: Any,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a model from scratch with given parameters.

    Args:
        model: Model to train (PT or SA).
        seed: Random seed.
        n_objects: Objects per training sequence.
        n_frames: Frames per training sequence.
        **gen_kw: Additional kwargs for sequence generation.

    Returns:
        Tuple of (trained_model, training_info_dict).
    """
    from prinet.utils.temporal_training import (
        generate_dataset,
        hungarian_similarity_loss,
    )
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    train_data = generate_dataset(
        TRAIN_SEQS, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, base_seed=seed, **gen_kw
    )
    val_data = generate_dataset(
        VAL_SEQS, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, base_seed=seed + 50000, **gen_kw
    )

    best_loss = float("inf")
    best_state: Optional[dict[str, Any]] = None
    patience_cnt = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ips: list[float] = []
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_losses: list[float] = []
        for seq in train_data:
            frames = [f.to(DEVICE) for f in seq.frames]
            total_loss = torch.tensor(0.0, device=DEVICE)
            for t in range(len(frames) - 1):
                _, sim = model(frames[t], frames[t + 1])
                loss = hungarian_similarity_loss(sim, n_objects)
                total_loss = total_loss + loss
            total_loss = total_loss / max(len(frames) - 1, 1)
            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_losses.append(float(total_loss.item()))
        train_losses.append(sum(epoch_losses) / max(len(epoch_losses), 1))

        model.eval()
        v_losses: list[float] = []
        v_ips_epoch: list[float] = []
        with torch.no_grad():
            for seq in val_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                res = model.track_sequence(frames)
                v_ips_epoch.append(res["identity_preservation"])
                t_loss = torch.tensor(0.0, device=DEVICE)
                for t in range(len(frames) - 1):
                    _, sim = model(frames[t], frames[t + 1])
                    t_loss = t_loss + hungarian_similarity_loss(sim, n_objects)
                v_losses.append(float(t_loss.item() / max(len(frames) - 1, 1)))
        val_loss = sum(v_losses) / max(len(v_losses), 1)
        val_ip = sum(v_ips_epoch) / max(len(v_ips_epoch), 1)
        val_losses.append(val_loss)
        val_ips.append(val_ip)

        if val_loss < best_loss and epoch >= WARMUP:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE and epoch >= WARMUP:
            break

    wall_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, {
        "epochs": len(train_losses),
        "wall_time_s": wall_time,
        "final_val_ip": val_ips[-1] if val_ips else 0.0,
        "best_val_loss": float(best_loss),
    }


# =========================================================================
# 2.1 — Object Count Scaling
# =========================================================================


def experiment_2_1_object_scaling() -> dict[str, Any]:
    """Experiment 2.1: Object count scaling (N=5,8,10,15,20).

    Identifies the binding capacity limit for PT vs SA by scaling
    the number of objects while keeping all other parameters at baseline.
    7 seeds per condition. Tracks IP, and phase coherence.

    Hypothesis: PT degrades gracefully (linear coherence loss) while
    SA hits a hard wall when GRU hidden state saturates. The crossover
    point N* is a publishable result.

    Returns:
        Dict with per-condition scaling results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.1: Object Count Scaling")
    _p("=" * 60)

    n_objects_list = [5, 8, 10, 15, 20]
    n_frames = 20
    sweep: list[dict[str, Any]] = []

    for n_obj in n_objects_list:
        _p(f"\n  --- N_objects = {n_obj} ---")
        pt_ips_all: list[float] = []
        sa_ips_all: list[float] = []
        pt_coherences: list[float] = []
        per_seed_data: list[dict[str, Any]] = []

        for si, seed in enumerate(SEEDS_7):
            _p(f"    Seed {seed} ({si+1}/{len(SEEDS_7)})")

            # Build and train for this object count
            # Scale slots for SA to match object count
            sa_kw = dict(SA_KWARGS)
            sa_kw["num_slots"] = max(n_obj + 2, 6)  # slots >= n_objects + 2

            pt = _build_pt(seed)
            sa = _build_sa(seed)
            if n_obj > 6:
                # Override SA slots for larger object counts
                from prinet.nn.slot_attention import TemporalSlotAttentionMOT
                torch.manual_seed(seed)
                sa = TemporalSlotAttentionMOT(
                    detection_dim=DET_DIM, **sa_kw
                )

            pt, pt_info = _train_model(pt, seed, n_objects=n_obj, n_frames=n_frames)
            _cleanup()
            sa, sa_info = _train_model(sa, seed, n_objects=n_obj, n_frames=n_frames)
            _cleanup()

            test_data = _gen(TEST_SEQS, n_objects=n_obj, n_frames=n_frames,
                             base_seed=seed + 70000)
            pt_ips = _eval_ip(pt, test_data)
            sa_ips = _eval_ip(sa, test_data)

            # Phase coherence for PT
            pt.eval().to(DEVICE)
            coherences = []
            with torch.no_grad():
                for seq in test_data[:5]:  # Subsample for speed
                    frames = [f.to(DEVICE) for f in seq.frames]
                    res = pt.track_sequence(frames)
                    rhos = res.get("per_frame_phase_correlation", [])
                    if rhos:
                        coherences.append(float(np.mean(rhos)))
            pt_coherence = float(np.mean(coherences)) if coherences else 0.0

            pt_mean = float(np.mean(pt_ips))
            sa_mean = float(np.mean(sa_ips))
            pt_ips_all.append(pt_mean)
            sa_ips_all.append(sa_mean)
            pt_coherences.append(pt_coherence)

            per_seed_data.append({
                "seed": seed,
                "pt_ip": pt_mean,
                "sa_ip": sa_mean,
                "pt_coherence": pt_coherence,
                "pt_epochs": pt_info["epochs"],
                "sa_epochs": sa_info["epochs"],
            })

            del pt, sa, test_data
            _cleanup()

        # Aggregate
        stats = _welch_t(pt_ips_all, sa_ips_all)
        cd = _cliffs_delta(pt_ips_all, sa_ips_all)

        entry = {
            "n_objects": n_obj,
            "pt_stats": _bootstrap_ci(pt_ips_all),
            "sa_stats": _bootstrap_ci(sa_ips_all),
            "pt_coherence_stats": _bootstrap_ci(pt_coherences),
            "welch_t": stats,
            "cliffs_delta": cd,
            "per_seed": per_seed_data,
        }
        sweep.append(entry)

        _p(f"    PT: {stats['mean_a']:.4f}, SA: {stats['mean_b']:.4f}, "
           f"p={stats['p_value']:.4f}, d_cliff={cd:.3f}")

    # Find crossover point N*
    crossover_n = None
    for i, entry in enumerate(sweep):
        if entry["welch_t"]["mean_a"] > entry["welch_t"]["mean_b"] + 0.01:
            crossover_n = entry["n_objects"]
            break

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_1_object_scaling",
        "description": "Object count scaling: IP vs N_objects for PT and SA",
        "n_objects_tested": n_objects_list,
        "n_seeds": len(SEEDS_7),
        "seeds": list(SEEDS_7),
        "sweep": sweep,
        "crossover_n_star": crossover_n,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("object_scaling", result)
    return result


# =========================================================================
# 2.2 — Sequence Length Scaling (Headline Result)
# =========================================================================


def experiment_2_2_sequence_scaling() -> dict[str, Any]:
    """Experiment 2.2: Sequence length scaling (T=50,100,200,500,1000).

    Tests whether PT's oscillatory coherence provides temporal advantage
    at long sequences. This is the potential headline result.

    Hypothesis: PT maintains IP at T=1000 while SA's GRU accumulates
    drift, showing that oscillatory binding is the correct inductive
    bias for ultra-long temporal tracking.

    For T>=500, uses chunked processing (200-frame windows with
    carry-forward state) if VRAM is tight.

    Returns:
        Dict with sequence length scaling results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.2: Sequence Length Scaling (Headline Result)")
    _p("=" * 60)

    frame_counts = [50, 100, 200, 500, 1000]
    n_objects = 5
    sweep: list[dict[str, Any]] = []

    for n_frames in frame_counts:
        _p(f"\n  --- T = {n_frames} frames ---")

        # For T > 200, train on T=100 (feasible) and evaluate on T=n_frames
        train_frames = min(n_frames, 100)

        pt_ips_all: list[float] = []
        sa_ips_all: list[float] = []
        per_seed_data: list[dict[str, Any]] = []

        for si, seed in enumerate(SEEDS_7):
            _p(f"    Seed {seed} ({si+1}/{len(SEEDS_7)})")

            pt = _build_pt(seed)
            sa = _build_sa(seed)

            # Train on shorter sequences (temporal generalization test)
            pt, pt_info = _train_model(
                pt, seed, n_objects=n_objects, n_frames=train_frames
            )
            _cleanup()
            sa, sa_info = _train_model(
                sa, seed, n_objects=n_objects, n_frames=train_frames
            )
            _cleanup()

            # Generate test data at target sequence length
            # For very long sequences, reduce test set size
            n_test = TEST_SEQS if n_frames <= 200 else max(5, TEST_SEQS // 4)
            test_data = _gen(
                n_test, n_objects=n_objects, n_frames=n_frames,
                base_seed=seed + 70000,
            )

            pt_ips = _eval_ip(pt, test_data)
            sa_ips = _eval_ip(sa, test_data)

            pt_mean = float(np.mean(pt_ips))
            sa_mean = float(np.mean(sa_ips))
            pt_ips_all.append(pt_mean)
            sa_ips_all.append(sa_mean)

            per_seed_data.append({
                "seed": seed,
                "pt_ip": pt_mean,
                "sa_ip": sa_mean,
                "train_frames": train_frames,
                "test_frames": n_frames,
                "pt_epochs": pt_info["epochs"],
                "sa_epochs": sa_info["epochs"],
            })

            del pt, sa, test_data
            _cleanup()

        stats = _welch_t(pt_ips_all, sa_ips_all)
        cd = _cliffs_delta(pt_ips_all, sa_ips_all)

        entry = {
            "n_frames": n_frames,
            "train_frames": train_frames,
            "pt_stats": _bootstrap_ci(pt_ips_all),
            "sa_stats": _bootstrap_ci(sa_ips_all),
            "welch_t": stats,
            "cliffs_delta": cd,
            "per_seed": per_seed_data,
        }
        sweep.append(entry)

        pt_mean_agg = stats["mean_a"]
        sa_mean_agg = stats["mean_b"]
        advantage = pt_mean_agg - sa_mean_agg
        _p(f"    PT: {pt_mean_agg:.4f}, SA: {sa_mean_agg:.4f}, "
           f"advantage={advantage:+.4f}, p={stats['p_value']:.4f}")

    # Determine headline result
    last = sweep[-1] if sweep else {}
    headline = {
        "pt_ip_at_T1000": last.get("pt_stats", {}).get("mean"),
        "sa_ip_at_T1000": last.get("sa_stats", {}).get("mean"),
        "advantage_at_T1000": (
            (last.get("pt_stats", {}).get("mean", 0) -
             last.get("sa_stats", {}).get("mean", 0))
            if last else None
        ),
        "significant": last.get("welch_t", {}).get("p_value", 1.0) < 0.05,
    }

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_2_sequence_scaling",
        "description": "Sequence length scaling: IP vs T for PT and SA",
        "frame_counts": frame_counts,
        "n_objects": n_objects,
        "n_seeds": len(SEEDS_7),
        "sweep": sweep,
        "headline_result": headline,
        "go_no_go": {
            "criterion": "PT IP at T=1000 > SA IP at T=1000 by >= 2%",
            "pass": (headline.get("advantage_at_T1000") or 0) >= 0.02,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("sequence_scaling", result)
    return result


# =========================================================================
# 2.3 — Occlusion Recovery Dynamics
# =========================================================================


def experiment_2_3_occlusion_recovery() -> dict[str, Any]:
    """Experiment 2.3: Occlusion recovery dynamics.

    Protocol: 20 frames clear -> 20 frames 60% occlusion -> 20 frames clear.
    Measures: (a) IP during occlusion, (b) phase coherence recovery time
    tau_r after occlusion ends, (c) final IP vs no-occlusion baseline.

    This tests a genuinely novel property: whether oscillatory binding
    is self-healing after disruption.

    Returns:
        Dict with recovery dynamics results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.3: Occlusion Recovery Dynamics")
    _p("=" * 60)

    n_objects = 5
    n_frames = 60  # 20 clear + 20 occluded + 20 recovery
    occlusion_start = 20
    occlusion_end = 40

    sweep: list[dict[str, Any]] = []

    for si, seed in enumerate(SEEDS_7):
        _p(f"  Seed {seed} ({si+1}/{len(SEEDS_7)})")

        pt = _build_pt(seed)
        sa = _build_sa(seed)

        # Train on clean data
        pt, _ = _train_model(pt, seed, n_objects=n_objects, n_frames=20)
        _cleanup()
        sa, _ = _train_model(sa, seed, n_objects=n_objects, n_frames=20)
        _cleanup()

        pt.eval().to(DEVICE)
        sa.eval().to(DEVICE)

        # Generate custom occlusion protocol sequences
        from prinet.utils.temporal_training import generate_temporal_clevr_n
        seq = generate_temporal_clevr_n(
            n_objects=n_objects, n_frames=n_frames, det_dim=DET_DIM,
            seed=seed + 90000,
        )

        # Apply occlusion mask manually: frames 20-39 get 60% occlusion
        occ_gen = torch.Generator()
        occ_gen.manual_seed(seed + 91000)
        for t in range(occlusion_start, occlusion_end):
            mask = (torch.rand(n_objects, generator=occ_gen) > 0.6).float()
            seq.frames[t] = seq.frames[t] * mask.unsqueeze(-1)

        frames = [f.to(DEVICE) for f in seq.frames]

        # Evaluate PT with per-frame coherence
        with torch.no_grad():
            pt_result = pt.track_sequence(frames)
            sa_result = sa.track_sequence(frames)

        # Per-phase coherence for PT
        pt_rhos = pt_result.get("per_frame_phase_correlation", [])
        pt_sims = pt_result.get("per_frame_similarity", [])
        sa_sims = sa_result.get("per_frame_similarity", [])

        # Compute recovery time tau_r for PT
        # tau_r = frames after occlusion ends for coherence to reach
        # 90% of pre-occlusion level
        pre_occ = pt_rhos[:occlusion_start - 1] if len(pt_rhos) >= occlusion_start else []
        pre_occ_mean = float(np.mean(pre_occ)) if pre_occ else 1.0
        target_90 = 0.9 * pre_occ_mean
        tau_r = None
        if len(pt_rhos) > occlusion_end:
            for dt, rho in enumerate(pt_rhos[occlusion_end - 1:]):
                if rho >= target_90:
                    tau_r = dt
                    break

        # Compute per-phase IPs (pre/during/post occlusion)
        pt_matches = pt_result.get("identity_matches", [])
        sa_matches = sa_result.get("identity_matches", [])

        def _phase_ip(matches: list, start: int, end: int) -> float:
            """Compute IP for a sub-range of frames."""
            total_m = 0
            total_p = 0
            for t in range(max(start - 1, 0), min(end - 1, len(matches))):
                m = matches[t]
                n_m = int((m >= 0).sum().item()) if hasattr(m, 'sum') else 0
                total_m += n_m
                total_p += len(m) if hasattr(m, '__len__') else 0
            return total_m / max(total_p, 1)

        seed_data = {
            "seed": seed,
            "pt_ip_total": pt_result["identity_preservation"],
            "sa_ip_total": sa_result["identity_preservation"],
            "pt_ip_pre_occ": _phase_ip(pt_matches, 0, occlusion_start),
            "pt_ip_during_occ": _phase_ip(pt_matches, occlusion_start, occlusion_end),
            "pt_ip_post_occ": _phase_ip(pt_matches, occlusion_end, n_frames),
            "sa_ip_pre_occ": _phase_ip(sa_matches, 0, occlusion_start),
            "sa_ip_during_occ": _phase_ip(sa_matches, occlusion_start, occlusion_end),
            "sa_ip_post_occ": _phase_ip(sa_matches, occlusion_end, n_frames),
            "pt_tau_r": tau_r,
            "pt_pre_occ_coherence": pre_occ_mean,
            "pt_per_frame_coherence": pt_rhos,
            "pt_per_frame_similarity": pt_sims,
            "sa_per_frame_similarity": sa_sims,
        }
        sweep.append(seed_data)

        _p(f"    PT IP: total={seed_data['pt_ip_total']:.4f}, "
           f"pre={seed_data['pt_ip_pre_occ']:.4f}, "
           f"during={seed_data['pt_ip_during_occ']:.4f}, "
           f"post={seed_data['pt_ip_post_occ']:.4f}, tau_r={tau_r}")
        _p(f"    SA IP: total={seed_data['sa_ip_total']:.4f}")

        del pt, sa
        _cleanup()

    # Aggregate
    tau_rs = [s["pt_tau_r"] for s in sweep if s["pt_tau_r"] is not None]
    pt_post_ips = [s["pt_ip_post_occ"] for s in sweep]
    sa_post_ips = [s["sa_ip_post_occ"] for s in sweep]

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_3_occlusion_recovery",
        "description": "Occlusion recovery dynamics: 20 clear + 20 occluded + 20 clear",
        "protocol": {
            "clear_frames": [0, occlusion_start],
            "occlusion_frames": [occlusion_start, occlusion_end],
            "recovery_frames": [occlusion_end, n_frames],
            "occlusion_rate": 0.6,
        },
        "n_objects": n_objects,
        "n_seeds": len(SEEDS_7),
        "sweep": sweep,
        "aggregate": {
            "pt_tau_r_mean": float(np.mean(tau_rs)) if tau_rs else None,
            "pt_tau_r_std": float(np.std(tau_rs, ddof=1)) if len(tau_rs) > 1 else None,
            "n_recovered": len(tau_rs),
            "n_total": len(sweep),
            "pt_post_occ_stats": _bootstrap_ci(pt_post_ips),
            "sa_post_occ_stats": _bootstrap_ci(sa_post_ips),
        },
        "go_no_go": {
            "criterion": "tau_r finite and < 20 frames",
            "pass": len(tau_rs) > 0 and (float(np.mean(tau_rs)) if tau_rs else 999) < 20,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("occlusion_recovery", result)
    return result


# =========================================================================
# 2.4 — Velocity Stress Test
# =========================================================================


def experiment_2_4_velocity_stress() -> dict[str, Any]:
    """Experiment 2.4: Velocity stress test (1x, 2x, 5x, 10x).

    Tests whether PT's temporal coupling provides implicit velocity
    estimation. Models trained at 1x speed, evaluated at higher speeds.

    Hypothesis: PT is more velocity-invariant due to phase dynamics.

    Returns:
        Dict with velocity stress test results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.4: Velocity Stress Test")
    _p("=" * 60)

    speed_multipliers = [1.0, 2.0, 5.0, 10.0]
    n_objects = 5
    n_frames = 50
    sweep: list[dict[str, Any]] = []

    # Train models at base velocity (1x), then evaluate at different speeds
    trained_models: dict[int, tuple] = {}  # seed -> (pt_state, sa_state)

    for si, seed in enumerate(SEEDS_7):
        _p(f"  Training seed {seed} ({si+1}/{len(SEEDS_7)}) at 1x speed...")
        pt = _build_pt(seed)
        pt, _ = _train_model(pt, seed, n_objects=n_objects, n_frames=n_frames)
        pt_state = {k: v.cpu() for k, v in pt.state_dict().items()}
        del pt
        _cleanup()

        sa = _build_sa(seed)
        sa, _ = _train_model(sa, seed, n_objects=n_objects, n_frames=n_frames)
        sa_state = {k: v.cpu() for k, v in sa.state_dict().items()}
        del sa
        _cleanup()

        trained_models[seed] = (pt_state, sa_state)

    for speed in speed_multipliers:
        _p(f"\n  --- Speed = {speed}x ---")
        pt_ips_all: list[float] = []
        sa_ips_all: list[float] = []

        for seed in SEEDS_7:
            pt_state, sa_state = trained_models[seed]
            pt = _build_pt(seed)
            pt.load_state_dict(pt_state)
            pt.eval().to(DEVICE)
            sa = _build_sa(seed)
            sa.load_state_dict(sa_state)
            sa.eval().to(DEVICE)

            # Generate test data with adjusted velocity range
            base_vel = (0.5, 2.0)
            adjusted_vel = (base_vel[0] * speed, base_vel[1] * speed)
            test_data = _gen(
                TEST_SEQS, n_objects=n_objects, n_frames=n_frames,
                base_seed=seed + 70000,
                velocity_range=adjusted_vel,
            )

            pt_ips = _eval_ip(pt, test_data)
            sa_ips = _eval_ip(sa, test_data)
            pt_ips_all.append(float(np.mean(pt_ips)))
            sa_ips_all.append(float(np.mean(sa_ips)))

            del pt, sa, test_data
            _cleanup()

        stats = _welch_t(pt_ips_all, sa_ips_all)
        cd = _cliffs_delta(pt_ips_all, sa_ips_all)

        entry = {
            "speed_multiplier": speed,
            "pt_stats": _bootstrap_ci(pt_ips_all),
            "sa_stats": _bootstrap_ci(sa_ips_all),
            "welch_t": stats,
            "cliffs_delta": cd,
        }
        sweep.append(entry)
        _p(f"    PT: {stats['mean_a']:.4f}, SA: {stats['mean_b']:.4f}, "
           f"p={stats['p_value']:.4f}")

    del trained_models
    _cleanup()

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_4_velocity_stress",
        "description": "Velocity stress test: PT vs SA at increasing speeds",
        "speed_multipliers": speed_multipliers,
        "n_objects": n_objects,
        "n_frames": n_frames,
        "n_seeds": len(SEEDS_7),
        "sweep": sweep,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("velocity_stress", result)
    return result


# =========================================================================
# 2.5 — Fine-Grained Noise Sweep with Exponential Fit
# =========================================================================


def experiment_2_5_noise_sweep() -> dict[str, Any]:
    """Experiment 2.5: Fine-grained noise sweep with exponential decay fit.

    Sigma in {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    0.8, 0.9, 1.0}. Fits IP(sigma) = IP_0 * exp(-sigma / sigma_c)
    to extract characteristic noise scale sigma_c for both PT and SA.

    Returns:
        Dict with noise sweep and fitted degradation models.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.5: Fine-Grained Noise Sweep")
    _p("=" * 60)

    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_objects = 5
    n_frames = 50
    sweep: list[dict[str, Any]] = []

    # Train at noise_sigma=0.0, evaluate across sigma levels
    trained_models: dict[int, tuple] = {}

    for si, seed in enumerate(SEEDS_7):
        _p(f"  Training seed {seed} ({si+1}/{len(SEEDS_7)}) at sigma=0.0...")
        pt = _build_pt(seed)
        pt, _ = _train_model(pt, seed, n_objects=n_objects, n_frames=n_frames)
        pt_state = {k: v.cpu() for k, v in pt.state_dict().items()}
        del pt
        _cleanup()

        sa = _build_sa(seed)
        sa, _ = _train_model(sa, seed, n_objects=n_objects, n_frames=n_frames)
        sa_state = {k: v.cpu() for k, v in sa.state_dict().items()}
        del sa
        _cleanup()

        trained_models[seed] = (pt_state, sa_state)

    for sigma in sigmas:
        _p(f"\n  --- sigma = {sigma} ---")
        pt_ips_all: list[float] = []
        sa_ips_all: list[float] = []

        for seed in SEEDS_7:
            pt_state, sa_state = trained_models[seed]
            pt = _build_pt(seed)
            pt.load_state_dict(pt_state)
            pt.eval().to(DEVICE)
            sa = _build_sa(seed)
            sa.load_state_dict(sa_state)
            sa.eval().to(DEVICE)

            test_data = _gen(
                TEST_SEQS, n_objects=n_objects, n_frames=n_frames,
                base_seed=seed + 80000, noise_sigma=sigma,
            )

            pt_ips = _eval_ip(pt, test_data)
            sa_ips = _eval_ip(sa, test_data)
            pt_ips_all.append(float(np.mean(pt_ips)))
            sa_ips_all.append(float(np.mean(sa_ips)))

            del pt, sa, test_data
            _cleanup()

        stats = _welch_t(pt_ips_all, sa_ips_all)
        cd = _cliffs_delta(pt_ips_all, sa_ips_all)

        entry = {
            "sigma": sigma,
            "pt_stats": _bootstrap_ci(pt_ips_all),
            "sa_stats": _bootstrap_ci(sa_ips_all),
            "welch_t": stats,
            "cliffs_delta": cd,
            "pt_raw": pt_ips_all,
            "sa_raw": sa_ips_all,
        }
        sweep.append(entry)
        _p(f"    PT: {stats['mean_a']:.4f}, SA: {stats['mean_b']:.4f}")

    del trained_models
    _cleanup()

    # Fit exponential decay: IP(sigma) = IP_0 * exp(-sigma / sigma_c)
    def _fit_exponential(
        sigmas_arr: np.ndarray, means: np.ndarray
    ) -> dict[str, float]:
        """Fit exponential decay model to IP vs sigma data.

        Args:
            sigmas_arr: Noise levels.
            means: Mean IP at each noise level.

        Returns:
            Dict with IP_0, sigma_c, and R-squared.
        """
        from scipy.optimize import curve_fit

        def exp_decay(x: np.ndarray, ip0: float, sigma_c: float) -> np.ndarray:
            return ip0 * np.exp(-x / max(sigma_c, 1e-10))

        try:
            valid = means > 0.01  # Only fit to non-zero points
            popt, _ = curve_fit(
                exp_decay, sigmas_arr[valid], means[valid],
                p0=[1.0, 0.5], bounds=([0.0, 0.01], [2.0, 10.0]),
                maxfev=5000,
            )
            ip0, sigma_c = popt
            predicted = exp_decay(sigmas_arr[valid], ip0, sigma_c)
            ss_res = np.sum((means[valid] - predicted) ** 2)
            ss_tot = np.sum((means[valid] - means[valid].mean()) ** 2)
            r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)
            return {"IP_0": float(ip0), "sigma_c": float(sigma_c),
                    "R_squared": float(r_squared)}
        except Exception as e:
            return {"IP_0": float("nan"), "sigma_c": float("nan"),
                    "R_squared": float("nan"), "error": str(e)}

    sigma_arr = np.array(sigmas)
    pt_means = np.array([e["pt_stats"]["mean"] for e in sweep])
    sa_means = np.array([e["sa_stats"]["mean"] for e in sweep])

    pt_fit = _fit_exponential(sigma_arr, pt_means)
    sa_fit = _fit_exponential(sigma_arr, sa_means)

    _p(f"\n  Exponential fit (PT): IP_0={pt_fit['IP_0']:.4f}, "
       f"sigma_c={pt_fit['sigma_c']:.4f}, R^2={pt_fit['R_squared']:.4f}")
    _p(f"  Exponential fit (SA): IP_0={sa_fit['IP_0']:.4f}, "
       f"sigma_c={sa_fit['sigma_c']:.4f}, R^2={sa_fit['R_squared']:.4f}")

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_5_noise_sweep",
        "description": "Fine-grained noise sweep with exponential decay fit",
        "sigmas": sigmas,
        "n_seeds": len(SEEDS_7),
        "sweep": sweep,
        "exponential_fit": {
            "model": "IP(sigma) = IP_0 * exp(-sigma / sigma_c)",
            "phase_tracker": pt_fit,
            "slot_attention": sa_fit,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("noise_sweep", result)
    return result


# =========================================================================
# 2.6 — Chimera N-Sweep
# =========================================================================


def experiment_2_6_chimera_nsweep() -> dict[str, Any]:
    """Experiment 2.6: Chimera N-sweep (128, 256, 512, 1024 oscillators).

    Establishes scaling behavior of chimera states with system size.
    If BC stabilizes with N, chimera is a genuine dynamical regime.

    Returns:
        Dict with chimera N-sweep results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 2.6: Chimera N-Sweep")
    _p("=" * 60)

    from prinet.utils.oscillosim import OscilloSim, bimodality_index

    n_oscillators_list = [128, 256, 512, 1024]
    n_steps = 5000
    dt = 0.01
    sweep: list[dict[str, Any]] = []

    for n_osc in n_oscillators_list:
        _p(f"\n  --- N = {n_osc} oscillators ---")
        bc_values: list[float] = []
        order_params: list[float] = []
        chimera_lifetimes: list[float] = []
        per_seed_data: list[dict[str, Any]] = []

        for si, seed in enumerate(SEEDS_7):
            _p(f"    Seed {seed} ({si+1}/{len(SEEDS_7)})")
            torch.manual_seed(seed)

            sim = OscilloSim(
                n_oscillators=n_osc,
                coupling_strength=4.0,
                coupling_mode="ring",
                k_neighbors=min(n_osc // 4, 64),
                phase_lag=1.457,  # Abrams & Strogatz chimera parameter
                integrator="rk4",
                device=DEVICE,
            )

            result_sim = sim.run(
                n_steps=n_steps, dt=dt,
                record_trajectory=True,
                record_interval=100,
            )

            # Compute bimodality coefficient of final local order parameter
            final_phase = result_sim.final_phase
            if final_phase.dim() > 1:
                final_phase = final_phase.squeeze()

            try:
                bc = bimodality_index(final_phase)
                bc_val = float(bc)
            except Exception:
                bc_val = 0.0

            r_final = float(result_sim.order_parameter[-1]) if result_sim.order_parameter else 0.0

            # Estimate chimera lifetime from order parameter trajectory
            # (duration for which 0.3 < r < 0.7, indicating partial sync)
            chimera_steps = sum(
                1 for r in result_sim.order_parameter
                if 0.3 < r < 0.7
            )
            chimera_lifetime = chimera_steps * dt * (n_steps / max(len(result_sim.order_parameter), 1))

            bc_values.append(bc_val)
            order_params.append(r_final)
            chimera_lifetimes.append(chimera_lifetime)

            per_seed_data.append({
                "seed": seed,
                "bc": bc_val,
                "r_final": r_final,
                "chimera_lifetime": chimera_lifetime,
            })

            del sim, result_sim
            _cleanup()

        entry = {
            "n_oscillators": n_osc,
            "bc_stats": _bootstrap_ci(bc_values),
            "order_param_stats": _bootstrap_ci(order_params),
            "chimera_lifetime_stats": _bootstrap_ci(chimera_lifetimes),
            "per_seed": per_seed_data,
        }
        sweep.append(entry)
        _p(f"    BC: {np.mean(bc_values):.4f} +/- {np.std(bc_values, ddof=1):.4f}")
        _p(f"    r: {np.mean(order_params):.4f}")

    # Check if BC stabilizes (compare slope from 128->1024)
    if len(sweep) >= 2:
        bc_first = sweep[0]["bc_stats"]["mean"]
        bc_last = sweep[-1]["bc_stats"]["mean"]
        bc_change = abs(bc_last - bc_first)
        stabilizes = bc_change < 0.1  # Less than 0.1 change = stable
    else:
        stabilizes = False

    result: dict[str, Any] = {
        "benchmark": "phase2_exp2_6_chimera_nsweep",
        "description": "Chimera state scaling with system size N",
        "n_oscillators_list": n_oscillators_list,
        "n_steps": n_steps,
        "dt": dt,
        "n_seeds": len(SEEDS_7),
        "sweep": sweep,
        "scaling_analysis": {
            "bc_stabilizes": stabilizes,
            "interpretation": (
                "Chimera is a genuine dynamical regime (BC stable with N)"
                if stabilizes
                else "Chimera may be a finite-size effect (BC changes with N)"
            ),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("chimera_nsweep", result)
    return result


# =========================================================================
# Unified Phase 2 Summary
# =========================================================================


def _generate_phase2_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Generate unified Phase 2 go/no-go summary.

    Args:
        results: Dict mapping experiment IDs to result dicts.

    Returns:
        Summary dict.
    """
    _p("\n" + "=" * 60)
    _p("PHASE 2 UNIFIED SUMMARY")
    _p("=" * 60)

    go_no_go: dict[str, Any] = {}

    # 2.1 Object scaling
    r21 = results.get("2.1", {})
    crossover = r21.get("crossover_n_star")
    go_no_go["object_scaling"] = {
        "criterion": "Clear crossover N* identified, or PT maintains advantage at N>=15",
        "crossover_n_star": crossover,
        "pass": crossover is not None,
    }

    # 2.2 Sequence scaling (headline)
    r22 = results.get("2.2", {})
    headline = r22.get("headline_result", {})
    adv = headline.get("advantage_at_T1000", 0) or 0
    go_no_go["sequence_scaling"] = {
        "criterion": "PT IP at T=1000 > SA IP at T=1000 by >= 2%",
        "advantage": adv,
        "significant": headline.get("significant", False),
        "pass": adv >= 0.02,
    }

    # 2.3 Occlusion recovery
    r23 = results.get("2.3", {})
    tau_r = r23.get("aggregate", {}).get("pt_tau_r_mean")
    go_no_go["occlusion_recovery"] = {
        "criterion": "tau_r finite and < 20 frames",
        "tau_r_mean": tau_r,
        "pass": tau_r is not None and tau_r < 20,
    }

    all_pass = all(v.get("pass", False) for v in go_no_go.values())

    for key, val in go_no_go.items():
        status = "PASS" if val.get("pass") else "FAIL"
        _p(f"  [{status}] {key}: {val.get('criterion')}")

    _p(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    summary: dict[str, Any] = {
        "benchmark": "phase2_unified_summary",
        "description": "Phase 2 Scaling Analysis go/no-go summary",
        "go_no_go": go_no_go,
        "all_pass": all_pass,
        "phase2_recommendation": (
            "Proceed to Phase 3. Distinct advantage regimes identified."
            if all_pass
            else "Review failed criteria. Paper may focus on parameter efficiency."
        ),
        "n_experiments_run": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("unified_summary", summary)
    return summary


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    """Run Phase 2 scaling analysis experiments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 2: Scaling Analysis (Experiments 2.1-2.6)"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--object-scaling", action="store_true", help="Exp 2.1")
    parser.add_argument("--sequence-scaling", action="store_true", help="Exp 2.2")
    parser.add_argument("--occlusion-recovery", action="store_true", help="Exp 2.3")
    parser.add_argument("--velocity-stress", action="store_true", help="Exp 2.4")
    parser.add_argument("--noise-sweep", action="store_true", help="Exp 2.5")
    parser.add_argument("--chimera-nsweep", action="store_true", help="Exp 2.6")
    args = parser.parse_args()

    run_all = args.all or not any([
        args.object_scaling, args.sequence_scaling, args.occlusion_recovery,
        args.velocity_stress, args.noise_sweep, args.chimera_nsweep,
    ])

    _p("=" * 60)
    _p("PRINet Phase 2: Scaling Analysis")
    _p("=" * 60)
    _p(f"Device: {DEVICE}")
    _p(f"Results directory: {RESULTS_DIR}")
    assert DEVICE == "cuda", (
        "GPU required for Phase 2 experiments. "
        "Install CUDA-enabled PyTorch: pip install torch --index-url "
        "https://download.pytorch.org/whl/cu128"
    )
    _p(f"GPU: {torch.cuda.get_device_name(0)}")
    _p(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    t0 = time.time()
    results: dict[str, Any] = {}

    if run_all or args.object_scaling:
        results["2.1"] = experiment_2_1_object_scaling()
    if run_all or args.sequence_scaling:
        results["2.2"] = experiment_2_2_sequence_scaling()
    if run_all or args.occlusion_recovery:
        results["2.3"] = experiment_2_3_occlusion_recovery()
    if run_all or args.velocity_stress:
        results["2.4"] = experiment_2_4_velocity_stress()
    if run_all or args.noise_sweep:
        results["2.5"] = experiment_2_5_noise_sweep()
    if run_all or args.chimera_nsweep:
        results["2.6"] = experiment_2_6_chimera_nsweep()

    # Generate unified summary if all experiments ran
    if run_all:
        _generate_phase2_summary(results)

    elapsed = time.time() - t0
    _p(f"\nPhase 2 complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
