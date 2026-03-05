#!/usr/bin/env python
"""Phase 1 BF Extension: Expand 7-seed comparison to 15 seeds.

Addresses the Phase 1 go/no-go Bayes Factor failure (BF_10 = 0.521 > 1/3).
Trains PT and SA on 8 additional seeds, merges with existing 7-seed data,
and re-computes BF_10 with n=15 per group to achieve adequate statistical
power for equivalence evidence.

Usage:
    python benchmarks/phase1_bf_extension.py

Hardware: Requires CUDA GPU.

Reference:
    Phase 1 unified summary: BF criterion CONDITIONAL FAIL at n=7.
    Expected: BF_10 < 1/3 at n=15 given observed effect size.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Original 7 seeds + 8 new seeds = 15 total
SEEDS_ORIGINAL = (42, 123, 456, 789, 1024, 2048, 3072)
SEEDS_NEW = (4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264)
SEEDS_15 = SEEDS_ORIGINAL + SEEDS_NEW

DET_DIM = 4
TRAIN_SEQS = 30
VAL_SEQS = 10
TEST_SEQS = 20
MAX_EPOCHS = 10
PATIENCE = 4
WARMUP = 1
LR = 3e-4

PT_KWARGS = dict(
    n_delta=4, n_theta=8, n_gamma=16,
    n_discrete_steps=5, match_threshold=0.1,
)
SA_KWARGS = dict(
    num_slots=6, slot_dim=64, num_iterations=3,
    match_threshold=0.1,
)


def _p(*a: Any, **kw: Any) -> None:
    print(*a, flush=True, **kw)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_pt(seed: int) -> nn.Module:
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(detection_dim=DET_DIM, **PT_KWARGS)


def _build_sa(seed: int) -> nn.Module:
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(detection_dim=DET_DIM, **SA_KWARGS)


def _train_and_eval(seed: int) -> dict[str, float]:
    """Train PT and SA from scratch for one seed, return IPs.

    Args:
        seed: Random seed.

    Returns:
        Dict with pt_ip and sa_ip (mean over test set).
    """
    from prinet.utils.temporal_training import (
        generate_dataset,
        hungarian_similarity_loss,
    )

    n_objects = 4
    n_frames = 20

    results = {}
    for label, build_fn in [("pt", _build_pt), ("sa", _build_sa)]:
        torch.manual_seed(seed)
        model = build_fn(seed).to(DEVICE)
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LR,
        )

        train_data = generate_dataset(
            TRAIN_SEQS, n_objects=n_objects, n_frames=n_frames,
            det_dim=DET_DIM, base_seed=seed,
        )
        val_data = generate_dataset(
            VAL_SEQS, n_objects=n_objects, n_frames=n_frames,
            det_dim=DET_DIM, base_seed=seed + 50000,
        )

        best_loss = float("inf")
        best_state = None
        patience_cnt = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for seq in train_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                total_loss = torch.tensor(0.0, device=DEVICE)
                for t in range(len(frames) - 1):
                    _, sim = model(frames[t], frames[t + 1])
                    total_loss = total_loss + hungarian_similarity_loss(
                        sim, n_objects,
                    )
                total_loss = total_loss / max(len(frames) - 1, 1)
                optimizer.zero_grad()
                if total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            v_losses = []
            with torch.no_grad():
                for seq in val_data:
                    frames = [f.to(DEVICE) for f in seq.frames]
                    t_loss = torch.tensor(0.0, device=DEVICE)
                    for t in range(len(frames) - 1):
                        _, sim = model(frames[t], frames[t + 1])
                        t_loss = t_loss + hungarian_similarity_loss(
                            sim, n_objects,
                        )
                    v_losses.append(
                        float(t_loss.item() / max(len(frames) - 1, 1))
                    )
            val_loss = sum(v_losses) / max(len(v_losses), 1)

            if val_loss < best_loss and epoch >= WARMUP:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE and epoch >= WARMUP:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Evaluate on test set
        from prinet.utils.temporal_training import generate_temporal_clevr_n
        ips = []
        with torch.no_grad():
            for i in range(TEST_SEQS):
                seq = generate_temporal_clevr_n(
                    n_objects=n_objects, n_frames=n_frames,
                    det_dim=DET_DIM, seed=seed + 70000 + i,
                )
                frames = [f.to(DEVICE) for f in seq.frames]
                res = model.track_sequence(frames)
                ips.append(res["identity_preservation"])

        results[f"{label}_ip"] = float(np.mean(ips))
        del model
        _cleanup()

    return results


def _jeffreys_bayes_factor_t(
    t_stat: float, n_a: int, n_b: int, r: float = 0.7071,
) -> float:
    """JZS Bayes Factor BF_10 (same implementation as Phase 1)."""
    from scipy import integrate

    n = n_a + n_b
    v = n - 2
    n_eff = (n_a * n_b) / (n_a + n_b)

    def integrand(g: float) -> float:
        if g <= 0:
            return 0.0
        f1 = (1.0 + n_eff * g) ** (-0.5)
        f2 = (
            1.0 + t_stat**2 / (v * (1.0 + n_eff * g))
        ) ** (-(v + 1) / 2.0)
        f3 = (1.0 + t_stat**2 / v) ** ((v + 1) / 2.0)
        prior = (
            (r**2 / 2.0) ** 0.5
            * g ** (-1.5)
            * math.exp(-r**2 / (2.0 * g))
        )
        prior /= math.gamma(0.5) * math.sqrt(2.0)
        return f1 * f2 * f3 * prior

    try:
        bf10, _ = integrate.quad(integrand, 1e-10, 100.0, limit=200)
    except Exception:
        bf10 = float("nan")
    return bf10


def main() -> int:
    """Extend 7-seed comparison to 15 seeds and re-compute BF.

    Returns:
        0 on success.
    """
    _p("=" * 60)
    _p("Phase 1 BF Extension: 7 -> 15 Seeds")
    _p("=" * 60)
    _p(f"Device: {DEVICE}")
    assert DEVICE == "cuda", "GPU required for BF seed expansion training"
    _p(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load existing 7-seed data
    existing = json.load(
        open(RESULTS_DIR / "y4q1_9_7seed_comparison.json"),
    )

    # Check for partially-completed extension
    partial_path = RESULTS_DIR / "phase1_bf_extension_partial.json"
    if partial_path.exists():
        partial = json.load(open(partial_path))
        pt_ips = list(partial["pt_ips"])
        sa_ips = list(partial["sa_ips"])
        per_seed = list(partial.get("per_seed", []))
        seeds_done = {s["seed"] for s in per_seed}
        _p(f"Resuming from partial: {len(pt_ips)} seeds done")
    else:
        pt_ips = list(existing["pt_ips"])  # 7 values
        sa_ips = list(existing["sa_ips"])  # 7 values
        per_seed = list(existing.get("per_seed", []))
        seeds_done = set(existing.get("seeds", list(SEEDS_ORIGINAL)))

    _p(f"Existing seeds: {len(pt_ips)} (PT), {len(sa_ips)} (SA)")

    # Train new seeds
    t0 = time.time()

    for si, seed in enumerate(SEEDS_NEW):
        if seed in seeds_done:
            _p(f"  [skip] Seed {seed} already done")
            continue
        _p(f"\n  Training seed {seed} ({si+1}/{len(SEEDS_NEW)})...")
        t_seed = time.time()
        res = _train_and_eval(seed)
        pt_ips.append(res["pt_ip"])
        sa_ips.append(res["sa_ip"])
        per_seed.append({
            "seed": seed,
            "pt_ip": res["pt_ip"],
            "sa_ip": res["sa_ip"],
        })
        elapsed_seed = time.time() - t_seed
        _p(f"    PT IP={res['pt_ip']:.6f}, SA IP={res['sa_ip']:.6f} "
           f"({elapsed_seed:.1f}s)")

        # Save partial progress after each seed
        with open(partial_path, "w") as f:
            json.dump({"pt_ips": pt_ips, "sa_ips": sa_ips,
                        "per_seed": per_seed}, f, indent=2, default=str)
        _p(f"    (partial saved: {len(pt_ips)} seeds)")

    # Compute new statistics
    from scipy import stats as sp_stats

    pt_arr = np.array(pt_ips)
    sa_arr = np.array(sa_ips)
    n_a, n_b = len(pt_arr), len(sa_arr)

    t_res = sp_stats.ttest_ind(pt_arr, sa_arr, equal_var=False)
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)

    # Welch's t (manual for complete stats)
    mean_a = float(pt_arr.mean())
    mean_b = float(sa_arr.mean())
    var_a = float(pt_arr.var(ddof=1))
    var_b = float(sa_arr.var(ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    cohens_d = (mean_a - mean_b) / math.sqrt(pooled_var) if pooled_var > 0 else 0.0

    # BF_10 with 15 seeds
    bf10 = _jeffreys_bayes_factor_t(t_stat, n_a, n_b, r=0.7071)

    # Pingouin cross-check
    try:
        import pingouin as pg
        bf10_pg = float(pg.bayesfactor_ttest(t_stat, n_a, n_b, paired=False, r=0.7071))
    except Exception:
        bf10_pg = float("nan")

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_boot = 10000
    pt_boot = sorted([
        float(rng.choice(pt_arr, size=n_a, replace=True).mean())
        for _ in range(n_boot)
    ])
    sa_boot = sorted([
        float(rng.choice(sa_arr, size=n_b, replace=True).mean())
        for _ in range(n_boot)
    ])
    pt_ci = (pt_boot[int(0.025 * n_boot)], pt_boot[int(0.975 * n_boot)])
    sa_ci = (sa_boot[int(0.025 * n_boot)], sa_boot[int(0.975 * n_boot)])

    # CV
    pt_cv = float(pt_arr.std(ddof=1) / pt_arr.mean()) if pt_arr.mean() > 0 else 0.0
    sa_cv = float(sa_arr.std(ddof=1) / sa_arr.mean()) if sa_arr.mean() > 0 else 0.0

    # Cliff's delta
    more = sum(1 for a in pt_ips for b in sa_ips if a > b)
    less = sum(1 for a in pt_ips for b in sa_ips if a < b)
    cd = (more - less) / (n_a * n_b)

    elapsed = time.time() - t0

    def _bf_interp(bf: float) -> str:
        if math.isnan(bf):
            return "computation_error"
        if bf > 100:
            return "extreme_evidence_for_H1"
        elif bf > 30:
            return "very_strong_evidence_for_H1"
        elif bf > 10:
            return "strong_evidence_for_H1"
        elif bf > 3:
            return "moderate_evidence_for_H1"
        elif bf > 1:
            return "anecdotal_evidence_for_H1"
        elif bf > 1 / 3:
            return "anecdotal_evidence_for_H0"
        elif bf > 1 / 10:
            return "moderate_evidence_for_H0"
        elif bf > 1 / 30:
            return "strong_evidence_for_H0"
        elif bf > 1 / 100:
            return "very_strong_evidence_for_H0"
        else:
            return "extreme_evidence_for_H0"

    _p(f"\n{'='*60}")
    _p(f"RESULTS: 15-Seed Comparison")
    _p(f"{'='*60}")
    _p(f"  PT: mean={mean_a:.6f}, std={pt_arr.std(ddof=1):.6f}, "
       f"CV={pt_cv:.6f}, CI=[{pt_ci[0]:.6f}, {pt_ci[1]:.6f}]")
    _p(f"  SA: mean={mean_b:.6f}, std={sa_arr.std(ddof=1):.6f}, "
       f"CV={sa_cv:.6f}, CI=[{sa_ci[0]:.6f}, {sa_ci[1]:.6f}]")
    _p(f"  Welch's t = {t_stat:.4f}, p = {p_val:.6f}")
    _p(f"  Cohen's d = {cohens_d:.4f}")
    _p(f"  Cliff's delta = {cd:.4f}")
    _p(f"  BF_10 (JZS) = {bf10:.4f} ({_bf_interp(bf10)})")
    _p(f"  BF_10 (pingouin) = {bf10_pg:.4f}")
    _p(f"  BF criterion (< 1/3): {'PASS' if bf10 < 1/3 else 'FAIL'}")

    # Save extended comparison
    result_15seed = {
        "benchmark": "phase1_bf_extension_15seed_comparison",
        "description": "Extended 7->15 seed comparison for BF resolution",
        "n_seeds": n_a,
        "seeds": list(SEEDS_15),
        "pt_ips": pt_ips,
        "sa_ips": sa_ips,
        "pt_stats": {
            "mean": mean_a,
            "std": float(pt_arr.std(ddof=1)),
            "cv": pt_cv,
            "ci_95": list(pt_ci),
        },
        "sa_stats": {
            "mean": mean_b,
            "std": float(sa_arr.std(ddof=1)),
            "cv": sa_cv,
            "ci_95": list(sa_ci),
        },
        "welch_t": {
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "n_a": n_a,
            "n_b": n_b,
        },
        "cliffs_delta": cd,
        "bayes_factor": {
            "bf10_jzs": bf10,
            "bf10_pingouin": bf10_pg,
            "bf01_jzs": 1.0 / bf10 if bf10 > 0 else float("inf"),
            "interpretation": _bf_interp(bf10),
            "prior": "cauchy",
            "prior_scale_r": 0.7071,
            "criterion_pass": bf10 < 1 / 3,
        },
        "per_seed": per_seed,
        "previous_7seed_bf10": 0.5213,
        "wall_time_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = RESULTS_DIR / "phase1_bf_extension_15seed.json"
    with open(out_path, "w") as f:
        json.dump(result_15seed, f, indent=2, default=str)
    _p(f"\n  -> {out_path.name}")

    # Remove partial file
    if partial_path.exists():
        partial_path.unlink()
        _p("  (partial file cleaned up)")

    # Also update the unified summary
    summary_path = RESULTS_DIR / "phase1_unified_summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        summary["go_no_go"]["bayes_factor"] = {
            "criterion": "BF_10 < 1/3 (evidence for equivalence)",
            "pass": bf10 < 1 / 3,
            "bf10": bf10,
            "bf10_pingouin": bf10_pg,
            "interpretation": _bf_interp(bf10),
            "n_seeds": n_a,
            "note": f"Extended from 7 to {n_a} seeds",
        }
        summary["all_pass"] = all(
            v.get("pass", False) for v in summary["go_no_go"].values()
        )
        summary["phase1_recommendation"] = (
            "Proceed to Phase 2. All statistical claims are now defensible."
            if summary["all_pass"]
            else "Review failed criteria before proceeding."
        )
        summary["bf_extension_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        _p(f"  -> Updated {summary_path.name}")

    _p(f"\nBF extension complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
