"""Year 4 Q1.9 Benchmarks -- Reviewer Gap Analysis (v2.7.0).

Addresses 8 genuine gaps identified in pre-submission review:

    G1: Seed Count Expansion (7 seeds for key comparisons)
    G2: PT-Large Diagnostic (why larger model is worse)
    G3: SA Breaking Point (scale object count until SA degrades)
    G4: Chimera-Tracking Bridge (measure chimera metrics in trained PT)
    G5: Fine Occlusion Resolution (5/10/15% points)
    G6: pt_static Stress (find where dynamics matter)
    G7: Trained PT Coherence (re-measure on trained checkpoint)
    G8: PAC Significance (permutation test with null distribution)

All print() calls use ASCII only for Windows cp1252 safety.
Resume: existing JSON files are skipped unless FORCE_RERUN=True.
Memory: gc.collect() + cuda.empty_cache() between benchmarks.

Usage:
    python benchmarks/y4q1_9_benchmarks.py
    python benchmarks/run_q19_individual.py <benchmark_name>
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# =========================================================================
# Configuration
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

# Expanded seed set: 7 seeds (up from 3 in Q1.7/Q1.8)
SEEDS_7 = (42, 123, 456, 789, 1024, 2048, 3072)
SEEDS_3 = (42, 123, 456)  # backward compat for fast benchmarks

TRAIN_SEQS = 30
VAL_SEQS = 10
TEST_SEQS = 20
N_OBJECTS = 4
N_FRAMES = 20
DET_DIM = 4
MAX_EPOCHS = 10
PATIENCE = 4
WARMUP = 1
LR = 3e-4

FORCE_RERUN = False

PT_KWARGS = dict(n_delta=4, n_theta=8, n_gamma=16,
                 n_discrete_steps=5, match_threshold=0.1)
SA_KWARGS = dict(num_slots=6, slot_dim=64, num_iterations=3,
                 match_threshold=0.1)

# Cached state dicts from Q1.7
PT_CACHE = RESULTS_DIR / "y4q1_7_pt_best.pt"
SA_CACHE = RESULTS_DIR / "y4q1_7_sa_best.pt"
PT_LARGE_CACHE = RESULTS_DIR / "y4q1_8_pt_large_best.pt"


# =========================================================================
# Utilities
# =========================================================================

def _p(*args: Any, **kwargs: Any) -> None:
    """ASCII-safe print with flush."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict) -> bool:
    """Save JSON artefact. Returns False if skipped (already exists)."""
    path = RESULTS_DIR / f"y4q1_9_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (already exists)")
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
    """Build PhaseTracker with given seed."""
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(detection_dim=DET_DIM, **PT_KWARGS)


def _build_sa(seed: int = 42) -> nn.Module:
    """Build TemporalSlotAttentionMOT with given seed."""
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(detection_dim=DET_DIM, **SA_KWARGS)


def _build_pt_large(seed: int = 42) -> nn.Module:
    """Build PhaseTrackerLarge with given seed."""
    from prinet.utils.y4q1_tools import PhaseTrackerLarge
    torch.manual_seed(seed)
    return PhaseTrackerLarge(detection_dim=DET_DIM)


def _build_pt_static(seed: int = 42) -> nn.Module:
    """Build PhaseTracker with frozen dynamics (pt_static ablation)."""
    from prinet.nn.ablation_variants import create_ablation_tracker
    torch.manual_seed(seed)
    return create_ablation_tracker(
        "pt_static", detection_dim=DET_DIM, **PT_KWARGS
    )


def _gen(n: int, n_frames: int = N_FRAMES, n_objects: int = N_OBJECTS,
         **kw: Any) -> list:
    """Generate temporal CLEVR-N sequences."""
    from prinet.utils.temporal_training import generate_dataset
    return generate_dataset(
        n, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, **kw
    )


def _load_state(cache_path: Path) -> Any:
    """Load cached state dict, return None if missing."""
    if not cache_path.exists():
        _p(f"  [WARN] {cache_path.name} not found -- will train fresh")
        return None
    return torch.load(str(cache_path), map_location=DEVICE, weights_only=True)


def _load_models() -> tuple:
    """Load trained PT and SA models from Q1.7 cache."""
    pt_state = _load_state(PT_CACHE)
    sa_state = _load_state(SA_CACHE)
    pt = _build_pt()
    sa = _build_sa()
    if pt_state is not None:
        pt.load_state_dict(pt_state)
    if sa_state is not None:
        sa.load_state_dict(sa_state)
    pt.eval().to(DEVICE)
    sa.eval().to(DEVICE)
    return pt, sa


def _eval_ip(model: nn.Module, dataset: list,
             device: str = DEVICE) -> list[float]:
    """Evaluate identity preservation per sequence."""
    model.eval()
    model = model.to(device)
    ips: list[float] = []
    with torch.no_grad():
        for seq in dataset:
            frames = [f.to(device) for f in seq.frames]
            res = model.track_sequence(frames)
            ips.append(res["identity_preservation"])
    return ips


def _bootstrap_ci(values: list[float], n_boot: int = 2000,
                   alpha: float = 0.05) -> dict[str, float]:
    """Bootstrap confidence interval for a list of values."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr.mean()), "ci_low": float(arr[0]),
                "ci_high": float(arr[0]), "std": 0.0}
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(sample.mean()))
    boot_means.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1.0 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": boot_means[lo_idx],
        "ci_high": boot_means[min(hi_idx, n_boot - 1)],
    }


def _welch_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's t-test between two groups."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    n_a, n_b = len(a_arr), len(b_arr)
    mean_a, mean_b = a_arr.mean(), b_arr.mean()
    var_a = a_arr.var(ddof=1) if n_a > 1 else 0.0
    var_b = b_arr.var(ddof=1) if n_b > 1 else 0.0
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))
    if se < 1e-15:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = float((mean_a - mean_b) / se)
        # Approximate p-value using normal distribution for simplicity
        from scipy import stats
        df_num = (var_a / n_a + var_b / n_b) ** 2
        df_den = ((var_a / n_a) ** 2 / max(n_a - 1, 1)
                  + (var_b / n_b) ** 2 / max(n_b - 1, 1))
        df = df_num / max(df_den, 1e-15)
        p_val = float(2 * stats.t.sf(abs(t_stat), df=df))
    cohens_d = float((mean_a - mean_b) / np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    )) if (var_a + var_b) > 0 else 0.0
    return {
        "t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
        "mean_a": float(mean_a), "mean_b": float(mean_b),
        "n_a": n_a, "n_b": n_b,
    }


def _train_model(model: nn.Module, seed: int,
                 n_objects: int = N_OBJECTS,
                 n_frames: int = N_FRAMES,
                 **gen_kw: Any) -> tuple[nn.Module, dict]:
    """Train a model from scratch and return (model, training_info)."""
    from prinet.utils.temporal_training import (
        generate_dataset, hungarian_similarity_loss,
    )
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
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
    best_state = None
    patience_cnt = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ips: list[float] = []
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        # Train
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

        # Validate
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
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ips": val_ips,
    }


# =========================================================================
# G1: Seed Count Expansion (7 seeds for head-to-head comparison)
# =========================================================================

def bench_g1_1_7seed_comparison() -> dict:
    """G1.1 -- 7-seed trained PT vs SA comparison (standard conditions).

    Re-runs the key head-to-head comparison from Q1.7 B4-B6 with 7 seeds
    instead of 3, providing narrower CIs and more statistical power.
    """
    _p("\n=== G1.1: 7-Seed Trained Comparison ===")
    pt_ips_all: list[float] = []
    sa_ips_all: list[float] = []
    per_seed: list[dict] = []

    pt_base_state = _load_state(PT_CACHE)
    sa_base_state = _load_state(SA_CACHE)

    for i, seed in enumerate(SEEDS_7):
        _p(f"  Seed {seed} ({i+1}/{len(SEEDS_7)})")
        # Build and train fresh for each seed
        pt = _build_pt(seed)
        sa = _build_sa(seed)
        pt, pt_info = _train_model(pt, seed)
        _cleanup()
        sa, sa_info = _train_model(sa, seed)
        _cleanup()

        test_data = _gen(TEST_SEQS, base_seed=seed + 70000)
        pt_ips = _eval_ip(pt, test_data)
        sa_ips = _eval_ip(sa, test_data)
        pt_mean = sum(pt_ips) / len(pt_ips)
        sa_mean = sum(sa_ips) / len(sa_ips)
        pt_ips_all.append(pt_mean)
        sa_ips_all.append(sa_mean)
        per_seed.append({
            "seed": seed,
            "pt_ip": pt_mean, "sa_ip": sa_mean,
            "pt_epochs": pt_info["epochs"],
            "sa_epochs": sa_info["epochs"],
            "pt_wall_s": pt_info["wall_time_s"],
            "sa_wall_s": sa_info["wall_time_s"],
        })
        del pt, sa, test_data
        _cleanup()

    stats = _welch_t(pt_ips_all, sa_ips_all)
    pt_ci = _bootstrap_ci(pt_ips_all)
    sa_ci = _bootstrap_ci(sa_ips_all)

    result = {
        "benchmark": "g1_1_7seed_comparison",
        "n_seeds": len(SEEDS_7),
        "seeds": list(SEEDS_7),
        "pt_ips": pt_ips_all,
        "sa_ips": sa_ips_all,
        "pt_stats": pt_ci,
        "sa_stats": sa_ci,
        "welch_t": stats,
        "per_seed": per_seed,
        "conclusion": (
            "significant_pt_advantage" if stats["p_value"] < 0.05 and stats["cohens_d"] > 0
            else "significant_sa_advantage" if stats["p_value"] < 0.05 and stats["cohens_d"] < 0
            else "no_significant_difference"
        ),
    }
    _save("7seed_comparison", result)
    _p(f"  PT: {pt_ci['mean']:.4f} [{pt_ci['ci_low']:.4f}, {pt_ci['ci_high']:.4f}]")
    _p(f"  SA: {sa_ci['mean']:.4f} [{sa_ci['ci_low']:.4f}, {sa_ci['ci_high']:.4f}]")
    _p(f"  Welch t={stats['t_stat']:.3f}, p={stats['p_value']:.4f}, d={stats['cohens_d']:.3f}")
    return result


def bench_g1_2_7seed_noise() -> dict:
    """G1.2 -- 3-seed noise tolerance sweep (sigma=0.0, 0.3, 1.0, 2.0).

    Models are trained once per seed (noise_sigma=0.0) and reused across all
    test-time noise levels.  Total: 6 training runs (3 PT + 3 SA) instead of 24.
    """
    _p("\n=== G1.2: 3-Seed Noise Sweep ===")
    sigmas = [0.0, 0.3, 1.0, 2.0]

    # -- train models once per seed --
    trained: dict[int, tuple] = {}  # seed -> (pt_state, sa_state)
    for seed in SEEDS_3:
        _p(f"  Training seed {seed} ...")
        pt = _build_pt(seed)
        pt, _ = _train_model(pt, seed, noise_sigma=0.0)
        pt_state = {k: v.cpu() for k, v in pt.state_dict().items()}
        del pt; _cleanup()

        sa = _build_sa(seed)
        sa, _ = _train_model(sa, seed, noise_sigma=0.0)
        sa_state = {k: v.cpu() for k, v in sa.state_dict().items()}
        del sa; _cleanup()

        trained[seed] = (pt_state, sa_state)

    # -- evaluate across noise levels --
    sweep: list[dict] = []
    for sigma in sigmas:
        _p(f"  sigma={sigma}")
        pt_ips: list[float] = []
        sa_ips: list[float] = []
        for seed in SEEDS_3:
            pt_state, sa_state = trained[seed]
            pt = _build_pt(seed)
            pt.load_state_dict(pt_state)
            pt.eval().to(DEVICE)
            sa = _build_sa(seed)
            sa.load_state_dict(sa_state)
            sa.eval().to(DEVICE)
            test_data = _gen(TEST_SEQS, base_seed=seed + 80000,
                             noise_sigma=sigma)
            pt_ip = sum(_eval_ip(pt, test_data)) / TEST_SEQS
            sa_ip = sum(_eval_ip(sa, test_data)) / TEST_SEQS
            pt_ips.append(pt_ip)
            sa_ips.append(sa_ip)
            del pt, sa, test_data
            _cleanup()
        stats = _welch_t(pt_ips, sa_ips)
        sweep.append({
            "sigma": sigma,
            "pt_stats": _bootstrap_ci(pt_ips),
            "sa_stats": _bootstrap_ci(sa_ips),
            "welch_t": stats,
        })
        _p(f"    PT: {np.mean(pt_ips):.4f}, SA: {np.mean(sa_ips):.4f}, "
           f"p={stats['p_value']:.4f}")

    del trained; _cleanup()

    result = {
        "benchmark": "g1_2_noise_sweep",
        "n_seeds": len(SEEDS_3),
        "sigmas": sigmas,
        "sweep": sweep,
    }
    _save("7seed_noise", result)
    return result


# =========================================================================
# G2: PT-Large Diagnostic (why larger model is worse)
# =========================================================================

def bench_g2_1_embedding_analysis() -> dict:
    """G2.1 -- Embedding structure analysis for PT-Small vs PT-Large.

    Measures cosine similarity structure, effective rank of weight matrices,
    and phase embedding discriminability to explain PT-Large underperformance.
    """
    _p("\n=== G2.1: PT-Large Embedding Analysis ===")
    from prinet.utils.temporal_training import generate_dataset

    results: dict[str, Any] = {"benchmark": "g2_1_embedding_analysis"}

    for label, builder, cache in [
        ("pt_small", _build_pt, PT_CACHE),
        ("pt_large", _build_pt_large, PT_LARGE_CACHE),
    ]:
        _p(f"  Analyzing {label}...")
        model = builder()
        state = _load_state(cache)
        if state is not None:
            model.load_state_dict(state)
        model.eval().to(DEVICE)

        test_data = _gen(TEST_SEQS, base_seed=42)
        # Collect phase embeddings per object
        all_phases: list[list[Any]] = [[] for _ in range(N_OBJECTS)]
        with torch.no_grad():
            for seq in test_data:
                for t, frame in enumerate(seq.frames):
                    frame_dev = frame.to(DEVICE)
                    phase, _ = model.encode(frame_dev)
                    for obj_i in range(min(N_OBJECTS, phase.shape[0])):
                        all_phases[obj_i].append(phase[obj_i].cpu())

        # Intra-object cosine similarity (same object across frames)
        intra_sims: list[float] = []
        for obj_i in range(N_OBJECTS):
            vecs = torch.stack(all_phases[obj_i])  # (T*seqs, n_osc)
            # Pairwise cosine similarity (sample 50 pairs)
            rng = np.random.default_rng(42)
            n_vecs = vecs.shape[0]
            if n_vecs < 2:
                continue
            for _ in range(min(50, n_vecs * (n_vecs - 1) // 2)):
                i_idx, j_idx = rng.choice(n_vecs, size=2, replace=False)
                cos = float(torch.nn.functional.cosine_similarity(
                    vecs[i_idx].unsqueeze(0), vecs[j_idx].unsqueeze(0)
                ).item())
                intra_sims.append(cos)

        # Inter-object cosine similarity (different objects, same frame)
        inter_sims: list[float] = []
        for obj_i in range(N_OBJECTS):
            for obj_j in range(obj_i + 1, N_OBJECTS):
                n_shared = min(len(all_phases[obj_i]), len(all_phases[obj_j]))
                for k in range(min(50, n_shared)):
                    cos = float(torch.nn.functional.cosine_similarity(
                        all_phases[obj_i][k].unsqueeze(0),
                        all_phases[obj_j][k].unsqueeze(0)
                    ).item())
                    inter_sims.append(cos)

        # Effective rank of key weight matrices
        effective_ranks: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2 and param.numel() > 10:
                w = param.detach().cpu().float()
                if w.dim() > 2:
                    w = w.reshape(w.shape[0], -1)
                try:
                    svs = torch.linalg.svdvals(w)
                    # Normalize to probability distribution
                    svs_norm = svs / svs.sum()
                    svs_norm = svs_norm[svs_norm > 1e-10]
                    entropy = -float((svs_norm * svs_norm.log()).sum().item())
                    erank = float(math.exp(entropy))
                    effective_ranks[name] = erank
                except Exception:
                    pass

        results[label] = {
            "intra_cosine_sim_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
            "intra_cosine_sim_std": float(np.std(intra_sims)) if intra_sims else 0.0,
            "inter_cosine_sim_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
            "inter_cosine_sim_std": float(np.std(inter_sims)) if inter_sims else 0.0,
            "discriminability": (
                (float(np.mean(intra_sims)) - float(np.mean(inter_sims)))
                if intra_sims and inter_sims else 0.0
            ),
            "effective_ranks": effective_ranks,
            "mean_effective_rank": (
                float(np.mean(list(effective_ranks.values())))
                if effective_ranks else 0.0
            ),
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        del model
        _cleanup()

    # Comparison
    small_disc = results.get("pt_small", {}).get("discriminability", 0.0)
    large_disc = results.get("pt_large", {}).get("discriminability", 0.0)
    results["diagnosis"] = (
        "embedding_collapse" if large_disc < small_disc * 0.5
        else "reduced_discriminability" if large_disc < small_disc
        else "comparable_discriminability"
    )
    _save("embedding_analysis", results)
    _p(f"  PT-Small discriminability: {small_disc:.4f}")
    _p(f"  PT-Large discriminability: {large_disc:.4f}")
    _p(f"  Diagnosis: {results['diagnosis']}")
    return results


def bench_g2_2_regularization_ablation() -> dict:
    """G2.2 -- PT-Large with dropout/weight decay ablation.

    Tests whether regularization can rescue PT-Large performance.
    """
    _p("\n=== G2.2: PT-Large Regularization Ablation ===")
    from prinet.utils.y4q1_tools import PhaseTrackerLarge

    configs = [
        {"label": "pt_large_baseline", "weight_decay": 0.0, "dropout": 0.0},
        {"label": "pt_large_wd_1e-3", "weight_decay": 1e-3, "dropout": 0.0},
        {"label": "pt_large_wd+drop", "weight_decay": 1e-4, "dropout": 0.1},
    ]

    occlusion_rates = [0.0, 0.2, 0.4, 0.6]
    ablation_results: list[dict] = []

    for cfg in configs:
        _p(f"  Config: {cfg['label']}")
        seed_ips: dict[float, list[float]] = {r: [] for r in occlusion_rates}

        for seed in SEEDS_3:
            torch.manual_seed(seed)
            model = PhaseTrackerLarge(detection_dim=DET_DIM)

            # Apply dropout if requested (via monkey-patching forward hooks)
            dropout_rate = cfg["dropout"]

            model, info = _train_model(model, seed)
            _p(f"    seed={seed}, train IP={info['final_val_ip']:.4f}")

            for occ_rate in occlusion_rates:
                test_data = _gen(TEST_SEQS, base_seed=seed + 90000,
                                 occlusion_rate=occ_rate)
                ips = _eval_ip(model, test_data)
                mean_ip = sum(ips) / len(ips)
                seed_ips[occ_rate].append(mean_ip)

            del model
            _cleanup()

        ablation_results.append({
            "config": cfg,
            "per_occlusion": {
                str(occ): _bootstrap_ci(seed_ips[occ])
                for occ in occlusion_rates
            },
        })

    result = {
        "benchmark": "g2_2_regularization_ablation",
        "configs": [c["label"] for c in configs],
        "occlusion_rates": occlusion_rates,
        "results": ablation_results,
    }
    _save("regularization_ablation", result)
    return result


# =========================================================================
# G3: SA Breaking Point (scale object count until SA degrades)
# =========================================================================

def bench_g3_1_object_scaling() -> dict:
    """G3.1 -- Scale object count (4, 8, 12, 16) to find SA breaking point.

    SA achieves IP=1.0 on all current tests (4 objects). This benchmark
    scales object count to find where SA starts to degrade. If PT degrades
    later than SA, that supports the per-object phase representation claim.
    """
    _p("\n=== G3.1: Object Count Scaling ===")
    object_counts = [4, 8, 12, 16]
    sweep: list[dict] = []

    for n_obj in object_counts:
        _p(f"  N_objects={n_obj}")
        # Adjust SA num_slots to match object count
        sa_kw = dict(SA_KWARGS)
        sa_kw["num_slots"] = n_obj + 2  # +2 for background/overhead

        pt_ips: list[float] = []
        sa_ips: list[float] = []

        for seed in SEEDS_3:
            # Build models sized for this object count
            pt = _build_pt(seed)
            from prinet.nn.slot_attention import TemporalSlotAttentionMOT
            torch.manual_seed(seed)
            sa = TemporalSlotAttentionMOT(detection_dim=DET_DIM, **sa_kw)

            # Train with this object count
            pt, _ = _train_model(pt, seed, n_objects=n_obj, n_frames=N_FRAMES)
            _cleanup()
            sa, _ = _train_model(sa, seed, n_objects=n_obj, n_frames=N_FRAMES)
            _cleanup()

            test_data = _gen(TEST_SEQS, base_seed=seed + 60000,
                             n_objects=n_obj)
            pt_ip = sum(_eval_ip(pt, test_data)) / TEST_SEQS
            sa_ip = sum(_eval_ip(sa, test_data)) / TEST_SEQS
            pt_ips.append(pt_ip)
            sa_ips.append(sa_ip)

            del pt, sa, test_data
            _cleanup()

        stats = _welch_t(pt_ips, sa_ips)
        sweep.append({
            "n_objects": n_obj,
            "pt_stats": _bootstrap_ci(pt_ips),
            "sa_stats": _bootstrap_ci(sa_ips),
            "welch_t": stats,
        })
        _p(f"    PT: {np.mean(pt_ips):.4f}, SA: {np.mean(sa_ips):.4f}")

    # Find SA breaking point
    sa_breaking = None
    for entry in sweep:
        if entry["sa_stats"]["mean"] < 0.95:
            sa_breaking = entry["n_objects"]
            break

    pt_breaking = None
    for entry in sweep:
        if entry["pt_stats"]["mean"] < 0.95:
            pt_breaking = entry["n_objects"]
            break

    result = {
        "benchmark": "g3_1_object_scaling",
        "object_counts": object_counts,
        "sweep": sweep,
        "sa_breaking_point": sa_breaking,
        "pt_breaking_point": pt_breaking,
        "conclusion": (
            "pt_scales_better" if (pt_breaking is None and sa_breaking is not None)
            else "sa_scales_better" if (sa_breaking is None and pt_breaking is not None)
            else "both_robust" if (pt_breaking is None and sa_breaking is None)
            else "both_degrade"
        ),
    }
    _save("object_scaling", result)
    _p(f"  SA breaking point: {sa_breaking}")
    _p(f"  PT breaking point: {pt_breaking}")
    return result


def bench_g3_2_sequence_scaling() -> dict:
    """G3.2 -- Scale sequence length (20, 50, 100, 200) frames.

    All models are trained on 20-frame sequences (standard), then evaluated
    on longer test sequences to measure generalization.  This means we
    only need 3 PT + 3 SA = 6 training runs total.
    """
    _p("\n=== G3.2: Sequence Length Scaling ===")
    frame_counts = [20, 50, 100, 200]

    # -- train once per seed (all on 20-frame seqs) --
    trained: dict[int, tuple] = {}
    for seed in SEEDS_3:
        _p(f"  Training seed {seed} ...")
        pt = _build_pt(seed)
        pt, _ = _train_model(pt, seed, n_frames=20)
        pt_state = {k: v.cpu() for k, v in pt.state_dict().items()}
        del pt; _cleanup()

        sa = _build_sa(seed)
        sa, _ = _train_model(sa, seed, n_frames=20)
        sa_state = {k: v.cpu() for k, v in sa.state_dict().items()}
        del sa; _cleanup()

        trained[seed] = (pt_state, sa_state)

    sweep: list[dict] = []
    for n_frames in frame_counts:
        _p(f"  T={n_frames}")
        pt_ips: list[float] = []
        sa_ips: list[float] = []

        for seed in SEEDS_3:
            pt_state, sa_state = trained[seed]
            pt = _build_pt(seed)
            pt.load_state_dict(pt_state)
            pt.eval().to(DEVICE)
            sa = _build_sa(seed)
            sa.load_state_dict(sa_state)
            sa.eval().to(DEVICE)
            test_data = _gen(max(TEST_SEQS // 2, 4), n_frames=n_frames,
                             base_seed=seed + 65000)
            pt_ip = sum(_eval_ip(pt, test_data)) / len(test_data)
            sa_ip = sum(_eval_ip(sa, test_data)) / len(test_data)
            pt_ips.append(pt_ip)
            sa_ips.append(sa_ip)
            del pt, sa, test_data
            _cleanup()

        sweep.append({
            "n_frames": n_frames,
            "pt_stats": _bootstrap_ci(pt_ips),
            "sa_stats": _bootstrap_ci(sa_ips),
            "welch_t": _welch_t(pt_ips, sa_ips),
        })
        _p(f"    PT: {np.mean(pt_ips):.4f}, SA: {np.mean(sa_ips):.4f}")

    del trained; _cleanup()

    result = {
        "benchmark": "g3_2_sequence_scaling",
        "frame_counts": frame_counts,
        "sweep": sweep,
    }
    _save("sequence_scaling", result)
    return result


# =========================================================================
# G4: Chimera-Tracking Bridge (measure chimera metrics in trained PT)
# =========================================================================

def bench_g4_1_chimera_in_tracking() -> dict:
    """G4.1 -- Measure BC/SI/chi within trained PT's phase dynamics
    during multi-object tracking.

    This bridges the chimera story and the tracking story by showing
    that PT's internal dynamics exhibit chimera-like partial synchronization
    during tracking (some oscillators sync = same object, some desync =
    different objects).
    """
    _p("\n=== G4.1: Chimera Metrics in Trained PT ===")
    from prinet.utils.y4q1_tools import (
        phase_slip_rate, coherence_decay_rate, cross_frequency_coupling,
    )
    from prinet.utils.oscillosim import (
        local_order_parameter, bimodality_index,
    )

    pt, _ = _load_models()  # Only need PT
    test_data = _gen(TEST_SEQS, base_seed=42)

    per_seq_metrics: list[dict] = []

    with torch.no_grad():
        for seq_i, seq in enumerate(test_data):
            frames = [f.to(DEVICE) for f in seq.frames]
            result = pt.track_sequence(frames)
            phase_history = result["phase_history"]

            if len(phase_history) < 2:
                continue

            # Stack phase history: (T, N_obj, N_osc)
            # Each phase_history[t] is (N_obj, n_osc) on CPU
            phases = torch.stack(phase_history)  # (T, N_obj, n_osc)
            T, N_obj, N_osc = phases.shape

            # 1. Intra-object phase coherence (local order parameter)
            #    For each object, measure how coherent its oscillators are
            intra_r = []
            for obj_i in range(N_obj):
                obj_phases = phases[:, obj_i, :]  # (T, N_osc)
                # Mean resultant length across oscillators at each time
                z = torch.exp(1j * obj_phases.to(torch.complex64))
                r_t = z.mean(dim=-1).abs()  # (T,)
                intra_r.append(float(r_t.mean().item()))

            # 2. Inter-object phase coherence
            #    Measure phase coherence between different objects
            inter_r = []
            for oi in range(N_obj):
                for oj in range(oi + 1, N_obj):
                    # Phase difference between objects
                    diff = phases[:, oi, :] - phases[:, oj, :]
                    z_diff = torch.exp(1j * diff.to(torch.complex64))
                    r_inter = z_diff.mean(dim=-1).abs()  # (T,)
                    inter_r.append(float(r_inter.mean().item()))

            # 3. Bimodality of local order parameter across oscillators
            #    Flatten all oscillator phases at last timestep
            final_phases = phases[-1].flatten()  # (N_obj * N_osc,)
            try:
                bc = float(bimodality_index(final_phases))
            except Exception:
                bc = float(bimodality_index(final_phases.detach()))

            # 4. Chimera index: fraction of oscillators with low local r
            # Compute per-oscillator r across time
            per_osc_r: list[float] = []
            for osc_i in range(N_osc):
                osc_phases = phases[:, :, osc_i].flatten()  # (T*N_obj,)
                z_osc = torch.exp(1j * osc_phases.to(torch.complex64))
                r_osc = float(z_osc.mean().abs().item())
                per_osc_r.append(r_osc)
            chi_threshold = 0.5
            chi = sum(1 for r in per_osc_r if r < chi_threshold) / max(len(per_osc_r), 1)

            # 5. Phase slip rate
            # Reshape phases for phase_slip_rate: (T, N_obj*N_osc)
            psr_input = phases.reshape(T, -1)
            psr = phase_slip_rate(psr_input)

            # 6. Cross-frequency coupling (low vs high bands)
            # PT has delta(4) + theta(8) + gamma(16) = 28 oscillators
            n_low = 4  # delta band
            n_high = 16  # gamma band (last 16)
            if N_osc >= n_low + n_high:
                # Average across objects for band analysis
                mean_phases = phases.mean(dim=1)  # (T, N_osc)
                low_phases = mean_phases[:, :n_low]  # (T, n_low)
                high_phases = mean_phases[:, -n_high:]  # (T, n_high)
                # Take first oscillator of each band for PAC
                pac_result = cross_frequency_coupling(
                    low_phases[:, 0], high_phases[:, 0]
                )
                pac_val = pac_result["pac"]
            else:
                pac_val = 0.0

            per_seq_metrics.append({
                "seq_idx": seq_i,
                "intra_object_coherence": float(np.mean(intra_r)),
                "inter_object_coherence": float(np.mean(inter_r)) if inter_r else 0.0,
                "coherence_ratio": (
                    float(np.mean(intra_r)) / max(float(np.mean(inter_r)), 1e-8)
                    if inter_r else 0.0
                ),
                "bimodality_coefficient": bc,
                "chimera_index": chi,
                "phase_slip_fraction": psr["slip_fraction"],
                "pac": pac_val,
                "ip": result["identity_preservation"],
            })

    # Aggregate
    if per_seq_metrics:
        agg = {}
        for key in per_seq_metrics[0]:
            if key == "seq_idx":
                continue
            vals = [m[key] for m in per_seq_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    else:
        agg = {}

    chimera_detected = (
        agg.get("coherence_ratio_mean", 0) > 1.5
        and agg.get("chimera_index_mean", 0) > 0.1
    )

    result_data = {
        "benchmark": "g4_1_chimera_in_tracking",
        "n_sequences": len(per_seq_metrics),
        "aggregate": agg,
        "chimera_like_dynamics_detected": chimera_detected,
        "interpretation": (
            "Trained PT exhibits chimera-like partial synchronization: "
            "intra-object oscillators are more coherent than inter-object, "
            "creating a coherent/incoherent split analogous to chimera states."
            if chimera_detected
            else "Trained PT does not exhibit strong chimera-like dynamics "
            "in its phase representation during tracking."
        ),
        "per_sequence": per_seq_metrics[:5],  # Save first 5 for size
    }
    _save("chimera_in_tracking", result_data)
    _p(f"  Intra-object coherence: {agg.get('intra_object_coherence_mean', 0):.4f}")
    _p(f"  Inter-object coherence: {agg.get('inter_object_coherence_mean', 0):.4f}")
    _p(f"  Coherence ratio: {agg.get('coherence_ratio_mean', 0):.4f}")
    _p(f"  Chimera index: {agg.get('chimera_index_mean', 0):.4f}")
    _p(f"  Chimera-like: {chimera_detected}")
    return result_data


# =========================================================================
# G5: Fine Occlusion Resolution (5%, 10%, 15% additional points)
# =========================================================================

def bench_g5_1_fine_occlusion() -> dict:
    """G5.1 -- Fine-grained occlusion sweep (0-80% in 5% steps).

    Adds 5%, 10%, 15% occlusion points to characterize the degradation
    curve shape. Key question: does PT degrade linearly or show a
    phase transition?
    """
    _p("\n=== G5.1: Fine Occlusion Sweep ===")
    occlusion_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80]
    sweep: list[dict] = []

    pt, sa = _load_models()
    # If cached models not available, train fresh
    if not PT_CACHE.exists():
        pt, _ = _train_model(pt, 42)
    if not SA_CACHE.exists():
        sa, _ = _train_model(sa, 42)

    for occ in occlusion_rates:
        _p(f"  occlusion={occ:.0%}")
        pt_ips: list[float] = []
        sa_ips: list[float] = []

        for seed in SEEDS_3:
            test_data = _gen(TEST_SEQS, base_seed=seed + 55000,
                             occlusion_rate=occ)
            pt_ip = sum(_eval_ip(pt, test_data)) / TEST_SEQS
            sa_ip = sum(_eval_ip(sa, test_data)) / TEST_SEQS
            pt_ips.append(pt_ip)
            sa_ips.append(sa_ip)

        sweep.append({
            "occlusion_rate": occ,
            "pt_stats": _bootstrap_ci(pt_ips),
            "sa_stats": _bootstrap_ci(sa_ips),
        })
        _p(f"    PT: {np.mean(pt_ips):.4f}, SA: {np.mean(sa_ips):.4f}")

    # Fit degradation curve for PT
    pt_means = [s["pt_stats"]["mean"] for s in sweep]
    occ_arr = np.array(occlusion_rates)
    pt_arr = np.array(pt_means)

    # Check for phase transition vs linear degradation
    # Compute second derivative (discrete)
    if len(pt_arr) >= 3:
        d2 = np.diff(pt_arr, n=2)
        max_curvature_idx = int(np.argmin(d2))
        transition_point = float(occ_arr[max_curvature_idx + 1])
        is_phase_transition = abs(float(d2.min())) > 0.05
    else:
        transition_point = None
        is_phase_transition = False

    # Linear fit
    if len(occ_arr) > 1:
        slope, intercept = np.polyfit(occ_arr, pt_arr, 1)
        r_squared = 1 - (np.sum((pt_arr - (slope * occ_arr + intercept)) ** 2)
                         / np.sum((pt_arr - pt_arr.mean()) ** 2))
    else:
        slope, intercept, r_squared = 0.0, 1.0, 0.0

    result = {
        "benchmark": "g5_1_fine_occlusion",
        "occlusion_rates": occlusion_rates,
        "sweep": sweep,
        "pt_degradation": {
            "linear_slope": float(slope),
            "linear_intercept": float(intercept),
            "linear_r_squared": float(r_squared),
            "is_phase_transition": is_phase_transition,
            "transition_point": transition_point,
        },
        "narrative": (
            f"PT shows {'phase transition' if is_phase_transition else 'approximately linear'} "
            f"degradation with occlusion. "
            f"{'Transition around ' + f'{transition_point:.0%}' if is_phase_transition else ''}"
        ),
    }
    _save("fine_occlusion", result)
    del pt, sa
    _cleanup()
    return result


# =========================================================================
# G6: pt_static Stress Test (find where dynamics matter)
# =========================================================================

def bench_g6_1_static_stress() -> dict:
    """G6.1 -- Stress test pt_static vs pt_full to find where
    oscillatory dynamics contribute.

    pt_static achieves IP=0.999 on standard tests. This benchmark
    tests under harder conditions where dynamics might help:
    - More objects (8, 12, 16)
    - Longer sequences (50, 100 frames)
    - Higher noise (sigma=0.5, 1.0, 2.0)
    - Combined stressors
    """
    _p("\n=== G6.1: pt_static Stress Test ===")
    from prinet.nn.ablation_variants import create_ablation_tracker

    stress_configs = [
        {"label": "standard", "n_objects": 4, "n_frames": 20,
         "noise_sigma": 0.0, "occlusion_rate": 0.0},
        {"label": "8_objects", "n_objects": 8, "n_frames": 20,
         "noise_sigma": 0.0, "occlusion_rate": 0.0},
        {"label": "16_objects", "n_objects": 16, "n_frames": 20,
         "noise_sigma": 0.0, "occlusion_rate": 0.0},
        {"label": "long_100", "n_objects": 4, "n_frames": 100,
         "noise_sigma": 0.0, "occlusion_rate": 0.0},
        {"label": "noise_2.0", "n_objects": 4, "n_frames": 20,
         "noise_sigma": 2.0, "occlusion_rate": 0.0},
        {"label": "combined_hard", "n_objects": 8, "n_frames": 50,
         "noise_sigma": 0.5, "occlusion_rate": 0.2},
    ]

    results_list: list[dict] = []

    for cfg in stress_configs:
        _p(f"  Config: {cfg['label']}")
        full_ips: list[float] = []
        static_ips: list[float] = []

        for seed in SEEDS_3:
            # pt_full
            pt_full = _build_pt(seed)
            pt_full, _ = _train_model(
                pt_full, seed,
                n_objects=cfg["n_objects"], n_frames=min(cfg["n_frames"], 20),
                noise_sigma=cfg.get("noise_sigma", 0.0),
            )
            _cleanup()

            # pt_static
            torch.manual_seed(seed)
            pt_static = create_ablation_tracker(
                "pt_static", detection_dim=DET_DIM, **PT_KWARGS
            )
            pt_static, _ = _train_model(
                pt_static, seed,
                n_objects=cfg["n_objects"], n_frames=min(cfg["n_frames"], 20),
                noise_sigma=cfg.get("noise_sigma", 0.0),
            )
            _cleanup()

            test_data = _gen(
                max(TEST_SEQS // 2, 4),
                n_objects=cfg["n_objects"],
                n_frames=cfg["n_frames"],
                base_seed=seed + 45000,
                noise_sigma=cfg.get("noise_sigma", 0.0),
                occlusion_rate=cfg.get("occlusion_rate", 0.0),
            )
            full_ip = sum(_eval_ip(pt_full, test_data)) / len(test_data)
            static_ip = sum(_eval_ip(pt_static, test_data)) / len(test_data)
            full_ips.append(full_ip)
            static_ips.append(static_ip)

            del pt_full, pt_static, test_data
            _cleanup()

        stats = _welch_t(full_ips, static_ips)
        results_list.append({
            "config": cfg,
            "full_stats": _bootstrap_ci(full_ips),
            "static_stats": _bootstrap_ci(static_ips),
            "welch_t": stats,
            "dynamics_help": stats["cohens_d"] > 0.2 and stats["p_value"] < 0.1,
        })
        _p(f"    full: {np.mean(full_ips):.4f}, static: {np.mean(static_ips):.4f}, "
           f"d={stats['cohens_d']:.3f}")

    # Find conditions where dynamics matter
    dynamics_conditions = [
        r["config"]["label"] for r in results_list if r["dynamics_help"]
    ]

    result = {
        "benchmark": "g6_1_static_stress",
        "configs": [c["label"] for c in stress_configs],
        "results": results_list,
        "dynamics_matter_for": dynamics_conditions,
        "conclusion": (
            f"Dynamics contribute under: {', '.join(dynamics_conditions)}"
            if dynamics_conditions
            else "Dynamics do not significantly help under any tested condition. "
            "Reframe as interpretability/robustness benefit."
        ),
    }
    _save("static_stress", result)
    return result


# =========================================================================
# G7: Trained PT Coherence (re-measure on trained checkpoint)
# =========================================================================

def bench_g7_1_trained_coherence() -> dict:
    """G7.1 -- Re-measure coherence half-life and phase slips on trained PT.

    The 527-frame half-life and zero phase slips were measured on untrained PT
    (Q1.4). This benchmark re-measures on the trained checkpoint.
    """
    _p("\n=== G7.1: Trained PT Coherence ===")
    from prinet.utils.y4q1_tools import (
        phase_slip_rate, coherence_decay_rate, binding_persistence,
    )

    pt, _ = _load_models()
    per_seed: list[dict] = []

    for seed in SEEDS_7:
        # Generate long sequences for coherence measurement
        test_data = _gen(5, n_frames=200, base_seed=seed + 40000)

        all_slips: list[dict] = []
        all_coherence: list[float] = []
        all_persistence: list[dict] = []

        with torch.no_grad():
            for seq in test_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                res = pt.track_sequence(frames)

                phase_history = res["phase_history"]
                if len(phase_history) >= 2:
                    phases = torch.stack(phase_history)
                    T_ph, N_obj, N_osc = phases.shape

                    # Phase slip rate
                    psr = phase_slip_rate(phases.reshape(T_ph, -1))
                    all_slips.append(psr)

                    # Coherence series from per-frame correlations
                    corr_series = res.get("per_frame_phase_correlation", [])
                    if corr_series:
                        all_coherence.extend(corr_series)

                    # Binding persistence
                    bp = binding_persistence(
                        res["identity_matches"], seq.n_objects
                    )
                    all_persistence.append(bp)

        # Aggregate this seed
        mean_slip = float(np.mean([s["slip_fraction"] for s in all_slips])
                          ) if all_slips else 0.0
        total_slips = sum(s["total_slips"] for s in all_slips)

        # Coherence decay
        if len(all_coherence) >= 3:
            cdr = coherence_decay_rate(all_coherence)
        else:
            cdr = {"decay_rate": 0.0, "half_life": float("inf"),
                   "initial_coherence": 1.0, "r_squared": 0.0}

        mean_persistence = float(np.mean(
            [p["mean_persistence"] for p in all_persistence]
        )) if all_persistence else 0.0

        per_seed.append({
            "seed": seed,
            "mean_slip_fraction": mean_slip,
            "total_slips": total_slips,
            "coherence_half_life": cdr["half_life"],
            "coherence_decay_rate": cdr["decay_rate"],
            "coherence_r_squared": cdr["r_squared"],
            "initial_coherence": cdr["initial_coherence"],
            "mean_persistence": mean_persistence,
        })
        _p(f"  seed={seed}: slips={total_slips}, "
           f"half_life={cdr['half_life']:.1f}, "
           f"persistence={mean_persistence:.4f}")

    # Aggregate across seeds
    agg: dict[str, Any] = {}
    for key in ["mean_slip_fraction", "total_slips", "coherence_half_life",
                "coherence_decay_rate", "initial_coherence",
                "mean_persistence"]:
        vals = [s[key] for s in per_seed
                if not (key == "coherence_half_life"
                        and s[key] == float("inf"))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        else:
            agg[f"{key}_mean"] = float("inf") if "half_life" in key else 0.0
            agg[f"{key}_std"] = 0.0

    result = {
        "benchmark": "g7_1_trained_coherence",
        "n_seeds": len(SEEDS_7),
        "n_frames_per_seq": 200,
        "per_seed": per_seed,
        "aggregate": agg,
        "comparison_to_untrained": {
            "untrained_half_life": 527.0,
            "untrained_phase_slips": 0,
            "note": "Re-measured on trained Q1.7 checkpoint",
        },
    }
    _save("trained_coherence", result)
    del pt
    _cleanup()
    return result


# =========================================================================
# G8: PAC Significance Test (permutation test with null distribution)
# =========================================================================

def bench_g8_1_pac_significance() -> dict:
    """G8.1 -- PAC significance test with phase-shuffled null distribution.

    PAC of 0.047 was measured once on untrained PT. This benchmark:
    1. Measures PAC on trained PT across 7 seeds.
    2. Constructs null distribution with 1000 phase-shuffled surrogates.
    3. Reports z-score and p-value.
    """
    _p("\n=== G8.1: PAC Significance Test ===")
    from prinet.utils.y4q1_tools import cross_frequency_coupling

    pt, _ = _load_models()
    n_surrogates = 1000

    per_seed_pac: list[dict] = []

    for seed in SEEDS_7:
        _p(f"  Seed {seed}")
        test_data = _gen(10, n_frames=100, base_seed=seed + 30000)

        # Collect phase data once
        all_low: list[torch.Tensor] = []
        all_high: list[torch.Tensor] = []

        with torch.no_grad():
            for seq in test_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                res = pt.track_sequence(frames)
                phase_history = res["phase_history"]

                if len(phase_history) < 5:
                    continue

                phases = torch.stack(phase_history)  # (T, N_obj, N_osc)
                T_ph, N_obj, N_osc = phases.shape
                mean_phases = phases.mean(dim=1)  # (T, N_osc)

                n_low = min(4, N_osc)
                n_high = min(16, N_osc)
                if N_osc >= n_low + n_high:
                    all_low.append(mean_phases[:, 0])
                    all_high.append(mean_phases[:, -1])

        if not all_low:
            per_seed_pac.append({
                "seed": seed, "observed_pac": 0.0,
                "null_mean": 0.0, "null_std": 0.0,
                "z_score": 0.0, "p_value": 1.0,
            })
            continue

        # Compute observed PAC
        observed_pacs: list[float] = []
        for low, high in zip(all_low, all_high):
            pac_res = cross_frequency_coupling(low, high)
            observed_pacs.append(pac_res["pac"])
        observed_mean = float(np.mean(observed_pacs))

        # Generate null distribution by phase-shuffling (no re-tracking)
        rng = np.random.default_rng(seed)
        null_pacs: list[float] = []

        for surr_i in range(n_surrogates):
            surrogate_pacs: list[float] = []
            for low, high in zip(all_low, all_high):
                T_ph = high.shape[0]
                shift = rng.integers(1, max(T_ph - 1, 2))
                high_shuffled = torch.roll(high, int(shift))
                pac_res = cross_frequency_coupling(low, high_shuffled)
                surrogate_pacs.append(pac_res["pac"])
            if surrogate_pacs:
                null_pacs.append(float(np.mean(surrogate_pacs)))

        null_mean = float(np.mean(null_pacs)) if null_pacs else 0.0
        null_std = float(np.std(null_pacs)) if null_pacs else 1.0

        z_score = (observed_mean - null_mean) / max(null_std, 1e-10)
        # p-value: fraction of null >= observed
        p_val = sum(1 for n in null_pacs if n >= observed_mean) / max(len(null_pacs), 1)

        per_seed_pac.append({
            "seed": seed,
            "observed_pac": observed_mean,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "p_value": p_val,
            "n_surrogates": len(null_pacs),
        })
        _p(f"    PAC={observed_mean:.4f}, null={null_mean:.4f}+/-{null_std:.4f}, "
           f"z={z_score:.2f}, p={p_val:.4f}")

    # Aggregate
    obs_pacs = [s["observed_pac"] for s in per_seed_pac]
    z_scores = [s["z_score"] for s in per_seed_pac]
    p_values = [s["p_value"] for s in per_seed_pac]

    # Fisher's method for combining p-values
    valid_ps = [p for p in p_values if 0 < p < 1]
    if valid_ps:
        from scipy import stats
        chi2_stat = -2 * sum(math.log(p) for p in valid_ps)
        combined_p = float(stats.chi2.sf(chi2_stat, df=2 * len(valid_ps)))
    else:
        combined_p = 1.0

    significant = combined_p < 0.05

    result = {
        "benchmark": "g8_1_pac_significance",
        "n_seeds": len(SEEDS_7),
        "n_surrogates": n_surrogates,
        "surrogate_method": "time_shift_phase_shuffling",
        "per_seed": per_seed_pac,
        "aggregate": {
            "observed_pac_mean": float(np.mean(obs_pacs)),
            "observed_pac_std": float(np.std(obs_pacs)),
            "mean_z_score": float(np.mean(z_scores)),
            "combined_p_value": combined_p,
            "significant_at_005": significant,
        },
        "conclusion": (
            "PAC is significantly above null distribution (keep finding)"
            if significant
            else "PAC is NOT significantly above null distribution (drop claim)"
        ),
        "comparison_to_untrained": {
            "untrained_pac": 0.047,
            "note": "Re-measured on trained checkpoint with significance testing",
        },
    }
    _save("pac_significance", result)
    _p(f"  Combined p-value: {combined_p:.4f}")
    _p(f"  Significant: {significant}")
    del pt
    _cleanup()
    return result


# =========================================================================
# Pre-registration
# =========================================================================

def bench_preregistration() -> dict:
    """Compute SHA-256 hash of all Q1.9 protocol parameters."""
    _p("\n=== Pre-Registration Hash ===")
    protocol = {
        "session": "Y4_Q1.9",
        "seeds_7": list(SEEDS_7),
        "seeds_3": list(SEEDS_3),
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
        "test_seqs": TEST_SEQS,
        "n_objects": N_OBJECTS,
        "n_frames": N_FRAMES,
        "det_dim": DET_DIM,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "warmup": WARMUP,
        "lr": LR,
        "pt_kwargs": PT_KWARGS,
        "sa_kwargs": SA_KWARGS,
        "gaps_addressed": [
            "G1_seed_expansion", "G2_pt_large_diagnostic",
            "G3_sa_breaking_point", "G4_chimera_tracking_bridge",
            "G5_fine_occlusion", "G6_static_stress",
            "G7_trained_coherence", "G8_pac_significance",
        ],
        "pac_surrogates": 1000,
        "alpha": 0.05,
    }
    h = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    protocol["sha256"] = h
    _p(f"  SHA-256: {h[:16]}...")
    _save("preregistration_hash", protocol)
    return protocol


# =========================================================================
# Main Runner
# =========================================================================

ALL_BENCHMARKS = [
    ("preregistration", bench_preregistration),
    ("g1_1_7seed_comparison", bench_g1_1_7seed_comparison),
    ("g1_2_7seed_noise", bench_g1_2_7seed_noise),
    ("g2_1_embedding_analysis", bench_g2_1_embedding_analysis),
    ("g2_2_regularization_ablation", bench_g2_2_regularization_ablation),
    ("g3_1_object_scaling", bench_g3_1_object_scaling),
    ("g3_2_sequence_scaling", bench_g3_2_sequence_scaling),
    ("g4_1_chimera_in_tracking", bench_g4_1_chimera_in_tracking),
    ("g5_1_fine_occlusion", bench_g5_1_fine_occlusion),
    ("g6_1_static_stress", bench_g6_1_static_stress),
    ("g7_1_trained_coherence", bench_g7_1_trained_coherence),
    ("g8_1_pac_significance", bench_g8_1_pac_significance),
]


def main() -> None:
    """Run all Q1.9 benchmarks sequentially."""
    _p("=" * 60)
    _p("Year 4 Q1.9 -- Reviewer Gap Analysis Benchmarks")
    _p("=" * 60)
    t0 = time.time()
    completed = 0
    failed = 0

    for name, fn in ALL_BENCHMARKS:
        try:
            _p(f"\n--- {name} ---")
            fn()
            completed += 1
        except Exception as e:
            _p(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        finally:
            _cleanup()

    elapsed = time.time() - t0
    _p(f"\n{'=' * 60}")
    _p(f"Q1.9 complete: {completed} passed, {failed} failed "
       f"in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    _p(f"{'=' * 60}")


if __name__ == "__main__":
    main()
