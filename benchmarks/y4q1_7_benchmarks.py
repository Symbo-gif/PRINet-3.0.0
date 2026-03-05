"""Year 4 Q1.7 Benchmarks -- Definitive Temporal Advantage Protocol (v3 resumable).

Scale calibration (per-benchmark timing on RTX 4060):
    - PT epoch / 20 seqs ~= 5.5 s  -> 50 seqs ~= 13.75 s
    - SA epoch / 20 seqs ~= 3.0 s  -> 50 seqs ~= 7.5 s
    - Full run estimate: ~50-60 min (3 seeds, 20 max epochs)

Resume capability:
    - State dicts are cached to results/y4q1_7_pt_best.pt and sa_best.pt
    - JSON files are skipped if they already exist (--resume mode default)
    - Re-run with FORCE_RERUN=True to overwrite everything

All print() calls use ASCII only (no Unicode) to avoid cp1252 encoding errors.

Usage:
    python benchmarks/y4q1_7_benchmarks.py
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

# =========================================================================
# Configuration
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

TRAIN_SEQS     = 50
VAL_SEQS       = 10
TEST_SEQS      = 20
TEST_LONG_SEQS = 8
TEST_OCC_SEQS  = 8
TEST_SWAP_SEQS = 8
N_OBJECTS      = 4
N_FRAMES_TRAIN = 20
N_FRAMES_LONG  = 80
DET_DIM        = 4
MAX_EPOCHS     = 20
PATIENCE       = 5
WARMUP         = 2
SEEDS          = (42, 123, 456)
LR             = 3e-4

FORCE_RERUN = False  # set True to overwrite existing JSON artefacts

PT_KWARGS = dict(n_delta=4, n_theta=8, n_gamma=16,
                 n_discrete_steps=5, match_threshold=0.1)
SA_KWARGS = dict(num_slots=6, slot_dim=64, num_iterations=3,
                 match_threshold=0.1)

PT_CACHE = RESULTS_DIR / "y4q1_7_pt_best.pt"
SA_CACHE = RESULTS_DIR / "y4q1_7_sa_best.pt"


# =========================================================================
# Utilities
# =========================================================================

def _p(*args, **kwargs):
    """print with flush=True for immediate log output."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict) -> bool:
    """Save JSON; returns False if already exists and FORCE_RERUN is off."""
    path = RESULTS_DIR / f"y4q1_7_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (already exists)")
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _p(f"  -> {path.name}")
    return True


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_pt(seed: int = 42):
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(detection_dim=DET_DIM, **PT_KWARGS)


def _build_sa(seed: int = 42):
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(detection_dim=DET_DIM, **SA_KWARGS)


def _build_variant(variant: str, seed: int = 42):
    from prinet.nn.ablation_variants import create_ablation_tracker
    torch.manual_seed(seed)
    kw: dict[str, Any] = {}
    if variant.startswith("pt"):
        kw.update(PT_KWARGS)
    else:
        kw.update(SA_KWARGS)
    return create_ablation_tracker(variant, detection_dim=DET_DIM, **kw)


def _gen(n: int, n_frames: int = N_FRAMES_TRAIN, **kw):
    from prinet.utils.temporal_training import generate_dataset
    return generate_dataset(n, n_objects=N_OBJECTS, n_frames=n_frames,
                            det_dim=DET_DIM, **kw)


def _eval_metrics(model, dataset, device: str = DEVICE) -> dict:
    from prinet.utils.temporal_metrics import (
        identity_switches, track_fragmentation_rate,
        identity_overcount, mostly_tracked_lost, track_duration_stats,
    )
    model.eval()
    model = model.to(device)
    rows: list[dict] = []
    with torch.no_grad():
        for seq in dataset:
            frames = [f.to(device) for f in seq.frames]
            res = model.track_sequence(frames)
            m = res["identity_matches"]
            ip = res["identity_preservation"]
            mt, ml = mostly_tracked_lost(m, seq.n_objects)
            dur_mean, dur_med = track_duration_stats(m, seq.n_objects)
            rows.append({
                "ip": ip, "idsw": identity_switches(m, seq.n_objects),
                "tfr": track_fragmentation_rate(m, seq.n_objects),
                "ioc": identity_overcount(m, seq.n_objects),
                "mt": mt, "ml": ml,
                "mean_dur": dur_mean, "median_dur": dur_med,
            })
    if not rows:
        return {}
    n = len(rows)
    agg: dict[str, float] = {}
    for k in rows[0]:
        vs = [r[k] for r in rows]
        mu = sum(vs) / n
        sd = (sum((v - mu) ** 2 for v in vs) / max(n - 1, 1)) ** 0.5
        agg[f"{k}_mean"] = mu
        agg[f"{k}_std"]  = sd
    return agg


def _tr_dict(tr) -> dict:
    return {
        "final_train_loss": tr.final_train_loss,
        "final_val_loss":   tr.final_val_loss,
        "final_val_ip":     tr.final_val_ip,
        "best_val_loss":    tr.best_val_loss,
        "best_epoch":       tr.best_epoch,
        "total_epochs":     tr.total_epochs,
        "wall_time_s":      tr.wall_time_s,
        "train_losses":     tr.train_losses,
        "val_losses":       tr.val_losses,
        "val_ips":          tr.val_ips,
        "snapshots": [{
            "epoch":           s.epoch,
            "train_loss":      s.train_loss,
            "val_loss":        s.val_loss,
            "val_ip":          s.val_ip,
            "val_idsw":        s.val_idsw,
            "gradient_norm":   s.gradient_norm,
            "param_norm":      s.param_norm,
            "phase_coherence": s.phase_coherence,
            "slot_entropy":    s.slot_entropy,
        } for s in tr.snapshots],
    }


def _ms_dict(ms) -> dict:
    return {
        "model_name":     ms.model_name,
        "seeds":          ms.seeds,
        "mean_ip":        ms.mean_ip,
        "std_ip":         ms.std_ip,
        "mean_epochs":    ms.mean_epochs,
        "mean_wall_time": ms.mean_wall_time,
        "per_seed":       [_tr_dict(r) for r in ms.per_seed],
    }


def _load_or_train(name, factory, cache_path, train_data, val_data) -> tuple:
    """Load cached state dict if available; otherwise train all seeds.

    Returns (MultiSeedResult, state_dict).
    """
    from prinet.utils.temporal_training import TemporalTrainer, MultiSeedResult

    _name_slug = {"PhaseTracker": "pt", "SlotAttention": "sa"}.get(name,
                  name.lower().replace(" ", "_"))
    json_path = RESULTS_DIR / f"y4q1_7_training_curves_{_name_slug}.json"
    if cache_path.exists() and json_path.exists() and not FORCE_RERUN:
        _p(f"  [resume] Loading cached {name} state dict from {cache_path.name}")
        state = torch.load(str(cache_path), map_location=DEVICE, weights_only=True)
        # Reconstruct MultiSeedResult from JSON
        with open(json_path) as f:
            d = json.load(f)
        # Minimal reconstruction for downstream consumers
        per_seed_fake = []
        for ps in d.get("per_seed", []):
            class _FakeTr:
                pass
            tr = _FakeTr()
            tr.final_train_loss = ps.get("final_train_loss", 0.0)
            tr.final_val_loss   = ps.get("final_val_loss", 0.0)
            tr.final_val_ip     = ps.get("final_val_ip", 0.0)
            tr.best_val_loss    = ps.get("best_val_loss", 0.0)
            tr.best_epoch       = ps.get("best_epoch", 0)
            tr.total_epochs     = ps.get("total_epochs", 0)
            tr.wall_time_s      = ps.get("wall_time_s", 0.0)
            tr.train_losses     = ps.get("train_losses", [])
            tr.val_losses       = ps.get("val_losses", [])
            tr.val_ips          = ps.get("val_ips", [])

            class _FakeSnap:
                pass
            snaps = []
            for s in ps.get("snapshots", []):
                sn = _FakeSnap()
                for k, v in s.items():
                    setattr(sn, k, v)
                snaps.append(sn)
            tr.snapshots = snaps
            per_seed_fake.append(tr)

        ms = MultiSeedResult(
            model_name=name,
            seeds=list(d.get("seeds", SEEDS)),
            mean_ip=d.get("mean_ip", 0.0),
            std_ip=d.get("std_ip", 0.0),
            mean_epochs=d.get("mean_epochs", 0.0),
            mean_wall_time=d.get("mean_wall_time", 0.0),
            per_seed=per_seed_fake,
        )
        _p(f"  Loaded {name}: mean_ip={ms.mean_ip:.4f}")
        return ms, state

    _p(f"\n  Training {name} ({len(SEEDS)} seeds, max={MAX_EPOCHS} ep) ...")
    per_seed = []
    best_loss  = float("inf")
    best_state = None
    wall_ips: list[float] = []
    wall_epochs: list[int] = []
    wall_times: list[float] = []

    for i, seed in enumerate(SEEDS):
        _p(f"  [{i+1}/{len(SEEDS)}]", end=" ")
        m = factory(seed)
        t0 = time.perf_counter()
        tr = TemporalTrainer(m, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                             warmup_epochs=WARMUP, device=DEVICE, seed=seed
                             ).train(train_data, val_data)
        dt = time.perf_counter() - t0
        _p(f"seed={seed}: ip={tr.final_val_ip:.4f}  ep={tr.total_epochs}  t={dt:.0f}s")
        per_seed.append(tr)
        wall_ips.append(tr.final_val_ip)
        wall_epochs.append(tr.total_epochs)
        wall_times.append(tr.wall_time_s)
        if tr.best_val_loss < best_loss:
            best_loss  = tr.best_val_loss
            best_state = copy.deepcopy(m.state_dict())
        del m
        _cleanup()

    n  = len(SEEDS)
    mu = sum(wall_ips) / n
    sd = (sum((v - mu)**2 for v in wall_ips) / max(n - 1, 1))**0.5
    ms = MultiSeedResult(
        model_name=name, seeds=list(SEEDS),
        mean_ip=mu, std_ip=sd,
        mean_epochs=sum(wall_epochs) / n,
        mean_wall_time=sum(wall_times) / n,
        per_seed=per_seed,
    )
    _p(f"  {name} mean_ip={mu:.4f} +/- {sd:.4f}")

    # Cache state dict
    torch.save(best_state, str(cache_path))
    _p(f"  Cached state dict -> {cache_path.name}")

    return ms, best_state


# =========================================================================
# Benchmark 20: Pre-registration hash
# =========================================================================

def bench_preregistration() -> str:
    _p("\n=== Benchmark 20: Pre-registration Hash ===")
    plan = (Path(__file__).parent.parent / "Docs" / "Planning_Documentation"
            / "Y4_Q1_7_Temporal_Advantage_Benchmark_Plan.md")
    sha = hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else "NOT_FOUND"
    _save("preregistration_hash", {
        "benchmark": "preregistration_hash",
        "sha256": sha,
        "plan_exists": plan.exists(),
        "expected": "ad1ed5f16e9fbcdd223b75f049f18e1b9ae436bdb79d7f67478da0e4a6ff2c19",
        "hash_verified": sha == "ad1ed5f16e9fbcdd223b75f049f18e1b9ae436bdb79d7f67478da0e4a6ff2c19",
    })
    _p(f"  SHA-256: {sha[:16]}...  verified: {sha == 'ad1ed5f16e9fbcdd223b75f049f18e1b9ae436bdb79d7f67478da0e4a6ff2c19'}")
    return sha


# =========================================================================
# Benchmark 1: Parameter budget
# =========================================================================

def bench_parameter_budget() -> dict:
    _p("\n=== Benchmark 1: Parameter Budget ===")
    from prinet.utils.temporal_training import count_parameters
    results: dict[str, dict] = {}
    for v in ("pt_full", "pt_frozen", "pt_static",
              "sa_full", "sa_no_gru", "sa_frozen"):
        m = _build_variant(v)
        c = count_parameters(m)
        results[v] = c
        _p(f"  {v:15s}: total={c['total']:,}  "
           f"trainable={c['trainable']:,}  adj={c['complex_adjusted']:,}")
        del m
    ratio = (results["pt_full"]["complex_adjusted"]
             / max(results["sa_full"]["complex_adjusted"], 1))
    data = {"benchmark": "parameter_budget", "variants": results,
            "pt_sa_ratio": ratio, "within_10pct": abs(ratio - 1.0) < 0.1}
    _save("parameter_budget", data)
    return data


# =========================================================================
# Benchmarks 2-3: Training curves (with resume support)
# =========================================================================

def bench_training_curves(train_data, val_data):
    _p("\n=== Benchmarks 2-3: Training Curves (PT + SA) ===")
    pt_ms, pt_state = _load_or_train(
        "PhaseTracker", _build_pt, PT_CACHE, train_data, val_data)
    _save("training_curves_pt",
          {"benchmark": "training_curves_pt", **_ms_dict(pt_ms)})

    sa_ms, sa_state = _load_or_train(
        "SlotAttention", _build_sa, SA_CACHE, train_data, val_data)
    _save("training_curves_sa",
          {"benchmark": "training_curves_sa", **_ms_dict(sa_ms)})

    return pt_ms, sa_ms, pt_state, sa_state


# =========================================================================
# Benchmark 4: Standard comparison
# =========================================================================

def bench_standard(pt_state, sa_state, test_data) -> dict:
    _p("\n=== Benchmark 4: Trained Comparison (Standard 20-frame) ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    pm = _eval_metrics(pt, test_data)
    sm = _eval_metrics(sa, test_data)
    data = {"benchmark": "trained_comparison_standard",
            "n_frames": N_FRAMES_TRAIN, "n_seqs": TEST_SEQS,
            "phase_tracker": pm, "slot_attention": sm,
            "ip_advantage": pm.get("ip_mean", 0) - sm.get("ip_mean", 0)}
    _p(f"  PT IP={pm.get('ip_mean',0):.4f}  SA IP={sm.get('ip_mean',0):.4f}")
    _save("trained_comparison_standard", data)
    del pt, sa; _cleanup()
    return data


# =========================================================================
# Benchmark 5: Long-sequence comparison
# =========================================================================

def bench_long(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 5: Trained Comparison (Long 80-frame) ===")
    td = _gen(TEST_LONG_SEQS, n_frames=N_FRAMES_LONG, base_seed=80000)
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    pm = _eval_metrics(pt, td)
    sm = _eval_metrics(sa, td)
    data = {"benchmark": "trained_comparison_long",
            "n_frames": N_FRAMES_LONG, "n_seqs": TEST_LONG_SEQS,
            "train_n_frames": N_FRAMES_TRAIN,
            "phase_tracker": pm, "slot_attention": sm,
            "ip_advantage": pm.get("ip_mean", 0) - sm.get("ip_mean", 0)}
    _p(f"  PT IP={pm.get('ip_mean',0):.4f}  SA IP={sm.get('ip_mean',0):.4f}")
    _save("trained_comparison_long", data)
    del pt, sa; _cleanup()
    return data


# =========================================================================
# Benchmark 6: Extrapolation
# =========================================================================

def bench_extrapolation(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 6: Sequence Length Extrapolation ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    rows: list[dict] = []
    for nf in (20, 40, 80, 120):
        td = _gen(8, n_frames=nf, base_seed=70000 + nf)
        pm = _eval_metrics(pt, td)
        sm = _eval_metrics(sa, td)
        adv = pm.get("ip_mean", 0) - sm.get("ip_mean", 0)
        rows.append({"n_frames": nf, "pt": pm, "sa": sm, "ip_advantage": adv})
        _p(f"  T={nf:3d}: PT={pm.get('ip_mean',0):.4f}"
           f"  SA={sm.get('ip_mean',0):.4f}  d={adv:+.4f}")
    data = {"benchmark": "extrapolation",
            "train_n_frames": N_FRAMES_TRAIN, "results": rows}
    _save("trained_comparison_extrapolation", data)
    del pt, sa; _cleanup()
    return data


# =========================================================================
# Benchmarks 7-8: Occlusion stress
# =========================================================================

def bench_occlusion(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 7-8: Occlusion Stress ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    pt_rows: list[dict] = []
    sa_rows: list[dict] = []
    for occ in (0.0, 0.2, 0.4, 0.6, 0.8):
        td = _gen(TEST_OCC_SEQS, occlusion_rate=occ, base_seed=60000)
        pm = _eval_metrics(pt, td)
        sm = _eval_metrics(sa, td)
        pt_rows.append({"occlusion_rate": occ, **pm})
        sa_rows.append({"occlusion_rate": occ, **sm})
        _p(f"  occ={occ:.1f}: PT={pm.get('ip_mean',0):.4f}"
           f"  SA={sm.get('ip_mean',0):.4f}")
    _save("occlusion_stress_pt",
          {"benchmark": "occlusion_stress_pt", "results": pt_rows})
    _save("occlusion_stress_sa",
          {"benchmark": "occlusion_stress_sa", "results": sa_rows})
    del pt, sa; _cleanup()
    return {"pt": pt_rows, "sa": sa_rows}


# =========================================================================
# Benchmark 9: Appearance swap stress
# =========================================================================

def bench_swap(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 9: Appearance Swap Stress ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    rows: list[dict] = []
    for sr in (0.0, 0.05, 0.10, 0.20):
        td = _gen(TEST_SWAP_SEQS, swap_rate=sr, base_seed=65000)
        pm = _eval_metrics(pt, td)
        sm = _eval_metrics(sa, td)
        rows.append({"swap_rate": sr, "pt": pm, "sa": sm,
                     "ip_advantage": pm.get("ip_mean", 0) - sm.get("ip_mean", 0)})
        _p(f"  swap={sr:.2f}: PT={pm.get('ip_mean',0):.4f}"
           f"  SA={sm.get('ip_mean',0):.4f}")
    _save("swap_stress", {"benchmark": "swap_stress", "results": rows})
    del pt, sa; _cleanup()
    return {"results": rows}


# =========================================================================
# Benchmark 10: Motion discontinuity
# =========================================================================

def bench_motion(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 10: Motion Discontinuity ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    rows: list[dict] = []
    for rc in (0, 1, 3, 5):
        td = _gen(TEST_SEQS, reversal_count=rc, base_seed=66000)
        pm = _eval_metrics(pt, td)
        sm = _eval_metrics(sa, td)
        rows.append({"reversal_count": rc, "pt": pm, "sa": sm,
                     "ip_advantage": pm.get("ip_mean", 0) - sm.get("ip_mean", 0)})
        _p(f"  rev={rc}: PT={pm.get('ip_mean',0):.4f}"
           f"  SA={sm.get('ip_mean',0):.4f}")
    _save("motion_discontinuity",
          {"benchmark": "motion_discontinuity", "results": rows})
    del pt, sa; _cleanup()
    return {"results": rows}


# =========================================================================
# Benchmark 11: Noise injection
# =========================================================================

def bench_noise(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 11: Noise Injection ===")
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)
    rows: list[dict] = []
    for sig in (0.0, 0.1, 0.3, 0.5):
        td = _gen(TEST_SEQS, noise_sigma=sig, base_seed=67000)
        pm = _eval_metrics(pt, td)
        sm = _eval_metrics(sa, td)
        rows.append({"noise_sigma": sig, "pt": pm, "sa": sm,
                     "ip_advantage": pm.get("ip_mean", 0) - sm.get("ip_mean", 0)})
        _p(f"  sigma={sig:.1f}: PT={pm.get('ip_mean',0):.4f}"
           f"  SA={sm.get('ip_mean',0):.4f}")
    _save("noise_injection",
          {"benchmark": "noise_injection", "results": rows})
    del pt, sa; _cleanup()
    return {"results": rows}


# =========================================================================
# Benchmarks 12-13: Ablation studies
# =========================================================================

def bench_ablation(train_data, val_data, test_data) -> dict:
    from prinet.utils.temporal_training import TemporalTrainer, count_parameters

    def _run_variant(v):
        _p(f"  {v:<20s}", end=" ")
        m = _build_variant(v, seed=42)
        params = count_parameters(m)
        t0 = time.perf_counter()
        tr = TemporalTrainer(m, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                             warmup_epochs=WARMUP, device=DEVICE, seed=42
                             ).train(train_data, val_data)
        met = _eval_metrics(m, test_data)
        _p(f"IP={met.get('ip_mean',0):.4f}  ep={tr.total_epochs}"
           f"  t={time.perf_counter()-t0:.0f}s")
        r = {"params": params, "training": _tr_dict(tr), "test_metrics": met}
        del m; _cleanup()
        return r

    _p("\n=== Benchmark 12: PT Ablation ===")
    pt_res = {v: _run_variant(v) for v in ("pt_full", "pt_frozen", "pt_static")}
    _save("ablation_pt", {"benchmark": "ablation_pt",
                           "description": "PT-full vs PT-frozen vs PT-static",
                           "results": pt_res})

    _p("\n=== Benchmark 13: SA Ablation ===")
    sa_res = {v: _run_variant(v) for v in ("sa_full", "sa_no_gru", "sa_frozen")}
    _save("ablation_sa", {"benchmark": "ablation_sa",
                           "description": "SA-full vs SA-no-GRU vs SA-frozen",
                           "results": sa_res})

    return {"pt": pt_res, "sa": sa_res}


# =========================================================================
# Benchmark 14: Convergence speed
# =========================================================================

def bench_convergence(pt_ms, sa_ms) -> dict:
    _p("\n=== Benchmark 14: Convergence Speed ===")

    def _e90(ips: list[float]) -> int:
        if not ips:
            return -1
        tgt = max(ips) * 0.9
        for i, v in enumerate(ips):
            if v >= tgt:
                return i
        return len(ips)

    pt_rows = [{"seed": SEEDS[i], "epochs_to_90pct": _e90(r.val_ips),
                "total_epochs": r.total_epochs, "final_ip": r.final_val_ip}
               for i, r in enumerate(pt_ms.per_seed)]
    sa_rows = [{"seed": SEEDS[i], "epochs_to_90pct": _e90(r.val_ips),
                "total_epochs": r.total_epochs, "final_ip": r.final_val_ip}
               for i, r in enumerate(sa_ms.per_seed)]

    pt_mean = sum(r["epochs_to_90pct"] for r in pt_rows) / max(len(pt_rows), 1)
    sa_mean = sum(r["epochs_to_90pct"] for r in sa_rows) / max(len(sa_rows), 1)
    faster  = "phase_tracker" if pt_mean < sa_mean else "slot_attention"

    data = {"benchmark": "convergence_speed",
            "phase_tracker": {"per_seed": pt_rows, "mean_epochs_to_90pct": pt_mean},
            "slot_attention": {"per_seed": sa_rows, "mean_epochs_to_90pct": sa_mean},
            "faster_model": faster}
    _p(f"  PT 90pct: {pt_mean:.1f} ep   SA 90pct: {sa_mean:.1f} ep   faster={faster}")
    _save("convergence_speed", data)
    return data


# =========================================================================
# Benchmark 15: Hidden state evolution
# =========================================================================

def bench_hidden_state(pt_ms, sa_ms) -> dict:
    _p("\n=== Benchmark 15: Hidden State Evolution ===")

    def _snaps(ms):
        s0 = ms.per_seed[0].snapshots if ms.per_seed else []
        return [{"epoch": s.epoch, "train_loss": s.train_loss,
                 "val_loss": s.val_loss, "val_ip": s.val_ip,
                 "gradient_norm": s.gradient_norm, "param_norm": s.param_norm,
                 "phase_coherence": s.phase_coherence,
                 "slot_entropy": s.slot_entropy}
                for s in s0]

    data = {"benchmark": "hidden_state_evolution",
            "phase_tracker": _snaps(pt_ms), "slot_attention": _snaps(sa_ms)}
    _p(f"  PT snaps={len(data['phase_tracker'])}"
       f"  SA snaps={len(data['slot_attention'])}")
    _save("hidden_state_evolution", data)
    return data


# =========================================================================
# Benchmark 16: Mutual information proxy (from training curve snapshots)
# =========================================================================

def bench_mi(pt_ms, sa_ms) -> dict:
    _p("\n=== Benchmark 16: Mutual Information Proxy ===")

    def _extract(ms):
        if ms.per_seed and ms.per_seed[0].snapshots:
            return [{"epoch": s.epoch, "mi_proxy": s.val_ip}
                    for s in ms.per_seed[0].snapshots]
        if ms.per_seed:
            return [{"epoch": ep, "mi_proxy": ip}
                    for ep, ip in enumerate(ms.per_seed[0].val_ips)]
        return []

    pt_mi = _extract(pt_ms)
    sa_mi = _extract(sa_ms)
    _p(f"  PT MI data points: {len(pt_mi)}   SA MI data points: {len(sa_mi)}")

    data = {"benchmark": "mutual_information",
            "description": ("Identity-preservation score as MI proxy over training."
                            " Derived from snapshot data (no additional training)."),
            "phase_tracker": pt_mi,
            "slot_attention": sa_mi}
    _save("mutual_information", data)
    return data


# =========================================================================
# Benchmark 17: Statistical summary
# =========================================================================

def bench_stats(pt_ms, sa_ms) -> dict:
    _p("\n=== Benchmark 17: Statistical Summary ===")
    from prinet.utils.y4q1_tools import bootstrap_ci, welch_t_test, cohens_d

    pt_ips = [r.final_val_ip for r in pt_ms.per_seed]
    sa_ips = [r.final_val_ip for r in sa_ms.per_seed]
    pt_ci  = bootstrap_ci(pt_ips)
    sa_ci  = bootstrap_ci(sa_ips)
    tt     = welch_t_test(pt_ips, sa_ips)
    d      = cohens_d(pt_ips, sa_ips)
    p      = tt["p_value"]
    pt_m   = sum(pt_ips) / len(pt_ips)
    sa_m   = sum(sa_ips) / len(sa_ips)

    if p < 0.01 and abs(d) > 0.5 and pt_m > sa_m:
        outcome = "A_temporal_advantage_confirmed"
    elif p > 0.05 or pt_m <= sa_m:
        outcome = "B_no_temporal_advantage"
    else:
        outcome = "C_conditional_advantage"

    data = {
        "benchmark": "statistical_summary",
        "phase_tracker": {"ips": pt_ips, "mean": pt_ci["mean"],
                          "ci_95": [pt_ci["ci_lower"], pt_ci["ci_upper"]]},
        "slot_attention": {"ips": sa_ips, "mean": sa_ci["mean"],
                           "ci_95": [sa_ci["ci_lower"], sa_ci["ci_upper"]]},
        "welch_t_test": tt, "cohens_d": d,
        "alpha": 0.01, "min_effect_size": 0.5,
        "outcome": outcome,
    }
    _p(f"  PT: {pt_ci['mean']:.4f} [{pt_ci['ci_lower']:.4f}, {pt_ci['ci_upper']:.4f}]")
    _p(f"  SA: {sa_ci['mean']:.4f} [{sa_ci['ci_lower']:.4f}, {sa_ci['ci_upper']:.4f}]")
    _p(f"  t={tt['t_stat']:.3f}  p={p:.5f}  d={d:.3f}   outcome: {outcome}")
    _save("statistical_summary", data)
    return data


# =========================================================================
# Benchmark 18: Recovery speed
# =========================================================================

def bench_recovery(pt_state, sa_state) -> dict:
    _p("\n=== Benchmark 18: Recovery Speed ===")
    from prinet.utils.temporal_metrics import recovery_speed as _rs

    td = _gen(TEST_OCC_SEQS, occlusion_rate=0.3, base_seed=77000)
    pt = _build_pt(); pt.load_state_dict(pt_state)
    sa = _build_sa(); sa.load_state_dict(sa_state)

    def _collect(model, ds):
        speeds: list[float] = []
        model.eval(); model = model.to(DEVICE)
        with torch.no_grad():
            for seq in ds:
                frames = [f.to(DEVICE) for f in seq.frames]
                res = model.track_sequence(frames)
                v = _rs(res["identity_matches"],
                        seq.occlusion_mask, seq.n_objects)
                if not math.isnan(v):
                    speeds.append(v)
        return speeds

    pt_sp = _collect(pt, td)
    sa_sp = _collect(sa, td)
    pt_mean = sum(pt_sp) / max(len(pt_sp), 1) if pt_sp else float("nan")
    sa_mean = sum(sa_sp) / max(len(sa_sp), 1) if sa_sp else float("nan")

    faster = "tied_or_nan"
    if not (math.isnan(pt_mean) or math.isnan(sa_mean)):
        faster = "phase_tracker" if pt_mean < sa_mean else "slot_attention"

    data = {
        "benchmark": "recovery_speed",
        "description": "Post-occlusion re-binding latency at 30% occlusion rate.",
        "phase_tracker": {"mean_recovery_frames": pt_mean,
                          "n_events": len(pt_sp), "sample": pt_sp[:20]},
        "slot_attention": {"mean_recovery_frames": sa_mean,
                           "n_events": len(sa_sp), "sample": sa_sp[:20]},
        "faster_recovery": faster,
    }
    _p(f"  PT recovery={pt_mean:.2f} f   SA recovery={sa_mean:.2f} f")
    _save("recovery_speed", data)
    del pt, sa; _cleanup()
    return data


# =========================================================================
# Benchmark 19: Stress test summary
# =========================================================================

def bench_stress_summary(occ, swap, motion, noise) -> dict:
    _p("\n=== Benchmark 19: Stress Test Summary ===")

    def _get_ip(rows, key, val, side):
        row = next((r for r in rows if r.get(key) == val), {})
        sub = row.get(side, {})
        return sub.get("ip_mean", 0) if isinstance(sub, dict) else 0

    conditions = [
        {"condition": "occlusion_60pct",
         "pt_ip": next((r.get("ip_mean", 0) for r in occ["pt"]
                        if r.get("occlusion_rate") == 0.6), 0),
         "sa_ip": next((r.get("ip_mean", 0) for r in occ["sa"]
                        if r.get("occlusion_rate") == 0.6), 0)},
        {"condition": "swap_10pct",
         "pt_ip": _get_ip(swap.get("results", []), "swap_rate", 0.10, "pt"),
         "sa_ip": _get_ip(swap.get("results", []), "swap_rate", 0.10, "sa")},
        {"condition": "motion_3_reversals",
         "pt_ip": _get_ip(motion.get("results", []), "reversal_count", 3, "pt"),
         "sa_ip": _get_ip(motion.get("results", []), "reversal_count", 3, "sa")},
        {"condition": "noise_sigma_0.3",
         "pt_ip": _get_ip(noise.get("results", []), "noise_sigma", 0.3, "pt"),
         "sa_ip": _get_ip(noise.get("results", []), "noise_sigma", 0.3, "sa")},
    ]
    for r in conditions:
        r["advantage"] = r["pt_ip"] - r["sa_ip"]

    pt_wins = sum(1 for r in conditions if r["advantage"] > 0.05)
    sa_wins = sum(1 for r in conditions if r["advantage"] < -0.05)
    conclusion = ("phase_tracker_advantage" if pt_wins >= 2
                  else "slot_attention_advantage" if sa_wins >= 2
                  else "mixed_or_tied")

    data = {"benchmark": "stress_test_summary", "conditions": conditions,
            "pt_wins": pt_wins, "sa_wins": sa_wins,
            "stress_conclusion": conclusion}
    for r in conditions:
        _p(f"  {r['condition']}: PT={r['pt_ip']:.4f}"
           f"  SA={r['sa_ip']:.4f}  d={r['advantage']:+.4f}")
    _p(f"  Conclusion: {conclusion}")
    _save("stress_test_summary", data)
    return data


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    t0 = time.perf_counter()
    _p("=" * 70)
    _p("Y4 Q1.7: Definitive Temporal Advantage Benchmark Protocol")
    _p(f"Device: {DEVICE}   Seeds: {SEEDS}   MaxEpochs: {MAX_EPOCHS}"
       f"   TrainSeqs: {TRAIN_SEQS}")
    _p(f"Resume mode: {not FORCE_RERUN}   (PT cache: {PT_CACHE.exists()}"
       f"  SA cache: {SA_CACHE.exists()})")
    _p("=" * 70)

    bench_preregistration()
    bench_parameter_budget()

    _p("\n  Generating shared datasets ...")
    t1 = time.perf_counter()
    train_data = _gen(TRAIN_SEQS, base_seed=0)
    val_data   = _gen(VAL_SEQS,   base_seed=50000)
    test_data  = _gen(TEST_SEQS,  base_seed=90000)
    _p(f"  Generated: train={len(train_data)} val={len(val_data)}"
       f" test={len(test_data)} in {time.perf_counter()-t1:.1f}s")

    pt_ms, sa_ms, pt_state, sa_state = bench_training_curves(train_data, val_data)

    bench_standard(pt_state, sa_state, test_data)
    bench_long(pt_state, sa_state)
    bench_extrapolation(pt_state, sa_state)

    occ   = bench_occlusion(pt_state, sa_state)
    swap  = bench_swap(pt_state, sa_state)
    mot   = bench_motion(pt_state, sa_state)
    noise = bench_noise(pt_state, sa_state)

    bench_ablation(train_data, val_data, test_data)
    bench_convergence(pt_ms, sa_ms)
    bench_hidden_state(pt_ms, sa_ms)
    bench_mi(pt_ms, sa_ms)
    stats = bench_stats(pt_ms, sa_ms)
    bench_recovery(pt_state, sa_state)
    bench_stress_summary(occ, swap, mot, noise)

    elapsed = time.perf_counter() - t0
    artefacts = sorted(RESULTS_DIR.glob("y4q1_7_*.json"))
    _p("\n" + "=" * 70)
    _p(f"Completed in {elapsed/60:.1f} min  |  outcome: {stats.get('outcome', '?')}")
    _p(f"JSON artefacts: {len(artefacts)}")
    for p in artefacts:
        _p(f"  {p.name}")
    _p("=" * 70)


if __name__ == "__main__":
    main()
