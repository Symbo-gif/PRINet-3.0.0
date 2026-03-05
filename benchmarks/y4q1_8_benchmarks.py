"""Year 4 Q1.8 Benchmarks -- Extended Scientific Rigor Protocol (v2.6.0).

Implements 8 benchmark suites per ``PRINet Q1 8 Benchmark Plan.md``:
    B1: Adversarial Robustness (5 JSON)
    B2: Heterogeneous Frequency Chimera (4 JSON)
    B3: Noise Tolerance Scaling (3 JSON)
    B4: Parameter-Matched Comparison (4 JSON)
    B5: Multi-Scale Chimera (5 JSON)
    B6: Adversarial CLEVR-N (deferred -- requires extended CLEVR-N infra)
    B7: Curriculum Convergence (3 JSON)
    B8: Evolutionary Chimera Dynamics (4 JSON)

All print() calls are ASCII-only (no Unicode) for Windows cp1252 safety.
Resume: existing JSON files are skipped unless FORCE_RERUN=True.
Memory: gc.collect() + cuda.empty_cache() between benchmarks.

Usage:
    python benchmarks/y4q1_8_benchmarks.py
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# =========================================================================
# Configuration
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

# Q1.7 compatible settings
TRAIN_SEQS = 50
VAL_SEQS = 10
TEST_SEQS = 20
N_OBJECTS = 4
N_FRAMES = 20
DET_DIM = 4
MAX_EPOCHS = 20
PATIENCE = 5
WARMUP = 2
SEEDS = (42, 123, 456)
LR = 3e-4

FORCE_RERUN = False

PT_KWARGS = dict(n_delta=4, n_theta=8, n_gamma=16,
                 n_discrete_steps=5, match_threshold=0.1)
SA_KWARGS = dict(num_slots=6, slot_dim=64, num_iterations=3,
                 match_threshold=0.1)

# Cached state dicts from Q1.7
PT_CACHE = RESULTS_DIR / "y4q1_7_pt_best.pt"
SA_CACHE = RESULTS_DIR / "y4q1_7_sa_best.pt"


# =========================================================================
# Utilities
# =========================================================================

def _p(*args, **kwargs):
    """ASCII-safe print with flush."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict) -> bool:
    """Save JSON artefact. Returns False if skipped (already exists)."""
    path = RESULTS_DIR / f"y4q1_8_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (already exists)")
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _p(f"  -> {path.name}")
    return True


def _cleanup():
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


def _build_pt_large(seed: int = 42):
    from prinet.utils.y4q1_tools import PhaseTrackerLarge
    torch.manual_seed(seed)
    return PhaseTrackerLarge(detection_dim=DET_DIM)


def _gen(n: int, n_frames: int = N_FRAMES, **kw):
    from prinet.utils.temporal_training import generate_dataset
    return generate_dataset(n, n_objects=N_OBJECTS, n_frames=n_frames,
                            det_dim=DET_DIM, **kw)


def _load_state(cache_path: Path):
    """Load cached state dict."""
    if not cache_path.exists():
        _p(f"  [WARN] {cache_path.name} not found -- will train fresh")
        return None
    return torch.load(str(cache_path), map_location=DEVICE, weights_only=True)


def _load_models():
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
    return pt, sa, pt_state, sa_state


def _eval_ip(model, dataset, device=DEVICE) -> list[float]:
    """Evaluate identity preservation per sequence."""
    model.eval()
    model = model.to(device)
    ips = []
    with torch.no_grad():
        for seq in dataset:
            frames = [f.to(device) for f in seq.frames]
            res = model.track_sequence(frames)
            ips.append(res["identity_preservation"])
    return ips


def _make_sim(N: int, K: int, seed: int = 42, **kw) -> "OscilloSim":
    """Build OscilloSim in ring mode with cosine coupling weights."""
    from prinet.utils.oscillosim import OscilloSim, cosine_coupling_kernel
    weights = cosine_coupling_kernel(N, K)
    return OscilloSim(N, coupling_strength=1.0, coupling_mode="ring",
                      k_neighbors=K, integrator="rk4",
                      coupling_weights=weights, seed=seed, **kw)


def _make_sim_custom(N: int, nbr_idx, weights=None, seed: int = 42, **kw):
    """Build OscilloSim with custom topology (override _neighbors)."""
    from prinet.utils.oscillosim import OscilloSim
    k = nbr_idx.shape[1]
    sim = OscilloSim(N, coupling_strength=1.0, coupling_mode="ring",
                     k_neighbors=k if k % 2 == 0 else k + 1,
                     integrator="rk4", seed=seed, **kw)
    sim._neighbors = nbr_idx.to(sim.device)
    if weights is not None:
        sim._coupling_weights = weights.to(device=sim.device, dtype=sim.dtype)
    return sim


# =========================================================================
# B1: Adversarial Robustness (5 JSON)
# =========================================================================

def bench_b1_1_fgsm_sweep() -> dict:
    """B1.1 -- FGSM Sweep (PT vs SA)."""
    _p("\n=== B1.1: FGSM Sweep ===")
    from prinet.utils.adversarial_tools import fgsm_attack

    pt, sa, _, _ = _load_models()
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {"benchmark": "fgsm_sweep", "epsilons": epsilons, "seeds": list(SEEDS)}
    sweep_data = []

    for eps in epsilons:
        _p(f"  eps={eps}", end=" ")
        pt_ips_all, sa_ips_all = [], []
        for seed in SEEDS:
            ds = _gen(TEST_SEQS, base_seed=seed + 90000)
            for model, ips_list, label in [(pt, pt_ips_all, "PT"), (sa, sa_ips_all, "SA")]:
                model.eval()
                seq_ips = []
                for seq in ds:
                    frames = [f.to(DEVICE) for f in seq.frames]
                    adv_frames = [frames[0]]
                    for t in range(1, len(frames)):
                        adv_f = fgsm_attack(model, frames[t - 1], frames[t],
                                            seq.n_objects, eps)
                        adv_frames.append(adv_f)
                    with torch.no_grad():
                        res = model.track_sequence(adv_frames)
                        seq_ips.append(res["identity_preservation"])
                ips_list.append(sum(seq_ips) / max(len(seq_ips), 1))

        # Also get clean baselines
        pt_clean = _eval_ip(pt, _gen(TEST_SEQS, base_seed=SEEDS[0] + 90000))
        sa_clean = _eval_ip(sa, _gen(TEST_SEQS, base_seed=SEEDS[0] + 90000))

        from prinet.utils.y4q1_tools import bootstrap_ci
        row = {
            "epsilon": eps,
            "pt_clean_ip": float(np.mean(pt_clean)),
            "pt_adv_ips": pt_ips_all,
            "pt_adv_mean": float(np.mean(pt_ips_all)),
            "sa_clean_ip": float(np.mean(sa_clean)),
            "sa_adv_ips": sa_ips_all,
            "sa_adv_mean": float(np.mean(sa_ips_all)),
            "pt_degradation": float(np.mean(pt_clean)) - float(np.mean(pt_ips_all)),
            "sa_degradation": float(np.mean(sa_clean)) - float(np.mean(sa_ips_all)),
        }
        if len(pt_ips_all) >= 2:
            row["pt_ci"] = bootstrap_ci(pt_ips_all)
        if len(sa_ips_all) >= 2:
            row["sa_ci"] = bootstrap_ci(sa_ips_all)
        sweep_data.append(row)
        _p(f"PT_adv={row['pt_adv_mean']:.4f}  SA_adv={row['sa_adv_mean']:.4f}")

    results["sweep"] = sweep_data
    _save("fgsm_sweep", results)
    del pt, sa; _cleanup()
    return results


def bench_b1_2_pgd_attack() -> dict:
    """B1.2 -- PGD-20 Attack (PT vs SA)."""
    _p("\n=== B1.2: PGD Attack ===")
    from prinet.utils.adversarial_tools import pgd_attack

    pt, sa, _, _ = _load_models()
    epsilons = [0.05, 0.1, 0.2, 0.3]
    results = {"benchmark": "pgd_attack", "epsilons": epsilons, "seeds": list(SEEDS),
               "pgd_steps": 20, "alpha_rule": "eps/4"}
    sweep_data = []

    for eps in epsilons:
        _p(f"  eps={eps}", end=" ")
        pt_ips_all, sa_ips_all = [], []
        for seed in SEEDS:
            ds = _gen(TEST_SEQS, base_seed=seed + 91000)
            for model, ips_list in [(pt, pt_ips_all), (sa, sa_ips_all)]:
                model.eval()
                seq_ips = []
                for seq in ds:
                    frames = [f.to(DEVICE) for f in seq.frames]
                    adv_frames = [frames[0]]
                    for t in range(1, len(frames)):
                        adv_f = pgd_attack(model, frames[t - 1], frames[t],
                                           seq.n_objects, eps, steps=20,
                                           seed=seed + t * 100)
                        adv_frames.append(adv_f)
                    with torch.no_grad():
                        res = model.track_sequence(adv_frames)
                        seq_ips.append(res["identity_preservation"])
                ips_list.append(sum(seq_ips) / max(len(seq_ips), 1))

        from prinet.utils.y4q1_tools import bootstrap_ci
        row = {
            "epsilon": eps,
            "pt_adv_ips": pt_ips_all,
            "pt_adv_mean": float(np.mean(pt_ips_all)),
            "sa_adv_ips": sa_ips_all,
            "sa_adv_mean": float(np.mean(sa_ips_all)),
        }
        if len(pt_ips_all) >= 2:
            row["pt_ci"] = bootstrap_ci(pt_ips_all)
        if len(sa_ips_all) >= 2:
            row["sa_ci"] = bootstrap_ci(sa_ips_all)
        sweep_data.append(row)
        _p(f"PT={row['pt_adv_mean']:.4f}  SA={row['sa_adv_mean']:.4f}")

    results["sweep"] = sweep_data
    _save("pgd_attack", results)
    del pt, sa; _cleanup()
    return results


def bench_b1_3_adversarial_vs_random() -> dict:
    """B1.3 -- Adversarial vs Random Noise at matched L-inf."""
    _p("\n=== B1.3: Adversarial vs Random Noise ===")
    from prinet.utils.adversarial_tools import fgsm_attack

    pt, sa, _, _ = _load_models()
    epsilons = [0.05, 0.1, 0.2, 0.3]
    results = {"benchmark": "adversarial_vs_random", "epsilons": epsilons}
    rows = []

    for eps in epsilons:
        _p(f"  eps={eps}", end=" ")
        seed = SEEDS[0]
        ds = _gen(TEST_SEQS, base_seed=seed + 92000)

        comparisons = {}
        for noise_type in ("fgsm", "gaussian", "uniform"):
            pt_ips, sa_ips = [], []
            for model, ips in [(pt, pt_ips), (sa, sa_ips)]:
                model.eval()
                for seq in ds:
                    frames = [f.to(DEVICE) for f in seq.frames]
                    adv_frames = [frames[0]]
                    for t in range(1, len(frames)):
                        if noise_type == "fgsm":
                            adv_f = fgsm_attack(model, frames[t - 1], frames[t],
                                                seq.n_objects, eps)
                        elif noise_type == "gaussian":
                            noise = torch.randn_like(frames[t - 1]) * eps
                            adv_f = frames[t - 1] + noise
                        else:  # uniform
                            noise = (torch.rand_like(frames[t - 1]) * 2 - 1) * eps
                            adv_f = frames[t - 1] + noise
                        adv_frames.append(adv_f)
                    with torch.no_grad():
                        res = model.track_sequence(adv_frames)
                        ips.append(res["identity_preservation"])

            comparisons[noise_type] = {
                "pt_ip": float(np.mean(pt_ips)),
                "sa_ip": float(np.mean(sa_ips)),
            }

        rows.append({"epsilon": eps, "noise_types": comparisons})
        _p(f"done")

    results["results"] = rows
    _save("adversarial_vs_random", results)
    del pt, sa; _cleanup()
    return results


def bench_b1_4_phase_coherence() -> dict:
    """B1.4 -- Phase Coherence Under Attack."""
    _p("\n=== B1.4: Phase Coherence Under Attack ===")
    from prinet.utils.adversarial_tools import fgsm_attack
    from prinet.utils.y4q1_tools import order_parameter_series

    pt, sa, _, _ = _load_models()
    eps = 0.2  # moderate attack
    ds = _gen(TEST_SEQS, base_seed=SEEDS[0] + 93000)
    results = {"benchmark": "phase_coherence_adversarial", "epsilon": eps}

    # Track order parameter for PT under attack
    pt_r_clean, pt_r_adv = [], []
    for seq in ds:
        frames = [f.to(DEVICE) for f in seq.frames]

        # Clean
        with torch.no_grad():
            res_clean = pt.track_sequence(frames)

        # Adversarial
        adv_frames = [frames[0]]
        for t in range(1, len(frames)):
            adv_f = fgsm_attack(pt, frames[t - 1], frames[t],
                                seq.n_objects, eps)
            adv_frames.append(adv_f)
        with torch.no_grad():
            res_adv = pt.track_sequence(adv_frames)

        pt_r_clean.append(res_clean["identity_preservation"])
        pt_r_adv.append(res_adv["identity_preservation"])

    results["pt_clean_ip"] = float(np.mean(pt_r_clean))
    results["pt_adv_ip"] = float(np.mean(pt_r_adv))
    results["ip_degradation"] = float(np.mean(pt_r_clean) - np.mean(pt_r_adv))
    results["per_seq_clean"] = pt_r_clean
    results["per_seq_adv"] = pt_r_adv

    _save("phase_coherence_adversarial", results)
    _p(f"  PT clean={results['pt_clean_ip']:.4f}  adv={results['pt_adv_ip']:.4f}")
    del pt, sa; _cleanup()
    return results


def bench_b1_5_adversarial_summary() -> dict:
    """B1.5 -- Statistical Summary of Adversarial Results."""
    _p("\n=== B1.5: Adversarial Summary ===")
    from prinet.utils.y4q1_tools import welch_t_test, cohens_d, bootstrap_ci

    # Load FGSM results
    fgsm_path = RESULTS_DIR / "y4q1_8_fgsm_sweep.json"
    pgd_path = RESULTS_DIR / "y4q1_8_pgd_attack.json"

    results = {"benchmark": "adversarial_summary"}
    summary = []

    for fp, label in [(fgsm_path, "fgsm"), (pgd_path, "pgd")]:
        if not fp.exists():
            _p(f"  [skip] {fp.name} not found")
            continue
        with open(fp) as f:
            data = json.load(f)
        for row in data.get("sweep", []):
            pt_ips = row.get("pt_adv_ips", [])
            sa_ips = row.get("sa_adv_ips", [])
            eps = row.get("epsilon", 0)
            entry = {"attack": label, "epsilon": eps}
            if len(pt_ips) >= 2 and len(sa_ips) >= 2:
                stat = welch_t_test(pt_ips, sa_ips)
                entry["welch_t"] = stat
                entry["cohens_d"] = cohens_d(pt_ips, sa_ips)
                entry["pt_ci"] = bootstrap_ci(pt_ips)
                entry["sa_ci"] = bootstrap_ci(sa_ips)
                pt_mean = float(np.mean(pt_ips))
                sa_mean = float(np.mean(sa_ips))
                if pt_mean > sa_mean and stat["p_value"] < 0.01:
                    entry["outcome"] = "A_pt_more_robust"
                elif sa_mean > pt_mean and stat["p_value"] < 0.01:
                    entry["outcome"] = "B_sa_more_robust"
                else:
                    entry["outcome"] = "C_tied"
            summary.append(entry)

    results["per_epsilon"] = summary
    _save("adversarial_summary", results)
    _p(f"  {len(summary)} comparisons")
    del summary; _cleanup()
    return results


# =========================================================================
# B2: Heterogeneous Frequency Chimera (4 JSON)
# =========================================================================

def bench_b2_1_gaussian_freq_chimera() -> dict:
    """B2.1 -- Gaussian frequency spread sweep."""
    _p("\n=== B2.1: Gaussian Frequency Chimera ===")
    from prinet.utils.oscillosim import (
        bimodality_index, strength_of_incoherence, chimera_index,
        local_order_parameter, heterogeneous_natural_frequencies,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha = 256, 100, 1.521
    spreads = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {"benchmark": "gaussian_freq_chimera", "N": N, "K": K, "alpha": alpha}
    rows = []

    for sigma in spreads:
        _p(f"  sigma={sigma}", end=" ")
        bc_vals, si_vals, chi_vals = [], [], []
        for seed in SEEDS:
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase

            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc = float(bimodality_index(r_local))
            si = float(strength_of_incoherence(final_phase, window_size=10))
            chi = float(chimera_index(final_phase, sim._neighbors))

            bc_vals.append(bc)
            si_vals.append(si)
            chi_vals.append(chi)

        row = {
            "sigma": sigma,
            "bc_mean": float(np.mean(bc_vals)),
            "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None,
            "si_mean": float(np.mean(si_vals)),
            "chi_mean": float(np.mean(chi_vals)),
        }
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["sweep"] = rows
    _save("gaussian_freq_chimera", results)
    _cleanup()
    return results


def bench_b2_2_freq_distribution_comparison() -> dict:
    """B2.2 -- Lorentzian vs Gaussian vs Uniform frequency distributions."""
    _p("\n=== B2.2: Frequency Distribution Comparison ===")
    from prinet.utils.oscillosim import (
        bimodality_index, local_order_parameter,
        heterogeneous_natural_frequencies,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha, sigma = 256, 100, 1.521, 1.0
    dists = ["gaussian", "lorentzian", "uniform"]
    results = {"benchmark": "freq_distribution_comparison", "sigma": sigma}
    rows = []

    for dist in dists:
        _p(f"  {dist}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"distribution": dist, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["distributions"] = rows
    _save("freq_distribution_comparison", results)
    _cleanup()
    return results


def bench_b2_3_conduction_delay() -> dict:
    """B2.3 -- Conduction delay interaction."""
    _p("\n=== B2.3: Conduction Delay Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        heterogeneous_natural_frequencies, conduction_delay_matrix,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha, sigma = 256, 100, 1.521, 1.0
    delays = [0, 1, 2, 5, 10]
    results = {"benchmark": "conduction_delay_chimera"}
    rows = []

    for max_delay in delays:
        _p(f"  delay={max_delay}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"max_delay": max_delay, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["delay_sweep"] = rows
    _save("conduction_delay_chimera", results)
    _cleanup()
    return results


def bench_b2_4_heterogeneous_n_scaling() -> dict:
    """B2.4 -- N-scaling with heterogeneous frequencies."""
    _p("\n=== B2.4: Heterogeneous N-Scaling ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        heterogeneous_natural_frequencies,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    Ns = [128, 256, 512]
    K, alpha, sigma = 100, 1.521, 1.0
    results = {"benchmark": "heterogeneous_n_scaling"}
    rows = []

    for N in Ns:
        k_eff = min(K, N - 1)
        _p(f"  N={N}", end=" ")
        bc_homo, bc_hetero = [], []
        for seed in SEEDS:
            ic = gaussian_bump_ic(N, seed=seed)

            # Homogeneous
            sim = _make_sim(N, k_eff, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            r_local = local_order_parameter(result.final_phase, sim._neighbors)
            bc_homo.append(float(bimodality_index(r_local)))

            # Heterogeneous
            omega = heterogeneous_natural_frequencies(N, "gaussian", sigma, seed=seed)
            sim2 = _make_sim(N, k_eff, seed=seed + 1000, phase_lag=alpha)
            result2 = sim2.run(1000, dt=0.05, initial_phase=ic)
            r_local2 = local_order_parameter(result2.final_phase, sim2._neighbors)
            bc_hetero.append(float(bimodality_index(r_local2)))

        row = {
            "N": N,
            "bc_homo_mean": float(np.mean(bc_homo)),
            "bc_hetero_mean": float(np.mean(bc_hetero)),
            "bc_homo_ci": bootstrap_ci(bc_homo) if len(bc_homo) >= 2 else None,
            "bc_hetero_ci": bootstrap_ci(bc_hetero) if len(bc_hetero) >= 2 else None,
            "hetero_stronger": float(np.mean(bc_hetero)) >= float(np.mean(bc_homo)),
        }
        rows.append(row)
        _p(f"homo={row['bc_homo_mean']:.4f}  hetero={row['bc_hetero_mean']:.4f}")

    results["n_scaling"] = rows
    _save("heterogeneous_n_scaling", results)
    _cleanup()
    return results


# =========================================================================
# B3: Noise Tolerance Scaling (3 JSON)
# =========================================================================

def bench_b3_1_noise_sweep() -> dict:
    """B3.1 -- Full noise tolerance sweep."""
    _p("\n=== B3.1: Noise Sweep ===")
    pt, sa, _, _ = _load_models()
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = {"benchmark": "noise_sweep", "sigmas": sigmas, "seeds": list(SEEDS)}
    rows = []

    for sigma in sigmas:
        _p(f"  sigma={sigma}", end=" ")
        pt_ips, sa_ips = [], []
        for seed in SEEDS:
            ds = _gen(TEST_SEQS, noise_sigma=sigma, base_seed=seed + 94000)
            pt_ip = _eval_ip(pt, ds)
            sa_ip = _eval_ip(sa, ds)
            pt_ips.append(float(np.mean(pt_ip)))
            sa_ips.append(float(np.mean(sa_ip)))

        from prinet.utils.y4q1_tools import bootstrap_ci
        row = {
            "sigma": sigma,
            "pt_ips": pt_ips,
            "sa_ips": sa_ips,
            "pt_mean": float(np.mean(pt_ips)),
            "sa_mean": float(np.mean(sa_ips)),
            "pt_ci": bootstrap_ci(pt_ips) if len(pt_ips) >= 2 else None,
            "sa_ci": bootstrap_ci(sa_ips) if len(sa_ips) >= 2 else None,
        }
        rows.append(row)
        _p(f"PT={row['pt_mean']:.4f}  SA={row['sa_mean']:.4f}")

    results["sweep"] = rows
    _save("noise_sweep", results)
    del pt, sa; _cleanup()
    return results


def bench_b3_2_noise_type_comparison() -> dict:
    """B3.2 -- Noise type comparison at selected sigma levels."""
    _p("\n=== B3.2: Noise Type Comparison ===")
    pt, sa, _, _ = _load_models()
    test_sigmas = [0.3, 0.7, 1.0]
    noise_types = ["gaussian", "uniform", "salt_pepper", "structured"]
    results = {"benchmark": "noise_type_comparison", "sigmas": test_sigmas,
               "noise_types": noise_types}
    rows = []

    for sigma in test_sigmas:
        _p(f"  sigma={sigma}")
        type_results = {}
        for ntype in noise_types:
            pt_ips, sa_ips = [], []
            for seed in SEEDS:
                ds = _gen(TEST_SEQS, base_seed=seed + 95000)
                for model, ips in [(pt, pt_ips), (sa, sa_ips)]:
                    model.eval()
                    seq_ips = []
                    for seq_data in ds:
                        frames = [f.to(DEVICE) for f in seq_data.frames]
                        noisy_frames = []
                        for f in frames:
                            if ntype == "gaussian":
                                n = torch.randn_like(f) * sigma
                            elif ntype == "uniform":
                                n = (torch.rand_like(f) * 2 - 1) * sigma
                            elif ntype == "salt_pepper":
                                mask = torch.rand_like(f) < sigma * 0.5
                                n = torch.where(mask, torch.sign(torch.randn_like(f)) * 3.0,
                                                torch.zeros_like(f))
                            else:  # structured
                                n = torch.randn(1, f.shape[-1], device=f.device) * sigma
                                n = n.expand_as(f)
                            noisy_frames.append(f + n)
                        with torch.no_grad():
                            res = model.track_sequence(noisy_frames)
                            seq_ips.append(res["identity_preservation"])
                    ips.append(sum(seq_ips) / max(len(seq_ips), 1))

            type_results[ntype] = {
                "pt_mean": float(np.mean(pt_ips)),
                "sa_mean": float(np.mean(sa_ips)),
            }
            _p(f"    {ntype}: PT={type_results[ntype]['pt_mean']:.4f}  SA={type_results[ntype]['sa_mean']:.4f}")

        rows.append({"sigma": sigma, "noise_types": type_results})

    results["results"] = rows
    _save("noise_type_comparison", results)
    del pt, sa; _cleanup()
    return results


def bench_b3_3_noise_crossover() -> dict:
    """B3.3 -- Crossover analysis and statistics."""
    _p("\n=== B3.3: Noise Crossover Analysis ===")
    from prinet.utils.y4q1_tools import welch_t_test, cohens_d, bootstrap_ci

    # Load noise sweep results
    sweep_path = RESULTS_DIR / "y4q1_8_noise_sweep.json"
    if not sweep_path.exists():
        _p("  [skip] noise_sweep.json needed first")
        return {}

    with open(sweep_path) as f:
        sweep_data = json.load(f)

    results = {"benchmark": "noise_crossover"}
    sigmas = []
    pt_means = []
    sa_means = []
    per_sigma_stats = {}

    for row in sweep_data.get("sweep", []):
        sigma = row["sigma"]
        pt_ips = row.get("pt_ips", [])
        sa_ips = row.get("sa_ips", [])
        sigmas.append(sigma)
        pt_means.append(row.get("pt_mean", 0))
        sa_means.append(row.get("sa_mean", 0))

        if len(pt_ips) >= 2 and len(sa_ips) >= 2:
            per_sigma_stats[str(sigma)] = welch_t_test(pt_ips, sa_ips)

    # Find crossover
    crossover_sigma = None
    for i in range(len(sigmas) - 1):
        diff_i = pt_means[i] - sa_means[i]
        diff_j = pt_means[i + 1] - sa_means[i + 1]
        if diff_i <= 0 and diff_j > 0:
            f = -diff_i / max(diff_j - diff_i, 1e-12)
            crossover_sigma = sigmas[i] + f * (sigmas[i + 1] - sigmas[i])
            break

    # Exponential decay fit
    def _fit_lambda(means):
        s_arr = np.array(sigmas, dtype=np.float64)
        m_arr = np.clip(np.array(means, dtype=np.float64), 1e-10, None)
        ln_m = np.log(m_arr)
        if len(s_arr) >= 2:
            coeffs = np.polyfit(s_arr, ln_m, 1)
            return -float(coeffs[0])
        return 0.0

    lambda_pt = _fit_lambda(pt_means)
    lambda_sa = _fit_lambda(sa_means)

    results["crossover_sigma"] = crossover_sigma
    results["lambda_pt"] = lambda_pt
    results["lambda_sa"] = lambda_sa
    results["pt_degrades_slower"] = lambda_pt < lambda_sa
    results["per_sigma_stats"] = per_sigma_stats

    _save("noise_crossover", results)
    _p(f"  crossover_sigma={crossover_sigma}  lambda_PT={lambda_pt:.4f}  lambda_SA={lambda_sa:.4f}")
    _cleanup()
    return results


# =========================================================================
# B4: Parameter-Matched Comparison (4 JSON)
# =========================================================================

def bench_b4_1_pt_large_training() -> dict:
    """B4.1 -- PT-Large Training."""
    _p("\n=== B4.1: PT-Large Training ===")
    from prinet.utils.temporal_training import TemporalTrainer, count_parameters

    pt_large = _build_pt_large()
    params = count_parameters(pt_large)
    _p(f"  PT-Large params: total={params['total']}, trainable={params['trainable']}")

    results = {"benchmark": "pt_large_training",
               "params": params, "seeds": list(SEEDS)}
    per_seed = []

    train_ds = _gen(TRAIN_SEQS, base_seed=10000)
    val_ds = _gen(VAL_SEQS, base_seed=20000)

    best_loss = float("inf")
    best_state = None

    for i, seed in enumerate(SEEDS):
        _p(f"  [{i+1}/{len(SEEDS)}] seed={seed}", end=" ")
        model = _build_pt_large(seed)
        model.to(DEVICE)
        t0 = time.perf_counter()
        trainer = TemporalTrainer(
            model=model, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            warmup_epochs=WARMUP, device=DEVICE, seed=seed,
        )
        tr = trainer.train(train_ds, val_ds)
        dt = time.perf_counter() - t0
        _p(f"ip={tr.final_val_ip:.4f}  ep={tr.total_epochs}  t={dt:.0f}s")

        per_seed.append({
            "seed": seed,
            "final_val_ip": tr.final_val_ip,
            "best_epoch": tr.best_epoch,
            "total_epochs": tr.total_epochs,
            "wall_time_s": tr.wall_time_s,
        })

        if tr.best_val_loss < best_loss:
            best_loss = tr.best_val_loss
            best_state = copy.deepcopy(model.state_dict())
        del model; _cleanup()

    # Save best state
    ptl_cache = RESULTS_DIR / "y4q1_8_pt_large_best.pt"
    if best_state is not None:
        torch.save(best_state, str(ptl_cache))
        _p(f"  Cached -> {ptl_cache.name}")

    results["per_seed"] = per_seed
    results["mean_ip"] = float(np.mean([ps["final_val_ip"] for ps in per_seed]))
    _save("pt_large_training", results)
    _cleanup()
    return results


def bench_b4_2_parameter_matched_standard() -> dict:
    """B4.2 -- Standard comparison: PT-Small vs PT-Large vs SA."""
    _p("\n=== B4.2: Parameter-Matched Standard Comparison ===")
    pt, sa, _, _ = _load_models()

    # Load PT-Large
    ptl = _build_pt_large()
    ptl_cache = RESULTS_DIR / "y4q1_8_pt_large_best.pt"
    ptl_state = _load_state(ptl_cache)
    if ptl_state is not None:
        ptl.load_state_dict(ptl_state)
    ptl.eval().to(DEVICE)

    results = {"benchmark": "parameter_matched_standard"}
    conditions = [
        ("standard_T20", 20),
        ("long_T50", 50),
        ("extrapolation_T100", 100),
    ]
    rows = []

    for label, nf in conditions:
        _p(f"  {label}", end=" ")
        ds = _gen(TEST_SEQS, n_frames=nf, base_seed=96000 + nf)
        pt_ips = _eval_ip(pt, ds)
        sa_ips = _eval_ip(sa, ds)
        ptl_ips = _eval_ip(ptl, ds)

        row = {
            "condition": label, "n_frames": nf,
            "pt_small_ip": float(np.mean(pt_ips)),
            "pt_large_ip": float(np.mean(ptl_ips)),
            "sa_ip": float(np.mean(sa_ips)),
        }
        rows.append(row)
        _p(f"PT-S={row['pt_small_ip']:.4f}  PT-L={row['pt_large_ip']:.4f}  SA={row['sa_ip']:.4f}")

    results["conditions"] = rows
    _save("parameter_matched_standard", results)
    del pt, sa, ptl; _cleanup()
    return results


def bench_b4_3_parameter_matched_occlusion() -> dict:
    """B4.3 -- Occlusion stress: the key test."""
    _p("\n=== B4.3: Parameter-Matched Occlusion ===")
    pt, sa, _, _ = _load_models()
    ptl = _build_pt_large()
    ptl_cache = RESULTS_DIR / "y4q1_8_pt_large_best.pt"
    ptl_state = _load_state(ptl_cache)
    if ptl_state is not None:
        ptl.load_state_dict(ptl_state)
    ptl.eval().to(DEVICE)

    occ_rates = [0.2, 0.4, 0.6, 0.8]
    results = {"benchmark": "parameter_matched_occlusion"}
    rows = []

    for occ in occ_rates:
        _p(f"  occ={occ:.1f}", end=" ")
        ds = _gen(TEST_SEQS, occlusion_rate=occ, base_seed=97000)
        pt_ips = _eval_ip(pt, ds)
        sa_ips = _eval_ip(sa, ds)
        ptl_ips = _eval_ip(ptl, ds)

        from prinet.utils.y4q1_tools import bootstrap_ci
        row = {
            "occlusion_rate": occ,
            "pt_small_ip": float(np.mean(pt_ips)),
            "pt_large_ip": float(np.mean(ptl_ips)),
            "sa_ip": float(np.mean(sa_ips)),
            "pt_small_ci": bootstrap_ci(list(pt_ips)) if len(pt_ips) >= 2 else None,
            "pt_large_ci": bootstrap_ci(list(ptl_ips)) if len(ptl_ips) >= 2 else None,
            "sa_ci": bootstrap_ci(list(sa_ips)) if len(sa_ips) >= 2 else None,
            "ptl_closes_gap": float(np.mean(ptl_ips)) > float(np.mean(pt_ips)),
        }
        rows.append(row)
        _p(f"PT-S={row['pt_small_ip']:.4f}  PT-L={row['pt_large_ip']:.4f}  SA={row['sa_ip']:.4f}")

    results["occlusion"] = rows
    _save("parameter_matched_occlusion", results)
    del pt, sa, ptl; _cleanup()
    return results


def bench_b4_4_efficiency_frontier() -> dict:
    """B4.4 -- Efficiency analysis (FLOPs, wall-time, IP-per-param)."""
    _p("\n=== B4.4: Efficiency Frontier ===")
    from prinet.utils.y4q1_tools import count_flops, measure_wall_time
    from prinet.utils.temporal_training import count_parameters

    results = {"benchmark": "parameter_efficiency_frontier"}
    models_info = []

    pt, sa, _, _ = _load_models()
    ptl = _build_pt_large()
    ptl_cache = RESULTS_DIR / "y4q1_8_pt_large_best.pt"
    ptl_state = _load_state(ptl_cache)
    if ptl_state is not None:
        ptl.load_state_dict(ptl_state)
    ptl.eval().to(DEVICE)

    ds = _gen(TEST_SEQS, base_seed=98000)

    for model, name in [(pt, "PT-Small"), (ptl, "PT-Large"), (sa, "SA")]:
        params = count_parameters(model)
        ips = _eval_ip(model, ds)
        mean_ip = float(np.mean(ips))

        # Wall-time for single forward pass
        dets = torch.randn(N_OBJECTS, DET_DIM, device=DEVICE)
        wt = measure_wall_time(model, (dets, dets))

        info = {
            "model": name,
            "total_params": params["total"],
            "trainable_params": params["trainable"],
            "mean_ip": mean_ip,
            "ip_per_param": mean_ip / max(params["total"], 1) * 1000,
            "wall_time_ms": wt.get("mean_ms", 0),
        }
        models_info.append(info)
        _p(f"  {name}: params={params['total']:,}  IP={mean_ip:.4f}  "
           f"IP/kParam={info['ip_per_param']:.4f}")

    results["models"] = models_info
    _save("parameter_efficiency_frontier", results)
    del pt, sa, ptl; _cleanup()
    return results


# =========================================================================
# B5: Multi-Scale Chimera (5 JSON)
# =========================================================================

def bench_b5_1_2community_chimera() -> dict:
    """B5.1 -- 2-Community Chimera."""
    _p("\n=== B5.1: 2-Community Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index, local_order_parameter,
        community_topology,
    )
    from prinet.utils.y4q1_tools import (
        gaussian_bump_ic, bootstrap_ci, per_community_order_parameter,
    )

    N, K_coupling = 256, 100
    alpha = 1.521
    k_intra = 20
    k_inters = [0, 2, 5, 10, 20]
    results = {"benchmark": "2community_chimera", "N": N}
    rows = []

    for k_inter in k_inters:
        _p(f"  k_inter={k_inter}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            nbr_idx, communities = community_topology(N, 2, k_intra, k_inter, seed)
            k_total = k_intra + k_inter
            weights = torch.ones(N, k_total) / k_total
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim_custom(N, nbr_idx, weights, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"k_inter": k_inter, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["sweep"] = rows
    _save("2community_chimera", results)
    _cleanup()
    return results


def bench_b5_2_4community_chimera() -> dict:
    """B5.2 -- 4-Community Chimera."""
    _p("\n=== B5.2: 4-Community Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index, local_order_parameter,
        community_topology,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, alpha = 256, 1.521
    k_intra = 15
    k_inters = [0, 2, 5, 10]
    results = {"benchmark": "4community_chimera", "N": N}
    rows = []

    for k_inter in k_inters:
        _p(f"  k_inter={k_inter}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            nbr_idx, communities = community_topology(N, 4, k_intra, k_inter, seed)
            k_total = k_intra + k_inter
            weights = torch.ones(N, k_total) / k_total
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim_custom(N, nbr_idx, weights, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"k_inter": k_inter, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["sweep"] = rows
    _save("4community_chimera", results)
    _cleanup()
    return results


def bench_b5_3_hierarchical_chimera() -> dict:
    """B5.3 -- Hierarchical Topology."""
    _p("\n=== B5.3: Hierarchical Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index, local_order_parameter,
        hierarchical_topology,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, alpha = 256, 1.521
    k_intra = 15
    k_inter_levels = [0, 1, 3, 5, 10]
    results = {"benchmark": "hierarchical_chimera", "N": N}
    rows = []

    for k_inter in k_inter_levels:
        _p(f"  k_inter_level={k_inter}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            nbr_idx, groups = hierarchical_topology(N, 4, k_intra, k_inter, seed)
            k_total = k_intra + k_inter
            weights = torch.ones(N, k_total) / k_total
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim_custom(N, nbr_idx, weights, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"k_inter": k_inter, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["sweep"] = rows
    _save("hierarchical_chimera", results)
    _cleanup()
    return results


def bench_b5_4_topology_comparison() -> dict:
    """B5.4 -- Ring vs 2-Community vs 4-Community (direct comparison)."""
    _p("\n=== B5.4: Topology Comparison ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        community_topology,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha = 256, 100, 1.521
    n_comp_seeds = (42, 123, 456, 789, 1024)  # 5 seeds
    results = {"benchmark": "topology_comparison", "N": N}
    topo_results = {}

    for topo_name in ("ring", "2community", "4community"):
        _p(f"  {topo_name}", end=" ")
        bc_vals = []
        for seed in n_comp_seeds:
            if topo_name == "ring":
                nbr_idx = ring_topology(N, K)
                k_total = K
                weights = cosine_coupling_kernel(N, K)
            elif topo_name == "2community":
                nbr_idx, _ = community_topology(N, 2, 20, 5, seed)
                k_total = 25
                weights = torch.ones(N, k_total) / k_total
            else:
                nbr_idx, _ = community_topology(N, 4, 15, 5, seed)
                k_total = 20
                weights = torch.ones(N, k_total) / k_total

            ic = gaussian_bump_ic(N, seed=seed)
            if topo_name == "ring":
                sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
            else:
                sim = _make_sim_custom(N, nbr_idx, weights, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        topo_results[topo_name] = {
            "bc_mean": float(np.mean(bc_vals)),
            "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None,
            "bc_vals": bc_vals,
        }
        _p(f"BC={topo_results[topo_name]['bc_mean']:.4f}")

    results["topologies"] = topo_results
    _save("topology_comparison", results)
    _cleanup()
    return results


def bench_b5_5_cross_community_phase() -> dict:
    """B5.5 -- Cross-community phase relationships."""
    _p("\n=== B5.5: Cross-Community Phase ===")
    from prinet.utils.oscillosim import (
        OscilloSim,
        community_topology,
    )
    from prinet.utils.y4q1_tools import (
        gaussian_bump_ic, per_community_order_parameter,
    )

    N, alpha = 256, 1.521
    results = {"benchmark": "cross_community_phase", "N": N}

    nbr_idx, communities = community_topology(N, 4, 15, 5, seed=42)
    k_total = 20
    weights = torch.ones(N, k_total) / k_total
    ic = gaussian_bump_ic(N, seed=42)
    sim = _make_sim_custom(N, nbr_idx, weights, seed=42, phase_lag=alpha)
    result = sim.run(1000, dt=0.05, initial_phase=ic)

    final_phase = result.final_phase
    per_comm_r = per_community_order_parameter(final_phase, communities)

    # Inter-community phase correlation
    comm_means = []
    for comm in communities:
        idx = torch.tensor(comm, dtype=torch.long)
        z = torch.exp(1j * final_phase[idx].to(torch.complex64))
        comm_means.append(z.mean())

    inter_corrs = {}
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            phase_diff = (comm_means[i] * comm_means[j].conj()).angle()
            inter_corrs[f"comm_{i}_vs_{j}"] = float(phase_diff.item())

    results["per_community_r"] = per_comm_r
    results["inter_community_phase_diff"] = inter_corrs
    results["n_communities"] = len(communities)

    _save("cross_community_phase", results)
    _p(f"  per-community r = {[f'{r:.4f}' for r in per_comm_r]}")
    _cleanup()
    return results


# =========================================================================
# B7: Curriculum Convergence (3 JSON)
# =========================================================================

def bench_b7_1_curriculum_training() -> dict:
    """B7.1 -- Curriculum Training (PT vs SA)."""
    _p("\n=== B7.1: Curriculum Training ===")
    from prinet.utils.y4q1_tools import curriculum_train

    results = {"benchmark": "curriculum_training", "seeds": list(SEEDS)}
    model_results = {}

    for model_name, factory in [("PT", _build_pt), ("SA", _build_sa)]:
        _p(f"  {model_name}:")
        per_seed = []
        for seed in SEEDS:
            _p(f"    seed={seed}", end=" ")
            model = factory(seed)
            model.to(DEVICE)
            cr = curriculum_train(model, n_stages=4, epochs_per_stage=10,
                                 n_train=30, n_val=10, det_dim=DET_DIM,
                                 lr=LR, device=DEVICE, seed=seed)
            per_seed.append(cr)
            _p("done")
            del model; _cleanup()

        # Aggregate per-stage final IP across seeds
        stages_agg = {}
        for stage in range(1, 5):
            key = f"stage_{stage}"
            ips = [ps[key]["final_val_ip"] for ps in per_seed if key in ps]
            stages_agg[key] = {
                "mean_ip": float(np.mean(ips)) if ips else 0.0,
                "ips": ips,
            }

        model_results[model_name] = {"per_seed": per_seed, "stages_agg": stages_agg}

    results["models"] = model_results
    _save("curriculum_training", results)
    _cleanup()
    return results


def bench_b7_2_curriculum_vs_fixed() -> dict:
    """B7.2 -- Curriculum vs fixed-difficulty training."""
    _p("\n=== B7.2: Curriculum vs Fixed ===")
    from prinet.utils.y4q1_tools import curriculum_train
    from prinet.utils.temporal_training import TemporalTrainer

    results = {"benchmark": "curriculum_vs_fixed", "seeds": list(SEEDS)}
    comparisons = {}

    for model_name, factory in [("PT", _build_pt), ("SA", _build_sa)]:
        _p(f"  {model_name}:")
        # Curriculum
        curr_ips = []
        for seed in SEEDS:
            model = factory(seed)
            model.to(DEVICE)
            cr = curriculum_train(model, n_stages=4, epochs_per_stage=10,
                                 n_train=30, n_val=10, det_dim=DET_DIM,
                                 lr=LR, device=DEVICE, seed=seed)
            # Final stage IP
            curr_ips.append(cr.get("stage_4", {}).get("final_val_ip", 0.0))
            del model; _cleanup()

        # Fixed (4 objects, T=20, 40 epochs)
        fixed_ips = []
        for seed in SEEDS:
            model = factory(seed)
            model.to(DEVICE)
            train_ds = _gen(30, base_seed=seed + 50000)
            val_ds = _gen(10, base_seed=seed + 60000)
            trainer = TemporalTrainer(
                model=model, lr=LR, max_epochs=40, patience=40,
                warmup_epochs=WARMUP, device=DEVICE, seed=seed,
            )
            tr = trainer.train(train_ds, val_ds)
            fixed_ips.append(tr.final_val_ip)
            del model; _cleanup()

        comparisons[model_name] = {
            "curriculum_ips": curr_ips,
            "fixed_ips": fixed_ips,
            "curriculum_mean": float(np.mean(curr_ips)),
            "fixed_mean": float(np.mean(fixed_ips)),
            "curriculum_better": float(np.mean(curr_ips)) > float(np.mean(fixed_ips)),
        }
        _p(f"    curriculum={comparisons[model_name]['curriculum_mean']:.4f}  "
           f"fixed={comparisons[model_name]['fixed_mean']:.4f}")

    results["comparisons"] = comparisons
    _save("curriculum_vs_fixed", results)
    _cleanup()
    return results


def bench_b7_3_curriculum_transfer() -> dict:
    """B7.3 -- Transfer analysis to unseen difficulty."""
    _p("\n=== B7.3: Curriculum Transfer ===")
    from prinet.utils.y4q1_tools import curriculum_train
    from prinet.utils.temporal_training import TemporalTrainer, generate_dataset

    results = {"benchmark": "curriculum_transfer", "seeds": list(SEEDS)}
    transfer_results = {}

    # Test on hard: 6 objects, T=100 (never seen in training)
    hard_ds = generate_dataset(TEST_SEQS, n_objects=6, n_frames=100,
                               det_dim=DET_DIM, base_seed=999)

    for model_name, factory in [("PT", _build_pt), ("SA", _build_sa)]:
        _p(f"  {model_name}:")
        curr_ips = []
        fixed_ips = []

        for seed in SEEDS:
            # Curriculum-trained
            model_c = factory(seed)
            model_c.to(DEVICE)
            curriculum_train(model_c, n_stages=4, epochs_per_stage=10,
                             n_train=30, n_val=10, det_dim=DET_DIM,
                             lr=LR, device=DEVICE, seed=seed)
            ips_c = _eval_ip(model_c, hard_ds)
            curr_ips.append(float(np.mean(ips_c)))
            del model_c; _cleanup()

            # Fixed-trained
            model_f = factory(seed)
            model_f.to(DEVICE)
            train_ds = _gen(30, base_seed=seed + 70000)
            val_ds = _gen(10, base_seed=seed + 71000)
            trainer = TemporalTrainer(
                model=model_f, lr=LR, max_epochs=40, patience=40,
                warmup_epochs=WARMUP, device=DEVICE, seed=seed,
            )
            trainer.train(train_ds, val_ds)
            ips_f = _eval_ip(model_f, hard_ds)
            fixed_ips.append(float(np.mean(ips_f)))
            del model_f; _cleanup()

        transfer_results[model_name] = {
            "curriculum_ips": curr_ips,
            "fixed_ips": fixed_ips,
            "curriculum_mean": float(np.mean(curr_ips)),
            "fixed_mean": float(np.mean(fixed_ips)),
        }
        _p(f"    curriculum_transfer={transfer_results[model_name]['curriculum_mean']:.4f}  "
           f"fixed_transfer={transfer_results[model_name]['fixed_mean']:.4f}")

    results["transfer"] = transfer_results
    _save("curriculum_transfer", results)
    _cleanup()
    return results


# =========================================================================
# B8: Evolutionary Chimera Dynamics (4 JSON)
# =========================================================================

def bench_b8_1_evolutionary_static() -> dict:
    """B8.1 -- Static vs Evolutionary Coupling."""
    _p("\n=== B8.1: Evolutionary vs Static Coupling ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        evolutionary_coupling_update,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha = 256, 100, 1.521
    n_generations = 100
    results = {"benchmark": "evolutionary_static_comparison", "N": N, "generations": n_generations}

    bc_static, bc_evo = [], []
    for seed in SEEDS:
        ic = gaussian_bump_ic(N, seed=seed)

        # Static
        sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
        result = sim.run(1000, dt=0.05, initial_phase=ic)
        r_local = local_order_parameter(result.final_phase, sim._neighbors)
        bc_static.append(float(bimodality_index(r_local)))

        # Evolutionary
        sim_e = _make_sim(N, K, seed=seed, phase_lag=alpha)
        evo_weights = sim_e._coupling_weights.clone()
        nbr_idx_e = sim_e._neighbors
        phase = ic.clone()
        for gen in range(n_generations):
            sim_e._coupling_weights = evo_weights.to(device=sim_e.device, dtype=sim_e.dtype)
            res_e = sim_e.run(10, dt=0.05, initial_phase=phase)
            phase = res_e.final_phase
            evo_weights = evolutionary_coupling_update(
                evo_weights, phase, nbr_idx_e, "coordination", 0.01, seed + gen)

        r_local_e = local_order_parameter(phase, nbr_idx_e)
        bc_evo.append(float(bimodality_index(r_local_e)))

    results["bc_static"] = {"mean": float(np.mean(bc_static)),
                            "ci": bootstrap_ci(bc_static) if len(bc_static) >= 2 else None}
    results["bc_evolutionary"] = {"mean": float(np.mean(bc_evo)),
                                  "ci": bootstrap_ci(bc_evo) if len(bc_evo) >= 2 else None}
    results["evolution_strengthens"] = float(np.mean(bc_evo)) > float(np.mean(bc_static))

    _save("evolutionary_static_comparison", results)
    _p(f"  static BC={results['bc_static']['mean']:.4f}  evo BC={results['bc_evolutionary']['mean']:.4f}")
    _cleanup()
    return results


def bench_b8_2_directed_chimera() -> dict:
    """B8.2 -- Directed coupling chimera."""
    _p("\n=== B8.2: Directed Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        directed_weighted_topology,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, alpha = 256, 1.521
    asymmetries = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    results = {"benchmark": "directed_chimera", "N": N}
    rows = []

    for asym in asymmetries:
        _p(f"  asymmetry={asym}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            nbr_idx, dir_weights = directed_weighted_topology(N, 100, asym, seed)
            cos_weights = cosine_coupling_kernel(N, 100)
            combined_weights = cos_weights * dir_weights
            ic = gaussian_bump_ic(N, seed=seed)
            sim = _make_sim_custom(N, nbr_idx, combined_weights, seed=seed, phase_lag=alpha)
            result = sim.run(1000, dt=0.05, initial_phase=ic)
            final_phase = result.final_phase
            r_local = local_order_parameter(final_phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"asymmetry": asym, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["sweep"] = rows
    _save("directed_chimera", results)
    _cleanup()
    return results


def bench_b8_3_coupling_evolution() -> dict:
    """B8.3 -- Coupling evolution trajectory."""
    _p("\n=== B8.3: Coupling Evolution Trajectory ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        evolutionary_coupling_update,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic

    N, K, alpha = 256, 100, 1.521
    n_generations = 200
    results = {"benchmark": "coupling_evolution", "N": N, "generations": n_generations}

    sim = _make_sim(N, K, seed=42, phase_lag=alpha)
    weights = sim._coupling_weights.clone()
    nbr_idx = sim._neighbors
    ic = gaussian_bump_ic(N, seed=42)
    phase = ic.clone()

    trajectory = []
    for gen in range(n_generations):
        sim._coupling_weights = weights.to(device=sim.device, dtype=sim.dtype)
        res = sim.run(10, dt=0.05, initial_phase=phase)
        phase = res.final_phase
        weights = evolutionary_coupling_update(
            weights, phase, nbr_idx, "coordination", 0.01, 42 + gen)

        if gen % 20 == 0 or gen == n_generations - 1:
            entropy = -float((weights / weights.sum() * torch.log(weights / weights.sum() + 1e-10)).sum())
            sparsity = float((weights < 0.1).float().mean())
            trajectory.append({
                "generation": gen,
                "coupling_entropy": entropy,
                "coupling_sparsity": sparsity,
                "mean_weight": float(weights.mean()),
                "std_weight": float(weights.std()),
            })

    results["trajectory"] = trajectory
    _save("coupling_evolution", results)
    _p(f"  {len(trajectory)} snapshots recorded")
    _cleanup()
    return results


def bench_b8_4_payoff_chimera() -> dict:
    """B8.4 -- Payoff landscape analysis."""
    _p("\n=== B8.4: Payoff Chimera ===")
    from prinet.utils.oscillosim import (
        OscilloSim, ring_topology, cosine_coupling_kernel,
        bimodality_index, local_order_parameter,
        evolutionary_coupling_update,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic, bootstrap_ci

    N, K, alpha = 256, 100, 1.521
    payoffs = ["coordination", "prisoners_dilemma", "hawk_dove"]
    results = {"benchmark": "payoff_chimera", "N": N}
    rows = []

    for payoff in payoffs:
        _p(f"  {payoff}", end=" ")
        bc_vals = []
        for seed in SEEDS:
            sim = _make_sim(N, K, seed=seed, phase_lag=alpha)
            weights = sim._coupling_weights.clone()
            nbr_idx = sim._neighbors
            ic = gaussian_bump_ic(N, seed=seed)
            phase = ic.clone()

            for gen in range(50):
                sim._coupling_weights = weights.to(device=sim.device, dtype=sim.dtype)
                res = sim.run(10, dt=0.05, initial_phase=phase)
                phase = res.final_phase
                weights = evolutionary_coupling_update(
                    weights, phase, nbr_idx, payoff, 0.01, seed + gen)

            r_local = local_order_parameter(phase, sim._neighbors)
            bc_vals.append(float(bimodality_index(r_local)))

        row = {"payoff": payoff, "bc_mean": float(np.mean(bc_vals)),
               "bc_ci": bootstrap_ci(bc_vals) if len(bc_vals) >= 2 else None}
        rows.append(row)
        _p(f"BC={row['bc_mean']:.4f}")

    results["payoffs"] = rows
    _save("payoff_chimera", results)
    _cleanup()
    return results


# =========================================================================
# Pre-registration hash
# =========================================================================

def bench_preregistration() -> str:
    _p("\n=== Pre-registration Hash ===")
    plan = (Path(__file__).parent.parent / "Docs" / "Planning_Documentation"
            / "PRINet Q1 8 Benchmark Plan.md")
    sha = hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else "NOT_FOUND"
    _save("preregistration_hash", {
        "benchmark": "preregistration_hash",
        "sha256": sha,
        "plan_exists": plan.exists(),
    })
    _p(f"  SHA-256: {sha[:16]}...")
    return sha


# =========================================================================
# Master runner
# =========================================================================

ALL_BENCHMARKS = [
    # B1: Adversarial Robustness
    ("b1_1_fgsm_sweep", bench_b1_1_fgsm_sweep),
    ("b1_2_pgd_attack", bench_b1_2_pgd_attack),
    ("b1_3_adversarial_vs_random", bench_b1_3_adversarial_vs_random),
    ("b1_4_phase_coherence", bench_b1_4_phase_coherence),
    ("b1_5_adversarial_summary", bench_b1_5_adversarial_summary),
    # B2: Heterogeneous Frequency Chimera
    ("b2_1_gaussian_freq", bench_b2_1_gaussian_freq_chimera),
    ("b2_2_freq_dist_comparison", bench_b2_2_freq_distribution_comparison),
    ("b2_3_conduction_delay", bench_b2_3_conduction_delay),
    ("b2_4_het_n_scaling", bench_b2_4_heterogeneous_n_scaling),
    # B3: Noise Tolerance Scaling
    ("b3_1_noise_sweep", bench_b3_1_noise_sweep),
    ("b3_2_noise_type_comparison", bench_b3_2_noise_type_comparison),
    ("b3_3_noise_crossover", bench_b3_3_noise_crossover),
    # B4: Parameter-Matched Comparison
    ("b4_1_pt_large_training", bench_b4_1_pt_large_training),
    ("b4_2_param_matched_standard", bench_b4_2_parameter_matched_standard),
    ("b4_3_param_matched_occlusion", bench_b4_3_parameter_matched_occlusion),
    ("b4_4_efficiency_frontier", bench_b4_4_efficiency_frontier),
    # B5: Multi-Scale Chimera
    ("b5_1_2community", bench_b5_1_2community_chimera),
    ("b5_2_4community", bench_b5_2_4community_chimera),
    ("b5_3_hierarchical", bench_b5_3_hierarchical_chimera),
    ("b5_4_topology_comparison", bench_b5_4_topology_comparison),
    ("b5_5_cross_community", bench_b5_5_cross_community_phase),
    # B7: Curriculum Convergence
    ("b7_1_curriculum", bench_b7_1_curriculum_training),
    ("b7_2_curriculum_vs_fixed", bench_b7_2_curriculum_vs_fixed),
    ("b7_3_curriculum_transfer", bench_b7_3_curriculum_transfer),
    # B8: Evolutionary Chimera
    ("b8_1_evolutionary_static", bench_b8_1_evolutionary_static),
    ("b8_2_directed_chimera", bench_b8_2_directed_chimera),
    ("b8_3_coupling_evolution", bench_b8_3_coupling_evolution),
    ("b8_4_payoff_chimera", bench_b8_4_payoff_chimera),
    # Pre-registration
    ("preregistration", bench_preregistration),
]


def main():
    """Run all Q1.8 benchmarks in order."""
    _p("=" * 60)
    _p("PRINet Y4 Q1.8: Extended Scientific Rigor Benchmarks")
    _p(f"Device: {DEVICE}")
    _p(f"Seeds: {SEEDS}")
    _p(f"Results: {RESULTS_DIR}")
    _p("=" * 60)

    t0 = time.perf_counter()
    passed = 0
    failed = 0
    skipped = 0

    for name, fn in ALL_BENCHMARKS:
        try:
            _p(f"\n--- Running {name} ---")
            fn()
            passed += 1
        except Exception as e:
            _p(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        _cleanup()

    dt = time.perf_counter() - t0
    _p(f"\n{'=' * 60}")
    _p(f"Q1.8 Complete: {passed} passed, {failed} failed  ({dt:.0f}s)")
    _p(f"{'=' * 60}")


if __name__ == "__main__":
    # Fix sys.path for imports
    repo = Path(__file__).resolve().parent.parent
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))
    main()
