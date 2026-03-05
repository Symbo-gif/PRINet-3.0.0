"""Year 4 Q1.4 Benchmarks — Temporal Advantage Deepening.

Demonstrates that PRINet's oscillatory PhaseTracker provides
measurable temporal advantages over GRU-based Slot Attention for
multi-object tracking.  Addresses the paper gap: "Temporal advantage
not directly demonstrated."

Metrics from Perplexity-researched literature:
- **Identity Preservation (IP)**: Fraction of correctly matched slots
  across frame transitions.
- **Phase-Slip Rate (PSR)**: Frequency of ±π phase discontinuities —
  adapted from clinical neurophysiology (epilepsy seizure analysis).
- **Binding Persistence (BP)**: Fraction of frames each object is
  continuously tracked.
- **Coherence Decay Rate (λ)**: Exponential decay rate of inter-frame
  phase correlation.  Lower λ = more persistent binding.
- **Cross-Frequency Coupling (PAC)**: Phase-amplitude coupling between
  delta/theta and gamma bands — hallmark of binding-by-synchrony.
- **Re-binding Speed**: Frames to recover identity matches after a
  perturbation (noise/occlusion).

Benchmarks:
1. Head-to-head: PhaseTracker vs TemporalSlotAttentionMOT (5→100 frames).
2. Temporal degradation: IP vs sequence length (scaling curves).
3. Phase-slip analysis: PSR across sequence lengths.
4. Coherence decay comparison.
5. Binding persistence comparison.
6. Cross-frequency coupling profile (PhaseTracker only).
7. Multi-seed head-to-head with Welch t-test + bootstrap CI.

Generates JSON result files in ``benchmarks/results/``.

Usage:
    python benchmarks/y4q1_4_benchmarks.py
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

# =========================================================================
# Setup
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)


def _save(name: str, data: dict) -> None:
    path = RESULTS_DIR / f"benchmark_y4q1_4_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


def _cleanup() -> None:
    """Free GPU/CPU memory between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_smooth_sequence(
    n_frames: int, n_objects: int, det_dim: int = 4, seed: int = 42,
) -> list[torch.Tensor]:
    """Generate slowly-drifting detections (smooth temporal trajectory)."""
    torch.manual_seed(seed)
    base = torch.randn(n_objects, det_dim)
    frames: list[torch.Tensor] = []
    for t in range(n_frames):
        noise = 0.02 * t * torch.randn(n_objects, det_dim)
        frames.append(base + noise)
    return frames


def _make_perturbed_sequence(
    n_frames: int,
    n_objects: int,
    perturb_frame: int,
    perturb_scale: float = 5.0,
    det_dim: int = 4,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Smooth sequence with a spike perturbation at ``perturb_frame``."""
    torch.manual_seed(seed)
    base = torch.randn(n_objects, det_dim)
    frames: list[torch.Tensor] = []
    for t in range(n_frames):
        noise = 0.02 * t * torch.randn(n_objects, det_dim)
        if t == perturb_frame:
            noise = perturb_scale * torch.randn(n_objects, det_dim)
        frames.append(base + noise)
    return frames


# =========================================================================
# Tracker factories
# =========================================================================


def _build_phase_tracker(seed: int = 42) -> "PhaseTracker":
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(
        detection_dim=4,
        n_delta=4,
        n_theta=8,
        n_gamma=16,
        n_discrete_steps=5,
        match_threshold=0.1,
    )


def _build_slot_tracker(seed: int = 42) -> "TemporalSlotAttentionMOT":
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(
        detection_dim=4,
        num_slots=6,
        slot_dim=64,
        num_iterations=3,
        match_threshold=0.1,
    )


# =========================================================================
# Benchmark 1: Head-to-head comparison (sequence length sweep)
# =========================================================================


def benchmark_head_to_head() -> None:
    """PhaseTracker vs TemporalSlotAttentionMOT across sequence lengths.

    Runs both trackers on identical sequences of 5, 10, 20, 40, 60, 100
    frames with 4 objects.  Reports IP, mean similarity, and
    per-frame phase correlation (PhaseTracker only).
    """
    print("\n=== Benchmark 1: Head-to-Head Comparison ===")
    seq_lengths = [5, 10, 20, 40, 60, 100]
    n_objects = 4
    results: list[dict[str, Any]] = []

    for n_frames in seq_lengths:
        print(f"  Seq length {n_frames} ...", end=" ")
        frames = _make_smooth_sequence(n_frames, n_objects, seed=42)

        pt = _build_phase_tracker()
        sa = _build_slot_tracker()

        t0 = time.perf_counter()
        pt_result = pt.track_sequence(frames)
        pt_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sa_result = sa.track_sequence(frames)
        sa_time = time.perf_counter() - t0

        from prinet.utils.y4q1_tools import temporal_advantage_report
        report = temporal_advantage_report(pt_result, sa_result)

        entry = {
            "n_frames": n_frames,
            "n_objects": n_objects,
            "ip_phase": report["ip_phase"],
            "ip_slot": report["ip_slot"],
            "ip_advantage": report["ip_advantage"],
            "mean_sim_phase": report["mean_sim_phase"],
            "mean_sim_slot": report["mean_sim_slot"],
            "mean_rho_phase": report["mean_rho_phase"],
            "pt_wall_ms": pt_time * 1000,
            "sa_wall_ms": sa_time * 1000,
        }
        results.append(entry)
        print(
            f"IP_phase={entry['ip_phase']:.4f}  "
            f"IP_slot={entry['ip_slot']:.4f}  "
            f"Δ={entry['ip_advantage']:+.4f}"
        )
        del pt, sa
        _cleanup()

    _save("head_to_head", {
        "benchmark": "head_to_head_phase_vs_slot",
        "description": (
            "PhaseTracker vs TemporalSlotAttentionMOT on identical "
            "smooth sequences, varying length."
        ),
        "results": results,
    })


# =========================================================================
# Benchmark 2: Temporal degradation curves
# =========================================================================


def benchmark_temporal_degradation() -> None:
    """IP vs sequence length — do trackers degrade over longer sequences?

    Tests the hypothesis that PhaseTracker's oscillatory dynamics
    provide more stable binding over long sequences compared to
    GRU-based carry-over which may drift.
    """
    print("\n=== Benchmark 2: Temporal Degradation Curves ===")
    seq_lengths = [5, 10, 20, 40, 60, 80, 100]
    n_objects = 4
    results: list[dict] = []

    for n_frames in seq_lengths:
        print(f"  T={n_frames} ...", end=" ")
        frames = _make_smooth_sequence(n_frames, n_objects, seed=99)

        pt = _build_phase_tracker(seed=99)
        sa = _build_slot_tracker(seed=99)

        pt_result = pt.track_sequence(frames)
        sa_result = sa.track_sequence(frames)

        # Binding persistence
        from prinet.utils.y4q1_tools import binding_persistence
        pt_bp = binding_persistence(pt_result["identity_matches"], n_objects)
        sa_bp = binding_persistence(sa_result["identity_matches"],
                                    sa.num_slots)

        entry = {
            "n_frames": n_frames,
            "ip_phase": pt_result["identity_preservation"],
            "ip_slot": sa_result["identity_preservation"],
            "bp_phase": pt_bp["mean_persistence"],
            "bp_slot": sa_bp["mean_persistence"],
        }
        results.append(entry)
        print(
            f"IP_phase={entry['ip_phase']:.4f}  "
            f"IP_slot={entry['ip_slot']:.4f}  "
            f"BP_phase={entry['bp_phase']:.4f}  "
            f"BP_slot={entry['bp_slot']:.4f}"
        )
        del pt, sa
        _cleanup()

    _save("temporal_degradation", {
        "benchmark": "temporal_degradation_curves",
        "description": (
            "Identity preservation and binding persistence vs sequence "
            "length for both trackers."
        ),
        "results": results,
    })


# =========================================================================
# Benchmark 3: Phase-slip analysis
# =========================================================================


def benchmark_phase_slip() -> None:
    """Phase-slip rate analysis for PhaseTracker across sequence lengths.

    Reports PSR (slips per step, slip fraction) — lower is better.
    GRU hidden-state based tracker has no meaningful phase to analyse,
    so this is PhaseTracker-specific.
    """
    print("\n=== Benchmark 3: Phase-Slip Analysis ===")
    from prinet.utils.y4q1_tools import phase_slip_rate

    seq_lengths = [10, 20, 40, 60, 100]
    n_objects = 4
    results: list[dict] = []

    for n_frames in seq_lengths:
        print(f"  T={n_frames} ...", end=" ")
        frames = _make_smooth_sequence(n_frames, n_objects, seed=42)
        pt = _build_phase_tracker()
        pt_result = pt.track_sequence(frames)

        # Stack phase history: (T, N_obj, n_osc)
        trajectory = torch.stack(pt_result["phase_history"])
        # Average over oscillators for per-object trajectory
        mean_phase = trajectory.mean(dim=-1)  # (T, N_obj)

        psr = phase_slip_rate(mean_phase)
        entry = {
            "n_frames": n_frames,
            "total_slips": psr["total_slips"],
            "slips_per_step": psr["slips_per_step"],
            "slip_fraction": psr["slip_fraction"],
        }
        results.append(entry)
        print(
            f"slips={psr['total_slips']}  "
            f"frac={psr['slip_fraction']:.4f}"
        )
        del pt
        _cleanup()

    _save("phase_slip", {
        "benchmark": "phase_slip_rate_analysis",
        "description": (
            "Phase-slip rate for PhaseTracker across sequence lengths. "
            "Adapted from clinical neurophysiology (epilepsy PSR)."
        ),
        "results": results,
    })


# =========================================================================
# Benchmark 4: Coherence decay comparison
# =========================================================================


def benchmark_coherence_decay() -> None:
    """Compare coherence decay between PhaseTracker and SlotAttention.

    For PhaseTracker: per-frame phase correlation ρ.
    For SlotAttention: per-frame cosine similarity.
    Fits exponential decay C(t) = C₀ exp(-λt) and compares λ.
    """
    print("\n=== Benchmark 4: Coherence Decay Comparison ===")
    from prinet.utils.y4q1_tools import coherence_decay_rate

    n_frames = 60
    n_objects = 4
    frames = _make_smooth_sequence(n_frames, n_objects, seed=42)

    pt = _build_phase_tracker()
    sa = _build_slot_tracker()

    pt_result = pt.track_sequence(frames)
    sa_result = sa.track_sequence(frames)

    # PhaseTracker: use per_frame_phase_correlation
    rho_series = pt_result["per_frame_phase_correlation"]
    # SlotAttention: use per_frame_similarity
    sim_series = sa_result["per_frame_similarity"]

    # Clamp to positive for log-space fit
    rho_clamped = [max(v, 1e-6) for v in rho_series]
    sim_clamped = [max(v, 1e-6) for v in sim_series]

    pt_decay = coherence_decay_rate(rho_clamped)
    sa_decay = coherence_decay_rate(sim_clamped)

    result = {
        "n_frames": n_frames,
        "phase_tracker": {
            "metric": "per_frame_phase_correlation",
            "decay_rate": pt_decay["decay_rate"],
            "half_life": pt_decay["half_life"],
            "initial_coherence": pt_decay["initial_coherence"],
            "r_squared": pt_decay["r_squared"],
            "series_mean": sum(rho_series) / max(len(rho_series), 1),
        },
        "slot_attention": {
            "metric": "per_frame_similarity",
            "decay_rate": sa_decay["decay_rate"],
            "half_life": sa_decay["half_life"],
            "initial_coherence": sa_decay["initial_coherence"],
            "r_squared": sa_decay["r_squared"],
            "series_mean": sum(sim_series) / max(len(sim_series), 1),
        },
        "advantage": (
            "phase_tracker" if pt_decay["decay_rate"] < sa_decay["decay_rate"]
            else "slot_attention"
        ),
    }

    print(
        f"  Phase decay λ={pt_decay['decay_rate']:.6f} "
        f"(half-life={pt_decay['half_life']:.1f} frames)"
    )
    print(
        f"  Slot  decay λ={sa_decay['decay_rate']:.6f} "
        f"(half-life={sa_decay['half_life']:.1f} frames)"
    )
    print(f"  Advantage: {result['advantage']}")

    del pt, sa
    _cleanup()

    _save("coherence_decay", {
        "benchmark": "coherence_decay_comparison",
        "description": (
            "Exponential coherence decay fit (λ) for PhaseTracker "
            "(phase correlation ρ) vs SlotAttention (cosine similarity)."
        ),
        "results": result,
    })


# =========================================================================
# Benchmark 5: Binding persistence comparison
# =========================================================================


def benchmark_binding_persistence() -> None:
    """Binding persistence comparison across sequence lengths.

    Reports the fraction of frames each object maintains a valid
    identity match for both trackers.
    """
    print("\n=== Benchmark 5: Binding Persistence Comparison ===")
    from prinet.utils.y4q1_tools import binding_persistence

    seq_lengths = [10, 20, 40, 60, 100]
    n_objects = 4
    results: list[dict] = []

    for n_frames in seq_lengths:
        print(f"  T={n_frames} ...", end=" ")
        frames = _make_smooth_sequence(n_frames, n_objects, seed=42)

        pt = _build_phase_tracker()
        sa = _build_slot_tracker()

        pt_result = pt.track_sequence(frames)
        sa_result = sa.track_sequence(frames)

        pt_bp = binding_persistence(pt_result["identity_matches"], n_objects)
        sa_bp = binding_persistence(sa_result["identity_matches"],
                                    sa.num_slots)

        entry = {
            "n_frames": n_frames,
            "bp_phase_mean": pt_bp["mean_persistence"],
            "bp_phase_min": pt_bp["min_persistence"],
            "bp_slot_mean": sa_bp["mean_persistence"],
            "bp_slot_min": sa_bp["min_persistence"],
        }
        results.append(entry)
        print(
            f"BP_phase={entry['bp_phase_mean']:.4f}  "
            f"BP_slot={entry['bp_slot_mean']:.4f}"
        )
        del pt, sa
        _cleanup()

    _save("binding_persistence", {
        "benchmark": "binding_persistence_comparison",
        "description": (
            "Per-object binding persistence for PhaseTracker vs "
            "SlotAttention across sequence lengths."
        ),
        "results": results,
    })


# =========================================================================
# Benchmark 6: Cross-frequency coupling profile
# =========================================================================


def benchmark_cross_frequency_coupling() -> None:
    """Cross-frequency coupling analysis (PhaseTracker only).

    Computes phase-amplitude coupling (PAC) between delta/theta bands
    (low frequency) and gamma band (high frequency) from the
    PhaseTracker's phase history.  High PAC during successful tracking
    is a hallmark of binding-by-synchrony.
    """
    print("\n=== Benchmark 6: Cross-Frequency Coupling ===")
    from prinet.utils.y4q1_tools import cross_frequency_coupling

    n_frames = 40
    n_objects = 4
    frames = _make_smooth_sequence(n_frames, n_objects, seed=42)

    pt = _build_phase_tracker()
    pt_result = pt.track_sequence(frames)

    # PhaseTracker has n_delta=4, n_theta=8, n_gamma=16  (total=28)
    n_delta = 4
    n_theta = 8
    n_gamma = 16

    trajectory = torch.stack(pt_result["phase_history"])  # (T, N, n_osc)

    # Flatten objects: (T, N*n_osc) → take per-band slices
    T, N, n_osc = trajectory.shape

    # Low = delta+theta (first 12), High = gamma (last 16)
    # Average across objects for a population-level measure
    phases_low = trajectory[:, :, :n_delta + n_theta].reshape(T, -1)  # (T, N*12)
    phases_high = trajectory[:, :, n_delta + n_theta:].reshape(T, -1)  # (T, N*16)

    # For PAC, need same shape — take min columns
    min_cols = min(phases_low.shape[1], phases_high.shape[1])
    phases_low = phases_low[:, :min_cols]
    phases_high = phases_high[:, :min_cols]

    pac_result = cross_frequency_coupling(phases_low, phases_high)

    result = {
        "n_frames": n_frames,
        "n_objects": n_objects,
        "n_delta": n_delta,
        "n_theta": n_theta,
        "n_gamma": n_gamma,
        "mean_pac": pac_result["pac"],
        "pac_per_step": pac_result["pac_per_step"],
        "identity_preservation": pt_result["identity_preservation"],
    }

    print(
        f"  Mean PAC = {pac_result['pac']:.4f}  "
        f"IP = {pt_result['identity_preservation']:.4f}"
    )

    del pt
    _cleanup()

    _save("cross_frequency_coupling", {
        "benchmark": "cross_frequency_coupling_profile",
        "description": (
            "Phase-amplitude coupling between low-freq (delta+theta) "
            "and high-freq (gamma) bands in PhaseTracker. "
            "High PAC → binding-by-synchrony signature."
        ),
        "results": result,
    })


# =========================================================================
# Benchmark 7: Multi-seed head-to-head with statistical tests
# =========================================================================


def benchmark_multi_seed_comparison() -> None:
    """Multi-seed head-to-head with bootstrap CI and Welch's t-test.

    Runs 5 seeds for both trackers on a 40-frame sequence.
    Reports per-seed IP, bootstrap 95% CI, Cohen's d, p-value.
    Target: PhaseTracker ≥ SlotAttention at p < 0.05.
    """
    print("\n=== Benchmark 7: Multi-Seed Statistical Comparison ===")
    from prinet.utils.y4q1_tools import bootstrap_ci, welch_t_test

    n_frames = 40
    n_objects = 4
    n_seeds = 5
    seeds = list(range(42, 42 + n_seeds))

    ip_phase_all: list[float] = []
    ip_slot_all: list[float] = []
    per_seed_results: list[dict] = []

    for seed in seeds:
        print(f"  Seed {seed} ...", end=" ")
        frames = _make_smooth_sequence(n_frames, n_objects, seed=seed)

        pt = _build_phase_tracker(seed=seed)
        sa = _build_slot_tracker(seed=seed)

        pt_result = pt.track_sequence(frames)
        sa_result = sa.track_sequence(frames)

        ip_pt = pt_result["identity_preservation"]
        ip_sa = sa_result["identity_preservation"]
        ip_phase_all.append(ip_pt)
        ip_slot_all.append(ip_sa)

        per_seed_results.append({
            "seed": seed,
            "ip_phase": ip_pt,
            "ip_slot": ip_sa,
            "ip_advantage": ip_pt - ip_sa,
        })
        print(f"IP_phase={ip_pt:.4f}  IP_slot={ip_sa:.4f}")
        del pt, sa
        _cleanup()

    # Bootstrap CI
    ci_phase = bootstrap_ci(ip_phase_all)
    ci_slot = bootstrap_ci(ip_slot_all)

    # Welch's t-test (phase vs slot)
    t_test = welch_t_test(ip_phase_all, ip_slot_all)

    result = {
        "n_frames": n_frames,
        "n_objects": n_objects,
        "n_seeds": n_seeds,
        "per_seed": per_seed_results,
        "phase_tracker_ci": ci_phase,
        "slot_attention_ci": ci_slot,
        "welch_t_test": t_test,
        "conclusion": (
            "phase_tracker_advantage_significant"
            if t_test["p_value"] < 0.05 and t_test["mean_diff"] > 0
            else "slot_attention_advantage_significant"
            if t_test["p_value"] < 0.05 and t_test["mean_diff"] < 0
            else "no_significant_difference"
        ),
    }

    print(f"\n  Phase IP: {ci_phase['mean']:.4f} "
          f"[{ci_phase['ci_lower']:.4f}, {ci_phase['ci_upper']:.4f}]")
    print(f"  Slot  IP: {ci_slot['mean']:.4f} "
          f"[{ci_slot['ci_lower']:.4f}, {ci_slot['ci_upper']:.4f}]")
    print(f"  Welch t={t_test['t_stat']:.3f}, p={t_test['p_value']:.4f}, "
          f"d={t_test['cohens_d']:.3f}")
    print(f"  Conclusion: {result['conclusion']}")

    _save("multi_seed_comparison", {
        "benchmark": "multi_seed_head_to_head",
        "description": (
            "5-seed head-to-head PhaseTracker vs SlotAttention with "
            "bootstrap 95% CI and Welch's t-test."
        ),
        "results": result,
    })


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    print("=" * 60)
    print("Y4 Q1.4: Temporal Advantage Deepening Benchmarks")
    print("=" * 60)

    benchmark_head_to_head()
    benchmark_temporal_degradation()
    benchmark_phase_slip()
    benchmark_coherence_decay()
    benchmark_binding_persistence()
    benchmark_cross_frequency_coupling()
    benchmark_multi_seed_comparison()

    print("\n" + "=" * 60)
    print("All 7 benchmarks complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
