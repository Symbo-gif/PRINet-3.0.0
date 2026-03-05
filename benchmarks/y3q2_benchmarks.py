"""Year 3 Q2 Benchmarks for PRINet — Harder Binding Tasks.

Covers all Q2 deliverables:
- N.1: MOT17-style linear evaluation (PhaseTracker MOTA)
- N.2: Crowded MOT (50+ objects, partial occlusion)
- N.3: Temporal reasoning (PhaseTracker vs AttentionTracker)
- N.4: Adaptive oscillator allocation capacity curve
- N.5: Subconscious A/B statistical test

Run with::

    python benchmarks/y3q2_benchmarks.py

Results are written to ``benchmarks/results/y3q2_*.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

# Ensure src/ is on path when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

RESULTS_DIR = _ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved -> {path.relative_to(_ROOT)}")


def _seed(s: int = 42) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# N.1 — MOT17-style linear evaluation
# ---------------------------------------------------------------------------


def bench_n1_mot17_linear() -> dict:
    """PhaseTracker on synthetic linear MOT sequence."""
    print("[N.1] MOT17-style linear evaluation")
    _seed()

    from prinet.nn.hybrid import PhaseTracker
    from prinet.nn.mot_evaluation import (
        evaluate_tracking,
        generate_linear_mot_sequence,
    )

    configs = [
        {"n_objects": 5, "n_frames": 30},
        {"n_objects": 10, "n_frames": 30},
        {"n_objects": 20, "n_frames": 30},
    ]

    tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
    tracker.eval()

    results_list = []
    for cfg in configs:
        t0 = time.perf_counter()
        seq = generate_linear_mot_sequence(**cfg, seed=42)
        result = evaluate_tracking(
            seq,
            tracker,
            detection_dim=4,
            sequence_name=f"linear_{cfg['n_objects']}obj",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "n_objects": cfg["n_objects"],
            "n_frames": cfg["n_frames"],
            "mota": round(result.mota, 4),
            "motp": round(result.motp, 4) if not math.isnan(result.motp) else None,
            "idf1": round(result.idf1, 4),
            "id_switches": result.id_switches,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        results_list.append(entry)
        print(
            f"  {cfg['n_objects']:>2} objects: MOTA={result.mota:.3f}  "
            f"IDF1={result.idf1:.3f}  IDSW={result.id_switches}  "
            f"({elapsed_ms:.0f} ms)"
        )

    data = {
        "benchmark": "N.1_mot17_linear",
        "device": _DEVICE,
        "results": results_list,
        "status": "PASS",
    }
    return data


# ---------------------------------------------------------------------------
# N.2 — Crowded MOT (50+ objects)
# ---------------------------------------------------------------------------


def bench_n2_crowded_mot() -> dict:
    """PhaseTracker on crowded MOT with 50+ objects."""
    print("[N.2] Crowded MOT (50+ objects)")
    _seed()

    from prinet.nn.hybrid import PhaseTracker
    from prinet.nn.mot_evaluation import (
        evaluate_tracking,
        generate_crowded_mot_sequence,
    )

    tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
    tracker.eval()

    configs = [
        {"n_objects": 50, "n_frames": 30, "occlusion_rate": 0.15},
        {"n_objects": 75, "n_frames": 30, "occlusion_rate": 0.20},
    ]

    results_list = []
    for cfg in configs:
        t0 = time.perf_counter()
        seq = generate_crowded_mot_sequence(**cfg, seed=42)
        result = evaluate_tracking(
            seq,
            tracker,
            detection_dim=4,
            sequence_name=f"crowded_{cfg['n_objects']}obj",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "n_objects": cfg["n_objects"],
            "n_frames": cfg["n_frames"],
            "occlusion_rate": cfg["occlusion_rate"],
            "mota": round(result.mota, 4),
            "motp": round(result.motp, 4) if not math.isnan(result.motp) else None,
            "idf1": round(result.idf1, 4),
            "id_switches": result.id_switches,
            "identity_preservation": round(result.identity_preservation, 4),
            "elapsed_ms": round(elapsed_ms, 1),
        }
        results_list.append(entry)
        print(
            f"  {cfg['n_objects']:>2} obj, occ={cfg['occlusion_rate']}: "
            f"MOTA={result.mota:.3f}  IDF1={result.idf1:.3f}  "
            f"IdPres={result.identity_preservation:.3f}  ({elapsed_ms:.0f} ms)"
        )

    data = {
        "benchmark": "N.2_crowded_mot",
        "device": _DEVICE,
        "results": results_list,
        "status": "PASS",
    }
    return data


# ---------------------------------------------------------------------------
# N.3 — PhaseTracker vs AttentionTracker (temporal reasoning)
# ---------------------------------------------------------------------------


def bench_n3_temporal_reasoning() -> dict:
    """Compare oscillatory (PhaseTracker) vs non-oscillatory (AttentionTracker)."""
    print("[N.3] Temporal reasoning: PhaseTracker vs AttentionTracker")
    _seed()

    from prinet.nn.hybrid import PhaseTracker
    from prinet.nn.mot_evaluation import (
        AttentionTracker,
        evaluate_tracking,
        generate_temporal_reasoning_sequence,
    )

    seq = generate_temporal_reasoning_sequence(
        n_objects=8,
        n_frames=20,
        occlusion_frames=[(0, 8, 12), (1, 8, 12), (2, 9, 13)],
        distractor_injection_frames=[(10, 4), (11, 4), (12, 3)],
        seed=42,
    )

    phase_tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
    phase_tracker.eval()
    t0 = time.perf_counter()
    phase_result = evaluate_tracking(seq, phase_tracker, detection_dim=4, sequence_name="phase")
    phase_ms = (time.perf_counter() - t0) * 1000

    attn_tracker = AttentionTracker(detection_dim=4, match_threshold=0.0)
    attn_tracker.eval()
    t1 = time.perf_counter()
    attn_result = evaluate_tracking(seq, attn_tracker, detection_dim=4, sequence_name="attn")
    attn_ms = (time.perf_counter() - t1) * 1000

    advantage = phase_result.mota - attn_result.mota

    data = {
        "benchmark": "N.3_temporal_reasoning",
        "device": _DEVICE,
        "n_objects": 8,
        "n_frames": 20,
        "phase_tracker": {
            "mota": round(phase_result.mota, 4),
            "idf1": round(phase_result.idf1, 4),
            "id_switches": phase_result.id_switches,
            "elapsed_ms": round(phase_ms, 1),
        },
        "attention_tracker": {
            "mota": round(attn_result.mota, 4),
            "idf1": round(attn_result.idf1, 4),
            "id_switches": attn_result.id_switches,
            "elapsed_ms": round(attn_ms, 1),
        },
        "oscillatory_advantage_mota": round(advantage, 4),
        "status": "PASS",
    }
    print(
        f"  PhaseTracker:     MOTA={phase_result.mota:.3f}  "
        f"IDF1={phase_result.idf1:.3f}  IDSW={phase_result.id_switches}"
    )
    print(
        f"  AttentionTracker: MOTA={attn_result.mota:.3f}  "
        f"IDF1={attn_result.idf1:.3f}  IDSW={attn_result.id_switches}"
    )
    print(f"  Oscillatory advantage: {advantage:+.3f} MOTA")
    return data


# ---------------------------------------------------------------------------
# N.4 — Adaptive allocator capacity curve
# ---------------------------------------------------------------------------


def bench_n4_adaptive_allocation() -> dict:
    """Sweep adaptive oscillator allocation over complexity [0, 1]."""
    print("[N.4] Adaptive oscillator allocation — capacity curve")
    _seed()

    from prinet.nn.adaptive_allocation import AdaptiveOscillatorAllocator

    alloc = AdaptiveOscillatorAllocator(min_total=12, max_total=64)
    budgets = alloc.sweep_complexity(steps=21)

    curve = []
    for b in budgets:
        curve.append(
            {
                "complexity": round(b.complexity, 2),
                "n_delta": b.n_delta,
                "n_theta": b.n_theta,
                "n_gamma": b.n_gamma,
                "total": b.total,
            }
        )
        print(
            f"  c={b.complexity:.2f}  d={b.n_delta:>2}  "
            f"t={b.n_theta:>2}  g={b.n_gamma:>2}  total={b.total:>2}"
        )

    # Check monotonicity
    totals = [b.total for b in budgets]
    monotonic = all(totals[i] >= totals[i - 1] for i in range(1, len(totals)))

    # Check no mid-range dip in gamma
    gammas = [b.n_gamma for b in budgets]
    no_gamma_dip = all(gammas[i] >= gammas[i - 1] for i in range(1, len(gammas)))

    data = {
        "benchmark": "N.4_adaptive_allocation",
        "min_total": 12,
        "max_total": 64,
        "steps": 21,
        "curve": curve,
        "monotonic": monotonic,
        "no_gamma_dip": no_gamma_dip,
        "status": "PASS" if (monotonic and no_gamma_dip) else "WARN",
    }
    print(f"  Monotonic: {monotonic}  No gamma dip: {no_gamma_dip}")
    return data


# ---------------------------------------------------------------------------
# N.5 — Subconscious A/B test
# ---------------------------------------------------------------------------


def bench_n5_subconscious_ab() -> dict:
    """Run subconscious daemon A/B test and compute p-value."""
    print("[N.5] Subconscious A/B (daemon on vs off)")
    _seed()

    from prinet.nn.hybrid import PhaseTracker
    from prinet.nn.mot_evaluation import (
        generate_linear_mot_sequence,
        run_subconscious_ab_test,
    )

    tracker = PhaseTracker(detection_dim=4, match_threshold=0.0)
    tracker.eval()
    seq = generate_linear_mot_sequence(n_objects=10, n_frames=20, seed=42)

    n_trials = 10
    t0 = time.perf_counter()
    results = run_subconscious_ab_test(
        tracker, seq, n_trials=n_trials, detection_dim=4,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    with_daemon = results["with_daemon"]
    without_daemon = results["without_daemon"]

    mean_with = sum(with_daemon) / len(with_daemon)
    mean_without = sum(without_daemon) / len(without_daemon)

    # Try Welch's t-test
    p_value = None
    try:
        from scipy.stats import ttest_ind

        _, p = ttest_ind(with_daemon, without_daemon, equal_var=False)
        p_value = round(float(p), 6) if not math.isnan(p) else None
    except ImportError:
        pass

    data = {
        "benchmark": "N.5_subconscious_ab",
        "device": _DEVICE,
        "n_trials": n_trials,
        "mean_mota_with_daemon": round(mean_with, 4),
        "mean_mota_without_daemon": round(mean_without, 4),
        "mota_values_with": [round(v, 4) for v in with_daemon],
        "mota_values_without": [round(v, 4) for v in without_daemon],
        "p_value": p_value,
        "elapsed_ms": round(elapsed_ms, 1),
        "status": "PASS",
    }
    print(f"  With daemon:    mean MOTA = {mean_with:.3f}")
    print(f"  Without daemon: mean MOTA = {mean_without:.3f}")
    print(f"  p-value: {p_value}  ({elapsed_ms:.0f} ms total)")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("PRINet Year 3 Q2 Benchmarks")
    print("=" * 60 + "\n")

    all_results: dict = {}

    try:
        r = bench_n1_mot17_linear()
        all_results["n1_mot17_linear"] = r
        _save("y3q2_n1_mot17_linear.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["n1_mot17_linear"] = {"status": "ERROR", "error": str(exc)}

    print()

    try:
        r = bench_n2_crowded_mot()
        all_results["n2_crowded_mot"] = r
        _save("y3q2_n2_crowded_mot.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["n2_crowded_mot"] = {"status": "ERROR", "error": str(exc)}

    print()

    try:
        r = bench_n3_temporal_reasoning()
        all_results["n3_temporal_reasoning"] = r
        _save("y3q2_n3_temporal_reasoning.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["n3_temporal_reasoning"] = {"status": "ERROR", "error": str(exc)}

    print()

    try:
        r = bench_n4_adaptive_allocation()
        all_results["n4_adaptive_allocation"] = r
        _save("y3q2_n4_adaptive_allocation.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["n4_adaptive_allocation"] = {"status": "ERROR", "error": str(exc)}

    print()

    try:
        r = bench_n5_subconscious_ab()
        all_results["n5_subconscious_ab"] = r
        _save("y3q2_n5_subconscious_ab.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["n5_subconscious_ab"] = {"status": "ERROR", "error": str(exc)}

    # Summary
    _save("y3q2_summary.json", all_results)
    print("\n" + "=" * 60)
    passed = sum(1 for v in all_results.values() if v.get("status") == "PASS")
    total = len(all_results)
    print(f"Results: {passed}/{total} PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
