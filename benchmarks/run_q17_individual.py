"""Individual benchmark runner for Y4 Q1.7.

Runs a single named benchmark function with proper error handling,
memory cleanup, and ASCII-only output for Windows cp1252 compatibility.

Usage:
    python benchmarks/run_q17_individual.py <benchmark_name>

Available benchmarks (run in order):
    preregistration    - Benchmark 20: Pre-registration hash
    parameter_budget   - Benchmark 1: Parameter budget
    training_curves    - Benchmarks 2-3: Train PT + SA (requires datasets)
    standard           - Benchmark 4: Standard 20-frame comparison
    long               - Benchmark 5: Long 80-frame comparison
    extrapolation      - Benchmark 6: Sequence length extrapolation
    occlusion          - Benchmarks 7-8: Occlusion stress
    swap               - Benchmark 9: Appearance swap stress
    motion             - Benchmark 10: Motion discontinuity
    noise              - Benchmark 11: Noise injection
    ablation           - Benchmarks 12-13: Ablation studies
    convergence        - Benchmark 14: Convergence speed
    hidden_state       - Benchmark 15: Hidden state evolution
    mi                 - Benchmark 16: Mutual information proxy
    stats              - Benchmark 17: Statistical summary
    recovery           - Benchmark 18: Recovery speed
    stress_summary     - Benchmark 19: Stress test summary
    all                - Run all benchmarks in order
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback

# Force UTF-8 output and ASCII-safe printing
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root and benchmarks dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cleanup():
    """Aggressive memory cleanup between benchmarks."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_single(name: str) -> bool:
    """Run a single benchmark by name. Returns True on success."""
    import torch
    from y4q1_7_benchmarks import (
        DEVICE, SEEDS, FORCE_RERUN, RESULTS_DIR,
        PT_CACHE, SA_CACHE,
        _p, _cleanup, _gen,
        TRAIN_SEQS, VAL_SEQS, TEST_SEQS,
        N_FRAMES_TRAIN,
        bench_preregistration,
        bench_parameter_budget,
        bench_training_curves,
        bench_standard,
        bench_long,
        bench_extrapolation,
        bench_occlusion,
        bench_swap,
        bench_motion,
        bench_noise,
        bench_ablation,
        bench_convergence,
        bench_hidden_state,
        bench_mi,
        bench_stats,
        bench_recovery,
        bench_stress_summary,
    )

    t0 = time.perf_counter()
    _p(f"\n{'='*60}")
    _p(f"Running: {name}")
    _p(f"Device: {DEVICE}  Resume: {not FORCE_RERUN}")
    _p(f"{'='*60}")

    try:
        if name == "preregistration":
            bench_preregistration()

        elif name == "parameter_budget":
            bench_parameter_budget()

        elif name == "training_curves":
            _p("  Generating datasets...")
            train_data = _gen(TRAIN_SEQS, base_seed=0)
            val_data = _gen(VAL_SEQS, base_seed=50000)
            pt_ms, sa_ms, pt_state, sa_state = bench_training_curves(
                train_data, val_data
            )
            _p(f"  PT mean_ip={pt_ms.mean_ip:.4f}  SA mean_ip={sa_ms.mean_ip:.4f}")

        elif name == "standard":
            # Load cached states
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            test_data = _gen(TEST_SEQS, base_seed=90000)
            bench_standard(pt_state, sa_state, test_data)

        elif name == "long":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_long(pt_state, sa_state)

        elif name == "extrapolation":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_extrapolation(pt_state, sa_state)

        elif name == "occlusion":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_occlusion(pt_state, sa_state)

        elif name == "swap":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_swap(pt_state, sa_state)

        elif name == "motion":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_motion(pt_state, sa_state)

        elif name == "noise":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_noise(pt_state, sa_state)

        elif name == "ablation":
            train_data = _gen(TRAIN_SEQS, base_seed=0)
            val_data = _gen(VAL_SEQS, base_seed=50000)
            test_data = _gen(TEST_SEQS, base_seed=90000)
            bench_ablation(train_data, val_data, test_data)

        elif name == "convergence":
            # Need training results - load from JSON
            pt_ms, sa_ms = _load_ms_from_json()
            bench_convergence(pt_ms, sa_ms)

        elif name == "hidden_state":
            pt_ms, sa_ms = _load_ms_from_json()
            bench_hidden_state(pt_ms, sa_ms)

        elif name == "mi":
            pt_ms, sa_ms = _load_ms_from_json()
            bench_mi(pt_ms, sa_ms)

        elif name == "stats":
            pt_ms, sa_ms = _load_ms_from_json()
            bench_stats(pt_ms, sa_ms)

        elif name == "recovery":
            pt_state = torch.load(str(PT_CACHE), map_location=DEVICE, weights_only=True)
            sa_state = torch.load(str(SA_CACHE), map_location=DEVICE, weights_only=True)
            bench_recovery(pt_state, sa_state)

        elif name == "stress_summary":
            # Load existing stress test JSONs
            import json
            occ_pt = json.load(open(RESULTS_DIR / "y4q1_7_occlusion_stress_pt.json"))
            occ_sa = json.load(open(RESULTS_DIR / "y4q1_7_occlusion_stress_sa.json"))
            swap = json.load(open(RESULTS_DIR / "y4q1_7_swap_stress.json"))
            motion = json.load(open(RESULTS_DIR / "y4q1_7_motion_discontinuity.json"))
            noise_data = json.load(open(RESULTS_DIR / "y4q1_7_noise_injection.json"))
            occ = {"pt": occ_pt["results"], "sa": occ_sa["results"]}
            bench_stress_summary(occ, swap, motion, noise_data)

        else:
            _p(f"  Unknown benchmark: {name}")
            return False

        dt = time.perf_counter() - t0
        _p(f"\n  COMPLETED: {name} in {dt:.1f}s")
        return True

    except Exception as e:
        dt = time.perf_counter() - t0
        _p(f"\n  FAILED: {name} after {dt:.1f}s")
        _p(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        cleanup()


def _load_ms_from_json():
    """Load MultiSeedResult objects from cached training curves JSONs."""
    import json
    from y4q1_7_benchmarks import RESULTS_DIR, SEEDS, _p

    from prinet.utils.temporal_training import MultiSeedResult

    results = []
    for model_name, slug in [("PhaseTracker", "pt"), ("SlotAttention", "sa")]:
        path = RESULTS_DIR / f"y4q1_7_training_curves_{slug}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name} - run training_curves first")
        with open(path) as f:
            d = json.load(f)

        per_seed_fake = []
        for ps in d.get("per_seed", []):
            class _FakeTr:
                pass
            tr = _FakeTr()
            tr.final_train_loss = ps.get("final_train_loss", 0.0)
            tr.final_val_loss = ps.get("final_val_loss", 0.0)
            tr.final_val_ip = ps.get("final_val_ip", 0.0)
            tr.best_val_loss = ps.get("best_val_loss", 0.0)
            tr.best_epoch = ps.get("best_epoch", 0)
            tr.total_epochs = ps.get("total_epochs", 0)
            tr.wall_time_s = ps.get("wall_time_s", 0.0)
            tr.train_losses = ps.get("train_losses", [])
            tr.val_losses = ps.get("val_losses", [])
            tr.val_ips = ps.get("val_ips", [])

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
            model_name=model_name,
            seeds=list(d.get("seeds", SEEDS)),
            mean_ip=d.get("mean_ip", 0.0),
            std_ip=d.get("std_ip", 0.0),
            mean_epochs=d.get("mean_epochs", 0.0),
            mean_wall_time=d.get("mean_wall_time", 0.0),
            per_seed=per_seed_fake,
        )
        results.append(ms)

    return results[0], results[1]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    name = sys.argv[1].lower()

    if name == "all":
        order = [
            "preregistration", "parameter_budget", "training_curves",
            "standard", "long", "extrapolation",
            "occlusion", "swap", "motion", "noise",
            "ablation", "convergence", "hidden_state", "mi",
            "stats", "recovery", "stress_summary",
        ]
        results = {}
        for bname in order:
            ok = run_single(bname)
            results[bname] = "PASS" if ok else "FAIL"
            cleanup()
            if not ok:
                print(f"\n  Stopping at {bname} due to failure.")
                break

        print("\n" + "=" * 60)
        print("Summary:")
        for bname, status in results.items():
            print(f"  {bname:25s} {status}")
        print("=" * 60)
    elif name == "status":
        from pathlib import Path
        results_dir = Path(__file__).parent / "results"
        expected = [
            "preregistration_hash", "parameter_budget",
            "training_curves_pt", "training_curves_sa",
            "trained_comparison_standard", "trained_comparison_long",
            "trained_comparison_extrapolation",
            "occlusion_stress_pt", "occlusion_stress_sa",
            "swap_stress", "motion_discontinuity", "noise_injection",
            "ablation_pt", "ablation_sa",
            "convergence_speed", "hidden_state_evolution",
            "mutual_information", "statistical_summary",
            "recovery_speed", "stress_test_summary",
        ]
        print("\nY4 Q1.7 Benchmark Status:")
        print("-" * 50)
        done = 0
        for name in expected:
            path = results_dir / f"y4q1_7_{name}.json"
            status = "DONE" if path.exists() else "MISSING"
            if status == "DONE":
                done += 1
            print(f"  {name:40s} {status}")
        # Also check state dicts
        for cache_name in ["pt_best.pt", "sa_best.pt"]:
            path = results_dir / f"y4q1_7_{cache_name}"
            status = "DONE" if path.exists() else "MISSING"
            print(f"  {cache_name:40s} {status}")
        print(f"\n  {done}/{len(expected)} JSON artefacts complete")
    else:
        run_single(name)


if __name__ == "__main__":
    main()
