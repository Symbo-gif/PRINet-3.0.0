"""Individual benchmark runner for Y4 Q1.8.

Runs a single named benchmark function with proper error handling,
memory cleanup, and ASCII-only output for Windows cp1252 compatibility.

Usage:
    python benchmarks/run_q18_individual.py <benchmark_name>

Available benchmarks:
    preregistration         - Pre-registration hash
    b1_1_fgsm_sweep         - FGSM sweep (PT vs SA)
    b1_2_pgd_attack         - PGD-20 attack
    b1_3_adversarial_random - Adversarial vs random noise
    b1_4_phase_coherence    - Phase coherence under attack
    b1_5_adversarial_summary - Adversarial statistical summary
    b2_1_gaussian_freq      - Gaussian frequency chimera
    b2_2_freq_dist          - Frequency distribution comparison
    b2_3_conduction_delay   - Conduction delay chimera
    b2_4_het_n_scaling      - Heterogeneous N-scaling
    b3_1_noise_sweep        - Noise tolerance sweep
    b3_2_noise_type         - Noise type comparison
    b3_3_noise_crossover    - Noise crossover analysis
    b4_1_pt_large           - PT-Large training
    b4_2_param_matched      - Parameter-matched standard
    b4_3_param_occlusion    - Parameter-matched occlusion
    b4_4_efficiency         - Efficiency frontier
    b5_1_2community         - 2-Community chimera
    b5_2_4community         - 4-Community chimera
    b5_3_hierarchical       - Hierarchical chimera
    b5_4_topology           - Topology comparison
    b5_5_cross_community    - Cross-community phase
    b7_1_curriculum         - Curriculum training
    b7_2_curriculum_fixed   - Curriculum vs fixed
    b7_3_curriculum_transfer - Curriculum transfer
    b8_1_evo_static         - Evolutionary vs static coupling
    b8_2_directed           - Directed chimera
    b8_3_coupling_evo       - Coupling evolution trajectory
    b8_4_payoff             - Payoff chimera
    all                     - Run all benchmarks in order
    status                  - Show which artefacts exist
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
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


# Map short names to (module_attr_name, display_name)
BENCHMARK_MAP = {
    "preregistration":          "bench_preregistration",
    "b1_1_fgsm_sweep":          "bench_b1_1_fgsm_sweep",
    "b1_2_pgd_attack":          "bench_b1_2_pgd_attack",
    "b1_3_adversarial_random":  "bench_b1_3_adversarial_vs_random",
    "b1_4_phase_coherence":     "bench_b1_4_phase_coherence",
    "b1_5_adversarial_summary": "bench_b1_5_adversarial_summary",
    "b2_1_gaussian_freq":       "bench_b2_1_gaussian_freq_chimera",
    "b2_2_freq_dist":           "bench_b2_2_freq_distribution_comparison",
    "b2_3_conduction_delay":    "bench_b2_3_conduction_delay",
    "b2_4_het_n_scaling":       "bench_b2_4_heterogeneous_n_scaling",
    "b3_1_noise_sweep":         "bench_b3_1_noise_sweep",
    "b3_2_noise_type":          "bench_b3_2_noise_type_comparison",
    "b3_3_noise_crossover":     "bench_b3_3_noise_crossover",
    "b4_1_pt_large":            "bench_b4_1_pt_large_training",
    "b4_2_param_matched":       "bench_b4_2_parameter_matched_standard",
    "b4_3_param_occlusion":     "bench_b4_3_parameter_matched_occlusion",
    "b4_4_efficiency":          "bench_b4_4_efficiency_frontier",
    "b5_1_2community":          "bench_b5_1_2community_chimera",
    "b5_2_4community":          "bench_b5_2_4community_chimera",
    "b5_3_hierarchical":        "bench_b5_3_hierarchical_chimera",
    "b5_4_topology":            "bench_b5_4_topology_comparison",
    "b5_5_cross_community":     "bench_b5_5_cross_community_phase",
    "b7_1_curriculum":          "bench_b7_1_curriculum_training",
    "b7_2_curriculum_fixed":    "bench_b7_2_curriculum_vs_fixed",
    "b7_3_curriculum_transfer": "bench_b7_3_curriculum_transfer",
    "b8_1_evo_static":          "bench_b8_1_evolutionary_static",
    "b8_2_directed":            "bench_b8_2_directed_chimera",
    "b8_3_coupling_evo":        "bench_b8_3_coupling_evolution",
    "b8_4_payoff":              "bench_b8_4_payoff_chimera",
}

# Canonical execution order
EXECUTION_ORDER = [
    "preregistration",
    # B1
    "b1_1_fgsm_sweep", "b1_2_pgd_attack", "b1_3_adversarial_random",
    "b1_4_phase_coherence", "b1_5_adversarial_summary",
    # B2
    "b2_1_gaussian_freq", "b2_2_freq_dist", "b2_3_conduction_delay",
    "b2_4_het_n_scaling",
    # B3
    "b3_1_noise_sweep", "b3_2_noise_type", "b3_3_noise_crossover",
    # B4
    "b4_1_pt_large", "b4_2_param_matched", "b4_3_param_occlusion",
    "b4_4_efficiency",
    # B5
    "b5_1_2community", "b5_2_4community", "b5_3_hierarchical",
    "b5_4_topology", "b5_5_cross_community",
    # B7
    "b7_1_curriculum", "b7_2_curriculum_fixed", "b7_3_curriculum_transfer",
    # B8
    "b8_1_evo_static", "b8_2_directed", "b8_3_coupling_evo", "b8_4_payoff",
]

# Expected JSON artefact basenames (without y4q1_8_ prefix and .json suffix)
EXPECTED_ARTEFACTS = [
    "preregistration_hash",
    "fgsm_sweep", "pgd_attack", "adversarial_vs_random",
    "phase_coherence_adversarial", "adversarial_summary",
    "gaussian_freq_chimera", "freq_distribution_comparison",
    "conduction_delay_chimera", "heterogeneous_n_scaling",
    "noise_sweep", "noise_type_comparison", "noise_crossover",
    "pt_large_training", "parameter_matched_standard",
    "parameter_matched_occlusion", "parameter_efficiency_frontier",
    "2community_chimera", "4community_chimera", "hierarchical_chimera",
    "topology_comparison", "cross_community_phase",
    "curriculum_training", "curriculum_vs_fixed", "curriculum_transfer",
    "evolutionary_static_comparison", "directed_chimera",
    "coupling_evolution", "payoff_chimera",
]


def run_single(name: str) -> bool:
    """Run a single benchmark by name. Returns True on success."""
    import y4q1_8_benchmarks as bm

    fn_name = BENCHMARK_MAP.get(name)
    if fn_name is None:
        print(f"Unknown benchmark: {name}")
        print(f"Available: {', '.join(sorted(BENCHMARK_MAP.keys()))}")
        return False

    fn = getattr(bm, fn_name, None)
    if fn is None:
        print(f"Function {fn_name} not found in y4q1_8_benchmarks")
        return False

    t0 = time.perf_counter()
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {name} ({fn_name})", flush=True)
    print(f"Device: {bm.DEVICE}  Resume: {not bm.FORCE_RERUN}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        fn()
        dt = time.perf_counter() - t0
        print(f"\n  COMPLETED: {name} in {dt:.1f}s", flush=True)
        return True
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"\n  FAILED: {name} after {dt:.1f}s", flush=True)
        print(f"  Error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False
    finally:
        cleanup()


def show_status():
    """Show which Q1.8 artefacts exist."""
    from pathlib import Path
    results_dir = Path(__file__).parent / "results"

    print("\nY4 Q1.8 Benchmark Status:")
    print("-" * 55, flush=True)
    done = 0
    for art in EXPECTED_ARTEFACTS:
        path = results_dir / f"y4q1_8_{art}.json"
        status = "DONE" if path.exists() else "MISSING"
        if status == "DONE":
            done += 1
        print(f"  {art:45s} {status}", flush=True)

    # PT-Large cache
    ptl_path = results_dir / "y4q1_8_pt_large_best.pt"
    status = "DONE" if ptl_path.exists() else "MISSING"
    print(f"  {'pt_large_best.pt':45s} {status}", flush=True)

    print(f"\n  {done}/{len(EXPECTED_ARTEFACTS)} JSON artefacts complete", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    name = sys.argv[1].lower()

    if name == "all":
        results = {}
        for bname in EXECUTION_ORDER:
            ok = run_single(bname)
            results[bname] = "PASS" if ok else "FAIL"
            cleanup()

        print(f"\n{'='*60}", flush=True)
        print("Q1.8 Summary:", flush=True)
        passed = sum(1 for v in results.values() if v == "PASS")
        for bname, status in results.items():
            print(f"  {bname:30s} {status}", flush=True)
        print(f"\n  {passed}/{len(results)} passed", flush=True)
        print(f"{'='*60}", flush=True)

    elif name == "status":
        show_status()

    else:
        run_single(name)


if __name__ == "__main__":
    main()
