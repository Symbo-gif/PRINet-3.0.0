#!/usr/bin/env python
"""Reproduce all paper figures and tables from stored JSON artefacts.

This script regenerates every figure and table used in the NeurIPS 2026
submission from the benchmark JSON artefacts stored in benchmarks/results/.
It requires no training, no GPU, and no random state -- the output is
fully deterministic.

Usage:
    python reproduce.py                         # all figures + tables
    python reproduce.py --figures-only           # figures only
    python reproduce.py --tables-only            # tables only
    python reproduce.py --output-dir paper/figs  # custom output

Requirements:
    pip install matplotlib seaborn numpy

Output:
    paper/figures/*.pdf, *.png  -- publication figures
    paper/tables/*.tex          -- LaTeX table fragments

The script validates that all required JSON artefacts exist before
generating outputs, and prints a manifest of generated files with
SHA-256 checksums for verification.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_artefacts(results_dir: Path) -> list[str]:
    """Check that required JSON artefacts exist.

    Args:
        results_dir: Directory containing benchmark results.

    Returns:
        List of missing artefact filenames.
    """
    required = [
        # Legacy artefacts
        "benchmark_y4q1_ablation_variants.json",
        "benchmark_y4q1_3_gold_standard_chimera.json",
        "benchmark_y4q1_3_k_alpha_sensitivity.json",
        "y4q1_7_training_curves_pt.json",
        "y4q1_7_training_curves_sa.json",
        "y4q1_8_parameter_efficiency_frontier.json",
        "y4q1_9_fine_occlusion.json",
        "y4q1_9_7seed_comparison.json",
        "y3q4_p4_oscillosim_scaling.json",
        # Phase 1: Statistical hardening
        "phase1_cliffs_delta.json",
        "phase1_chimera_seeds.json",
        "phase1_clevr_seeds.json",
        "phase1_bayes_factor.json",
        "phase1_holm_bonferroni.json",
        "phase1_bf_tost_resolution.json",
        # Phase 2: Scaling analysis
        "phase2_object_scaling.json",
        "phase2_sequence_scaling.json",
        "phase2_occlusion_recovery.json",
        "phase2_velocity_stress.json",
        "phase2_noise_sweep.json",
        "phase2_chimera_nsweep.json",
        # Phase 3: Scientific experiments
        "phase3_profiling.json",
        "phase3_gradient_flow.json",
        "phase3_representation_geometry.json",
        # Phase 4: Theoretical grounding
        "phase4_convergence_verification.json",
        "phase4_parameter_scaling.json",
    ]
    missing = []
    for name in required:
        if not (results_dir / name).exists():
            missing.append(name)
    return missing


def main() -> int:
    """Entry point for reproduce.py.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Reproduce all NeurIPS 2026 paper figures and tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results",
        help="Directory containing JSON benchmark artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper",
        help="Base output directory for figures and tables.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate only figures, skip tables.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate only tables, skip figures.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = args.output_dir / "figures"
    tab_dir = args.output_dir / "tables"

    print("=" * 60)
    print("PRINet NeurIPS 2026 -- Reproducibility Script")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print()

    # Validate artefacts
    missing = validate_artefacts(results_dir)
    if missing:
        print("ERROR: Missing required artefacts:")
        for m in missing:
            print(f"  - {m}")
        return 1

    n_required = 26  # 9 legacy + 6 phase1 + 6 phase2 + 3 phase3 + 2 phase4
    print(f"All {n_required - len(missing)} / {n_required} required artefacts found.")
    print()

    # Import generation modules  
    from prinet.utils.figure_generation import generate_all_figures
    from prinet.utils.table_generation import generate_all_tables

    generated_files: list[Path] = []
    start_time = time.time()

    # Generate figures
    if not args.tables_only:
        print("Generating figures...")
        fig_paths = generate_all_figures(
            results_dir=results_dir, output_dir=fig_dir
        )
        for name, paths in fig_paths.items():
            for p in paths:
                generated_files.append(p)
                print(f"  [OK] {p.name}")
            if not paths:
                print(f"  [SKIP] {name} (data not available)")
        print()

    # Generate tables
    if not args.figures_only:
        print("Generating LaTeX tables...")
        tab_paths = generate_all_tables(
            results_dir=results_dir, output_dir=tab_dir
        )
        for name, path in tab_paths.items():
            generated_files.append(path)
            print(f"  [OK] {path.name}")
        print()

    elapsed = time.time() - start_time

    # Print manifest
    print("=" * 60)
    print("MANIFEST")
    print("=" * 60)
    for p in sorted(generated_files):
        sha = compute_sha256(p)
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<45} {size_kb:>7.1f} KB  sha256:{sha[:16]}...")

    print()
    print(f"Total files generated: {len(generated_files)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print("Reproduction complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
