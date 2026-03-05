"""Year 4 Q2 Benchmarks -- Publication Figure & Table Generation.

Validates that all figure and table generation functions produce
correct output from stored JSON artefacts.  This benchmark suite
does NOT train models or run simulations; it tests the paper
reproducibility pipeline.

Generates JSON result files covering:
1. Figure generation validation (all 9 figures)
2. Table generation validation (all 6 tables)
3. Reproduce pipeline validation
4. Artefact completeness check
5. Style consistency check

Usage:
    python benchmarks/y4q2_benchmarks.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _save(name: str, data: dict) -> None:
    """Save benchmark result as JSON."""
    path = RESULTS_DIR / f"benchmark_y4q2_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Benchmark 1: Figure Generation Validation
# ---------------------------------------------------------------------------

def benchmark_figure_generation() -> None:
    """Validate all figure generators produce output files."""
    import tempfile
    from prinet.utils.figure_generation import (
        configure_neurips_style,
        fig_clevr_n_capacity,
        fig_chimera_heatmap,
        fig_mot_identity_preservation,
        fig_oscillosim_scaling,
        fig_ablation_results,
        fig_parameter_efficiency,
        fig_training_curves,
        fig_gold_standard_chimera,
        fig_statistical_summary,
        generate_all_figures,
    )

    print("=== Benchmark 1: Figure Generation Validation ===")
    results_dir = RESULTS_DIR
    results_list = []

    figure_funcs = [
        ("fig_clevr_n_capacity", fig_clevr_n_capacity),
        ("fig_chimera_heatmap", fig_chimera_heatmap),
        ("fig_mot_identity_preservation", fig_mot_identity_preservation),
        ("fig_oscillosim_scaling", fig_oscillosim_scaling),
        ("fig_ablation_results", fig_ablation_results),
        ("fig_parameter_efficiency", fig_parameter_efficiency),
        ("fig_training_curves", fig_training_curves),
        ("fig_gold_standard_chimera", fig_gold_standard_chimera),
        ("fig_statistical_summary", fig_statistical_summary),
    ]

    for name, func in figure_funcs:
        entry = {"name": name, "status": "UNKNOWN", "files": [], "error": None}
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = func(results_dir=results_dir,
                             output_dir=Path(tmpdir))
                entry["files"] = [p.name for p in paths]
                entry["file_count"] = len(paths)
                # Check files are non-empty
                sizes = []
                for p in paths:
                    sz = p.stat().st_size
                    sizes.append(sz)
                entry["file_sizes_bytes"] = sizes
                entry["all_nonempty"] = all(s > 0 for s in sizes)
                entry["status"] = "OK" if entry["all_nonempty"] else "EMPTY_FILE"
        except FileNotFoundError as e:
            entry["status"] = "MISSING_DATA"
            entry["error"] = str(e)
        except Exception as e:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["elapsed_s"] = time.time() - t0
        results_list.append(entry)
        status_icon = "[OK]" if entry["status"] == "OK" else "[!!]"
        print(f"  {status_icon} {name}: {entry['status']}")

    # Test generate_all_figures
    all_entry = {"name": "generate_all_figures", "status": "UNKNOWN"}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            all_paths = generate_all_figures(
                results_dir=results_dir, output_dir=Path(tmpdir)
            )
            all_entry["figures_attempted"] = len(all_paths)
            all_entry["figures_generated"] = sum(
                1 for paths in all_paths.values() if paths
            )
            all_entry["status"] = "OK"
    except Exception as e:
        all_entry["status"] = "ERROR"
        all_entry["error"] = f"{type(e).__name__}: {e}"
    all_entry["elapsed_s"] = time.time() - t0
    results_list.append(all_entry)

    ok_count = sum(1 for r in results_list if r["status"] == "OK")
    data_missing = sum(1 for r in results_list if r["status"] == "MISSING_DATA")

    _save("figure_generation", {
        "benchmark": "figure_generation_validation",
        "total_generators": len(figure_funcs) + 1,
        "ok_count": ok_count,
        "missing_data_count": data_missing,
        "results": results_list,
    })


# ---------------------------------------------------------------------------
# Benchmark 2: Table Generation Validation
# ---------------------------------------------------------------------------

def benchmark_table_generation() -> None:
    """Validate all table generators produce valid LaTeX output."""
    import tempfile
    from prinet.utils.table_generation import (
        table_ablation_variants,
        table_parameter_efficiency,
        table_chimera_gold_standard,
        table_statistical_summary,
        table_occlusion_sweep,
        table_oscillosim_scaling,
        generate_all_tables,
    )

    print("\n=== Benchmark 2: Table Generation Validation ===")
    results_dir = RESULTS_DIR
    results_list = []

    table_funcs = [
        ("table_ablation_variants", table_ablation_variants),
        ("table_parameter_efficiency", table_parameter_efficiency),
        ("table_chimera_gold_standard", table_chimera_gold_standard),
        ("table_statistical_summary", table_statistical_summary),
        ("table_occlusion_sweep", table_occlusion_sweep),
        ("table_oscillosim_scaling", table_oscillosim_scaling),
    ]

    for name, func in table_funcs:
        entry = {"name": name, "status": "UNKNOWN", "error": None}
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = func(results_dir=results_dir,
                            output_dir=Path(tmpdir))
                content = path.read_text(encoding="utf-8")
                entry["file_name"] = path.name
                entry["file_size_bytes"] = len(content.encode("utf-8"))
                entry["has_begin_table"] = r"\begin{table}" in content
                entry["has_end_table"] = r"\end{table}" in content
                entry["has_toprule"] = r"\toprule" in content
                entry["has_bottomrule"] = r"\bottomrule" in content
                entry["line_count"] = content.count("\n")
                valid_latex = (
                    entry["has_begin_table"]
                    and entry["has_end_table"]
                    and entry["has_toprule"]
                    and entry["has_bottomrule"]
                )
                entry["status"] = "OK" if valid_latex else "INVALID_LATEX"
        except FileNotFoundError as e:
            entry["status"] = "MISSING_DATA"
            entry["error"] = str(e)
        except Exception as e:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["elapsed_s"] = time.time() - t0
        results_list.append(entry)
        status_icon = "[OK]" if entry["status"] == "OK" else "[!!]"
        print(f"  {status_icon} {name}: {entry['status']}")

    # Test generate_all_tables
    all_entry = {"name": "generate_all_tables", "status": "UNKNOWN"}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            all_tables = generate_all_tables(
                results_dir=results_dir, output_dir=Path(tmpdir)
            )
            all_entry["tables_generated"] = len(all_tables)
            all_entry["status"] = "OK"
    except Exception as e:
        all_entry["status"] = "ERROR"
        all_entry["error"] = f"{type(e).__name__}: {e}"
    all_entry["elapsed_s"] = time.time() - t0
    results_list.append(all_entry)

    ok_count = sum(1 for r in results_list if r["status"] == "OK")

    _save("table_generation", {
        "benchmark": "table_generation_validation",
        "total_generators": len(table_funcs) + 1,
        "ok_count": ok_count,
        "results": results_list,
    })


# ---------------------------------------------------------------------------
# Benchmark 3: Reproduce Pipeline Validation
# ---------------------------------------------------------------------------

def benchmark_reproduce_pipeline() -> None:
    """Validate the reproduce.py pipeline end-to-end."""
    import tempfile
    import subprocess

    print("\n=== Benchmark 3: Reproduce Pipeline Validation ===")

    entry = {"benchmark": "reproduce_pipeline", "status": "UNKNOWN"}
    t0 = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "reproduce.py"),
                "--results-dir", str(RESULTS_DIR),
                "--output-dir", str(tmpdir),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(PROJECT_ROOT),
            )
            entry["return_code"] = result.returncode
            entry["stdout_lines"] = result.stdout.count("\n")
            entry["stderr_lines"] = result.stderr.count("\n") if result.stderr else 0

            # Count generated files
            out_path = Path(tmpdir)
            fig_files = list((out_path / "figures").glob("*")) if (out_path / "figures").exists() else []
            tab_files = list((out_path / "tables").glob("*")) if (out_path / "tables").exists() else []
            entry["figures_generated"] = len(fig_files)
            entry["tables_generated"] = len(tab_files)
            entry["total_files"] = len(fig_files) + len(tab_files)

            if result.returncode == 0:
                entry["status"] = "OK"
            else:
                entry["status"] = "NONZERO_EXIT"
                # Capture last 10 lines of stdout for diagnostics
                stdout_lines = result.stdout.strip().split("\n")
                entry["last_stdout"] = stdout_lines[-10:] if len(stdout_lines) > 10 else stdout_lines

    except subprocess.TimeoutExpired:
        entry["status"] = "TIMEOUT"
    except Exception as e:
        entry["status"] = "ERROR"
        entry["error"] = f"{type(e).__name__}: {e}"

    entry["elapsed_s"] = time.time() - t0
    status_icon = "[OK]" if entry["status"] == "OK" else "[!!]"
    print(f"  {status_icon} reproduce.py: {entry['status']}")

    _save("reproduce_pipeline", entry)


# ---------------------------------------------------------------------------
# Benchmark 4: Artefact Completeness Check
# ---------------------------------------------------------------------------

def benchmark_artefact_completeness() -> None:
    """Check completeness of all JSON artefacts in results/."""
    print("\n=== Benchmark 4: Artefact Completeness Check ===")

    # All required artefacts from Q1.1-Q1.8 and earlier
    required_prefixes = {
        "q1.1": ["benchmark_y4q1_ring_chimera", "benchmark_y4q1_small_world_chimera",
                  "benchmark_y4q1_phase_lag_sweep", "benchmark_y4q1_temporal_mot",
                  "benchmark_y4q1_ablation_variants", "benchmark_y4q1_flops_efficiency",
                  "benchmark_y4q1_wall_time", "benchmark_y4q1_ring_scaling"],
        "q1.3": ["benchmark_y4q1_3_gold_standard_chimera",
                  "benchmark_y4q1_3_k_alpha_sensitivity"],
        "q1.7": ["y4q1_7_statistical_summary", "y4q1_7_training_curves_pt",
                  "y4q1_7_training_curves_sa"],
        "q1.8": ["y4q1_8_parameter_efficiency_frontier"],
        "figure_deps": ["y4q1_9_fine_occlusion", "y3q4_p4_oscillosim_scaling"],
    }

    results = {"benchmark": "artefact_completeness", "groups": {}}
    total_required = 0
    total_found = 0

    for group, prefixes in required_prefixes.items():
        group_results = []
        for prefix in prefixes:
            found = list(RESULTS_DIR.glob(f"{prefix}*.json"))
            exists = len(found) > 0
            group_results.append({
                "prefix": prefix,
                "found": exists,
                "matched_files": [f.name for f in found],
            })
            total_required += 1
            if exists:
                total_found += 1
        results["groups"][group] = group_results

    # Count all JSON files
    all_json = list(RESULTS_DIR.glob("*.json"))
    results["total_json_files"] = len(all_json)
    results["total_required"] = total_required
    results["total_found"] = total_found
    results["completeness_pct"] = (total_found / total_required * 100) if total_required > 0 else 0

    print(f"  JSON files in results/: {len(all_json)}")
    print(f"  Required artefacts found: {total_found}/{total_required} "
          f"({results['completeness_pct']:.0f}%)")

    _save("artefact_completeness", results)


# ---------------------------------------------------------------------------
# Benchmark 5: Style Consistency Check
# ---------------------------------------------------------------------------

def benchmark_style_consistency() -> None:
    """Validate NeurIPS style settings are internally consistent."""
    print("\n=== Benchmark 5: Style Consistency Check ===")

    from prinet.utils.figure_generation import (
        configure_neurips_style,
        COLORS,
    )
    import matplotlib.pyplot as plt

    checks = []

    # Check NeurIPS style application
    configure_neurips_style()
    checks.append({
        "check": "dpi_300",
        "expected": 300,
        "actual": plt.rcParams["figure.dpi"],
        "passed": plt.rcParams["figure.dpi"] == 300,
    })
    checks.append({
        "check": "font_size_9",
        "expected": 9,
        "actual": plt.rcParams["font.size"],
        "passed": plt.rcParams["font.size"] == 9,
    })
    family = plt.rcParams["font.family"]
    family_match = ("serif" in family) if isinstance(family, list) else family == "serif"
    checks.append({
        "check": "font_serif",
        "expected": "serif",
        "actual": str(family),
        "passed": family_match,
    })
    checks.append({
        "check": "savefig_dpi_300",
        "expected": 300,
        "actual": plt.rcParams["savefig.dpi"],
        "passed": plt.rcParams["savefig.dpi"] == 300,
    })

    # Check color palette completeness
    required_colors = ["pt", "sa", "pt_large", "chimera", "coherent",
                       "full", "attention_only", "oscillator_only",
                       "shared_phase", "mean_field", "sparse_knn", "csr"]
    for key in required_colors:
        checks.append({
            "check": f"color_{key}",
            "expected": "hex string",
            "actual": COLORS.get(key, "MISSING"),
            "passed": key in COLORS and COLORS[key].startswith("#"),
        })

    all_passed = all(c["passed"] for c in checks)
    pass_count = sum(1 for c in checks if c["passed"])

    print(f"  Checks passed: {pass_count}/{len(checks)}")
    print(f"  Overall: {'OK' if all_passed else 'FAIL'}")

    _save("style_consistency", {
        "benchmark": "style_consistency",
        "total_checks": len(checks),
        "passed": pass_count,
        "all_passed": all_passed,
        "checks": checks,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Y4Q2 benchmarks."""
    print("=" * 60)
    print("PRINet Year 4 Q2 -- Publication Pipeline Benchmarks")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print()

    t0 = time.time()

    benchmark_figure_generation()
    benchmark_table_generation()
    benchmark_reproduce_pipeline()
    benchmark_artefact_completeness()
    benchmark_style_consistency()

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"All Y4Q2 benchmarks complete in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
