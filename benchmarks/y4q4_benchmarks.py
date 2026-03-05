"""Year 4 Q4 Benchmarks: Review Response & Archival Close.

Validates all Q4 deliverables:
  B.1  Year 4 comprehensive report (W.4)
  B.2  Project retrospective (W.5)
  B.3  Archive files -- CITATION.cff, CODE_OF_CONDUCT.md (W.6)
  B.4  SHA-256 manifest of benchmark artefacts (W.6)
  B.5  Final regression gate (W.7)
  B.6  Documentation completeness audit
  B.7  Project metrics summary
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "benchmarks", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(name: str, data: dict) -> None:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    print(f"  -> {path}")


# =========================================================================
# B.1 Year 4 Comprehensive Report
# =========================================================================


def benchmark_year4_report() -> dict:
    """B.1: Validate Year 4 comprehensive report."""
    report_path = os.path.join(ROOT, "Docs", "Year_4_Comprehensive_Report.md")
    exists = os.path.isfile(report_path)

    sections_found = []
    word_count = 0
    if exists:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        word_count = len(content.split())
        required = [
            "Executive Summary",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Benchmark",
            "Metrics",
        ]
        for s in required:
            if s.lower() in content.lower():
                sections_found.append(s)

    result = {
        "benchmark": "B.1_year4_report",
        "exists": exists,
        "word_count": word_count,
        "sections_found": sections_found,
        "sections_expected": 7,
        "gate_passed": exists and word_count >= 500 and len(sections_found) >= 6,
    }
    print(f"  Report: exists={exists}, words={word_count}, "
          f"sections={len(sections_found)}/7")
    return result


# =========================================================================
# B.2 Project Retrospective
# =========================================================================


def benchmark_retrospective() -> dict:
    """B.2: Validate project retrospective."""
    retro_path = os.path.join(ROOT, "Docs", "Project_Retrospective.md")
    exists = os.path.isfile(retro_path)

    sections_found = []
    word_count = 0
    if exists:
        with open(retro_path, "r", encoding="utf-8") as f:
            content = f.read()
        word_count = len(content.split())
        required = [
            "What Worked",
            "What Didn't",
            "Technical Decisions",
            "Extension",
        ]
        for s in required:
            if s.lower() in content.lower():
                sections_found.append(s)

    result = {
        "benchmark": "B.2_retrospective",
        "exists": exists,
        "word_count": word_count,
        "sections_found": sections_found,
        "gate_passed": exists and word_count >= 300 and len(sections_found) >= 3,
    }
    print(f"  Retrospective: exists={exists}, words={word_count}, "
          f"sections={len(sections_found)}/4")
    return result


# =========================================================================
# B.3 Archive Files
# =========================================================================


def benchmark_archive_files() -> dict:
    """B.3: Validate CITATION.cff and CODE_OF_CONDUCT.md."""
    cit_path = os.path.join(ROOT, "CITATION.cff")
    coc_path = os.path.join(ROOT, "CODE_OF_CONDUCT.md")

    cit_exists = os.path.isfile(cit_path)
    coc_exists = os.path.isfile(coc_path)

    cit_valid = False
    if cit_exists:
        with open(cit_path, "r", encoding="utf-8") as f:
            cit_text = f.read()
        cit_valid = all(
            k in cit_text for k in ["cff-version", "title", "version", "license"]
        )

    coc_valid = False
    if coc_exists:
        with open(coc_path, "r", encoding="utf-8") as f:
            coc_text = f.read()
        coc_valid = "pledge" in coc_text.lower() and "enforcement" in coc_text.lower()

    result = {
        "benchmark": "B.3_archive_files",
        "citation_cff_exists": cit_exists,
        "citation_cff_valid": cit_valid,
        "code_of_conduct_exists": coc_exists,
        "code_of_conduct_valid": coc_valid,
        "gate_passed": cit_valid and coc_valid,
    }
    print(f"  CITATION.cff: exists={cit_exists}, valid={cit_valid}")
    print(f"  CODE_OF_CONDUCT.md: exists={coc_exists}, valid={coc_valid}")
    return result


# =========================================================================
# B.4 SHA-256 Manifest
# =========================================================================


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def benchmark_sha256_manifest() -> dict:
    """B.4: Generate and validate SHA-256 manifest for benchmark artefacts."""
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    manifest = {}
    for jf in json_files:
        name = os.path.basename(jf)
        # Skip the manifest itself if it exists from a prior run
        if name == "sha256_manifest.json":
            continue
        manifest[name] = _sha256_file(jf)

    # Save the manifest
    manifest_path = os.path.join(RESULTS_DIR, "sha256_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    print(f"  -> {manifest_path}")

    result = {
        "benchmark": "B.4_sha256_manifest",
        "artefacts_hashed": len(manifest),
        "manifest_path": manifest_path,
        "gate_passed": len(manifest) >= 100,
    }
    print(f"  Manifest: {len(manifest)} artefacts hashed")
    return result


# =========================================================================
# B.5 Final Regression Gate
# =========================================================================


def benchmark_final_regression() -> dict:
    """B.5: Run full test suite and check 0 failures, <=14 skips."""
    t0 = time.perf_counter()
    cmd = [
        sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"),
        "-v", "--tb=short", "-q",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT, timeout=600,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    run_time = time.perf_counter() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Parse pytest summary line, e.g. "1582 passed, 14 skipped"
    passed = 0
    failed = 0
    skipped = 0
    import re
    summary_match = re.search(
        r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+skipped)?",
        stdout + stderr,
    )
    if not summary_match:
        # Try alternate patterns
        summary_match = re.search(
            r"(\d+)\s+passed",
            stdout + stderr,
        )
    if summary_match:
        passed = int(summary_match.group(1))
        if summary_match.lastindex and summary_match.lastindex >= 2 and summary_match.group(2):
            failed = int(summary_match.group(2))
        if summary_match.lastindex and summary_match.lastindex >= 3 and summary_match.group(3):
            skipped = int(summary_match.group(3))

    # Also check for "failed" separately
    fail_match = re.search(r"(\d+)\s+failed", stdout + stderr)
    if fail_match:
        failed = int(fail_match.group(1))
    skip_match = re.search(r"(\d+)\s+skipped", stdout + stderr)
    if skip_match:
        skipped = int(skip_match.group(1))

    gate = proc.returncode == 0 and failed == 0 and skipped <= 14

    result = {
        "benchmark": "B.5_final_regression",
        "return_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "run_time_s": round(run_time, 2),
        "gate_passed": gate,
    }
    print(f"  Tests: {passed} passed, {failed} failed, {skipped} skipped "
          f"(rc={proc.returncode}, {run_time:.1f}s)")
    return result


# =========================================================================
# B.6 Documentation Completeness
# =========================================================================


def benchmark_doc_completeness() -> dict:
    """B.6: Audit documentation completeness."""
    required_files = {
        "LICENSE": os.path.join(ROOT, "LICENSE"),
        "README.md": os.path.join(ROOT, "README.md"),
        "CITATION.cff": os.path.join(ROOT, "CITATION.cff"),
        "CODE_OF_CONDUCT.md": os.path.join(ROOT, "CODE_OF_CONDUCT.md"),
        "pyproject.toml": os.path.join(ROOT, "pyproject.toml"),
        "reproduce.py": os.path.join(ROOT, "reproduce.py"),
        "Year-4-Plan.md": os.path.join(ROOT, "Docs", "Planning_Documentation",
                                        "Year-4-Plan.md"),
        "Arxiv_Preprint_Outline.md": os.path.join(ROOT, "Docs",
                                                    "Arxiv_Preprint_Outline.md"),
        "Year_4_Comprehensive_Report.md": os.path.join(ROOT, "Docs",
                                                         "Year_4_Comprehensive_Report.md"),
        "Project_Retrospective.md": os.path.join(ROOT, "Docs",
                                                   "Project_Retrospective.md"),
    }

    triad_dirs = [
        os.path.join(ROOT, "benchmarks"),
        os.path.join(ROOT, "tests"),
        os.path.join(ROOT, "src", "prinet"),
    ]
    triad_files = ["README.md", "CHANGELOG.md"]

    found = {}
    for name, path in required_files.items():
        found[name] = os.path.isfile(path)

    triad_ok = 0
    triad_total = 0
    for d in triad_dirs:
        for tf in triad_files:
            triad_total += 1
            if os.path.isfile(os.path.join(d, tf)):
                triad_ok += 1

    n_found = sum(found.values())
    gate = n_found >= 9 and triad_ok >= 5

    result = {
        "benchmark": "B.6_doc_completeness",
        "root_files_found": n_found,
        "root_files_total": len(required_files),
        "root_files_detail": {k: v for k, v in found.items()},
        "triad_ok": triad_ok,
        "triad_total": triad_total,
        "gate_passed": gate,
    }
    print(f"  Root files: {n_found}/{len(required_files)}")
    print(f"  Triad (README+CHANGELOG): {triad_ok}/{triad_total}")
    return result


# =========================================================================
# B.7 Project Metrics Summary
# =========================================================================


def benchmark_project_metrics() -> dict:
    """B.7: Collect final project metrics."""
    # Version
    version = "unknown"
    try:
        import prinet
        version = getattr(prinet, "__version__", "unknown")
    except ImportError:
        pass

    # API surface
    api_count = 0
    try:
        import prinet
        api_count = len([a for a in dir(prinet) if not a.startswith("_")])
    except ImportError:
        pass

    # Benchmark artefacts
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    n_artefacts = len(json_files)

    # Test files
    test_files = glob.glob(os.path.join(ROOT, "tests", "test_*.py"))
    n_test_files = len(test_files)

    # Benchmark scripts
    bench_files = glob.glob(os.path.join(ROOT, "benchmarks", "*.py"))
    n_bench_files = len(bench_files)

    # Notebooks
    nb_files = glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))
    n_notebooks = len(nb_files)

    # Source lines of code (rough)
    src_dir = os.path.join(ROOT, "src", "prinet")
    sloc = 0
    if os.path.isdir(src_dir):
        for py in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
            try:
                with open(py, "r", encoding="utf-8", errors="replace") as f:
                    sloc += sum(1 for line in f if line.strip()
                                and not line.strip().startswith("#"))
            except OSError:
                pass

    result = {
        "benchmark": "B.7_project_metrics",
        "version": version,
        "api_symbols": api_count,
        "benchmark_artefacts": n_artefacts,
        "test_files": n_test_files,
        "benchmark_scripts": n_bench_files,
        "notebooks": n_notebooks,
        "source_lines": sloc,
        "gate_passed": (
            version >= "3.0.0"
            and api_count >= 120
            and n_artefacts >= 100
            and n_test_files >= 25
        ),
    }
    print(f"  Version: {version}")
    print(f"  API symbols: {api_count}")
    print(f"  Benchmark artefacts: {n_artefacts}")
    print(f"  Test files: {n_test_files}")
    print(f"  Benchmark scripts: {n_bench_files}")
    print(f"  Notebooks: {n_notebooks}")
    print(f"  Source lines: {sloc}")
    return result


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    print("=" * 60)
    print("Year 4 Q4 Benchmarks: Review Response & Archival Close")
    print("=" * 60)

    all_results = []

    benchmarks = [
        ("B.1 Year 4 Report", benchmark_year4_report),
        ("B.2 Retrospective", benchmark_retrospective),
        ("B.3 Archive Files", benchmark_archive_files),
        ("B.4 SHA-256 Manifest", benchmark_sha256_manifest),
        ("B.5 Final Regression", benchmark_final_regression),
        ("B.6 Doc Completeness", benchmark_doc_completeness),
        ("B.7 Project Metrics", benchmark_project_metrics),
    ]

    for name, fn in benchmarks:
        print(f"\n--- {name} ---")
        try:
            result = fn()
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"benchmark": name, "error": str(e),
                                "gate_passed": False})

    # Save combined results
    combined = {
        "suite": "y4q4_archival_close",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
        "gates_passed": sum(1 for r in all_results if r.get("gate_passed")),
        "gates_total": len(all_results),
    }
    _save("y4q4_benchmarks.json", combined)

    print("\n" + "=" * 60)
    gates = sum(1 for r in all_results if r.get("gate_passed"))
    total = len(all_results)
    print(f"Gates passed: {gates}/{total}")
    if gates == total:
        print("ALL GATES PASSED -- Year 4 Q4 archival close COMPLETE")
    else:
        failed_gates = [
            r["benchmark"] for r in all_results if not r.get("gate_passed")
        ]
        print(f"Failed gates: {', '.join(failed_gates)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
