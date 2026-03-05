"""Year 4 Q3 Benchmarks: v3.0.0 Release Readiness.

Validates all Q3 deliverables:
  B.1  CUDA JIT compilation and performance (V.1)
  B.2  Test skip reduction assessment (V.2)
  B.3  Sphinx documentation build (V.3)
  B.4  Notebook validation (V.4)
  B.5  Version and API surface (V.5)
  B.6  Reproducibility script (V.6)
"""

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
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> {path}")


def benchmark_cuda_jit() -> dict:
    """B.1: CUDA JIT compile + benchmark."""
    import torch
    from prinet.utils.fused_kernels import (
        cuda_fused_kernel_available,
        fused_discrete_step_cuda,
        pytorch_fused_discrete_step_full,
    )

    available = cuda_fused_kernel_available()
    print(f"  CUDA kernel available: {available}")

    B, nd, nt, ng = 32, 4, 8, 32
    N = nd + nt + ng
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    phase = (torch.rand(B, N) * 6.2832).to(device)
    amp = torch.ones(B, N, device=device)
    fd = torch.full((nd,), 2.0, device=device)
    ft = torch.full((nt,), 6.0, device=device)
    fg = torch.full((ng,), 40.0, device=device)
    Wd = (torch.randn(nd, nd) * 0.5).to(device)
    Wt = (torch.randn(nt, nt) * 0.25).to(device)
    Wg = (torch.randn(ng, ng) * 0.125).to(device)
    W_pac_dt_w = (torch.randn(nt, 2 * nd) * 0.1).to(device)
    W_pac_dt_b = torch.zeros(nt, device=device)
    W_pac_tg_w = (torch.randn(ng, 2 * nt) * 0.1).to(device)
    W_pac_tg_b = torch.zeros(ng, device=device)

    args = (phase, amp, fd, ft, fg, Wd, Wt, Wg,
            W_pac_dt_w, W_pac_dt_b, W_pac_tg_w, W_pac_tg_b,
            1.0, 1.0, 1.0, 0.01, nd, nt, ng)

    # Warmup
    for _ in range(10):
        pytorch_fused_discrete_step_full(*args)
    if available:
        for _ in range(10):
            fused_discrete_step_cuda(*args)
        torch.cuda.synchronize()

    # Benchmark PyTorch
    n_iter = 500
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pytorch_fused_discrete_step_full(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pt_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark CUDA
    cuda_ms = None
    speedup = None
    if available:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fused_discrete_step_cuda(*args)
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) / n_iter * 1000
        speedup = pt_ms / cuda_ms

    result = {
        "benchmark": "B.1_cuda_jit",
        "cuda_available": available,
        "pytorch_ms_per_step": round(pt_ms, 4),
        "cuda_ms_per_step": round(cuda_ms, 4) if cuda_ms else None,
        "speedup": round(speedup, 2) if speedup else None,
        "batch_size": B,
        "n_oscillators": N,
        "n_iterations": n_iter,
        "device": device,
        "gate_passed": available and speedup is not None and speedup > 1.0,
    }
    print(f"  PyTorch: {pt_ms:.4f} ms/step")
    if cuda_ms:
        print(f"  CUDA:    {cuda_ms:.4f} ms/step ({speedup:.1f}x)")
    return result


def benchmark_test_skips() -> dict:
    """B.2: Run test suite, count skips."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-rs"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          timeout=600)
    output = proc.stdout + proc.stderr
    lines = output.strip().split("\n")

    # Parse final summary line (may have '=' decorations)
    summary = ""
    for line in reversed(lines):
        if "passed" in line or "failed" in line:
            summary = line.strip().strip("=").strip()
            break
    passed = skipped = failed = 0
    for part in summary.split(","):
        part = part.strip()
        tokens = [t for t in part.split() if t.isdigit()]
        if not tokens:
            continue
        count = int(tokens[0])
        if "passed" in part:
            passed = count
        elif "skipped" in part:
            skipped = count
        elif "failed" in part:
            failed = count

    skip_reasons = [l for l in lines if "SKIPPED" in l]
    result = {
        "benchmark": "B.2_test_skips",
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "target_skips": 10,
        "skip_reasons": skip_reasons,
        "gate_passed": failed == 0,
    }
    print(f"  Passed: {passed}, Skipped: {skipped}, Failed: {failed}")
    return result


def benchmark_sphinx_build() -> dict:
    """B.3: Build Sphinx docs, count pages."""
    docs_dir = os.path.join(ROOT, "Docs")
    build_dir = os.path.join(docs_dir, "_build")

    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", docs_dir, build_dir,
           "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          timeout=300)
    build_time = time.perf_counter() - t0

    # Count HTML files
    html_count = 0
    if os.path.isdir(build_dir):
        for dirpath, _, filenames in os.walk(build_dir):
            html_count += sum(1 for f in filenames if f.endswith(".html"))

    result = {
        "benchmark": "B.3_sphinx_build",
        "build_time_s": round(build_time, 2),
        "html_pages": html_count,
        "return_code": proc.returncode,
        "gate_passed": proc.returncode == 0 and html_count > 0,
    }
    print(f"  Build: {build_time:.1f}s, {html_count} HTML pages, rc={proc.returncode}")
    return result


def benchmark_notebooks() -> dict:
    """B.4: Validate notebook JSON structure."""
    nb_dir = os.path.join(ROOT, "notebooks")
    notebooks = {}
    for f in sorted(os.listdir(nb_dir)):
        if not f.endswith(".ipynb"):
            continue
        path = os.path.join(nb_dir, f)
        try:
            with open(path) as fh:
                data = json.load(fh)
            n_cells = len(data.get("cells", []))
            n_code = sum(1 for c in data["cells"] if c["cell_type"] == "code")
            n_md = sum(1 for c in data["cells"] if c["cell_type"] == "markdown")
            notebooks[f] = {
                "cells": n_cells,
                "code_cells": n_code,
                "markdown_cells": n_md,
                "valid": True,
            }
        except Exception as e:
            notebooks[f] = {"valid": False, "error": str(e)}

    result = {
        "benchmark": "B.4_notebooks",
        "notebooks": notebooks,
        "total": len(notebooks),
        "all_valid": all(v.get("valid", False) for v in notebooks.values()),
        "gate_passed": len(notebooks) >= 3 and all(
            v.get("valid", False) for v in notebooks.values()
        ),
    }
    for name, info in notebooks.items():
        status = "OK" if info.get("valid") else "FAIL"
        print(f"  {name}: {status} ({info.get('cells', '?')} cells)")
    return result


def benchmark_version_api() -> dict:
    """B.5: Check version and API surface."""
    import prinet

    version = prinet.__version__
    n_symbols = len(prinet.__all__)
    parts = version.split(".")
    is_semver = len(parts) == 3 and all(p.isdigit() for p in parts)
    major = int(parts[0]) if is_semver else 0

    result = {
        "benchmark": "B.5_version_api",
        "version": version,
        "is_semver": is_semver,
        "major_version": major,
        "public_symbols": n_symbols,
        "target_symbols": 120,
        "gate_passed": is_semver and major >= 3 and n_symbols >= 120,
    }
    print(f"  Version: {version}, Symbols: {n_symbols}")
    return result


def benchmark_reproduce() -> dict:
    """B.6: Run reproduce.py and check output."""
    reproduce_py = os.path.join(ROOT, "reproduce.py")
    if not os.path.exists(reproduce_py):
        return {
            "benchmark": "B.6_reproduce",
            "exists": False,
            "gate_passed": False,
        }

    t0 = time.perf_counter()
    cmd = [sys.executable, reproduce_py]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          timeout=120)
    run_time = time.perf_counter() - t0

    result = {
        "benchmark": "B.6_reproduce",
        "exists": True,
        "return_code": proc.returncode,
        "run_time_s": round(run_time, 2),
        "gate_passed": proc.returncode == 0,
    }
    print(f"  reproduce.py: rc={proc.returncode}, {run_time:.1f}s")
    return result


def main() -> None:
    print("=" * 60)
    print("Year 4 Q3 Benchmarks: v3.0.0 Release Readiness")
    print("=" * 60)

    all_results = []

    benchmarks = [
        ("B.1 CUDA JIT", benchmark_cuda_jit),
        ("B.2 Test Skips", benchmark_test_skips),
        ("B.3 Sphinx Build", benchmark_sphinx_build),
        ("B.4 Notebooks", benchmark_notebooks),
        ("B.5 Version/API", benchmark_version_api),
        ("B.6 Reproduce", benchmark_reproduce),
    ]

    for name, fn in benchmarks:
        print(f"\n--- {name} ---")
        try:
            result = fn()
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"benchmark": name, "error": str(e)})

    # Save combined results
    combined = {
        "suite": "y4q3_release_readiness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
        "gates_passed": sum(1 for r in all_results if r.get("gate_passed")),
        "gates_total": len(all_results),
    }
    _save("y4q3_benchmarks.json", combined)

    print("\n" + "=" * 60)
    gates = sum(1 for r in all_results if r.get("gate_passed"))
    print(f"Gates passed: {gates}/{len(all_results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
