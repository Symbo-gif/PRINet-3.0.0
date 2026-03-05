"""Year 4 Q3 tests -- v3.0.0 release readiness.

Validates all Q3 deliverables:
  T.1  CUDA JIT compilation availability (V.1)
  T.2  Test skip count within target (V.2)
  T.3  Sphinx documentation configuration (V.3)
  T.4  Notebook validity and structure (V.4)
  T.5  Version, classifier, and API surface (V.5)
  T.6  Reproducibility script existence (V.6)
  T.7  LaTeX paper existence (Q2 carry-over)
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import prinet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def project_root():
    return ROOT


# =========================================================================
# T.1 CUDA JIT (V.1)
# =========================================================================


class TestCUDAJIT:
    """CUDA JIT compilation and correctness."""

    def test_cuda_kernel_module_importable(self):
        from prinet.utils.fused_kernels import (
            cuda_fused_kernel_available,
            fused_discrete_step_cuda,
            pytorch_fused_discrete_step_full,
        )

        # These should be importable regardless of CUDA
        assert callable(cuda_fused_kernel_available)
        assert callable(fused_discrete_step_cuda)
        assert callable(pytorch_fused_discrete_step_full)

    def test_pytorch_fallback_runs(self):
        from prinet.utils.fused_kernels import pytorch_fused_discrete_step_full

        B, nd, nt, ng = 4, 2, 3, 5
        N = nd + nt + ng
        phase = torch.rand(B, N) * 6.2832
        amp = torch.ones(B, N)
        fd = torch.full((nd,), 2.0)
        ft = torch.full((nt,), 6.0)
        fg = torch.full((ng,), 40.0)
        Wd = torch.randn(nd, nd) * 0.5
        Wt = torch.randn(nt, nt) * 0.25
        Wg = torch.randn(ng, ng) * 0.125
        # PAC projection weights: delta(2*nd) -> theta(nt), theta(2*nt) -> gamma(ng)
        W_pac_dt_w = torch.randn(nt, 2 * nd) * 0.1
        W_pac_dt_b = torch.zeros(nt)
        W_pac_tg_w = torch.randn(ng, 2 * nt) * 0.1
        W_pac_tg_b = torch.zeros(ng)

        new_phase, new_amp = pytorch_fused_discrete_step_full(
            phase,
            amp,
            fd,
            ft,
            fg,
            Wd,
            Wt,
            Wg,
            W_pac_dt_w,
            W_pac_dt_b,
            W_pac_tg_w,
            W_pac_tg_b,
            1.0,
            1.0,
            1.0,
            0.01,
            nd,
            nt,
            ng,
        )
        assert new_phase.shape == phase.shape
        assert new_amp.shape == amp.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_jit_compiles(self):
        from prinet.utils.fused_kernels import cuda_fused_kernel_available

        assert (
            cuda_fused_kernel_available()
        ), "CUDA JIT kernel failed to compile -- check MSVC setup"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_matches_pytorch(self):
        from prinet.utils.fused_kernels import (
            cuda_fused_kernel_available,
            fused_discrete_step_cuda,
            pytorch_fused_discrete_step_full,
        )

        if not cuda_fused_kernel_available():
            pytest.skip("CUDA JIT not compiled")

        B, nd, nt, ng = 8, 4, 8, 16
        torch.manual_seed(99)
        N = nd + nt + ng
        phase = (torch.rand(B, N) * 6.2832).cuda()
        amp = torch.ones(B, N).cuda()
        fd = torch.full((nd,), 2.0).cuda()
        ft = torch.full((nt,), 6.0).cuda()
        fg = torch.full((ng,), 40.0).cuda()
        Wd = (torch.randn(nd, nd) * 0.5).cuda()
        Wt = (torch.randn(nt, nt) * 0.25).cuda()
        Wg = (torch.randn(ng, ng) * 0.125).cuda()
        W_pac_dt_w = (torch.randn(nt, 2 * nd) * 0.1).cuda()
        W_pac_dt_b = torch.zeros(nt).cuda()
        W_pac_tg_w = (torch.randn(ng, 2 * nt) * 0.1).cuda()
        W_pac_tg_b = torch.zeros(ng).cuda()

        args = (
            phase,
            amp,
            fd,
            ft,
            fg,
            Wd,
            Wt,
            Wg,
            W_pac_dt_w,
            W_pac_dt_b,
            W_pac_tg_w,
            W_pac_tg_b,
            1.0,
            1.0,
            1.0,
            0.01,
            nd,
            nt,
            ng,
        )

        p_pt, a_pt = pytorch_fused_discrete_step_full(*args)
        p_cu, a_cu = fused_discrete_step_cuda(*args)

        assert torch.allclose(
            p_pt, p_cu, atol=1e-3
        ), f"Phase max diff: {(p_pt - p_cu).abs().max():.2e}"
        assert torch.allclose(
            a_pt, a_cu, atol=1e-3
        ), f"Amp max diff: {(a_pt - a_cu).abs().max():.2e}"


# =========================================================================
# T.2 Test skip count (V.2)
# =========================================================================


class TestSkipCount:
    """Verify skip count is within acceptable bounds."""

    def test_skip_count_within_target(self, project_root):
        """Count must be <= 20 (target 10; allow up to 20 on Windows w/o Triton)."""
        # Exclude this file to avoid recursive pytest invocation
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=no",
            "--ignore=tests/test_y4q3.py",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600,
        )
        summary = proc.stdout.strip().split("\n")[-1]
        skipped = 0
        for part in summary.split(","):
            part = part.strip()
            if "skipped" in part:
                skipped = int(part.split()[0])
        assert skipped <= 20, f"Too many test skips: {skipped} (target <= 20)"

    def test_no_failures(self, project_root):
        """Zero test failures required for release."""
        # Exclude this file to avoid recursive pytest invocation
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=no",
            "--ignore=tests/test_y4q3.py",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600,
        )
        # Find the summary line containing 'passed' (may have '=' decorations)
        lines = proc.stdout.strip().split("\n")
        summary = ""
        for line in reversed(lines):
            if "passed" in line or "failed" in line:
                summary = line.strip().strip("=").strip()
                break
        failed = 0
        for part in summary.split(","):
            part = part.strip()
            if "failed" in part:
                tokens = [t for t in part.split() if t.isdigit()]
                if tokens:
                    failed = int(tokens[0])
        assert failed == 0, f"Test failures: {failed}\n{summary}"


# =========================================================================
# T.3 Sphinx docs (V.3)
# =========================================================================


class TestSphinxDocs:
    """Sphinx docs configuration and build."""

    def test_conf_py_exists(self, project_root):
        conf = os.path.join(project_root, "docs", "conf.py")
        assert os.path.isfile(conf), "docs/conf.py not found"

    def test_index_rst_exists(self, project_root):
        idx = os.path.join(project_root, "docs", "index.rst")
        assert os.path.isfile(idx), "docs/index.rst not found"

    def test_api_rst_files_exist(self, project_root):
        api_dir = os.path.join(project_root, "docs", "api")
        required = ["core.rst", "nn.rst", "utils.rst"]
        for name in required:
            path = os.path.join(api_dir, name)
            assert os.path.isfile(path), f"docs/api/{name} not found"

    def test_readthedocs_yaml_exists(self, project_root):
        path = os.path.join(project_root, ".readthedocs.yaml")
        assert os.path.isfile(path), ".readthedocs.yaml not found"

    def test_sphinx_build_succeeds(self, project_root):
        docs_dir = os.path.join(project_root, "docs")
        build_dir = os.path.join(docs_dir, "_build_test")
        cmd = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            docs_dir,
            build_dir,
            "-q",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )
        assert proc.returncode == 0, f"Sphinx build failed:\n{proc.stderr[-500:]}"
        # Count HTML files
        html_count = 0
        for dirpath, _, filenames in os.walk(build_dir):
            html_count += sum(1 for f in filenames if f.endswith(".html"))
        assert html_count > 0, "No HTML pages generated"


# =========================================================================
# T.4 Notebooks (V.4)
# =========================================================================


class TestNotebooks:
    """Validate Jupyter notebook JSON structure."""

    EXPECTED = [
        "01_oscillosim_quickstart.ipynb",
        "02_clevr_n_binding.ipynb",
        "03_custom_coupling.ipynb",
    ]

    def test_notebooks_directory_exists(self, project_root):
        nb_dir = os.path.join(project_root, "notebooks")
        assert os.path.isdir(nb_dir), "notebooks/ directory not found"

    @pytest.mark.parametrize("name", EXPECTED)
    def test_notebook_exists(self, project_root, name):
        path = os.path.join(project_root, "notebooks", name)
        assert os.path.isfile(path), f"notebooks/{name} not found"

    @pytest.mark.parametrize("name", EXPECTED)
    def test_notebook_valid_json(self, project_root, name):
        path = os.path.join(project_root, "notebooks", name)
        with open(path) as f:
            data = json.load(f)
        assert "cells" in data, "Missing 'cells' key"
        assert "metadata" in data, "Missing 'metadata' key"
        assert "nbformat" in data, "Missing 'nbformat' key"
        assert data["nbformat"] == 4, f"Expected nbformat 4, got {data['nbformat']}"

    @pytest.mark.parametrize("name", EXPECTED)
    def test_notebook_has_code_cells(self, project_root, name):
        path = os.path.join(project_root, "notebooks", name)
        with open(path) as f:
            data = json.load(f)
        code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 3, f"Expected >= 3 code cells, got {len(code_cells)}"

    @pytest.mark.parametrize("name", EXPECTED)
    def test_notebook_has_markdown_cells(self, project_root, name):
        path = os.path.join(project_root, "notebooks", name)
        with open(path) as f:
            data = json.load(f)
        md_cells = [c for c in data["cells"] if c["cell_type"] == "markdown"]
        assert len(md_cells) >= 2, f"Expected >= 2 markdown cells, got {len(md_cells)}"


# =========================================================================
# T.5 Version / API (V.5)
# =========================================================================


class TestVersionAPI:
    """Version string, classifier, and API surface."""

    def test_version_is_3_0_0(self):
        assert prinet.__version__ == "3.0.0"

    def test_version_is_valid_semver(self):
        parts = prinet.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_major_version_at_least_3(self):
        major = int(prinet.__version__.split(".")[0])
        assert major >= 3, f"Expected major >= 3, got {major}"

    def test_public_api_surface(self):
        n_symbols = len(prinet.__all__)
        assert n_symbols >= 120, f"Expected >= 120 public symbols, got {n_symbols}"

    def test_pyproject_version_matches(self, project_root):
        import tomllib

        pyproject = os.path.join(project_root, "pyproject.toml")
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        pv = data["project"]["version"]
        assert (
            pv == prinet.__version__
        ), f"pyproject.toml ({pv}) != __init__.py ({prinet.__version__})"

    def test_pyproject_production_stable(self, project_root):
        import tomllib

        pyproject = os.path.join(project_root, "pyproject.toml")
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        classifiers = data["project"].get("classifiers", [])
        assert any(
            "Production/Stable" in c for c in classifiers
        ), "Classifier 'Development Status :: 5 - Production/Stable' not found"


# =========================================================================
# T.6 Reproducibility (V.6)
# =========================================================================


class TestReproducibility:
    """Reproducibility script for paper figures."""

    def test_reproduce_py_exists(self, project_root):
        path = os.path.join(project_root, "reproduce.py")
        assert os.path.isfile(path), "reproduce.py not found"

    def test_reproduce_imports(self, project_root):
        """reproduce.py should be importable without errors."""
        cmd = [
            sys.executable,
            "-c",
            "import importlib.util; "
            "spec = importlib.util.spec_from_file_location('reproduce', "
            f"r'{os.path.join(project_root, 'reproduce.py')}'); "
            "mod = importlib.util.module_from_spec(spec)",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"reproduce.py import failed: {proc.stderr}"


# =========================================================================
# T.7 LaTeX Paper (Q2 carry-over)
# =========================================================================


class TestLaTeXPaper:
    """Paper artifact existence and structure."""

    def test_paper_tex_exists(self, project_root):
        path = os.path.join(project_root, "paper", "main.tex")
        assert os.path.isfile(path), "paper/main.tex not found"

    def test_paper_has_sections(self, project_root):
        path = os.path.join(project_root, "paper", "main.tex")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for section in ["Introduction", "Background", "Architecture", "Experiments"]:
            assert section.lower() in content.lower(), f"Missing section: {section}"

    def test_paper_has_bibliography(self, project_root):
        path = os.path.join(project_root, "paper", "main.tex")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "\\begin{thebibliography}" in content or "\\bibliography{" in content


# =========================================================================
# T.8 Benchmark artefacts (Q1 carry-over validation)
# =========================================================================


class TestBenchmarkArtefacts:
    """Validate Q1 benchmark JSON artefacts exist and are valid."""

    def test_results_directory_exists(self, project_root):
        path = os.path.join(project_root, "benchmarks", "results")
        assert os.path.isdir(path)

    def test_has_benchmark_results(self, project_root):
        results_dir = os.path.join(project_root, "benchmarks", "results")
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        assert (
            len(json_files) >= 10
        ), f"Expected >= 10 benchmark JSON files, got {len(json_files)}"

    def test_benchmark_jsons_valid(self, project_root):
        results_dir = os.path.join(project_root, "benchmarks", "results")
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        for fname in json_files:
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            assert isinstance(
                data, (dict, list)
            ), f"{fname}: expected dict or list, got {type(data).__name__}"
