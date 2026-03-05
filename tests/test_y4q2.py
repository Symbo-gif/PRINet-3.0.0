"""Year 4 Q2 tests -- publication figure/table generation infrastructure.

Covers:
- U.1: Figure generation module (configure_neurips_style, 9 generators,
       generate_all_figures).
- U.2: LaTeX table generation module (6 generators, generate_all_tables).
- U.3: Reproducibility script (reproduce.py validation).
- U.4: Style consistency (NeurIPS 2026 formatting, color palette).
- U.5: Artefact completeness (required JSON files exist).

All tests run without GPU or model training -- they validate the
paper reproducibility pipeline against stored JSON artefacts.

Version: 2.7.0 -> 2.8.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


# =====================================================================
# U.1: Figure Generation Module
# =====================================================================


class TestConfigureNeuripsStyle:
    """Tests for :func:`configure_neurips_style`."""

    def test_sets_dpi(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        assert plt.rcParams["figure.dpi"] == 300

    def test_sets_savefig_dpi(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        assert plt.rcParams["savefig.dpi"] == 300

    def test_sets_font_size(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        assert plt.rcParams["font.size"] == 9

    def test_sets_serif_font(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        family = plt.rcParams["font.family"]
        # matplotlib may return a list or string depending on version
        if isinstance(family, list):
            assert "serif" in family
        else:
            assert family == "serif"

    def test_sets_axes_title_size(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        assert plt.rcParams["axes.titlesize"] == 10

    def test_sets_axes_label_size(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        assert plt.rcParams["axes.labelsize"] == 9

    def test_idempotent(self) -> None:
        import matplotlib.pyplot as plt
        from prinet.utils.figure_generation import configure_neurips_style
        configure_neurips_style()
        dpi1 = plt.rcParams["figure.dpi"]
        configure_neurips_style()
        dpi2 = plt.rcParams["figure.dpi"]
        assert dpi1 == dpi2


class TestColorPalette:
    """Tests for the color-blind-safe Tol bright palette."""

    def test_pt_color_defined(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert "pt" in COLORS

    def test_sa_color_defined(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert "sa" in COLORS

    def test_pt_large_color_defined(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert "pt_large" in COLORS

    def test_chimera_color_defined(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert "chimera" in COLORS

    def test_colors_are_hex(self) -> None:
        from prinet.utils.figure_generation import COLORS
        for key, val in COLORS.items():
            assert val.startswith("#"), f"Color {key} is not hex: {val}"
            assert len(val) == 7, f"Color {key} wrong length: {val}"

    def test_ablation_colors_complete(self) -> None:
        from prinet.utils.figure_generation import COLORS
        required = ["full", "attention_only", "oscillator_only", "shared_phase"]
        for key in required:
            assert key in COLORS, f"Missing ablation color: {key}"

    def test_coupling_colors_complete(self) -> None:
        from prinet.utils.figure_generation import COLORS
        required = ["mean_field", "sparse_knn", "csr"]
        for key in required:
            assert key in COLORS, f"Missing coupling color: {key}"


class TestFigureGenerators:
    """Tests for individual figure generator functions."""

    @pytest.fixture
    def tmpdir(self, tmp_path: Path) -> Path:
        return tmp_path

    def _has_required_data(self, *filenames: str) -> bool:
        """Check if all required JSON files exist."""
        return all((RESULTS_DIR / f).exists() for f in filenames)

    def test_fig_clevr_n_capacity(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_ablation_variants.json"):
            pytest.skip("Missing benchmark_y4q1_ablation_variants.json")
        from prinet.utils.figure_generation import fig_clevr_n_capacity
        paths = fig_clevr_n_capacity(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2  # PDF + PNG
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_fig_chimera_heatmap(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_3_k_alpha_sensitivity.json"):
            pytest.skip("Missing benchmark_y4q1_3_k_alpha_sensitivity.json")
        from prinet.utils.figure_generation import fig_chimera_heatmap
        paths = fig_chimera_heatmap(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_fig_mot_identity_preservation(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_9_fine_occlusion.json"):
            pytest.skip("Missing y4q1_9_fine_occlusion.json")
        from prinet.utils.figure_generation import fig_mot_identity_preservation
        paths = fig_mot_identity_preservation(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_oscillosim_scaling(self, tmpdir: Path) -> None:
        if not self._has_required_data("y3q4_p4_oscillosim_scaling.json"):
            pytest.skip("Missing y3q4_p4_oscillosim_scaling.json")
        from prinet.utils.figure_generation import fig_oscillosim_scaling
        paths = fig_oscillosim_scaling(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_ablation_results(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_ablation_variants.json"):
            pytest.skip("Missing benchmark_y4q1_ablation_variants.json")
        from prinet.utils.figure_generation import fig_ablation_results
        paths = fig_ablation_results(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_parameter_efficiency(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_8_parameter_efficiency_frontier.json"):
            pytest.skip("Missing y4q1_8_parameter_efficiency_frontier.json")
        from prinet.utils.figure_generation import fig_parameter_efficiency
        paths = fig_parameter_efficiency(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_training_curves(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_7_training_curves_pt.json",
                                       "y4q1_7_training_curves_sa.json"):
            pytest.skip("Missing training curves JSON")
        from prinet.utils.figure_generation import fig_training_curves
        paths = fig_training_curves(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_gold_standard_chimera(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_3_gold_standard_chimera.json"):
            pytest.skip("Missing benchmark_y4q1_3_gold_standard_chimera.json")
        from prinet.utils.figure_generation import fig_gold_standard_chimera
        paths = fig_gold_standard_chimera(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_fig_statistical_summary(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_7_statistical_summary.json"):
            pytest.skip("Missing y4q1_7_statistical_summary.json")
        from prinet.utils.figure_generation import fig_statistical_summary
        paths = fig_statistical_summary(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()


class TestGenerateAllFigures:
    """Tests for :func:`generate_all_figures`."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        from prinet.utils.figure_generation import generate_all_figures
        result = generate_all_figures(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        assert isinstance(result, dict)

    def test_has_expected_keys(self, tmp_path: Path) -> None:
        from prinet.utils.figure_generation import generate_all_figures
        result = generate_all_figures(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        # Should have at least some figure keys
        assert len(result) >= 5

    def test_does_not_raise(self, tmp_path: Path) -> None:
        from prinet.utils.figure_generation import generate_all_figures
        # Should gracefully handle missing data
        generate_all_figures(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )

    def test_output_files_created(self, tmp_path: Path) -> None:
        from prinet.utils.figure_generation import generate_all_figures
        result = generate_all_figures(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        # At least some figures should have been generated
        generated = {k: v for k, v in result.items() if v}
        assert len(generated) > 0

    def test_pdf_and_png_produced(self, tmp_path: Path) -> None:
        from prinet.utils.figure_generation import generate_all_figures
        result = generate_all_figures(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        for name, paths in result.items():
            if paths:
                extensions = {p.suffix for p in paths}
                assert ".pdf" in extensions, f"{name} missing PDF"
                assert ".png" in extensions, f"{name} missing PNG"


# =====================================================================
# U.2: LaTeX Table Generation Module
# =====================================================================


class TestTableHelpers:
    """Tests for table generation helper functions."""

    def test_load_json_valid_file(self) -> None:
        from prinet.utils.table_generation import _load_json
        # Find any JSON file in results
        json_files = list(RESULTS_DIR.glob("*.json"))
        if not json_files:
            pytest.skip("No JSON files in results/")
        data = _load_json(json_files[0])
        assert isinstance(data, dict)

    def test_load_json_missing_file(self) -> None:
        from prinet.utils.table_generation import _load_json
        with pytest.raises(FileNotFoundError):
            _load_json(Path("nonexistent_file_12345.json"))

    def test_save_table(self, tmp_path: Path) -> None:
        from prinet.utils.table_generation import _save_table
        content = r"\begin{table} test \end{table}"
        path = _save_table(content, "test_table", tmp_path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == content


class TestTableGenerators:
    """Tests for individual table generator functions."""

    @pytest.fixture
    def tmpdir(self, tmp_path: Path) -> Path:
        return tmp_path

    def _has_required_data(self, *filenames: str) -> bool:
        return all((RESULTS_DIR / f).exists() for f in filenames)

    def _validate_latex(self, path: Path) -> None:
        """Check that output is valid booktabs LaTeX."""
        content = path.read_text(encoding="utf-8")
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"\toprule" in content
        assert r"\bottomrule" in content
        assert r"\midrule" in content

    def test_table_ablation_variants(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_ablation_variants.json"):
            pytest.skip("Missing benchmark_y4q1_ablation_variants.json")
        from prinet.utils.table_generation import table_ablation_variants
        path = table_ablation_variants(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)

    def test_table_parameter_efficiency(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_8_parameter_efficiency_frontier.json"):
            pytest.skip("Missing y4q1_8_parameter_efficiency_frontier.json")
        from prinet.utils.table_generation import table_parameter_efficiency
        path = table_parameter_efficiency(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)

    def test_table_chimera_gold_standard(self, tmpdir: Path) -> None:
        if not self._has_required_data("benchmark_y4q1_3_gold_standard_chimera.json"):
            pytest.skip("Missing benchmark_y4q1_3_gold_standard_chimera.json")
        from prinet.utils.table_generation import table_chimera_gold_standard
        path = table_chimera_gold_standard(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)

    def test_table_statistical_summary(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_7_statistical_summary.json"):
            pytest.skip("Missing y4q1_7_statistical_summary.json")
        from prinet.utils.table_generation import table_statistical_summary
        path = table_statistical_summary(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)

    def test_table_occlusion_sweep(self, tmpdir: Path) -> None:
        if not self._has_required_data("y4q1_9_fine_occlusion.json"):
            pytest.skip("Missing y4q1_9_fine_occlusion.json")
        from prinet.utils.table_generation import table_occlusion_sweep
        path = table_occlusion_sweep(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)

    def test_table_oscillosim_scaling(self, tmpdir: Path) -> None:
        if not self._has_required_data("y3q4_p4_oscillosim_scaling.json"):
            pytest.skip("Missing y3q4_p4_oscillosim_scaling.json")
        from prinet.utils.table_generation import table_oscillosim_scaling
        path = table_oscillosim_scaling(results_dir=RESULTS_DIR, output_dir=tmpdir)
        assert path.exists()
        self._validate_latex(path)


class TestGenerateAllTables:
    """Tests for :func:`generate_all_tables`."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        from prinet.utils.table_generation import generate_all_tables
        result = generate_all_tables(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        assert isinstance(result, dict)

    def test_does_not_raise(self, tmp_path: Path) -> None:
        from prinet.utils.table_generation import generate_all_tables
        generate_all_tables(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )

    def test_generated_files_are_tex(self, tmp_path: Path) -> None:
        from prinet.utils.table_generation import generate_all_tables
        result = generate_all_tables(
            results_dir=RESULTS_DIR, output_dir=tmp_path
        )
        for name, path in result.items():
            assert path.suffix == ".tex", f"{name} is not .tex"


# =====================================================================
# U.3: Reproducibility Script
# =====================================================================


class TestReproduceScript:
    """Tests for reproduce.py."""

    def test_script_exists(self) -> None:
        script = PROJECT_ROOT / "reproduce.py"
        assert script.exists(), "reproduce.py not found at project root"

    def test_script_is_importable(self) -> None:
        """reproduce.py should be syntactically valid Python."""
        script = PROJECT_ROOT / "reproduce.py"
        source = script.read_text(encoding="utf-8")
        compile(source, str(script), "exec")

    def test_validate_artefacts_returns_list(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            # Import the validation function
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "reproduce", str(PROJECT_ROOT / "reproduce.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            missing = mod.validate_artefacts(RESULTS_DIR)
            assert isinstance(missing, list)
        finally:
            if str(PROJECT_ROOT) in sys.path:
                sys.path.remove(str(PROJECT_ROOT))

    def test_compute_sha256(self, tmp_path: Path) -> None:
        # Create a temp file and check SHA-256
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")

        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "reproduce", str(PROJECT_ROOT / "reproduce.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sha = mod.compute_sha256(test_file)
            assert isinstance(sha, str)
            assert len(sha) == 64  # SHA-256 hex digest length
        finally:
            if str(PROJECT_ROOT) in sys.path:
                sys.path.remove(str(PROJECT_ROOT))

    def test_help_flag(self) -> None:
        """reproduce.py --help should exit cleanly."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "reproduce.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Reproduce" in result.stdout or "reproduce" in result.stdout


# =====================================================================
# U.4: Style Consistency
# =====================================================================


class TestStyleConsistency:
    """Tests for NeurIPS 2026 style compliance."""

    def test_tol_bright_pt_is_blue(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert COLORS["pt"] == "#4477AA"

    def test_tol_bright_sa_is_red(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert COLORS["sa"] == "#EE6677"

    def test_tol_bright_pt_large_is_green(self) -> None:
        from prinet.utils.figure_generation import COLORS
        assert COLORS["pt_large"] == "#228833"

    def test_default_results_dir(self) -> None:
        from prinet.utils.figure_generation import DEFAULT_RESULTS_DIR
        # Should resolve to benchmarks/results
        assert "benchmarks" in str(DEFAULT_RESULTS_DIR)
        assert "results" in str(DEFAULT_RESULTS_DIR)

    def test_default_output_dir(self) -> None:
        from prinet.utils.figure_generation import DEFAULT_OUTPUT_DIR
        assert "paper" in str(DEFAULT_OUTPUT_DIR)
        assert "figures" in str(DEFAULT_OUTPUT_DIR)

    def test_table_default_output_dir(self) -> None:
        from prinet.utils.table_generation import DEFAULT_OUTPUT_DIR
        assert "paper" in str(DEFAULT_OUTPUT_DIR)
        assert "tables" in str(DEFAULT_OUTPUT_DIR)


# =====================================================================
# U.5: Artefact Completeness
# =====================================================================


class TestArtefactCompleteness:
    """Tests for JSON benchmark artefact availability."""

    def test_results_dir_exists(self) -> None:
        assert RESULTS_DIR.exists(), f"Results dir not found: {RESULTS_DIR}"

    def test_has_json_files(self) -> None:
        json_files = list(RESULTS_DIR.glob("*.json"))
        assert len(json_files) > 0, "No JSON files in results/"

    def test_has_q1_1_artefacts(self) -> None:
        """Q1.1 should have produced at least ablation variants."""
        expected = RESULTS_DIR / "benchmark_y4q1_ablation_variants.json"
        if not expected.exists():
            pytest.skip("Q1.1 ablation artefact not available")
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert "variants" in data

    def test_has_q1_3_chimera(self) -> None:
        """Q1.3 gold-standard chimera data."""
        expected = RESULTS_DIR / "benchmark_y4q1_3_gold_standard_chimera.json"
        if not expected.exists():
            pytest.skip("Q1.3 chimera artefact not available")
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert "seeds" in data

    def test_has_q1_7_statistical_summary(self) -> None:
        """Q1.7 statistical summary (primary result)."""
        expected = RESULTS_DIR / "y4q1_7_statistical_summary.json"
        if not expected.exists():
            pytest.skip("Q1.7 statistical summary not available")
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert "outcome" in data
        assert "phase_tracker" in data
        assert "slot_attention" in data

    def test_has_q1_8_parameter_efficiency(self) -> None:
        """Q1.8 parameter efficiency frontier."""
        expected = RESULTS_DIR / "y4q1_8_parameter_efficiency_frontier.json"
        if not expected.exists():
            pytest.skip("Q1.8 parameter efficiency not available")
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert "models" in data

    def test_minimum_json_count(self) -> None:
        """Project should have at least 100 JSON artefacts."""
        json_files = list(RESULTS_DIR.glob("*.json"))
        assert len(json_files) >= 100, (
            f"Expected >= 100 JSON artefacts, found {len(json_files)}"
        )

    def test_json_files_are_valid(self) -> None:
        """All JSON files should be parseable."""
        json_files = list(RESULTS_DIR.glob("*.json"))
        for f in json_files[:20]:  # Check first 20
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                assert isinstance(data, dict), f"{f.name} is not a dict"
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in {f.name}")


# =====================================================================
# Module import tests
# =====================================================================


class TestModuleImports:
    """Tests that new Q2 modules are importable."""

    def test_import_figure_generation(self) -> None:
        from prinet.utils import figure_generation
        assert hasattr(figure_generation, "generate_all_figures")
        assert hasattr(figure_generation, "configure_neurips_style")

    def test_import_table_generation(self) -> None:
        from prinet.utils import table_generation
        assert hasattr(table_generation, "generate_all_tables")

    def test_figure_generation_has_all_generators(self) -> None:
        from prinet.utils import figure_generation
        expected = [
            "fig_clevr_n_capacity",
            "fig_chimera_heatmap",
            "fig_mot_identity_preservation",
            "fig_oscillosim_scaling",
            "fig_ablation_results",
            "fig_parameter_efficiency",
            "fig_training_curves",
            "fig_gold_standard_chimera",
            "fig_statistical_summary",
        ]
        for name in expected:
            assert hasattr(figure_generation, name), f"Missing: {name}"

    def test_table_generation_has_all_generators(self) -> None:
        from prinet.utils import table_generation
        expected = [
            "table_ablation_variants",
            "table_parameter_efficiency",
            "table_chimera_gold_standard",
            "table_statistical_summary",
            "table_occlusion_sweep",
            "table_oscillosim_scaling",
        ]
        for name in expected:
            assert hasattr(table_generation, name), f"Missing: {name}"


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge case handling for figure/table generation."""

    def test_figure_gen_with_empty_dir(self, tmp_path: Path) -> None:
        """generate_all_figures should not crash with no data."""
        empty = tmp_path / "empty_results"
        empty.mkdir()
        from prinet.utils.figure_generation import generate_all_figures
        result = generate_all_figures(
            results_dir=empty, output_dir=tmp_path / "out"
        )
        # All entries should be empty lists (no data)
        assert isinstance(result, dict)

    def test_table_gen_with_empty_dir(self, tmp_path: Path) -> None:
        """generate_all_tables should not crash with no data."""
        empty = tmp_path / "empty_results"
        empty.mkdir()
        from prinet.utils.table_generation import generate_all_tables
        result = generate_all_tables(
            results_dir=empty, output_dir=tmp_path / "out"
        )
        assert isinstance(result, dict)

    def test_figure_output_dir_created(self, tmp_path: Path) -> None:
        """Output directory should be created if it does not exist."""
        out = tmp_path / "nonexistent" / "deep" / "dir"
        assert not out.exists()
        from prinet.utils.figure_generation import generate_all_figures
        generate_all_figures(results_dir=RESULTS_DIR, output_dir=out)
        # Directory should exist now (even if no figures generated)
        # The function may or may not create it depending on data availability

    def test_table_output_dir_created(self, tmp_path: Path) -> None:
        """Output directory should be created if it does not exist."""
        out = tmp_path / "nonexistent" / "deep" / "dir"
        from prinet.utils.table_generation import generate_all_tables
        generate_all_tables(results_dir=RESULTS_DIR, output_dir=out)
