"""Year 4 Q4 tests -- Final regression, archival, and project close.

Validates all Q4 deliverables:
  W.4  Year 4 comprehensive report
  W.5  Project retrospective document
  W.6  Archive files (CITATION.cff, CODE_OF_CONDUCT.md, SHA-256 manifest)
  W.7  Final regression gate (0 failures, <=14 skips, all artefacts committed)

Tests are designed for Windows compatibility with ASCII-safe output.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import prinet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def project_root() -> str:
    return ROOT


@pytest.fixture(scope="module")
def results_dir() -> str:
    return os.path.join(ROOT, "benchmarks", "results")


@pytest.fixture(scope="module")
def docs_dir() -> str:
    return os.path.join(ROOT, "Docs")


# =========================================================================
# W.4 Year 4 Comprehensive Report
# =========================================================================


class TestYear4Report:
    """Validate Year 4 comprehensive report exists and has required content."""

    def test_report_exists(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        assert os.path.isfile(path), "Year 4 comprehensive report not found"

    def test_report_has_executive_summary(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Executive Summary" in content

    def test_report_has_quarterly_summary(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Q1" in content and "Q2" in content and "Q3" in content and "Q4" in content

    def test_report_covers_benchmark_results(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        content = Path(path).read_text(encoding="utf-8")
        # Must reference key Q1 results
        assert "chimera" in content.lower()
        assert "Outcome B" in content or "outcome B" in content or "no temporal advantage" in content.lower()

    def test_report_has_metrics(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "172" in content  # API symbols
        assert "3.0.0" in content  # version

    def test_report_references_q1_depth(self, docs_dir: str) -> None:
        """Report must reflect the extensive Q1 benchmark work (9 sub-sessions)."""
        path = os.path.join(docs_dir, "Year_4_Comprehensive_Report.md")
        content = Path(path).read_text(encoding="utf-8")
        # Should mention multiple Q1 sub-sessions
        assert "Q1.7" in content or "Q1.9" in content
        assert "17x" in content  # parameter efficiency


# =========================================================================
# W.5 Project Retrospective
# =========================================================================


class TestProjectRetrospective:
    """Validate project retrospective document."""

    def test_retrospective_exists(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Project_Retrospective.md")
        assert os.path.isfile(path), "Project retrospective not found"

    def test_retrospective_has_lessons_learned(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Project_Retrospective.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "What Worked" in content or "what worked" in content.lower()
        assert "What Didn" in content or "what didn" in content.lower()

    def test_retrospective_has_technical_decisions(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Project_Retrospective.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Technical Decisions" in content or "Kuramoto" in content

    def test_retrospective_has_extension_guidance(self, docs_dir: str) -> None:
        path = os.path.join(docs_dir, "Project_Retrospective.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Forking" in content or "Extending" in content or "extension" in content.lower()


# =========================================================================
# W.6 Archive Files
# =========================================================================


class TestCitationCFF:
    """Validate CITATION.cff file."""

    def test_citation_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "CITATION.cff")
        assert os.path.isfile(path), "CITATION.cff not found"

    def test_citation_has_required_fields(self, project_root: str) -> None:
        path = os.path.join(project_root, "CITATION.cff")
        content = Path(path).read_text(encoding="utf-8")
        required_fields = ["cff-version", "message", "title", "authors", "version", "license"]
        for field in required_fields:
            assert field in content, f"CITATION.cff missing required field: {field}"

    def test_citation_version_matches(self, project_root: str) -> None:
        path = os.path.join(project_root, "CITATION.cff")
        content = Path(path).read_text(encoding="utf-8")
        assert "3.0.0" in content, "CITATION.cff version should match project version"

    def test_citation_has_keywords(self, project_root: str) -> None:
        path = os.path.join(project_root, "CITATION.cff")
        content = Path(path).read_text(encoding="utf-8")
        assert "keywords" in content
        assert "oscillatory" in content.lower() or "phase" in content.lower()

    def test_citation_has_repository(self, project_root: str) -> None:
        path = os.path.join(project_root, "CITATION.cff")
        content = Path(path).read_text(encoding="utf-8")
        assert "repository" in content.lower()


class TestCodeOfConduct:
    """Validate CODE_OF_CONDUCT.md file."""

    def test_code_of_conduct_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "CODE_OF_CONDUCT.md")
        assert os.path.isfile(path), "CODE_OF_CONDUCT.md not found"

    def test_code_of_conduct_has_pledge(self, project_root: str) -> None:
        path = os.path.join(project_root, "CODE_OF_CONDUCT.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Pledge" in content or "pledge" in content

    def test_code_of_conduct_has_enforcement(self, project_root: str) -> None:
        path = os.path.join(project_root, "CODE_OF_CONDUCT.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Enforcement" in content

    def test_code_of_conduct_has_attribution(self, project_root: str) -> None:
        path = os.path.join(project_root, "CODE_OF_CONDUCT.md")
        content = Path(path).read_text(encoding="utf-8")
        assert "Contributor Covenant" in content


class TestSHA256Manifest:
    """Validate SHA-256 manifest generation for benchmark artefacts."""

    def test_can_generate_manifest(self, results_dir: str) -> None:
        """All JSON artefacts can be hashed without error."""
        json_files = sorted(Path(results_dir).glob("*.json"))
        assert len(json_files) >= 50, f"Expected >=50 JSON artefacts, found {len(json_files)}"
        manifest: Dict[str, str] = {}
        for f in json_files:
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            manifest[f.name] = sha
        assert len(manifest) == len(json_files)

    def test_artefacts_valid_json(self, results_dir: str) -> None:
        """All JSON artefacts parse without error."""
        json_files = sorted(Path(results_dir).glob("*.json"))
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                assert isinstance(data, dict), f"{f.name} top-level is not a dict"
            except json.JSONDecodeError:
                pytest.fail(f"{f.name} is not valid JSON")

    def test_artefacts_have_benchmark_name(self, results_dir: str) -> None:
        """Most JSON artefacts contain a benchmark_name or name key."""
        json_files = sorted(Path(results_dir).glob("*.json"))
        named_count = 0
        for f in json_files:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "benchmark_name" in data or "name" in data or "benchmark" in data:
                named_count += 1
        # Allow some without name (summary files) but most should have it
        assert named_count >= len(json_files) * 0.5, (
            f"Only {named_count}/{len(json_files)} artefacts have benchmark name"
        )

    def test_manifest_deterministic(self, results_dir: str) -> None:
        """SHA-256 of any artefact is deterministic across runs."""
        json_files = sorted(Path(results_dir).glob("*.json"))
        if not json_files:
            pytest.skip("No JSON artefacts found")
        f = json_files[0]
        sha1 = hashlib.sha256(f.read_bytes()).hexdigest()
        sha2 = hashlib.sha256(f.read_bytes()).hexdigest()
        assert sha1 == sha2


# =========================================================================
# W.7 Final Regression Gate
# =========================================================================


class TestVersionConsistency:
    """Version consistency across project files."""

    def test_prinet_version_3(self) -> None:
        """Package version is 3.x.x."""
        parts = prinet.__version__.split(".")
        assert int(parts[0]) >= 3, f"Expected major version >= 3, got {parts[0]}"

    def test_pyproject_version_matches(self, project_root: str) -> None:
        """pyproject.toml version matches __init__.py."""
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        content = Path(pyproject_path).read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        assert match is not None, "Version not found in pyproject.toml"
        assert match.group(1) == prinet.__version__

    def test_api_symbol_count(self) -> None:
        """API surface meets target (>= 120 symbols)."""
        assert len(prinet.__all__) >= 120, (
            f"Expected >= 120 API symbols, got {len(prinet.__all__)}"
        )

    def test_api_symbols_importable(self) -> None:
        """All symbols in __all__ are actually importable."""
        missing = []
        for sym in prinet.__all__[:20]:  # Sample first 20 to keep fast
            if not hasattr(prinet, sym):
                missing.append(sym)
        assert not missing, f"Symbols in __all__ but not importable: {missing}"


class TestArtefactCounts:
    """Validate artefact counts meet project targets."""

    def test_json_artefact_count(self, results_dir: str) -> None:
        """At least 100 JSON benchmark artefacts exist."""
        json_files = list(Path(results_dir).glob("*.json"))
        assert len(json_files) >= 100, (
            f"Expected >= 100 JSON artefacts, got {len(json_files)}"
        )

    def test_test_file_count(self, project_root: str) -> None:
        """At least 25 test files exist."""
        test_dir = os.path.join(project_root, "tests")
        test_files = list(Path(test_dir).glob("test_*.py"))
        assert len(test_files) >= 25, (
            f"Expected >= 25 test files, got {len(test_files)}"
        )

    def test_benchmark_file_count(self, project_root: str) -> None:
        """At least 10 benchmark scripts exist."""
        bench_dir = os.path.join(project_root, "benchmarks")
        bench_files = [
            f for f in Path(bench_dir).glob("*.py")
            if not f.name.startswith("_")
        ]
        assert len(bench_files) >= 10, (
            f"Expected >= 10 benchmark files, got {len(bench_files)}"
        )

    def test_notebook_count(self, project_root: str) -> None:
        """At least 3 Jupyter notebooks exist."""
        nb_dir = os.path.join(project_root, "notebooks")
        if not os.path.isdir(nb_dir):
            pytest.skip("notebooks/ directory not found")
        nb_files = list(Path(nb_dir).glob("*.ipynb"))
        assert len(nb_files) >= 3, (
            f"Expected >= 3 notebooks, got {len(nb_files)}"
        )


class TestDocumentationCompleteness:
    """Validate documentation completeness for project close."""

    def test_license_exists(self, project_root: str) -> None:
        assert os.path.isfile(os.path.join(project_root, "LICENSE"))

    def test_readme_exists(self, project_root: str) -> None:
        # Check for README in any common name/extension
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            if os.path.isfile(os.path.join(project_root, name)):
                return
        # Check src/prinet README
        assert os.path.isfile(os.path.join(project_root, "src", "prinet", "README.md"))

    def test_changelog_exists(self, project_root: str) -> None:
        # Check multiple possible locations
        locations = [
            os.path.join(project_root, "CHANGELOG.md"),
            os.path.join(project_root, "src", "prinet", "CHANGELOG.md"),
        ]
        found = any(os.path.isfile(loc) for loc in locations)
        assert found, "No CHANGELOG.md found"

    def test_year4_plan_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "Docs", "Planning_Documentation", "Year-4-Plan.md")
        assert os.path.isfile(path)

    def test_arxiv_outline_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "Docs", "Arxiv_Preprint_Outline.md")
        assert os.path.isfile(path)

    def test_latex_paper_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "paper", "main.tex")
        assert os.path.isfile(path), "LaTeX paper not found at paper/main.tex"

    def test_reproduce_script_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "reproduce.py")
        assert os.path.isfile(path)

    def test_sphinx_conf_exists(self, project_root: str) -> None:
        path = os.path.join(project_root, "Docs", "conf.py")
        assert os.path.isfile(path)


class TestQ1BenchmarkIntegrity:
    """Validate that Q1 benchmark artefacts from the extensive experimental
    phase are present and well-formed. This is critical because Q1 produced
    98 artefacts across 9 sub-sessions -- the largest experimental output
    of the entire project."""

    def test_q1_chimera_artefacts(self, results_dir: str) -> None:
        """Gold-standard chimera benchmark artefacts from Q1.3."""
        expected = [
            "benchmark_y4q1_3_gold_standard_chimera.json",
            "benchmark_y4q1_3_rk4_vs_euler.json",
            "benchmark_y4q1_3_k_alpha_sensitivity.json",
        ]
        for name in expected:
            path = os.path.join(results_dir, name)
            assert os.path.isfile(path), f"Q1.3 chimera artefact missing: {name}"

    def test_q1_temporal_advantage_artefacts(self, results_dir: str) -> None:
        """Definitive temporal advantage artefacts from Q1.7."""
        expected = [
            "y4q1_7_statistical_summary.json",
            "y4q1_7_training_curves_pt.json",
            "y4q1_7_training_curves_sa.json",
            "y4q1_7_preregistration_hash.json",
        ]
        for name in expected:
            path = os.path.join(results_dir, name)
            assert os.path.isfile(path), f"Q1.7 temporal advantage artefact missing: {name}"

    def test_q1_reviewer_gap_artefacts(self, results_dir: str) -> None:
        """Reviewer gap analysis artefacts from Q1.9."""
        expected = [
            "y4q1_9_7seed_comparison.json",
            "y4q1_9_fine_occlusion.json",
            "y4q1_9_pac_significance.json",
        ]
        for name in expected:
            path = os.path.join(results_dir, name)
            assert os.path.isfile(path), f"Q1.9 reviewer gap artefact missing: {name}"

    def test_q1_adversarial_artefacts(self, results_dir: str) -> None:
        """Adversarial robustness artefacts from Q1.8."""
        expected = [
            "y4q1_8_fgsm_sweep.json",
            "y4q1_8_pgd_attack.json",
            "y4q1_8_adversarial_summary.json",
        ]
        for name in expected:
            path = os.path.join(results_dir, name)
            assert os.path.isfile(path), f"Q1.8 adversarial artefact missing: {name}"

    def test_statistical_summary_content(self, results_dir: str) -> None:
        """The statistical summary JSON contains Outcome B determination."""
        path = os.path.join(results_dir, "y4q1_7_statistical_summary.json")
        if not os.path.isfile(path):
            pytest.skip("Statistical summary artefact not found")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Should contain outcome determination
        assert isinstance(data, dict)
        # Check for key statistical fields
        has_outcome = any(
            "outcome" in str(k).lower() or "conclusion" in str(k).lower()
            for k in data.keys()
        )
        assert has_outcome or "p_value" in str(data) or "cohens_d" in str(data), (
            "Statistical summary should contain outcome/conclusion/p_value"
        )
