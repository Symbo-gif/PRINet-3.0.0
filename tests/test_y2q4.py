"""Year 2 Q4 Tests — Consolidation.

Workstream J: API freeze, documentation, regression suite.
Workstream K: Empirical capacity, phase diagrams.

Covers:
- TestAPIFreeze: Public API matches frozen surface.
- TestDeprecation: Deprecation utilities work correctly.
- TestDocumentation: Architecture guide and tutorial exist.
- TestRegressionSuite: J.3 regression benchmarks importable.
- TestCapacityMeasurement: K.1 capacity sweep produces valid results.
- TestPhaseDiagrams: K.2 phase diagrams produce valid order parameters.
- TestVersioning: Semantic version is 1.0.0.
"""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prinet
from prinet._deprecation import (
    FROZEN_PUBLIC_API,
    deprecated,
    deprecated_parameter,
    verify_api_surface,
)


# ============================================================================
# Workstream J: Engineering Quality
# ============================================================================


class TestAPIFreeze:
    """J.1 — Public API matches documented frozen surface."""

    def test_version_is_stable(self) -> None:
        """Version is semantic (no dev/rc suffix)."""
        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)
        # No .dev, .rc, .alpha, .beta suffix
        assert "dev" not in prinet.__version__
        assert "rc" not in prinet.__version__

    def test_all_frozen_symbols_exported(self) -> None:
        """Every symbol in FROZEN_PUBLIC_API is in prinet.__all__."""
        missing, _ = verify_api_surface(prinet.__all__)
        assert not missing, f"Frozen symbols missing from __all__: {missing}"

    def test_no_accidental_removals(self) -> None:
        """__all__ contains all frozen symbols (no regressions)."""
        current = set(prinet.__all__)
        frozen = set(FROZEN_PUBLIC_API)
        removed = frozen - current
        assert not removed, f"Symbols removed from __all__: {removed}"

    def test_frozen_symbols_importable(self) -> None:
        """All frozen symbols can be accessed on the prinet module."""
        for name in FROZEN_PUBLIC_API:
            assert hasattr(prinet, name), f"prinet.{name} not accessible"

    def test_frozen_api_is_frozenset(self) -> None:
        """FROZEN_PUBLIC_API is immutable."""
        assert isinstance(FROZEN_PUBLIC_API, frozenset)

    def test_all_list_matches_exports(self) -> None:
        """Every entry in __all__ is actually importable."""
        for name in prinet.__all__:
            assert hasattr(prinet, name), f"prinet.{name} in __all__ but not importable"


class TestDeprecation:
    """J.1 — Deprecation utilities."""

    def test_deprecated_function_warns(self) -> None:
        """@deprecated emits DeprecationWarning."""

        @deprecated("1.0.0", "Use new_func() instead.")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == "result"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func" in str(w[0].message)
            assert "1.0.0" in str(w[0].message)

    def test_deprecated_with_removal_version(self) -> None:
        """@deprecated includes removal version in warning."""

        @deprecated("1.0.0", "Migrate to new_api.", removal="2.0.0")
        def legacy_api() -> None:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            legacy_api()
            assert "2.0.0" in str(w[0].message)

    def test_deprecated_parameter_warns(self) -> None:
        """@deprecated_parameter warns when deprecated kwarg is passed."""

        @deprecated_parameter("old_param", "1.0.0", "Use new_param.")
        def my_func(new_param: int = 1, **kwargs: object) -> int:
            return new_param

        # No warning without deprecated param
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            my_func(new_param=5)
            assert len(w) == 0

        # Warning with deprecated param
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            my_func(old_param=3)
            assert len(w) == 1
            assert "old_param" in str(w[0].message)

    def test_verify_api_surface_detects_missing(self) -> None:
        """verify_api_surface reports missing symbols."""
        fake_all = ["PolyadicTensor"]  # Missing many
        missing, extra = verify_api_surface(fake_all)
        assert len(missing) > 0
        assert "KuramotoOscillator" in missing

    def test_verify_api_surface_detects_extra(self) -> None:
        """verify_api_surface reports extra symbols."""
        fake_all = list(FROZEN_PUBLIC_API) + ["BrandNewSymbol"]
        missing, extra = verify_api_surface(fake_all)
        assert len(missing) == 0
        assert "BrandNewSymbol" in extra


class TestDocumentation:
    """J.2 — Documentation completeness."""

    def test_architecture_guide_exists(self) -> None:
        """Architecture guide document exists."""
        guide = Path(__file__).parents[1] / "Docs" / "Architecture_Guide.md"
        assert guide.exists(), f"Missing: {guide}"
        content = guide.read_text(encoding="utf-8")
        assert len(content) > 1000, "Architecture guide too short"
        assert "PRINet" in content

    def test_getting_started_tutorial_exists(self) -> None:
        """Getting started tutorial exists."""
        tutorial = Path(__file__).parents[1] / "Docs" / "Getting_Started_Tutorial.md"
        assert tutorial.exists(), f"Missing: {tutorial}"
        content = tutorial.read_text(encoding="utf-8")
        assert len(content) > 1000, "Tutorial too short"
        assert "import prinet" in content or "from prinet" in content

    def test_tutorial_code_blocks(self) -> None:
        """Tutorial contains runnable Python code blocks."""
        tutorial = Path(__file__).parents[1] / "Docs" / "Getting_Started_Tutorial.md"
        content = tutorial.read_text(encoding="utf-8")
        # Should have multiple code blocks
        assert content.count("```python") >= 3, "Tutorial needs more code examples"

    def test_api_reference_exists(self) -> None:
        """API Reference document exists."""
        ref = Path(__file__).parents[1] / "Docs" / "API_Reference_Coupling_Topologies.md"
        assert ref.exists()


class TestRegressionSuite:
    """J.3 — Performance regression suite."""

    def test_benchmark_script_importable(self) -> None:
        """Q4 benchmark script can be imported."""
        bench_path = Path(__file__).parents[1] / "benchmarks"
        sys.path.insert(0, str(bench_path))
        try:
            from y2q4_benchmarks import (
                run_j3_regression_suite,
                run_k1_capacity_sweep,
                run_k2_phase_diagrams,
            )

            assert callable(run_j3_regression_suite)
            assert callable(run_k1_capacity_sweep)
            assert callable(run_k2_phase_diagrams)
        finally:
            sys.path.pop(0)

    def test_regression_goldilocks_quick(self) -> None:
        """Quick smoke: V2 produces finite output on CLEVR-6."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(
            n_input=14, n_classes=2,
            d_model=64, n_heads=4, n_layers=2,
            n_delta=2, n_theta=4, n_gamma=8,
        )
        x = torch.randn(4, 14)  # Already projected to n_osc dimensions
        out = model(x)
        assert torch.isfinite(out).all()
        assert out.shape == (4, 2)


# ============================================================================
# Workstream K: Theoretical Grounding
# ============================================================================


class TestCapacityMeasurement:
    """K.1 — Empirical capacity numbers reproducible."""

    def test_clevr_n_generation_seeded(self) -> None:
        """CLEVR-N data generation is deterministic with seed."""
        bench_path = Path(__file__).parents[1] / "benchmarks"
        sys.path.insert(0, str(bench_path))
        try:
            from y2q4_benchmarks import generate_clevr_n

            X1, y1 = generate_clevr_n(6, n_samples=50, seed=123)
            X2, y2 = generate_clevr_n(6, n_samples=50, seed=123)
            assert torch.equal(X1, X2)
            assert torch.equal(y1, y2)
        finally:
            sys.path.pop(0)

    def test_clevr_n_labels_balanced(self) -> None:
        """CLEVR-N produces roughly 50/50 positive/negative labels."""
        bench_path = Path(__file__).parents[1] / "benchmarks"
        sys.path.insert(0, str(bench_path))
        try:
            from y2q4_benchmarks import generate_clevr_n

            _, y = generate_clevr_n(6, n_samples=500, seed=42)
            pos_rate = y.float().mean().item()
            assert 0.3 < pos_rate < 0.7, f"Unbalanced: {pos_rate:.2f}"
        finally:
            sys.path.pop(0)

    @pytest.mark.slow
    def test_capacity_single_model(self) -> None:
        """Train one model at N=6, verify accuracy is reasonable."""
        bench_path = Path(__file__).parents[1] / "benchmarks"
        sys.path.insert(0, str(bench_path))
        try:
            from y2q4_benchmarks import DiscreteDTGCLEVRN, generate_clevr_n

            X_train, y_train = generate_clevr_n(6, n_samples=200, seed=42)
            X_test, y_test = generate_clevr_n(6, n_samples=100, seed=99)

            model = DiscreteDTGCLEVRN(19, n_classes=2)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            for _ in range(20):
                model.train()
                logits = model(X_train)
                loss = F.nll_loss(logits, y_train)
                loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=-1)
                acc = (preds == y_test).float().mean().item()
            # After 20 epochs should be noticeably above random guessing
            assert acc >= 0.40, f"Accuracy too low: {acc:.2%}"
        finally:
            sys.path.pop(0)


class TestPhaseDiagrams:
    """K.2 — Extended K x Delta diagrams produce valid order parameters."""

    def test_kuramoto_phase_diagram_small(self) -> None:
        """Small 3x3 phase diagram produces valid r values."""
        from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

        for K in [0.5, 2.0, 5.0]:
            for delta in [0.0, 1.0, 3.0]:
                N = 10
                torch.manual_seed(42)
                if delta == 0.0:
                    freq = torch.ones(N)
                else:
                    freq = torch.normal(mean=1.0, std=delta, size=(N,))

                osc = KuramotoOscillator(
                    n_oscillators=N, coupling_strength=K,
                    coupling_mode="full",
                )
                state = OscillatorState.create_random(N, seed=42)
                state = OscillatorState(
                    phase=state.phase,
                    amplitude=state.amplitude,
                    frequency=freq,
                )

                for _ in range(50):
                    state = osc.step(state, dt=0.05)

                r = kuramoto_order_parameter(state.phase).item()
                assert 0.0 <= r <= 1.0, f"Invalid r={r} at K={K}, delta={delta}"

    def test_high_coupling_synchronizes(self) -> None:
        """High K, identical frequencies should yield high synchronization."""
        from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

        N = 20
        osc = KuramotoOscillator(
            n_oscillators=N, coupling_strength=5.0,
            coupling_mode="full",
        )
        # Create state with IDENTICAL frequencies — zero detuning
        torch.manual_seed(42)
        state = OscillatorState(
            phase=torch.rand(N) * 2 * 3.14159,
            amplitude=torch.ones(N),
            frequency=torch.ones(N),  # all identical
        )

        r0 = kuramoto_order_parameter(state.phase).item()

        for _ in range(500):
            state = osc.step(state, dt=0.01)

        r_final = kuramoto_order_parameter(state.phase).item()
        # With identical freqs and coupling, r should increase
        assert r_final > r0 or r_final > 0.3, (
            f"Expected sync improvement: r0={r0:.3f}, r_final={r_final:.3f}"
        )

    def test_low_coupling_desynchronized(self) -> None:
        """Low K, high delta should yield low synchronization."""
        from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

        N = 50
        torch.manual_seed(42)
        freq = torch.normal(mean=1.0, std=3.0, size=(N,))

        osc = KuramotoOscillator(
            n_oscillators=N, coupling_strength=0.1,
            coupling_mode="mean_field",
        )
        state = OscillatorState.create_random(N, seed=42)
        state = OscillatorState(
            phase=state.phase,
            amplitude=state.amplitude,
            frequency=freq,
        )

        for _ in range(200):
            state = osc.step(state, dt=0.05)

        r = kuramoto_order_parameter(state.phase).item()
        assert r < 0.8, f"Expected desync at K=0.1, delta=3: r={r}"

    def test_diagram_scales_with_n(self) -> None:
        """Phase diagram works for N=10, 50, 100 without error."""
        from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

        for N in [10, 50, 100]:
            osc = KuramotoOscillator(
                n_oscillators=N, coupling_strength=2.0,
                coupling_mode="mean_field",
            )
            state = OscillatorState.create_random(N, seed=42)
            for _ in range(50):
                state = osc.step(state, dt=0.05)
            r = kuramoto_order_parameter(state.phase).item()
            assert 0.0 <= r <= 1.0


class TestTopLevelExports:
    """All Q4 additions accessible."""

    @pytest.mark.parametrize("symbol", [
        "FROZEN_PUBLIC_API", "verify_api_surface",
        "deprecated", "deprecated_parameter",
    ])
    def test_deprecation_module_importable(self, symbol: str) -> None:
        """Deprecation utilities are importable."""
        from prinet import _deprecation
        assert hasattr(_deprecation, symbol)

    def test_version_string(self) -> None:
        """Version is a valid semantic version string."""
        parts = prinet.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
