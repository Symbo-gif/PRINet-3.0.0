"""Year 3 Q4.5 Tests: Bug Fix Verification & Comprehensive Validation.

Covers:
    - Fix 1: MSVC cl.exe auto-detection via vswhere
    - Fix 2: torch.compile Triton auto-downgrade (max-autotune → reduce-overhead)
    - Fix 3: HybridPRINetV2CLEVRN query usage (query_proj + merge)
    - Fix 4: CSR beta warning suppression
    - Comprehensive subsystem integration validation
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ── Guards ──────────────────────────────────────────────────────
_CUDA = torch.cuda.is_available()
_DEVICE = "cuda" if _CUDA else "cpu"
_WINDOWS = sys.platform == "win32"

SEED = 42


def _seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if _CUDA:
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Fix 1: MSVC Auto-Detection
# ================================================================


class TestMSVCAutoDetection:
    """Verify _find_msvc_cl() and _ensure_msvc_on_path() helpers."""

    def test_find_msvc_cl_returns_string_or_none(self) -> None:
        """_find_msvc_cl() returns a valid path string or None."""
        from prinet.utils.fused_kernels import _find_msvc_cl

        result = _find_msvc_cl()
        if result is not None:
            assert isinstance(result, str)
            assert "cl" in Path(result).name.lower()

    @pytest.mark.skipif(not _WINDOWS, reason="MSVC only on Windows")
    def test_find_msvc_cl_windows(self) -> None:
        """On Windows with VS installed, should find cl.exe."""
        from prinet.utils.fused_kernels import _find_msvc_cl

        result = _find_msvc_cl()
        # May be None if no Visual Studio installed; skip in that case
        if result is None:
            pytest.skip("No Visual Studio installation found")
        assert "VC" in result or "vc" in result.lower()

    def test_ensure_msvc_on_path_returns_bool(self) -> None:
        """_ensure_msvc_on_path() returns a boolean."""
        from prinet.utils.fused_kernels import _ensure_msvc_on_path

        result = _ensure_msvc_on_path()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _WINDOWS, reason="MSVC only on Windows")
    def test_ensure_msvc_on_path_modifies_env(self) -> None:
        """After _ensure_msvc_on_path(), cl.exe directory is on PATH."""
        import shutil

        from prinet.utils.fused_kernels import _ensure_msvc_on_path, _find_msvc_cl

        cl = _find_msvc_cl()
        if cl is None:
            pytest.skip("No Visual Studio installation found")

        original_path = os.environ.get("PATH", "")
        result = _ensure_msvc_on_path()
        assert result is True

        cl_dir = str(Path(cl).parent)
        assert cl_dir in os.environ.get("PATH", "")

        # Restore
        os.environ["PATH"] = original_path


# ================================================================
# Fix 2: torch.compile Triton Auto-Downgrade
# ================================================================


class TestTorchCompileAutoDowngrade:
    """Verify torch.compile gracefully handles missing Triton."""

    def test_triton_available_returns_bool(self) -> None:
        """_triton_available() returns a boolean."""
        from prinet.nn.hybrid import HybridPRINetV2

        result = HybridPRINetV2._triton_available()
        assert isinstance(result, bool)

    def test_compile_succeeds(self) -> None:
        """compile() succeeds regardless of Triton availability."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=32, n_classes=5, d_model=32)
        model.compile(mode="max-autotune")
        assert model.is_compiled is True

    def test_compile_is_compiled_flag(self) -> None:
        """is_compiled is False before compile, True after."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=32, n_classes=5, d_model=32)
        assert model.is_compiled is False
        model.compile(backend="eager")
        assert model.is_compiled is True

    def test_compiled_forward_produces_output(self) -> None:
        """compiled_forward() works after compile()."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=32, n_classes=5, d_model=32)
        # Don't call .eval() after compile — PyTorch's eager backend
        # has a known recursion issue when traversing compiled sub-modules.
        model.eval()
        model.compile(backend="eager")

        x = torch.randn(4, 32)
        with torch.no_grad():
            out = model.compiled_forward(x)
        assert out.shape == (4, 5)

    def test_compile_returns_self(self) -> None:
        """compile() returns self for method chaining."""
        from prinet.nn.hybrid import HybridPRINetV2

        model = HybridPRINetV2(n_input=32, n_classes=5, d_model=32)
        result = model.compile(backend="eager")
        assert result is model


# ================================================================
# Fix 3: HybridPRINetV2CLEVRN Query Usage
# ================================================================


class TestCLEVRNQueryFix:
    """Verify HybridPRINetV2CLEVRN actually uses query input."""

    def test_forward_shape_default_query_dim(self) -> None:
        """Forward produces correct shape with default query_dim=60."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16)
        scene = torch.randn(4, 6, 16)
        query = torch.randn(4, 60)  # default query_dim
        out = model(scene, query)
        assert out.shape == (4, 2)

    def test_forward_shape_custom_query_dim(self) -> None:
        """Forward works with a custom query_dim."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16, query_dim=44, d_model=32)
        scene = torch.randn(2, 6, 16)
        query = torch.randn(2, 44)
        out = model(scene, query)
        assert out.shape == (2, 2)

    def test_query_affects_output(self) -> None:
        """Different queries produce different outputs (query is used)."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16, query_dim=60, d_model=32)
        model.eval()

        scene = torch.randn(2, 6, 16)
        query_a = torch.randn(2, 60)
        query_b = torch.randn(2, 60) * 3.0  # very different query

        with torch.no_grad():
            out_a = model(scene, query_a)
            out_b = model(scene, query_b)

        # Different queries → different outputs (proves query is used)
        assert not torch.allclose(
            out_a, out_b, atol=1e-6
        ), "Same output for different queries — query may not be used"

    def test_has_query_proj_and_merge(self) -> None:
        """Model has query_proj and merge layers (architecture fix)."""
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16, query_dim=60, d_model=32)
        assert hasattr(model, "query_proj"), "Missing query_proj layer"
        assert hasattr(model, "merge"), "Missing merge layer"
        assert isinstance(model.query_proj, nn.Linear)
        assert isinstance(model.merge, nn.Linear)

    def test_2d_scene_input(self) -> None:
        """Handles collapsed (B, scene_dim) scene input."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16, query_dim=60, d_model=32)
        scene_2d = torch.randn(4, 16)  # already aggregated
        query = torch.randn(4, 60)
        out = model(scene_2d, query)
        assert out.shape == (4, 2)

    def test_3d_scene_input(self) -> None:
        """Handles (B, N_objects, scene_dim) scene input."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=16, query_dim=60, d_model=32)
        scene_3d = torch.randn(4, 10, 16)
        query = torch.randn(4, 60)
        out = model(scene_3d, query)
        assert out.shape == (4, 2)

    def test_backward_gradient_flow(self) -> None:
        """Gradients flow through scene AND query paths."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=8, query_dim=16, d_model=16)
        scene = torch.randn(2, 4, 8, requires_grad=True)
        query = torch.randn(2, 16, requires_grad=True)

        out = model(scene, query)
        loss = out.sum()
        loss.backward()

        assert scene.grad is not None, "No gradient on scene input"
        assert query.grad is not None, "No gradient on query input"
        assert scene.grad.abs().sum() > 0, "Zero gradient on scene"
        assert query.grad.abs().sum() > 0, "Zero gradient on query"


# ================================================================
# Fix 4: CSR Beta Warning Suppression
# ================================================================


class TestCSRWarningSuppression:
    """Verify CSR beta warnings are suppressed in fused_kernels and oscillosim."""

    def test_sparse_coupling_no_warning(self) -> None:
        """sparse_coupling_matrix_csr() produces no beta warnings."""
        from prinet.utils.fused_kernels import sparse_coupling_matrix_csr

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = sparse_coupling_matrix_csr(100, sparsity=0.9, seed=42)

        csr_warnings = [x for x in w if "Sparse CSR" in str(x.message)]
        assert len(csr_warnings) == 0, f"Got {len(csr_warnings)} CSR beta warning(s)"

    def test_csr_coupling_step_no_warning(self) -> None:
        """csr_coupling_step() produces no beta warnings."""
        from prinet.utils.fused_kernels import (
            csr_coupling_step,
            sparse_coupling_matrix_csr,
        )

        csr = sparse_coupling_matrix_csr(100, sparsity=0.9, seed=42)
        phase = torch.randn(100)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = csr_coupling_step(phase, csr)

        csr_warnings = [x for x in w if "Sparse CSR" in str(x.message)]
        assert len(csr_warnings) == 0, f"Got {len(csr_warnings)} CSR beta warning(s)"

    def test_oscillosim_csr_mode_no_warning(self) -> None:
        """OscilloSim in csr mode produces no beta warnings."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim = OscilloSim(
                n_oscillators=100,
                coupling_strength=2.0,
                coupling_mode="csr",
                k_neighbors=8,
                device="cpu",
            )
            _ = sim.run(n_steps=10, dt=0.01)

        csr_warnings = [x for x in w if "Sparse CSR" in str(x.message)]
        assert len(csr_warnings) == 0, f"Got {len(csr_warnings)} CSR beta warning(s)"


# ================================================================
# Cross-integration: Comprehensive Subsystem Validation
# ================================================================


class TestQ45CrossIntegration:
    """Verify cross-subsystem interactions after all Q4.5 fixes."""

    def test_clevrn_training_loop(self) -> None:
        """HybridPRINetV2CLEVRN can train in a full loop."""
        _seed()
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        model = HybridPRINetV2CLEVRN(scene_dim=8, query_dim=16, d_model=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(5):
            scene = torch.randn(4, 3, 8)
            query = torch.randn(4, 16)
            target = torch.randint(0, 2, (4,))

            log_probs = model(scene, query)
            loss = torch.nn.functional.nll_loss(log_probs, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Just ensure it doesn't crash and loss is finite
        assert torch.isfinite(torch.tensor(loss.item()))

    def test_subconscious_state_to_tensor(self) -> None:
        """SubconsciousState.to_tensor() returns correct shape."""
        from prinet.core.subconscious import SubconsciousState

        state = SubconsciousState.default()
        tensor = state.to_tensor()
        assert tensor.shape == (32,), f"Expected (32,), got {tensor.shape}"

    def test_oscillosim_all_modes_finish(self) -> None:
        """All three OscilloSim coupling modes complete without error."""
        from prinet.utils.oscillosim import OscilloSim

        _seed()
        for mode in ["mean_field", "sparse_knn", "csr"]:
            sim = OscilloSim(
                n_oscillators=50,
                coupling_strength=2.0,
                coupling_mode=mode,
                k_neighbors=8,
                device="cpu",
            )
            result = sim.run(n_steps=10, dt=0.01)
            r = result.order_parameter[-1]
            assert 0.0 <= r <= 1.0, f"{mode}: r={r} not in [0, 1]"

    def test_api_version(self) -> None:
        """prinet.__version__ is a valid semver."""
        import prinet

        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)
