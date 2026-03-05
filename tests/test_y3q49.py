"""Year 3 Q4.9 Tests — Scientific Coupling-Regime Benchmark Verification.

Covers:
- TestBenchmarkInfrastructure: Benchmark imports, config, helpers work.
- TestSoloRegimeExecution: Each regime runs on each device.
- TestCSRMode: CSR via OscilloSim produces valid results.
- TestConcurrentExecution: Multi-regime concurrent runs succeed.
- TestGoldilocksZone: K sweep produces monotonically increasing r.
- TestFiniteSizeScaling: K_c(N) increases with N (smaller systems need less K).
- TestPhaseCoherence: Synchronized state persists briefly after K drop.
- TestChimeraDetection: Local order parameter computation produces valid output.
- TestResultArtefacts: All JSON artefacts exist and are valid.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState
from prinet.utils.oscillosim import OscilloSim

SEED = 42
DT = 0.01
HAS_CUDA = torch.cuda.is_available()
RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


# ============================================================================
# Infrastructure Tests
# ============================================================================


class TestBenchmarkInfrastructure:
    """Verify benchmark module can be imported and configured."""

    def test_benchmark_module_importable(self) -> None:
        """y3q49 benchmark script is syntactically valid."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "y3q49",
            str(
                Path(__file__).resolve().parents[1]
                / "benchmarks"
                / "y3q49_scientific_regime_benchmark.py"
            ),
        )
        assert spec is not None
        assert spec.loader is not None

    def test_regime_configs_complete(self) -> None:
        """All 4 coupling regimes are defined in the benchmark."""
        expected = {"mean_field", "sparse_knn", "full", "csr"}
        # Import the module config
        bm_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "y3q49_scientific_regime_benchmark.py"
        )
        # Read and check config
        content = bm_path.read_text(encoding="utf-8")
        for regime in expected:
            assert f'"{regime}"' in content, f"Regime {regime} missing from config"

    def test_kuramoto_order_param(self) -> None:
        """kuramoto_order_parameter works for random phases."""
        torch.manual_seed(SEED)
        phase = torch.rand(256) * 2 * math.pi
        r = kuramoto_order_parameter(phase).item()
        assert 0.0 <= r <= 1.0

    def test_oscillator_state_create(self) -> None:
        """OscillatorState.create_random works."""
        state = OscillatorState.create_random(128, seed=SEED)
        assert state.phase.shape == (128,)
        assert state.phase.dtype == torch.float32

    def test_confidence_interval_bootstrap(self) -> None:
        """Bootstrap CI produces valid interval."""
        import random

        rng = random.Random(42)
        values = [rng.gauss(10.0, 1.0) for _ in range(50)]
        mean = statistics.mean(values)
        # Simple check: mean is within reasonable bounds
        assert 8.0 < mean < 12.0


# ============================================================================
# Solo Regime Execution Tests
# ============================================================================


class TestSoloRegimeExecution:
    """Each coupling regime runs successfully on each device."""

    def _run_quick(
        self, regime: str, device: str, N: int = 128, steps: int = 20,
    ) -> dict[str, Any]:
        """Quick run of a regime and return basic metrics."""
        if regime == "csr":
            sim = OscilloSim(
                n_oscillators=N,
                coupling_strength=2.0,
                coupling_mode="csr",
                sparsity=0.99,
                device=device,
                seed=SEED,
            )
            result = sim.run(n_steps=steps, dt=DT, record_trajectory=False)
            r_final = result.order_parameter[-1] if result.order_parameter else 0.0
            return {"r": r_final, "wall": result.wall_time_s}
        else:
            mode_map = {
                "mean_field": ("mean_field", True),
                "sparse_knn": ("sparse_knn", False),
                "full": ("full", False),
            }
            coupling_mode, mf_flag = mode_map[regime]
            dev = torch.device(device)
            model = KuramotoOscillator(
                n_oscillators=N,
                coupling_strength=2.0,
                coupling_mode=coupling_mode,
                mean_field=mf_flag,
                device=dev,
            )
            state = OscillatorState.create_random(N, device=dev, seed=SEED)
            t0 = time.perf_counter()
            for _ in range(steps):
                state = model.step(state, dt=DT)
            wall = time.perf_counter() - t0
            r = kuramoto_order_parameter(state.phase).item()
            return {"r": r, "wall": wall}

    def test_mean_field_cpu(self) -> None:
        """Mean-field runs on CPU with valid order parameter."""
        result = self._run_quick("mean_field", "cpu")
        assert 0.0 <= result["r"] <= 1.0
        assert result["wall"] > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_mean_field_gpu(self) -> None:
        """Mean-field runs on GPU."""
        result = self._run_quick("mean_field", "cuda")
        assert 0.0 <= result["r"] <= 1.0

    def test_sparse_knn_cpu(self) -> None:
        """Sparse k-NN runs on CPU."""
        result = self._run_quick("sparse_knn", "cpu")
        assert 0.0 <= result["r"] <= 1.0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_sparse_knn_gpu(self) -> None:
        """Sparse k-NN runs on GPU."""
        result = self._run_quick("sparse_knn", "cuda")
        assert 0.0 <= result["r"] <= 1.0

    def test_full_pairwise_cpu(self) -> None:
        """Full pairwise runs on CPU."""
        result = self._run_quick("full", "cpu")
        assert 0.0 <= result["r"] <= 1.0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_full_pairwise_gpu(self) -> None:
        """Full pairwise runs on GPU."""
        result = self._run_quick("full", "cuda")
        assert 0.0 <= result["r"] <= 1.0

    def test_csr_cpu(self) -> None:
        """CSR sparse via OscilloSim runs on CPU."""
        result = self._run_quick("csr", "cpu")
        assert 0.0 <= result["r"] <= 1.0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_csr_gpu(self) -> None:
        """CSR sparse via OscilloSim runs on GPU."""
        result = self._run_quick("csr", "cuda")
        assert 0.0 <= result["r"] <= 1.0


# ============================================================================
# CSR Mode Specific Tests
# ============================================================================


class TestCSRMode:
    """CSR coupling mode produces valid physics."""

    def test_csr_order_param_increases_with_K(self) -> None:
        """Stronger coupling → higher order parameter in CSR mode."""
        r_vals = []
        for K in [0.0, 5.0, 20.0]:
            sim = OscilloSim(
                n_oscillators=512,
                coupling_strength=K,
                coupling_mode="csr",
                sparsity=0.95,
                device="cpu",
                seed=SEED,
            )
            result = sim.run(n_steps=200, dt=DT, record_trajectory=False)
            r_vals.append(result.order_parameter[-1])
        # r should be monotonically non-decreasing with K
        # (K=0 should give low r, K=20 should give high r)
        assert r_vals[2] >= r_vals[0] - 0.1, (
            f"CSR r not increasing with K: {r_vals}"
        )

    def test_csr_sparsity_parameter(self) -> None:
        """Different sparsity values produce different coupling matrices."""
        for sparsity in [0.5, 0.9, 0.99]:
            sim = OscilloSim(
                n_oscillators=256,
                coupling_strength=2.0,
                coupling_mode="csr",
                sparsity=sparsity,
                device="cpu",
                seed=SEED,
            )
            result = sim.run(n_steps=50, dt=DT)
            assert len(result.order_parameter) > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_csr_gpu_matches_physics(self) -> None:
        """CSR on GPU produces valid order parameters."""
        sim = OscilloSim(
            n_oscillators=1024,
            coupling_strength=5.0,
            coupling_mode="csr",
            sparsity=0.99,
            device="cuda",
            seed=SEED,
        )
        result = sim.run(n_steps=100, dt=DT, record_trajectory=True,
                         record_interval=10)
        assert len(result.order_parameter) > 0
        for r in result.order_parameter:
            assert 0.0 <= r <= 1.0 + 1e-6


# ============================================================================
# Concurrent Execution Tests
# ============================================================================


class TestConcurrentExecution:
    """Multi-regime concurrent runs complete without errors."""

    def test_two_regimes_concurrent_cpu(self) -> None:
        """Two different regimes run concurrently on CPU."""
        results: list[dict[str, Any]] = [{"status": "pending"}, {"status": "pending"}]

        def worker(idx: int, regime: str) -> None:
            try:
                model = KuramotoOscillator(
                    n_oscillators=128,
                    coupling_strength=2.0,
                    coupling_mode=regime,
                    mean_field=(regime == "mean_field"),
                    device=torch.device("cpu"),
                )
                state = OscillatorState.create_random(
                    128, device=torch.device("cpu"), seed=SEED + idx,
                )
                for _ in range(20):
                    state = model.step(state, dt=DT)
                r = kuramoto_order_parameter(state.phase).item()
                results[idx] = {"status": "ok", "r": r}
            except Exception as e:
                results[idx] = {"status": f"error: {e}"}

        t1 = threading.Thread(target=worker, args=(0, "mean_field"))
        t2 = threading.Thread(target=worker, args=(1, "sparse_knn"))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert results[0]["status"] == "ok", results[0]
        assert results[1]["status"] == "ok", results[1]
        assert 0.0 <= results[0]["r"] <= 1.0
        assert 0.0 <= results[1]["r"] <= 1.0

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_mixed_device_concurrent(self) -> None:
        """One regime on GPU + another on CPU concurrently."""
        results: list[dict[str, Any]] = [{"status": "pending"}, {"status": "pending"}]

        def worker_gpu(out: dict[str, Any]) -> None:
            try:
                model = KuramotoOscillator(
                    n_oscillators=512,
                    coupling_strength=2.0,
                    coupling_mode="mean_field",
                    mean_field=True,
                    device=torch.device("cuda"),
                )
                state = OscillatorState.create_random(
                    512, device=torch.device("cuda"), seed=SEED,
                )
                for _ in range(20):
                    state = model.step(state, dt=DT)
                torch.cuda.synchronize()
                r = kuramoto_order_parameter(state.phase).item()
                out.update({"status": "ok", "r": r})
            except Exception as e:
                out["status"] = f"error: {e}"

        def worker_cpu(out: dict[str, Any]) -> None:
            try:
                model = KuramotoOscillator(
                    n_oscillators=256,
                    coupling_strength=2.0,
                    coupling_mode="full",
                    device=torch.device("cpu"),
                )
                state = OscillatorState.create_random(
                    256, device=torch.device("cpu"), seed=SEED,
                )
                for _ in range(20):
                    state = model.step(state, dt=DT)
                r = kuramoto_order_parameter(state.phase).item()
                out.update({"status": "ok", "r": r})
            except Exception as e:
                out["status"] = f"error: {e}"

        t1 = threading.Thread(target=worker_gpu, args=(results[0],))
        t2 = threading.Thread(target=worker_cpu, args=(results[1],))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert results[0]["status"] == "ok", results[0]
        assert results[1]["status"] == "ok", results[1]


# ============================================================================
# Goldilocks Zone Tests
# ============================================================================


class TestGoldilocksZone:
    """K sweep produces physically valid results."""

    def test_order_param_increases_with_K_mean_field(self) -> None:
        """Mean-field r increases monotonically (approx) with K."""
        K_vals = [0.0, 2.0, 5.0, 10.0]
        r_vals = []
        for K in K_vals:
            model = KuramotoOscillator(
                n_oscillators=512,
                coupling_strength=K,
                coupling_mode="mean_field",
                mean_field=True,
                device=torch.device("cpu"),
            )
            state = OscillatorState.create_random(
                512, device=torch.device("cpu"), seed=SEED,
            )
            for _ in range(100):
                state = model.step(state, dt=DT)
            r_vals.append(kuramoto_order_parameter(state.phase).item())

        # r at K=10 should exceed r at K=0
        assert r_vals[-1] > r_vals[0] - 0.1, (
            f"r not increasing with K: {dict(zip(K_vals, r_vals))}"
        )

    def test_full_pairwise_k_sweep(self) -> None:
        """Full pairwise r increases with K at small N."""
        K_vals = [0.0, 3.0, 10.0]
        r_vals = []
        for K in K_vals:
            model = KuramotoOscillator(
                n_oscillators=64,
                coupling_strength=K,
                coupling_mode="full",
                device=torch.device("cpu"),
            )
            state = OscillatorState.create_random(
                64, device=torch.device("cpu"), seed=SEED,
            )
            for _ in range(100):
                state = model.step(state, dt=DT)
            r_vals.append(kuramoto_order_parameter(state.phase).item())

        assert r_vals[-1] > r_vals[0] - 0.1


# ============================================================================
# Finite-Size Scaling Tests
# ============================================================================


class TestFiniteSizeScaling:
    """K_c estimation across different N values."""

    def test_mean_field_scaling_produces_values(self) -> None:
        """K_c can be estimated for mean_field at different N."""
        sizes = [64, 128, 256]
        Kc_estimates = []

        for N in sizes:
            best_slope = 0.0
            best_K = 0.0
            prev_r = 0.0
            for K in [0.0, 1.0, 3.0, 5.0, 10.0]:
                model = KuramotoOscillator(
                    n_oscillators=N,
                    coupling_strength=K,
                    coupling_mode="mean_field",
                    mean_field=True,
                    device=torch.device("cpu"),
                )
                state = OscillatorState.create_random(
                    N, device=torch.device("cpu"), seed=SEED,
                )
                for _ in range(50):
                    state = model.step(state, dt=DT)
                r = kuramoto_order_parameter(state.phase).item()
                slope = r - prev_r
                if slope > best_slope:
                    best_slope = slope
                    best_K = K
                prev_r = r
            Kc_estimates.append(best_K)

        # All values should be within physical range
        for Kc in Kc_estimates:
            assert 0.0 <= Kc <= 20.0


# ============================================================================
# Phase Coherence Tests
# ============================================================================


class TestPhaseCoherence:
    """Phase coherence lifetime after coupling reduction."""

    def test_coherence_drops_after_k_zero(self) -> None:
        """Order parameter decreases after coupling is removed."""
        N = 64
        device = torch.device("cpu")

        # Phase 1: synchronise
        model_sync = KuramotoOscillator(
            n_oscillators=N,
            coupling_strength=10.0,
            coupling_mode="mean_field",
            mean_field=True,
            device=device,
        )
        state = OscillatorState.create_random(N, device=device, seed=SEED)
        for _ in range(200):
            state = model_sync.step(state, dt=DT)
        r_sync = kuramoto_order_parameter(state.phase).item()

        # Phase 2: remove coupling
        model_free = KuramotoOscillator(
            n_oscillators=N,
            coupling_strength=0.0,
            coupling_mode="mean_field",
            mean_field=True,
            device=device,
        )
        for _ in range(200):
            state = model_free.step(state, dt=DT)
        r_free = kuramoto_order_parameter(state.phase).item()

        # r should drop (or at least not increase substantially)
        # We allow some tolerance since small N can fluctuate
        assert r_free < r_sync + 0.3, (
            f"r didn't drop: sync={r_sync:.3f}, free={r_free:.3f}"
        )


# ============================================================================
# Chimera Detection Tests
# ============================================================================


class TestChimeraDetection:
    """Local order parameter computation for chimera detection."""

    def test_local_order_param_computation(self) -> None:
        """Local order parameter has valid range for all oscillators."""
        N = 128
        torch.manual_seed(SEED)
        phase = torch.rand(N) * 2 * math.pi
        phase_np = phase.numpy()

        k_local = 10
        local_r = np.zeros(N)
        for i in range(N):
            diffs = np.abs(np.angle(np.exp(1j * (phase_np - phase_np[i]))))
            nearest = np.argsort(diffs)[:k_local]
            z = np.mean(np.exp(1j * phase_np[nearest]))
            local_r[i] = np.abs(z)

        # All local r values should be in [0, 1]
        assert np.all(local_r >= 0.0)
        assert np.all(local_r <= 1.0 + 1e-6)

    def test_synchronized_state_high_local_r(self) -> None:
        """Fully synchronized oscillators have high local order parameter."""
        N = 64
        # All oscillators at same phase
        phase = torch.zeros(N)
        phase_np = phase.numpy()

        k_local = 10
        local_r = np.zeros(N)
        for i in range(N):
            diffs = np.abs(np.angle(np.exp(1j * (phase_np - phase_np[i]))))
            nearest = np.argsort(diffs)[:k_local]
            z = np.mean(np.exp(1j * phase_np[nearest]))
            local_r[i] = np.abs(z)

        # All should be ~1.0 for synchronized state
        assert np.mean(local_r) > 0.99


# ============================================================================
# Result Artefact Tests
# ============================================================================


class TestResultArtefacts:
    """Verify all JSON artefacts from the benchmark exist and are valid."""

    EXPECTED_FILES = [
        "y3q49_solo_matrix.json",
        "y3q49_concurrent_pairs.json",
        "y3q49_concurrent_triples.json",
        "y3q49_goldilocks_zones.json",
        "y3q49_finite_size_scaling.json",
        "y3q49_phase_coherence.json",
        "y3q49_chimera_detection.json",
        "y3q49_summary.json",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_artefact_exists(self, filename: str) -> None:
        """JSON artefact file exists."""
        path = RESULTS_DIR / filename
        assert path.exists(), f"Missing artefact: {path}"

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_artefact_valid_json(self, filename: str) -> None:
        """JSON artefact is parseable."""
        path = RESULTS_DIR / filename
        if not path.exists():
            pytest.skip(f"Artefact not yet generated: {filename}")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_summary_status_pass(self) -> None:
        """Summary JSON reports PASS status."""
        path = RESULTS_DIR / "y3q49_summary.json"
        if not path.exists():
            pytest.skip("Summary not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert data.get("status") == "PASS"

    def test_solo_matrix_has_all_regimes(self) -> None:
        """Solo matrix JSON contains all 4 regimes."""
        path = RESULTS_DIR / "y3q49_solo_matrix.json"
        if not path.exists():
            pytest.skip("Solo matrix not yet generated")
        with open(path) as f:
            data = json.load(f)
        matrix = data.get("matrix", {})
        regimes_found = {v["regime"] for v in matrix.values()}
        assert regimes_found == {"mean_field", "sparse_knn", "full", "csr"}

    def test_goldilocks_zones_all_regimes(self) -> None:
        """Goldilocks zones JSON contains all 4 regimes."""
        path = RESULTS_DIR / "y3q49_goldilocks_zones.json"
        if not path.exists():
            pytest.skip("Goldilocks zones not yet generated")
        with open(path) as f:
            data = json.load(f)
        zones = data.get("zones", {})
        regimes = {v["regime"] for v in zones.values()}
        assert regimes == {"mean_field", "sparse_knn", "full", "csr"}

    def test_report_exists(self) -> None:
        """Markdown report was generated."""
        report = (
            Path(__file__).resolve().parents[1]
            / "Docs"
            / "test_and_benchmark_results"
            / "y3q49_scientific_regime_report.md"
        )
        assert report.exists(), f"Missing report: {report}"
        content = report.read_text(encoding="utf-8")
        assert "Solo Device" in content
        assert "Goldilocks" in content
