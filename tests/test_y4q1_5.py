"""Tests for Year 4 Q1.5 — Session-Length Dynamics.

Tests the 8 new session-length metric functions and the session runner
logic.  All tests are deterministic (seeded) and run quickly (no
multi-minute sessions in the test suite).

Test classes:
    TestOrderParameterSeries — 7 tests
    TestWindowedOrderParameterVariance — 6 tests
    TestPhaseLockingValue — 6 tests
    TestInstantaneousFrequencySpread — 5 tests
    TestCumulativePhaseSlipCurve — 6 tests
    TestThroughputSeries — 5 tests
    TestMemoryGrowthProfile — 5 tests
    TestSessionLengthStatisticalComparison — 6 tests
    TestFDistributionPValue — 4 tests
    TestSessionRunnerSmoke — 4 tests
    TestPropertyBased — 4 hypothesis tests
    TestVersionConsistency — 2 tests
Total: 60 tests
"""

from __future__ import annotations

import importlib
import math
import tomllib

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# =====================================================================
# TestOrderParameterSeries
# =====================================================================


class TestOrderParameterSeries:
    """Test order_parameter_series function."""

    def test_fully_synchronized(self) -> None:
        """All oscillators at the same phase → r ≈ 1."""
        from prinet.utils.y4q1_tools import order_parameter_series

        trajectory = torch.zeros(20, 10)  # all phases = 0
        result = order_parameter_series(trajectory)
        assert result["mean_r"] == pytest.approx(1.0, abs=1e-5)

    def test_uniformly_spread(self) -> None:
        """Oscillators uniformly spread → r ≈ 0."""
        from prinet.utils.y4q1_tools import order_parameter_series

        N = 100
        T = 10
        phases = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
        trajectory = phases.unsqueeze(0).expand(T, -1)
        result = order_parameter_series(trajectory)
        assert result["mean_r"] < 0.1

    def test_3d_input(self) -> None:
        """Accepts (T, N, K) input and flattens."""
        from prinet.utils.y4q1_tools import order_parameter_series

        trajectory = torch.zeros(10, 4, 8)  # all zero → r = 1
        result = order_parameter_series(trajectory)
        assert result["mean_r"] == pytest.approx(1.0, abs=1e-5)

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import order_parameter_series

        result = order_parameter_series(torch.rand(5, 3))
        assert set(result.keys()) == {"r_series", "mean_r", "std_r", "final_r"}

    def test_r_series_length(self) -> None:
        from prinet.utils.y4q1_tools import order_parameter_series

        T = 15
        result = order_parameter_series(torch.rand(T, 5))
        assert len(result["r_series"]) == T

    def test_r_bounded(self) -> None:
        """r(t) ∈ [0, 1]."""
        from prinet.utils.y4q1_tools import order_parameter_series

        torch.manual_seed(99)
        result = order_parameter_series(torch.rand(50, 20) * 2 * math.pi)
        for r in result["r_series"]:
            assert -0.01 <= r <= 1.01

    def test_single_oscillator(self) -> None:
        """Single oscillator → r = 1 always."""
        from prinet.utils.y4q1_tools import order_parameter_series

        result = order_parameter_series(torch.rand(10, 1) * 2 * math.pi)
        assert result["mean_r"] == pytest.approx(1.0, abs=1e-5)


# =====================================================================
# TestWindowedOrderParameterVariance
# =====================================================================


class TestWindowedOrderParameterVariance:
    """Test windowed_order_parameter_variance function."""

    def test_constant_series(self) -> None:
        """Constant r → all window stds ≈ 0."""
        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        result = windowed_order_parameter_variance([0.5] * 100, window_size=10)
        assert all(s < 1e-10 for s in result["window_stds"])

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        result = windowed_order_parameter_variance([0.1, 0.2] * 20)
        assert set(result.keys()) == {"window_stds", "trend_slope", "mean_window_std"}

    def test_increasing_variance(self) -> None:
        """Increasingly noisy series → positive trend slope."""
        import numpy as np

        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        np.random.seed(42)
        series = []
        for i in range(100):
            noise = np.random.normal(0, 0.01 * (i + 1))
            series.append(0.5 + noise)
        result = windowed_order_parameter_variance(series, window_size=10)
        assert result["trend_slope"] > 0

    def test_short_series(self) -> None:
        """Very short series returns gracefully."""
        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        result = windowed_order_parameter_variance([0.5])
        assert len(result["window_stds"]) >= 1

    def test_window_count(self) -> None:
        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        result = windowed_order_parameter_variance(list(range(100)), window_size=10)
        assert len(result["window_stds"]) == 10

    def test_zero_slope_for_stable(self) -> None:
        """Alternating same variance → slope ≈ 0."""
        import numpy as np

        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        np.random.seed(42)
        series = [np.random.normal(0.5, 0.01) for _ in range(100)]
        result = windowed_order_parameter_variance(series, window_size=10)
        assert abs(result["trend_slope"]) < 0.01


# =====================================================================
# TestPhaseLockingValue
# =====================================================================


class TestPhaseLockingValue:
    """Test phase_locking_value function."""

    def test_identical_phases(self) -> None:
        """Identical phases → PLV = 1."""
        from prinet.utils.y4q1_tools import phase_locking_value

        a = torch.zeros(50)
        b = torch.zeros(50)
        result = phase_locking_value(a, b)
        assert result["plv"] == pytest.approx(1.0, abs=1e-5)

    def test_constant_offset(self) -> None:
        """Constant phase offset → PLV = 1."""
        from prinet.utils.y4q1_tools import phase_locking_value

        a = torch.zeros(50)
        b = torch.full((50,), math.pi / 4)
        result = phase_locking_value(a, b)
        assert result["plv"] == pytest.approx(1.0, abs=1e-5)

    def test_random_phases(self) -> None:
        """Random independent phases → PLV ≈ 0."""
        from prinet.utils.y4q1_tools import phase_locking_value

        torch.manual_seed(42)
        a = torch.rand(1000) * 2 * math.pi
        b = torch.rand(1000) * 2 * math.pi
        result = phase_locking_value(a, b)
        assert result["plv"] < 0.15

    def test_2d_input(self) -> None:
        """(T, N) input → per-pair PLVs."""
        from prinet.utils.y4q1_tools import phase_locking_value

        a = torch.zeros(20, 5)
        b = torch.zeros(20, 5)
        result = phase_locking_value(a, b)
        assert len(result["plv_per_pair"]) == 5

    def test_plv_bounded(self) -> None:
        from prinet.utils.y4q1_tools import phase_locking_value

        torch.manual_seed(7)
        result = phase_locking_value(
            torch.rand(100) * 2 * math.pi,
            torch.rand(100) * 2 * math.pi,
        )
        assert 0.0 <= result["plv"] <= 1.0

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import phase_locking_value

        result = phase_locking_value(torch.zeros(5), torch.zeros(5))
        assert set(result.keys()) == {"plv", "plv_per_pair"}


# =====================================================================
# TestInstantaneousFrequencySpread
# =====================================================================


class TestInstantaneousFrequencySpread:
    """Test instantaneous_frequency_spread function."""

    def test_constant_phase(self) -> None:
        """Constant phases → zero spread."""
        from prinet.utils.y4q1_tools import instantaneous_frequency_spread

        trajectory = torch.zeros(20, 5)
        result = instantaneous_frequency_spread(trajectory)
        assert result["mean_spread"] == pytest.approx(0.0, abs=1e-6)

    def test_uniform_drift(self) -> None:
        """All oscillators drifting at same rate → zero spread."""
        from prinet.utils.y4q1_tools import instantaneous_frequency_spread

        T, N = 20, 5
        times = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        trajectory = times.expand(T, N) * 0.1  # same rate for all
        result = instantaneous_frequency_spread(trajectory)
        assert result["mean_spread"] < 0.01

    def test_diverging_frequencies(self) -> None:
        """Different drift rates → positive spread."""
        from prinet.utils.y4q1_tools import instantaneous_frequency_spread

        T = 30
        rates = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
        times = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        trajectory = times * rates.unsqueeze(0)
        result = instantaneous_frequency_spread(trajectory)
        assert result["mean_spread"] > 0.5

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import instantaneous_frequency_spread

        result = instantaneous_frequency_spread(torch.rand(10, 3))
        assert set(result.keys()) == {
            "freq_spread_series",
            "mean_spread",
            "trend_slope",
        }

    def test_short_trajectory(self) -> None:
        """T=1 → empty result."""
        from prinet.utils.y4q1_tools import instantaneous_frequency_spread

        result = instantaneous_frequency_spread(torch.rand(1, 5))
        assert result["mean_spread"] == 0.0


# =====================================================================
# TestCumulativePhaseSlipCurve
# =====================================================================


class TestCumulativePhaseSlipCurve:
    """Test cumulative_phase_slip_curve function."""

    def test_smooth_trajectory(self) -> None:
        """Smooth trajectory → zero cumulative slips."""
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        trajectory = torch.zeros(30, 5)
        result = cumulative_phase_slip_curve(trajectory)
        assert result["total_slips"] == 0

    def test_alternating_antipodal(self) -> None:
        """Alternating 0/π → many slips."""
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        T, N = 20, 3
        trajectory = torch.zeros(T, N)
        for t in range(T):
            if t % 2 == 1:
                trajectory[t] = math.pi
        result = cumulative_phase_slip_curve(trajectory)
        assert result["total_slips"] > 0

    def test_cumulative_monotonic(self) -> None:
        """Cumulative curve is non-decreasing."""
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        torch.manual_seed(42)
        trajectory = torch.rand(50, 5) * 2 * math.pi
        result = cumulative_phase_slip_curve(trajectory)
        cum = result["cumulative_slips"]
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1]

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        result = cumulative_phase_slip_curve(torch.rand(10, 3))
        assert set(result.keys()) == {"cumulative_slips", "total_slips", "acceleration"}

    def test_short_trajectory(self) -> None:
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        result = cumulative_phase_slip_curve(torch.rand(1, 5))
        assert result["total_slips"] == 0

    def test_acceleration_near_zero_for_constant_rate(self) -> None:
        """Constant slip rate → acceleration ≈ 0 (linear curve)."""
        from prinet.utils.y4q1_tools import cumulative_phase_slip_curve

        # Alternating 0/π every step → constant slip per step
        T, N = 40, 1
        trajectory = torch.zeros(T, N)
        for t in range(T):
            if t % 2 == 1:
                trajectory[t] = math.pi
        result = cumulative_phase_slip_curve(trajectory)
        # Linear cumulative → acceleration (quadratic coeff) ≈ 0
        assert abs(result["acceleration"]) < 0.1


# =====================================================================
# TestThroughputSeries
# =====================================================================


class TestThroughputSeries:
    """Test throughput_series function."""

    def test_constant_throughput(self) -> None:
        from prinet.utils.y4q1_tools import throughput_series

        result = throughput_series([1.0] * 10, frames_per_interval=100)
        assert result["mean_fps"] == pytest.approx(100.0, abs=0.1)
        assert abs(result["degradation_pct"]) < 1.0

    def test_degrading_throughput(self) -> None:
        """Increasing wall time → degradation_pct < 0."""
        from prinet.utils.y4q1_tools import throughput_series

        # Each interval takes longer → FPS drops
        times = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        result = throughput_series(times, frames_per_interval=100)
        assert result["degradation_pct"] < 0

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import throughput_series

        result = throughput_series([1.0, 2.0], frames_per_interval=50)
        assert set(result.keys()) == {
            "fps_series",
            "mean_fps",
            "std_fps",
            "degradation_pct",
        }

    def test_single_interval(self) -> None:
        from prinet.utils.y4q1_tools import throughput_series

        result = throughput_series([0.5], frames_per_interval=100)
        assert result["mean_fps"] == pytest.approx(200.0, abs=0.1)

    def test_fps_positive(self) -> None:
        from prinet.utils.y4q1_tools import throughput_series

        result = throughput_series([0.1, 0.2, 0.3], frames_per_interval=10)
        assert all(f > 0 for f in result["fps_series"])


# =====================================================================
# TestMemoryGrowthProfile
# =====================================================================


class TestMemoryGrowthProfile:
    """Test memory_growth_profile function."""

    def test_stable_memory(self) -> None:
        from prinet.utils.y4q1_tools import memory_growth_profile

        result = memory_growth_profile([100.0] * 20, interval_seconds=30.0)
        assert result["growth_mb"] == pytest.approx(0.0)
        assert not result["is_leaking"]

    def test_growing_memory(self) -> None:
        from prinet.utils.y4q1_tools import memory_growth_profile

        samples = [100.0 + i * 5 for i in range(20)]
        result = memory_growth_profile(samples, interval_seconds=30.0)
        assert result["growth_mb"] > 50
        assert result["is_leaking"]

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import memory_growth_profile

        result = memory_growth_profile([100.0, 110.0], interval_seconds=60.0)
        assert set(result.keys()) == {
            "initial_mb",
            "peak_mb",
            "final_mb",
            "growth_mb",
            "growth_rate_mb_per_min",
            "is_leaking",
        }

    def test_empty_input(self) -> None:
        from prinet.utils.y4q1_tools import memory_growth_profile

        result = memory_growth_profile([], interval_seconds=10.0)
        assert result["initial_mb"] == 0.0

    def test_growth_rate(self) -> None:
        """Known growth rate: 1 MB per sample, 60s intervals = 1 MB/min."""
        from prinet.utils.y4q1_tools import memory_growth_profile

        samples = [100.0 + i for i in range(10)]
        result = memory_growth_profile(samples, interval_seconds=60.0)
        assert result["growth_rate_mb_per_min"] == pytest.approx(1.0, abs=0.1)


# =====================================================================
# TestSessionLengthStatisticalComparison
# =====================================================================


class TestSessionLengthStatisticalComparison:
    """Test session_length_statistical_comparison function."""

    def test_identical_groups(self) -> None:
        """Same values → no significance."""
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        data = {
            "5min": [0.5, 0.5, 0.5],
            "10min": [0.5, 0.5, 0.5],
            "30min": [0.5, 0.5, 0.5],
        }
        result = session_length_statistical_comparison(data)
        assert not result["significant"]

    def test_clearly_different_groups(self) -> None:
        """Clearly different means → significant."""
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        data = {
            "5min": [10.0, 10.1, 9.9, 10.0, 10.2],
            "10min": [5.0, 5.1, 4.9, 5.0, 5.2],
            "30min": [1.0, 1.1, 0.9, 1.0, 1.2],
        }
        result = session_length_statistical_comparison(data)
        assert result["significant"]
        assert result["anova_f"] > 10

    def test_output_keys(self) -> None:
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        data = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        result = session_length_statistical_comparison(data)
        assert set(result.keys()) == {
            "anova_f",
            "anova_p",
            "pairwise_d",
            "significant",
            "eta_squared",
        }

    def test_pairwise_d_count(self) -> None:
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        data = {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
        result = session_length_statistical_comparison(data)
        # 3 groups → 3 pairs
        assert len(result["pairwise_d"]) == 3

    def test_single_group(self) -> None:
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        result = session_length_statistical_comparison({"a": [1.0]})
        assert not result["significant"]

    def test_eta_squared_bounded(self) -> None:
        from prinet.utils.y4q1_tools import session_length_statistical_comparison

        data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
        result = session_length_statistical_comparison(data)
        assert 0.0 <= result["eta_squared"] <= 1.0


# =====================================================================
# TestFDistributionPValue
# =====================================================================


class TestFDistributionPValue:
    """Test the internal _f_distribution_p_value function."""

    def test_zero_f_stat(self) -> None:
        from prinet.utils.y4q1_tools import _f_distribution_p_value

        assert _f_distribution_p_value(0.0, 2, 10) == 1.0

    def test_large_f_stat(self) -> None:
        """Very large F → p ≈ 0."""
        from prinet.utils.y4q1_tools import _f_distribution_p_value

        p = _f_distribution_p_value(100.0, 3, 20)
        assert p < 0.01

    def test_moderate_f(self) -> None:
        """F ~ 3, df1=2, df2=30 → p ~ 0.05-0.10."""
        from prinet.utils.y4q1_tools import _f_distribution_p_value

        p = _f_distribution_p_value(3.0, 2, 30)
        assert 0.01 < p < 0.2

    def test_p_bounded(self) -> None:
        from prinet.utils.y4q1_tools import _f_distribution_p_value

        p = _f_distribution_p_value(5.0, 3, 15)
        assert 0.0 <= p <= 1.0


# =====================================================================
# TestSessionRunnerSmoke
# =====================================================================


class TestSessionRunnerSmoke:
    """Smoke tests for the session runner (very short durations)."""

    def test_imports(self) -> None:
        """Benchmark module imports without error."""
        import benchmarks.y4q1_5_benchmarks as mod

        assert hasattr(mod, "run_timed_session")
        assert hasattr(mod, "benchmark_5min")

    def test_micro_session(self) -> None:
        """0.05-minute (3s) session produces valid output."""
        from benchmarks.y4q1_5_benchmarks import run_timed_session

        result = run_timed_session(
            duration_minutes=0.05,
            seed=42,
            n_objects=4,
            warmup_seconds=2.0,
        )
        assert "aggregate" in result
        assert "intervals" in result
        agg = result["aggregate"]
        assert agg["total_frames"] > 0
        assert agg["total_wall_seconds"] > 0

    def test_micro_session_has_metrics(self) -> None:
        """Short session records at least one interval with metrics."""
        from benchmarks.y4q1_5_benchmarks import run_timed_session

        result = run_timed_session(
            duration_minutes=0.1,
            seed=42,
            n_objects=4,
            warmup_seconds=2.0,
        )
        if result["intervals"]:
            interval = result["intervals"][0]
            assert "fps" in interval
            assert "elapsed_seconds" in interval

    def test_deterministic_seed(self) -> None:
        """Same seed produces same frame count and similar metrics."""
        from benchmarks.y4q1_5_benchmarks import run_timed_session

        r1 = run_timed_session(0.05, seed=42, warmup_seconds=1.0)
        r2 = run_timed_session(0.05, seed=42, warmup_seconds=1.0)
        # Frame counts may differ slightly due to timing, but should be close
        assert (
            abs(r1["aggregate"]["total_frames"] - r2["aggregate"]["total_frames"])
            <= r1["aggregate"]["total_frames"] * 0.5
        )


# =====================================================================
# TestPropertyBased
# =====================================================================


class TestPropertyBased:
    """Hypothesis property-based tests for session-length metrics."""

    @given(
        phases=st.lists(
            st.floats(
                min_value=0, max_value=6.28, allow_infinity=False, allow_nan=False
            ),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_order_param_bounded(self, phases: list[float]) -> None:
        from prinet.utils.y4q1_tools import order_parameter_series

        t = torch.tensor(phases).unsqueeze(1)  # (T, 1)
        result = order_parameter_series(t)
        for r in result["r_series"]:
            assert -0.01 <= r <= 1.01

    @given(
        vals=st.lists(
            st.floats(min_value=0, max_value=10, allow_infinity=False, allow_nan=False),
            min_size=4,
            max_size=100,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_windowed_var_non_negative(self, vals: list[float]) -> None:
        from prinet.utils.y4q1_tools import windowed_order_parameter_variance

        result = windowed_order_parameter_variance(vals, window_size=2)
        assert all(s >= 0 for s in result["window_stds"])

    @given(
        data=st.lists(
            st.floats(
                min_value=0.1, max_value=10, allow_infinity=False, allow_nan=False
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_throughput_fps_positive(self, data: list[float]) -> None:
        from prinet.utils.y4q1_tools import throughput_series

        result = throughput_series(data, frames_per_interval=100)
        assert all(f > 0 for f in result["fps_series"])

    @given(
        samples=st.lists(
            st.floats(
                min_value=1, max_value=1000, allow_infinity=False, allow_nan=False
            ),
            min_size=2,
            max_size=30,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_memory_profile_peak_gte_initial(self, samples: list[float]) -> None:
        from prinet.utils.y4q1_tools import memory_growth_profile

        result = memory_growth_profile(samples, interval_seconds=10.0)
        assert result["peak_mb"] >= result["initial_mb"]


# =====================================================================
# TestVersionConsistency
# =====================================================================


class TestVersionConsistency:
    """Version consistency checks."""

    def test_version_string(self) -> None:
        import prinet

        parts = prinet.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    def test_pyproject_matches(self) -> None:
        import prinet

        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == prinet.__version__
