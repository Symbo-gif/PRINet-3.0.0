"""Run individual Q1.9 benchmarks by name.

Usage:
    python benchmarks/run_q19_individual.py <benchmark_name>
    python benchmarks/run_q19_individual.py --list

Example:
    python benchmarks/run_q19_individual.py g1_1_7seed_comparison
    python benchmarks/run_q19_individual.py g4_1_chimera_in_tracking
"""

from __future__ import annotations

import sys
import traceback

from y4q1_9_benchmarks import ALL_BENCHMARKS, _cleanup, _p


def main() -> None:
    """Run a single Q1.9 benchmark by name."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        _p("Usage: python run_q19_individual.py <benchmark_name>")
        _p("       python run_q19_individual.py --list")
        sys.exit(1)

    name = sys.argv[1]

    if name == "--list":
        _p("Available Q1.9 benchmarks:")
        for bname, _ in ALL_BENCHMARKS:
            _p(f"  {bname}")
        sys.exit(0)

    bench_map = dict(ALL_BENCHMARKS)
    if name not in bench_map:
        _p(f"Unknown benchmark: {name}")
        _p("Available:")
        for bname, _ in ALL_BENCHMARKS:
            _p(f"  {bname}")
        sys.exit(1)

    try:
        _p(f"Running Q1.9 benchmark: {name}")
        bench_map[name]()
        _p(f"\n[OK] {name} completed.")
    except Exception as e:
        _p(f"\n[FAIL] {name}: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
