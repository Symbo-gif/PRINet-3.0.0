"""Quick CLEVR-N sweep runner for all 5 baselines.

Run with: python benchmarks/run_clevr_n_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.clevr_n import (
    CLEVRNResult,
    DeltaThetaGammaCLEVRN,
    HopfieldCLEVRNBaseline,
    LSTMCLEVRNBaseline,
    ThetaGammaCLEVRN,
    TransformerCLEVRN,
    run_clevr_n_sweep,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    all_results: dict[str, list[CLEVRNResult]] = {}

    # Fast baselines: full N range
    fast_baselines = {
        "LSTM": lambda scene_dim, query_dim: LSTMCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "Transformer": lambda scene_dim, query_dim: TransformerCLEVRN(
            scene_dim=scene_dim, query_dim=query_dim
        ),
        "Hopfield": lambda scene_dim, query_dim: HopfieldCLEVRNBaseline(
            scene_dim=scene_dim, query_dim=query_dim
        ),
    }

    for name, factory in fast_baselines.items():
        print(f"\n--- {name} ---")
        results = run_clevr_n_sweep(
            factory, name, n_items_list=[2, 4, 6, 8], n_epochs=15, device=DEVICE
        )
        all_results[name] = results
        for r in results:
            print(f"  N={r.n_items}: test_acc={r.test_acc:.3f}")

    # Hierarchical baselines: reduced scope (per-element ODE is expensive)
    # ThetaGamma: N=2 only with 3 integration steps, 3 epochs
    # DeltaThetaGamma: skipped (3-level ODE needs vectorized batch for speed)
    hierarchical = {
        "ThetaGamma": lambda scene_dim, query_dim: ThetaGammaCLEVRN(
            scene_dim=scene_dim,
            query_dim=query_dim,
            n_integration_steps=3,
        ),
    }

    for name, factory in hierarchical.items():
        print(f"\n--- {name} ---")
        results = run_clevr_n_sweep(
            factory, name, n_items_list=[2], n_epochs=3, device=DEVICE
        )
        all_results[name] = results
        for r in results:
            print(f"  N={r.n_items}: test_acc={r.test_acc:.3f}")

    # Save
    save_path = Path("Docs/test_and_benchmark_results/benchmark_clevr_n_sweep.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        n: [r.to_dict() for r in rs] for n, rs in all_results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
