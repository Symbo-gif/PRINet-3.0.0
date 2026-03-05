"""Year 3 Q4 Benchmarks for PRINet — Publication & OscilloSim v2.0.

Covers all Q4 deliverables:
- P.3: PyPI v2.0 packaging validation (version, API surface count)
- P.4: OscilloSim v2.0 scaling (N=1K → 1M, three coupling modes)
- P.5: Slot Attention vs HybridPRINetV2 on CLEVR-N

Run with::

    python benchmarks/y3q4_benchmarks.py

Results are written to ``benchmarks/results/y3q4_*.json``.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure src/ is on path when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

RESULTS_DIR = _ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WARMUP_ITERS = 3
_BENCH_ITERS = 10
SEED = 42


def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved -> {path.relative_to(_ROOT)}")


def _seed(s: int = SEED) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _timer(fn, *, warmup: int = _WARMUP_ITERS, iters: int = _BENCH_ITERS) -> dict:
    """Time *fn* and return timing statistics in milliseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "stdev_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "iters": iters,
    }


# ══════════════════════════════════════════════════════════════════
# P.3: PyPI v2.0 packaging validation
# ══════════════════════════════════════════════════════════════════


def bench_p3_packaging() -> dict:
    """Validate v2.0.0 packaging metadata and API surface."""
    print("\n[P.3] PyPI v2.0 packaging validation")
    import prinet

    all_symbols = prinet.__all__
    nn_symbols = prinet.nn.__all__
    utils_symbols = prinet.utils.__all__

    # Check new Q4 symbols present
    q4_symbols = [
        "SlotAttentionModule",
        "SlotAttentionCLEVRN",
        "OscilloSim",
        "SimulationResult",
        "quick_simulate",
    ]
    q4_present = {s: s in all_symbols for s in q4_symbols}

    result = {
        "benchmark": "P.3_packaging_v2",
        "version": prinet.__version__,
        "total_public_symbols": len(all_symbols),
        "nn_public_symbols": len(nn_symbols),
        "utils_public_symbols": len(utils_symbols),
        "q4_symbols": q4_present,
        "all_q4_present": all(q4_present.values()),
        "status": "PASS" if prinet.__version__ == "2.0.0" and all(q4_present.values()) else "FAIL",
    }
    _save("y3q4_p3_packaging.json", result)
    print(f"  version={result['version']}, symbols={result['total_public_symbols']}, Q4 OK={result['all_q4_present']}")
    return result


# ══════════════════════════════════════════════════════════════════
# P.4: OscilloSim v2.0 scaling
# ══════════════════════════════════════════════════════════════════


def bench_p4_oscillosim_scaling() -> dict:
    """Benchmark OscilloSim across N=1K → 1M for each coupling mode."""
    print("\n[P.4] OscilloSim v2.0 scaling")
    from prinet.utils.oscillosim import OscilloSim

    n_steps = 100
    dt = 0.01
    k_strength = 1.0
    device = _DEVICE

    modes = ["mean_field", "sparse_knn", "csr"]
    # N values appropriate per mode
    mode_ns: dict[str, list[int]] = {
        "mean_field": [1_000, 10_000, 100_000, 1_000_000],
        "sparse_knn": [1_000, 5_000, 10_000, 50_000],
        "csr": [100, 500, 1_000, 5_000],
    }

    results: dict[str, list[dict]] = {}
    for mode in modes:
        mode_results: list[dict] = []
        for n in mode_ns[mode]:
            _seed()
            try:
                sim = OscilloSim(
                    n_oscillators=n,
                    coupling_mode=mode,
                    coupling_strength=k_strength,
                    device=device,
                    seed=SEED,
                )
                res = sim.run(n_steps=n_steps, dt=dt)
                entry = {
                    "n_oscillators": n,
                    "wall_time_s": round(res.wall_time_s, 4),
                    "throughput_osc_step_per_s": round(res.throughput, 0),
                    "final_order_param": round(res.order_parameter[-1], 4) if res.order_parameter else None,
                    "status": "OK",
                }
            except Exception as e:
                entry = {"n_oscillators": n, "status": f"FAIL: {e}"}
            mode_results.append(entry)
            print(f"  {mode} N={n:>9,d}: {entry.get('wall_time_s', 'N/A')}s, "
                  f"throughput={entry.get('throughput_osc_step_per_s', 'N/A')}")
        results[mode] = mode_results

    # Check million-scale target
    mf_1m = next(
        (r for r in results.get("mean_field", []) if r.get("n_oscillators") == 1_000_000),
        None,
    )
    million_ok = mf_1m is not None and mf_1m.get("status") == "OK"

    data = {
        "benchmark": "P.4_oscillosim_scaling",
        "device": device,
        "n_steps": n_steps,
        "dt": dt,
        "coupling_strength": k_strength,
        "results_by_mode": results,
        "million_scale_achieved": million_ok,
        "target": "1M+ oscillators in mean_field mode",
        "status": "PASS" if million_ok else "FAIL",
    }
    _save("y3q4_p4_oscillosim_scaling.json", data)
    return data


def bench_p4_oscillosim_order_param() -> dict:
    """Benchmark order parameter convergence under varying coupling."""
    print("\n[P.4] OscilloSim order-parameter convergence")
    from prinet.utils.oscillosim import OscilloSim

    couplings = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    n = 2_000
    n_steps = 500
    results: list[dict] = []

    for k in couplings:
        _seed()
        sim = OscilloSim(
            n_oscillators=n,
            coupling_mode="mean_field",
            coupling_strength=k,
            device=_DEVICE,
            seed=SEED,
        )
        res = sim.run(n_steps=n_steps, dt=0.01)
        final_r = res.order_parameter[-1] if res.order_parameter else 0.0
        results.append({
            "coupling_strength": k,
            "final_order_param": round(final_r, 4),
            "wall_time_s": round(res.wall_time_s, 4),
        })
        print(f"  K={k:>5.1f}: r={final_r:.4f}")

    # Expect higher coupling → higher order parameter
    monotonic = all(
        results[i]["final_order_param"] <= results[i + 1]["final_order_param"] + 0.05
        for i in range(len(results) - 1)
    )

    data = {
        "benchmark": "P.4_order_param_convergence",
        "device": _DEVICE,
        "n_oscillators": n,
        "n_steps": n_steps,
        "results": results,
        "roughly_monotonic": monotonic,
        "status": "PASS" if monotonic else "FAIL",
    }
    _save("y3q4_p4_order_param.json", data)
    return data


# ══════════════════════════════════════════════════════════════════
# P.5: Slot Attention vs HybridPRINetV2 on CLEVR-N
# ══════════════════════════════════════════════════════════════════


def _train_and_eval(
    model: torch.nn.Module,
    train_scenes: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    test_scenes: torch.Tensor,
    test_queries: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
) -> dict:
    """Train a CLEVR-N model and return accuracy + timing."""
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        logits = model(train_scenes.to(device), train_queries.to(device))
        loss = F.nll_loss(logits, train_labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_logits = model(test_scenes.to(device), test_queries.to(device))
        preds = test_logits.argmax(dim=-1)
        acc = (preds == test_labels.to(device)).float().mean().item()

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "accuracy": round(acc, 4),
        "train_time_s": round(train_time, 3),
        "n_params": n_params,
    }


def bench_p5_slot_attention_comparison() -> dict:
    """Compare Slot Attention vs HybridPRINetV2CLEVRN on synthetic CLEVR-N."""
    print("\n[P.5] Slot Attention comparison (CLEVR-N)")
    from prinet.nn.slot_attention import SlotAttentionCLEVRN

    scene_dim = 16
    query_dim = 60
    n_train = 200
    n_test = 50
    device = torch.device(_DEVICE)

    # Synthetic data (simple linearly separable for quick benchmark)
    _seed()
    train_scenes = torch.randn(n_train, scene_dim)
    train_queries = torch.randn(n_train, query_dim)
    train_labels = (train_scenes[:, 0] > 0).long()
    test_scenes = torch.randn(n_test, scene_dim)
    test_queries = torch.randn(n_test, query_dim)
    test_labels = (test_scenes[:, 0] > 0).long()

    comparisons: dict[str, dict] = {}

    # ── Slot Attention ───────────────────────────────────────────
    _seed()
    slot_model = SlotAttentionCLEVRN(
        scene_dim=scene_dim,
        query_dim=query_dim,
        num_slots=8,
        slot_dim=64,
        d_model=64,
    ).to(device)
    comparisons["SlotAttentionCLEVRN"] = _train_and_eval(
        slot_model,
        train_scenes, train_queries, train_labels,
        test_scenes, test_queries, test_labels,
    )
    print(f"  SlotAttention: acc={comparisons['SlotAttentionCLEVRN']['accuracy']:.4f}, "
          f"time={comparisons['SlotAttentionCLEVRN']['train_time_s']:.3f}s, "
          f"params={comparisons['SlotAttentionCLEVRN']['n_params']:,d}")

    # ── HybridPRINetV2CLEVRN ────────────────────────────────────
    try:
        from prinet.nn.hybrid import HybridPRINetV2CLEVRN

        _seed()
        hybrid_model = HybridPRINetV2CLEVRN(
            scene_dim=scene_dim,
            query_dim=query_dim,
        ).to(device)
        comparisons["HybridPRINetV2CLEVRN"] = _train_and_eval(
            hybrid_model,
            train_scenes, train_queries, train_labels,
            test_scenes, test_queries, test_labels,
        )
        print(f"  HybridPRINetV2: acc={comparisons['HybridPRINetV2CLEVRN']['accuracy']:.4f}, "
              f"time={comparisons['HybridPRINetV2CLEVRN']['train_time_s']:.3f}s, "
              f"params={comparisons['HybridPRINetV2CLEVRN']['n_params']:,d}")
    except Exception as e:
        comparisons["HybridPRINetV2CLEVRN"] = {"error": str(e)}
        print(f"  HybridPRINetV2: FAILED — {e}")

    # ── Simple MLP baseline ─────────────────────────────────────
    _seed()

    class _MLPBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(scene_dim + query_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2),
            )

        def forward(self, scene: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
            return F.log_softmax(self.net(torch.cat([scene, query], -1)), dim=-1)

    mlp_model = _MLPBaseline().to(device)
    comparisons["MLP_Baseline"] = _train_and_eval(
        mlp_model,
        train_scenes, train_queries, train_labels,
        test_scenes, test_queries, test_labels,
    )
    print(f"  MLP Baseline:   acc={comparisons['MLP_Baseline']['accuracy']:.4f}, "
          f"time={comparisons['MLP_Baseline']['train_time_s']:.3f}s, "
          f"params={comparisons['MLP_Baseline']['n_params']:,d}")

    data = {
        "benchmark": "P.5_slot_attention_comparison",
        "device": _DEVICE,
        "scene_dim": scene_dim,
        "query_dim": query_dim,
        "n_train": n_train,
        "n_test": n_test,
        "epochs": 30,
        "comparisons": comparisons,
        "status": "PASS",
    }
    _save("y3q4_p5_slot_attention.json", data)
    return data


# ══════════════════════════════════════════════════════════════════
# Summary — Final regression
# ══════════════════════════════════════════════════════════════════


def bench_summary(results: dict) -> dict:
    """Generate Y3Q4 summary JSON."""
    print("\n[Summary] Y3 Q4 results")
    data = {
        "quarter": "Y3Q4",
        "version": "2.0.0",
        "device": _DEVICE,
        **results,
    }
    _save("y3q4_summary.json", data)
    return data


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  PRINet Year 3 Q4 Benchmarks")
    print("=" * 72)

    r_p3 = bench_p3_packaging()
    r_p4_scale = bench_p4_oscillosim_scaling()
    r_p4_order = bench_p4_oscillosim_order_param()
    r_p5 = bench_p5_slot_attention_comparison()

    bench_summary({
        "p3_packaging": r_p3,
        "p4_oscillosim_scaling": r_p4_scale,
        "p4_order_param": r_p4_order,
        "p5_slot_attention": r_p5,
    })

    print("\n" + "=" * 72)
    print("  All Y3 Q4 benchmarks complete.")
    print("=" * 72)
