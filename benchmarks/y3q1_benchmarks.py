"""Year 3 Q1 Benchmarks for PRINet.

Covers all Q1 deliverables:
- R.1: propagation package import-time baseline
- R.3: pseudo-phase vs. real-phase phase-to-rate latency
- M.1: CIFAR-10 conv-stem smoke run (untrained accuracy, architecture check)
- M.2: Fashion-MNIST conv-stem smoke run
- M.3: CLEVR-N extended 24-colour palette capacity
- M.4: Adversarial CLEVR similar-colour distractor setup
- M.5: torch.profiler training-loop analysis on HybridPRINetV2

Run with::

    python benchmarks/y3q1_benchmarks.py

Results are written to ``benchmarks/results/y3q1_*.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

# Ensure src/ is on path when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

RESULTS_DIR = _ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(filename: str, data: dict) -> None:
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# R.1 — propagation package import-time baseline
# ---------------------------------------------------------------------------


def bench_r1_import() -> dict:
    """Measure import time of the new propagation package vs legacy."""
    print("[R.1] propagation package import-time benchmark")

    # Force re-import by clearing cached modules
    import importlib
    mods_to_clear = [k for k in sys.modules if "prinet.core.propagation" in k]
    for mod in mods_to_clear:
        del sys.modules[mod]

    t0 = time.perf_counter()
    from prinet.core.propagation import (  # noqa: F401
        OscillatorState, OscillatorSyncError,
        KuramotoOscillator, StuartLandauOscillator, HopfOscillator,
        ExponentialIntegrator, MultiRateIntegrator,
        PhaseAmplitudeCoupling,
        ThetaGammaNetwork, DeltaThetaGammaNetwork, DiscreteDeltaThetaGamma,
        TemporalPhasePropagator,
        FeedforwardInhibition, FeedbackInhibition, DentateGyrusConverter,
        detect_oscillation, phase_to_rate, sweep_coupling_params,
    )
    import_ms = (time.perf_counter() - t0) * 1000

    # Minimal functional test
    t1 = time.perf_counter()
    net = DeltaThetaGammaNetwork(n_delta=8, n_theta=16, n_gamma=32)
    init = net.create_initial_state(seed=0)
    final, hist = net.integrate(init, n_steps=20, dt=0.01)
    r_d, r_t, r_g = net.order_parameters(final)
    functional_ms = (time.perf_counter() - t1) * 1000

    result = {
        "benchmark": "R.1_propagation_import",
        "import_ms": round(import_ms, 2),
        "functional_ms": round(functional_ms, 2),
        "r_delta": round(r_d.item(), 4),
        "r_theta": round(r_t.item(), 4),
        "r_gamma": round(r_g.item(), 4),
        "status": "PASS",
    }
    print(f"  import: {import_ms:.1f} ms  functional: {functional_ms:.1f} ms")
    return result


# ---------------------------------------------------------------------------
# R.3 — pseudo-phase vs. real-phase latency
# ---------------------------------------------------------------------------


def bench_r3_phase_fix() -> dict:
    """Compare old zero-phase with new real-phase variant."""
    print("[R.3] pseudo-phase vs. real-phase benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from prinet.nn.hybrid import HybridPRINet
    from prinet.nn.layers import HierarchicalResonanceLayer, PhaseToRateConverter

    B, D = 4, 32
    x = torch.randn(B, D, device=device)

    layer = HierarchicalResonanceLayer(
        n_delta=4, n_theta=8, n_gamma=16, n_dims=D, n_steps=3
    ).to(device)
    ptr = PhaseToRateConverter(n_oscillators=28, mode="soft").to(device)

    # Old: zeros
    N_REPS = 10
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        with torch.no_grad():
            h = layer(x)
            pseudo = torch.zeros_like(h)
            _ = ptr(pseudo, h)
    pseudo_ms = (time.perf_counter() - t0) / N_REPS * 1000

    # New: real phase
    t1 = time.perf_counter()
    for _ in range(N_REPS):
        with torch.no_grad():
            h, phases = layer(x, return_phase=True)
            _ = ptr(phases, h)
    real_ms = (time.perf_counter() - t1) / N_REPS * 1000

    overhead_pct = (real_ms - pseudo_ms) / max(pseudo_ms, 1e-6) * 100
    result = {
        "benchmark": "R.3_pseudo_vs_real_phase",
        "device": str(device),
        "pseudo_phase_ms": round(pseudo_ms, 3),
        "real_phase_ms": round(real_ms, 3),
        "overhead_pct": round(overhead_pct, 2),
        "status": "PASS",
    }
    print(f"  pseudo: {pseudo_ms:.2f} ms  real: {real_ms:.2f} ms  overhead: {overhead_pct:+.1f}%")
    return result


# ---------------------------------------------------------------------------
# M.1 — CIFAR-10 architecture smoke run
# ---------------------------------------------------------------------------


def bench_m1_cifar10() -> dict:
    """Verify HybridPRINetV2 + conv stem works on CIFAR-10-shaped inputs."""
    print("[M.1] CIFAR-10 HybridPRINetV2 conv-stem architecture check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from prinet.nn.hybrid import HybridPRINetV2

    model = HybridPRINetV2(
        n_input=64 * 16,  # stem output: 64 ch × (4×4) spatial
        n_classes=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_delta=4,
        n_theta=8,
        n_gamma=16,
        n_discrete_steps=3,
        use_conv_stem=True,
        stem_channels=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # Fake CIFAR-10 mini-batch: (B, 3, 32, 32)
    x = torch.randn(8, 3, 32, 32, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        log_probs = model(x)
    fwd_ms = (time.perf_counter() - t0) * 1000

    assert log_probs.shape == (8, 10), f"bad shape {log_probs.shape}"
    # Untrained model: check it produces valid log-probabilities
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones(8, device=device), atol=1e-3), \
        "log_softmax sum != 1"

    result = {
        "benchmark": "M.1_cifar10_arch_check",
        "device": str(device),
        "n_params": n_params,
        "forward_ms": round(fwd_ms, 2),
        "output_shape": list(log_probs.shape),
        "log_probs_mean": round(log_probs.mean().item(), 4),
        "status": "PASS",
    }
    print(f"  params: {n_params:,}  fwd: {fwd_ms:.1f} ms  shape: {log_probs.shape}")
    return result


# ---------------------------------------------------------------------------
# M.2 — Fashion-MNIST architecture smoke run
# ---------------------------------------------------------------------------


def bench_m2_fashion_mnist() -> dict:
    """Verify HybridPRINetV2 works on Fashion-MNIST-shaped (3×32×32) inputs."""
    print("[M.2] Fashion-MNIST HybridPRINetV2 architecture check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from prinet.nn.hybrid import HybridPRINetV2

    model = HybridPRINetV2(
        n_input=64 * 16,
        n_classes=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_delta=4,
        n_theta=8,
        n_gamma=16,
        n_discrete_steps=3,
        use_conv_stem=True,
        stem_channels=64,
    ).to(device)

    # Fashion-MNIST resized to 3×32×32 (as per datasets.py)
    x = torch.randn(8, 3, 32, 32, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        log_probs = model(x)
    fwd_ms = (time.perf_counter() - t0) * 1000

    assert log_probs.shape == (8, 10)

    result = {
        "benchmark": "M.2_fashion_mnist_arch_check",
        "device": str(device),
        "forward_ms": round(fwd_ms, 2),
        "output_shape": list(log_probs.shape),
        "status": "PASS",
    }
    print(f"  fwd: {fwd_ms:.1f} ms  shape: {log_probs.shape}")
    return result


# ---------------------------------------------------------------------------
# M.3 — CLEVR-N extended palette capacity curve
# ---------------------------------------------------------------------------


def bench_m3_clevr_extended() -> dict:
    """Benchmark extended 24-colour CLEVR-N dataset generation."""
    print("[M.3] CLEVR-N extended 24-colour palette")

    # Inline import to avoid cluttering top-level
    sys.path.insert(0, str(_ROOT))
    from benchmarks.clevr_n import (
        COLORS_24, D_COLOR_24, make_clevr_n_extended,
    )

    assert D_COLOR_24 == 24, f"expected 24 colours, got {D_COLOR_24}"
    assert len(COLORS_24) == 24

    results_per_n: list[dict] = []
    for n in [4, 8, 12, 16]:
        t0 = time.perf_counter()
        ds = make_clevr_n_extended(n_items=n, n_samples=200, seed=42)
        gen_ms = (time.perf_counter() - t0) * 1000
        results_per_n.append({"n_items": n, "n_samples": 200, "gen_ms": round(gen_ms, 1)})
        print(f"  N={n:2d}: generated 200 samples in {gen_ms:.0f} ms")

    result = {
        "benchmark": "M.3_clevr_n_extended_palette",
        "n_colors": 24,
        "per_n": results_per_n,
        "status": "PASS",
    }
    return result


# ---------------------------------------------------------------------------
# M.4 — Adversarial CLEVR
# ---------------------------------------------------------------------------


def bench_m4_adversarial_clevr() -> dict:
    """Benchmark adversarial CLEVR with similar-colour distractors."""
    print("[M.4] Adversarial CLEVR similar-colour distractors")

    sys.path.insert(0, str(_ROOT))
    from benchmarks.clevr_n import (
        build_adversarial_colour_pairs, make_adversarial_clevr,
    )

    pairs = build_adversarial_colour_pairs(max_delta_e=25.0)
    assert len(pairs) > 0, "no adversarial pairs found"
    top5 = [(a, b, round(de, 2)) for a, b, de in pairs[:5]]
    print(f"  adversarial pairs found: {len(pairs)}  top-5 ΔE: {top5}")

    t0 = time.perf_counter()
    ds = make_adversarial_clevr(n_items=4, n_samples=200, seed=0)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  generated 200 adversarial samples in {gen_ms:.0f} ms")

    result = {
        "benchmark": "M.4_adversarial_clevr",
        "n_adversarial_pairs": len(pairs),
        "top5_pairs": [{"a": a, "b": b, "delta_e": de} for a, b, de in top5],
        "n_samples": 200,
        "gen_ms": round(gen_ms, 1),
        "status": "PASS",
    }
    return result


# ---------------------------------------------------------------------------
# M.5 — torch.profiler training-loop analysis
# ---------------------------------------------------------------------------


def bench_m5_profiler() -> dict:
    """Run torch.profiler on HybridPRINetV2 training loop."""
    print("[M.5] torch.profiler training-loop analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from prinet.nn.hybrid import HybridPRINetV2
    from prinet.utils.profiler import PRINetProfiler

    model = HybridPRINetV2(
        n_input=64 * 16,
        n_classes=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_delta=4,
        n_theta=8,
        n_gamma=16,
        n_discrete_steps=3,
        use_conv_stem=True,
        stem_channels=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Fake CIFAR-10 data
    fakeX = torch.randn(32, 3, 32, 32, device=device)
    fakeY = torch.randint(0, 10, (32,), device=device)

    profiler = PRINetProfiler(
        out_dir=None,
        warmup_steps=2,
        active_steps=6,
    )

    with profiler:
        for step in range(8):  # 2 warmup + 6 active
            optimizer.zero_grad()
            log_probs = model(fakeX)
            loss = loss_fn(log_probs, fakeY)
            loss.backward()
            optimizer.step()
            profiler.step()

    report = profiler.report(top_n=10)

    print(f"  avg step: {report.avg_step_ms:.1f} ms  bottleneck: {report.bottleneck_op}")
    print(report.top_ops_table)

    result = {
        "benchmark": "M.5_profiler_cifar10",
        "device": str(device),
        "avg_step_ms": round(report.avg_step_ms, 2),
        "total_wall_ms": round(report.total_wall_ms, 2),
        "bottleneck_op": report.bottleneck_op,
        "top_ops": [
            {"op": op, "cpu_ms": round(c, 3), "cuda_ms": round(g, 3)}
            for op, c, g in report.top_ops[:10]
        ],
        "status": "PASS",
    }
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("PRINet Year 3 Q1 Benchmarks")
    print("=" * 60)

    all_results: dict[str, dict] = {}

    try:
        r = bench_r1_import()
        all_results["r1_import"] = r
        _save("y3q1_r1_propagation_import.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["r1_import"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_r3_phase_fix()
        all_results["r3_phase"] = r
        _save("y3q1_r3_phase_fix.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["r3_phase"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_m1_cifar10()
        all_results["m1_cifar10"] = r
        _save("y3q1_m1_cifar10.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["m1_cifar10"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_m2_fashion_mnist()
        all_results["m2_fmnist"] = r
        _save("y3q1_m2_fashion_mnist.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["m2_fmnist"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_m3_clevr_extended()
        all_results["m3_clevr_extended"] = r
        _save("y3q1_m3_clevr_extended.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["m3_clevr_extended"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_m4_adversarial_clevr()
        all_results["m4_adversarial"] = r
        _save("y3q1_m4_adversarial_clevr.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["m4_adversarial"] = {"status": "ERROR", "error": str(exc)}

    try:
        r = bench_m5_profiler()
        all_results["m5_profiler"] = r
        _save("y3q1_m5_profiler.json", r)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_results["m5_profiler"] = {"status": "ERROR", "error": str(exc)}

    # Summary
    _save("y3q1_summary.json", all_results)
    print("\n" + "=" * 60)
    passed = sum(1 for v in all_results.values() if v.get("status") == "PASS")
    total = len(all_results)
    print(f"Results: {passed}/{total} PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
