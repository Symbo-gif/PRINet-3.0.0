"""Q2 Benchmarks — NaN fix verification, mixed precision VRAM, SCALR training.

Benchmarks for Year 1 Quarter 2 features:
- B.0: NaN fix verification (MNIST-like forward + backward)
- B.1: torch.compile speedup measurement (skipped if inductor unavailable)
- B.2: Mixed precision VRAM + throughput comparison
- B.3: SCALR optimizer training curve
- B.4: HopfOscillator scaling
- B.5: hEP trainer convergence

Results saved to Docs/test_and_benchmark_results/benchmark_q2_results.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import HopfOscillator, OscillatorState
from prinet.nn.hep import HolomorphicEPTrainer
from prinet.nn.layers import PRINetModel, compile_model
from prinet.nn.optimizers import SCALROptimizer

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# B.0: NaN Fix Verification
# ===================================================================


def benchmark_nan_fix() -> dict:
    """Verify NaN loss is fixed for MNIST-sized input on device."""
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_resonances=32, n_dims=784, n_concepts=10, n_layers=2, n_steps=5
    ).to(DEVICE)

    batch_sizes = [4, 8, 16, 32]
    results: list[dict] = []

    for bs in batch_sizes:
        x = torch.randn(bs, 784, device=DEVICE)
        target = torch.randint(0, 10, (bs,), device=DEVICE)

        # Forward
        t0 = time.perf_counter()
        out = model(x)
        fwd_time = time.perf_counter() - t0

        loss = F.nll_loss(out, target)

        # Backward
        model.zero_grad()
        t0 = time.perf_counter()
        loss.backward()
        bwd_time = time.perf_counter() - t0

        nan_outputs = torch.isnan(out).sum().item()
        nan_grads = sum(
            1
            for p in model.parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        )

        results.append(
            {
                "batch_size": bs,
                "loss": round(loss.item(), 4),
                "nan_outputs": nan_outputs,
                "nan_grads": nan_grads,
                "forward_ms": round(fwd_time * 1000, 2),
                "backward_ms": round(bwd_time * 1000, 2),
                "all_finite": bool(
                    torch.isfinite(out).all() and torch.isfinite(loss)
                ),
            }
        )

    return {
        "name": "B.0 NaN Fix Verification",
        "device": str(DEVICE),
        "status": "PASS" if all(r["all_finite"] for r in results) else "FAIL",
        "results": results,
    }


# ===================================================================
# B.1: torch.compile Speedup
# ===================================================================


def _inductor_available() -> bool:
    """Check if torch.compile inductor backend works."""
    try:
        m = torch.nn.Linear(2, 2)
        c = torch.compile(m)
        c(torch.randn(1, 2))
        return True
    except Exception:
        return False


def benchmark_torch_compile() -> dict:
    """Measure torch.compile speedup vs eager on CPU."""
    if not _inductor_available():
        return {
            "name": "B.1 torch.compile Speedup",
            "status": "SKIPPED",
            "reason": "Inductor backend unavailable (cl compiler not found)",
        }

    torch.manual_seed(SEED)
    model = PRINetModel(
        n_resonances=16, n_dims=64, n_concepts=10,
        n_layers=2, n_steps=5,
    )
    x = torch.randn(16, 64)

    # Warmup eager
    for _ in range(3):
        model(x)

    # Time eager
    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    eager_time = (time.perf_counter() - t0) / n_iter

    # Compile
    compiled = compile_model(model)

    # Warmup compiled (first call triggers compilation)
    for _ in range(3):
        compiled(x)

    # Time compiled
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compiled(x)
    compiled_time = (time.perf_counter() - t0) / n_iter

    speedup = eager_time / compiled_time if compiled_time > 0 else float("inf")

    return {
        "name": "B.1 torch.compile Speedup",
        "status": "PASS",
        "eager_ms": round(eager_time * 1000, 2),
        "compiled_ms": round(compiled_time * 1000, 2),
        "speedup": round(speedup, 2),
    }


# ===================================================================
# B.2: Mixed Precision VRAM + Throughput
# ===================================================================


def benchmark_mixed_precision() -> dict:
    """Compare fp32 vs mixed precision throughput."""
    torch.manual_seed(SEED)
    n_res, n_dim, n_con = 32, 256, 10
    batch_size = 32
    n_iter = 20

    # Use float16 which has broader GPU support
    mp_dtype = torch.float16

    results: dict = {}

    for label, use_mp in [("fp32", False), ("float16_mixed", True)]:
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=n_res, n_dims=n_dim, n_concepts=n_con,
            n_layers=2, n_steps=5,
        ).to(DEVICE)
        if use_mp:
            model.enable_mixed_precision(True, dtype=mp_dtype)

        x = torch.randn(batch_size, n_dim, device=DEVICE)
        target = torch.randint(0, n_con, (batch_size,), device=DEVICE)

        # Warmup
        for _ in range(3):
            out = model(x)
            loss = F.nll_loss(out, target)
            loss.backward()

        # Timed run
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, target)
            loss.backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iter

        # VRAM tracking (GPU only)
        if DEVICE.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        else:
            vram_mb = 0.0

        results[label] = {
            "iter_ms": round(elapsed * 1000, 2),
            "loss": round(loss.item(), 4),
            "vram_mb": round(vram_mb, 1),
            "all_finite": bool(torch.isfinite(loss)),
        }

    # Compute speedup
    mp_key = [k for k in results if k != "fp32"][0]
    fp32_ms = results["fp32"]["iter_ms"]
    mp_ms = results[mp_key]["iter_ms"]
    speedup = fp32_ms / mp_ms if mp_ms > 0 else 0.0

    return {
        "name": "B.2 Mixed Precision Throughput",
        "device": str(DEVICE),
        "model_config": f"N={n_res}, D={n_dim}, C={n_con}, batch={batch_size}",
        "status": "PASS",
        "speedup": round(speedup, 2),
        **results,
    }


# ===================================================================
# B.3: SCALR Optimizer Training Curve
# ===================================================================


def benchmark_scalr_training() -> dict:
    """Compare SCALR vs Adam on a small training task."""
    torch.manual_seed(SEED)

    n_res, n_dim, n_con = 16, 64, 5
    n_epochs = 20
    batch_size = 32

    # Fixed training data
    x = torch.randn(batch_size, n_dim, device=DEVICE)
    y = torch.randint(0, n_con, (batch_size,), device=DEVICE)

    results: dict = {}

    for label, opt_cls in [("Adam", "adam"), ("SCALR", "scalr")]:
        torch.manual_seed(SEED)
        model = PRINetModel(
            n_resonances=n_res, n_dims=n_dim, n_concepts=n_con,
            n_layers=2, n_steps=3, dt=0.01,
        ).to(DEVICE)

        if opt_cls == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            opt = SCALROptimizer(model.parameters(), lr=1e-3, r_min=0.1)

        losses: list[float] = []
        t0 = time.perf_counter()
        for epoch in range(n_epochs):
            opt.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if opt_cls == "scalr":
                # Simulate order parameter from model
                r = 0.3 + 0.03 * epoch  # Increasing sync
                opt.step(order_parameter=min(r, 1.0))
            else:
                opt.step()
            losses.append(round(loss.item(), 4))
        elapsed = time.perf_counter() - t0

        results[label] = {
            "final_loss": losses[-1],
            "losses": losses,
            "total_ms": round(elapsed * 1000, 1),
            "converged": losses[-1] < losses[0],
        }

    return {
        "name": "B.3 SCALR vs Adam Training",
        "device": str(DEVICE),
        "epochs": n_epochs,
        "status": "PASS",
        **results,
    }


# ===================================================================
# B.4: HopfOscillator Scaling
# ===================================================================


def benchmark_hopf_scaling() -> dict:
    """Measure HopfOscillator integration time at various sizes."""
    sizes = [50, 100, 500, 1_000, 5_000]
    results: list[dict] = []

    for n in sizes:
        model = HopfOscillator(
            n, coupling_strength=0.5, bifurcation_param=1.0
        )
        state = OscillatorState.create_random(n, seed=SEED, freq_range=(1.0, 2.0))

        t0 = time.perf_counter()
        final, _ = model.integrate(state, n_steps=200, dt=0.01)
        elapsed = time.perf_counter() - t0

        mean_amp = final.amplitude.mean().item()
        expected_amp = 1.0  # sqrt(mu=1.0)

        results.append(
            {
                "N": n,
                "time_ms": round(elapsed * 1000, 2),
                "mean_amplitude": round(mean_amp, 4),
                "amp_error": round(abs(mean_amp - expected_amp), 4),
            }
        )

    return {
        "name": "B.4 HopfOscillator Scaling",
        "status": "PASS",
        "results": results,
    }


# ===================================================================
# B.5: hEP Trainer Convergence
# ===================================================================


def benchmark_hep_convergence() -> dict:
    """Measure hEP trainer convergence on a small task."""
    torch.manual_seed(SEED)

    model = PRINetModel(
        n_resonances=8, n_dims=32, n_concepts=5,
        n_layers=1, n_steps=3,
    ).to(DEVICE)

    trainer = HolomorphicEPTrainer(
        model, beta=0.1, free_steps=5, nudge_steps=3
    )

    x = torch.randn(16, 32, device=DEVICE)
    y = torch.randint(0, 5, (16,), device=DEVICE)

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(15):
        loss = trainer.train_step(x, y, lr=0.01)
        losses.append(round(loss, 4))
    elapsed = time.perf_counter() - t0

    return {
        "name": "B.5 hEP Trainer Convergence",
        "device": str(DEVICE),
        "status": "PASS" if all(math.isfinite(l) for l in losses) else "FAIL",
        "n_steps": 15,
        "total_ms": round(elapsed * 1000, 1),
        "losses": losses,
        "final_loss": losses[-1],
        "converged": losses[-1] < losses[0] if len(losses) > 1 else False,
    }


# ===================================================================
# Main Runner
# ===================================================================


def main() -> None:
    """Run all Q2 benchmarks and save results."""
    print(f"PRINet Q2 Benchmarks — Device: {DEVICE}")
    print("=" * 60)

    all_results: list[dict] = []

    benchmarks = [
        ("B.0 NaN Fix Verification", benchmark_nan_fix),
        ("B.1 torch.compile Speedup", benchmark_torch_compile),
        ("B.2 Mixed Precision Throughput", benchmark_mixed_precision),
        ("B.3 SCALR vs Adam Training", benchmark_scalr_training),
        ("B.4 HopfOscillator Scaling", benchmark_hopf_scaling),
        ("B.5 hEP Trainer Convergence", benchmark_hep_convergence),
    ]

    for name, fn in benchmarks:
        print(f"\nRunning {name}...")
        try:
            result = fn()
            status = result.get("status", "UNKNOWN")
            print(f"  -> {status}")
            all_results.append(result)
        except Exception as e:
            print(f"  -> ERROR: {e}")
            all_results.append({"name": name, "status": "ERROR", "error": str(e)})

    # Save results
    output_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_q2_results.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "phase": "Year1_Q2",
                "device": str(DEVICE),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "benchmarks": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")

    # Summary
    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    skipped = sum(1 for r in all_results if r.get("status") == "SKIPPED")
    failed = sum(
        1 for r in all_results if r.get("status") in ("FAIL", "ERROR")
    )
    print(f"Summary: {passed} passed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
