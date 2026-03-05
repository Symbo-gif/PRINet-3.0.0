"""GPU Benchmark — O(N²) Full Pairwise Coupling Scaling.

Measures wall-clock time and VRAM for the full pairwise coupling matrix
path at increasing oscillator counts on GPU.  Compares against the O(N)
mean-field approximation to quantify the quadratic scaling factor.

The key cost center is ``KuramotoOscillator._compute_derivatives_full``,
which builds an (N, N) phase-difference matrix and performs element-wise
coupling at every integration step — giving O(N²) time **and** memory
per derivative evaluation.

Saves results to:
    Docs/test_and_benchmark_results/benchmark_pairwise_coupling_gpu.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState

# ───────────────────────── configuration ──────────────────────────

SEED = 42
DEVICE = torch.device("cuda")
N_STEPS = 200          # integration steps per trial
DT = 0.01
COUPLING_K = 2.0

# Sizes deliberately chosen to expose the N² wall.
# mean_field=False for ALL of them (that's the whole point).
SIZES_FULL = [64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192]

# For the comparison run, same sizes with mean_field=True (O(N)).
SIZES_MF = SIZES_FULL

WARMUP_STEPS = 5       # GPU JIT warm-up (excluded from timing)


# ───────────────────────── helpers ────────────────────────────────

def _flush_gpu() -> None:
    """Release cached CUDA memory so the next trial starts clean."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _run_trial(
    n: int,
    mean_field: bool,
    n_steps: int = N_STEPS,
    dt: float = DT,
) -> dict:
    """Run a single integration trial and return timing / memory stats."""
    _flush_gpu()
    torch.manual_seed(SEED)

    model = KuramotoOscillator(
        n_oscillators=n,
        coupling_strength=COUPLING_K / n,
        decay_rate=0.1,
        mean_field=mean_field,
        device=DEVICE,
    )
    state = OscillatorState.create_random(n, device=DEVICE, seed=SEED)

    # Warm-up (exclude from timing)
    for _ in range(WARMUP_STEPS):
        state = model.step(state, dt=dt)
    torch.cuda.synchronize()

    # Reset peak after warm-up so we measure the actual integration
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Full integration
    final_state, _ = model.integrate(state, n_steps=n_steps, dt=dt, method="rk4")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    r = kuramoto_order_parameter(final_state.phase).item()
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "n_oscillators": n,
        "mean_field": mean_field,
        "n_steps": n_steps,
        "dt": dt,
        "wall_seconds": round(elapsed, 5),
        "final_order_param": round(r, 6),
        "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 2),
    }


# ───────────────────────── main ───────────────────────────────────

def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print("=" * 72)
    print("  PRINet — O(N²) Full Pairwise Coupling GPU Benchmark")
    print(f"  GPU: {gpu_name}  |  VRAM: {vram_total_mb:.0f} MB")
    print(f"  CUDA: {cuda_ver}  |  PyTorch: {torch.__version__}")
    print(f"  Steps: {N_STEPS}  |  dt: {DT}  |  K: {COUPLING_K}")
    print("=" * 72)

    # ---- Full pairwise O(N²) runs ----
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Full Pairwise O(N²) — mean_field=False                    ║")
    print("╠═══════════╦══════════╦════════════╦═════════╦══════════════╣")
    print("║      N    ║  Time(s) ║  VRAM(MB)  ║   r     ║  time/N²(ns)║")
    print("╠═══════════╬══════════╬════════════╬═════════╬══════════════╣")

    full_results: list[dict] = []
    for n in SIZES_FULL:
        try:
            rec = _run_trial(n, mean_field=False)
            # Compute normalised time per N² to verify quadratic scaling
            t_per_n2_ns = (rec["wall_seconds"] / (n * n)) * 1e9
            rec["time_per_n2_ns"] = round(t_per_n2_ns, 4)
            full_results.append(rec)
            print(
                f"║  {n:>7d}  ║ {rec['wall_seconds']:>8.4f} ║ "
                f"{rec['peak_vram_allocated_mb']:>9.1f}  ║ "
                f"{rec['final_order_param']:>7.4f} ║ "
                f"{t_per_n2_ns:>11.4f}  ║"
            )
        except torch.cuda.OutOfMemoryError:
            oom_rec = {
                "n_oscillators": n,
                "mean_field": False,
                "oom": True,
                "note": "CUDA OOM",
            }
            full_results.append(oom_rec)
            print(f"║  {n:>7d}  ║   OOM    ║    OOM     ║   OOM   ║     OOM      ║")
            break  # no point trying larger sizes
    print("╚═══════════╩══════════╩════════════╩═════════╩══════════════╝")

    # ---- Mean-field O(N) comparison runs ----
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Mean-Field O(N) — mean_field=True                         ║")
    print("╠═══════════╦══════════╦════════════╦═════════╦══════════════╣")
    print("║      N    ║  Time(s) ║  VRAM(MB)  ║   r     ║  time/N(ns) ║")
    print("╠═══════════╬══════════╬════════════╬═════════╬══════════════╣")

    mf_results: list[dict] = []
    for n in SIZES_MF:
        rec = _run_trial(n, mean_field=True)
        t_per_n_ns = (rec["wall_seconds"] / n) * 1e9
        rec["time_per_n_ns"] = round(t_per_n_ns, 4)
        mf_results.append(rec)
        print(
            f"║  {n:>7d}  ║ {rec['wall_seconds']:>8.4f} ║ "
            f"{rec['peak_vram_allocated_mb']:>9.1f}  ║ "
            f"{rec['final_order_param']:>7.4f} ║ "
            f"{t_per_n_ns:>11.4f}  ║"
        )
    print("╚═══════════╩══════════╩════════════╩═════════╩══════════════╝")

    # ---- Speedup table ----
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Speedup: mean-field vs full pairwise                  ║")
    print("╠═══════════╦════════════╦════════════╦══════════════════╣")
    print("║      N    ║  Full (s)  ║  MF (s)    ║  Speedup (×)    ║")
    print("╠═══════════╬════════════╬════════════╬══════════════════╣")

    for fr, mr in zip(full_results, mf_results):
        if fr.get("oom"):
            print(f"║  {fr['n_oscillators']:>7d}  ║    OOM     ║ {mr['wall_seconds']:>9.4f}  ║       —          ║")
        else:
            speedup = fr["wall_seconds"] / max(mr["wall_seconds"], 1e-9)
            print(
                f"║  {fr['n_oscillators']:>7d}  ║ {fr['wall_seconds']:>9.4f}  ║ "
                f"{mr['wall_seconds']:>9.4f}  ║  {speedup:>13.1f}×  ║"
            )
    print("╚═══════════╩════════════╩════════════╩══════════════════╝")

    # ---- Scaling verification ----
    # If time scales as O(N²), then doubling N should ~4× the time.
    # We compute the empirical exponent from consecutive size doublings.
    exponents: list[dict] = []
    for i in range(1, len(full_results)):
        prev, curr = full_results[i - 1], full_results[i]
        if prev.get("oom") or curr.get("oom"):
            break
        n_ratio = curr["n_oscillators"] / prev["n_oscillators"]
        t_ratio = curr["wall_seconds"] / max(prev["wall_seconds"], 1e-9)
        import math
        exponent = math.log(t_ratio) / math.log(n_ratio)
        exponents.append({
            "n_from": prev["n_oscillators"],
            "n_to": curr["n_oscillators"],
            "n_ratio": round(n_ratio, 2),
            "time_ratio": round(t_ratio, 4),
            "empirical_exponent": round(exponent, 3),
        })

    if exponents:
        avg_exp = sum(e["empirical_exponent"] for e in exponents) / len(exponents)
        print(f"\n  Empirical scaling exponent (avg): {avg_exp:.3f}")
        print(f"  Expected for O(N²): 2.000")
        for e in exponents:
            print(
                f"    N {e['n_from']:>5d} → {e['n_to']:>5d}:  "
                f"time ×{e['time_ratio']:<7.3f}  exponent = {e['empirical_exponent']:.3f}"
            )

    # ---- Save results ----
    out_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "pairwise_coupling_scaling_gpu",
        "description": "O(N²) full pairwise coupling vs O(N) mean-field on GPU",
        "environment": {
            "gpu": gpu_name,
            "vram_total_mb": round(vram_total_mb, 0),
            "cuda_version": cuda_ver,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
        "config": {
            "n_steps": N_STEPS,
            "dt": DT,
            "coupling_K": COUPLING_K,
            "warmup_steps": WARMUP_STEPS,
            "seed": SEED,
        },
        "full_pairwise_O_N2": full_results,
        "mean_field_O_N": mf_results,
        "scaling_analysis": {
            "empirical_exponents": exponents,
            "average_exponent": round(avg_exp, 3) if exponents else None,
            "expected_exponent": 2.0,
        },
    }

    out_file = out_dir / "benchmark_pairwise_coupling_gpu.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Results saved to {out_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
