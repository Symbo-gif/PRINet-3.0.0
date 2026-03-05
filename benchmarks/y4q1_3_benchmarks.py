"""Year 4 Q1.3 Benchmarks — Chimera State Deepening.

Applies the literature-correct chimera formation protocol:

- **Cosine nonlocal coupling kernel** (Abrams & Strogatz 2004):
  G(i-j) = (1/2π)[1 + A·cos(2π(i-j)/N)], A=0.995
- **4th-order Runge-Kutta** integration (vs Euler, 4-8 orders of
  magnitude more accurate for chimera simulations)
- **High coupling strength** K=100 (literature range 50-150)
- **Optimal phase lag** α = π/2 − 0.05 ≈ 1.521 rad
- **Smooth Gaussian-bump IC** seeding chimera formation
- **New detection metrics**: Strength of Incoherence (SI), Discontinuity
  Measure (η), Chimera Index (χ), alongside existing BC and r_local

Benchmarks:
1. Gold-standard chimera detection with optimised params (K=100, RK4, cosine).
2. RK4 vs Euler accuracy comparison for chimera BC.
3. Cosine kernel vs uniform coupling comparison.
4. K–α sensitivity sweep around optimal parameters.
5. IC comparison (Gaussian bump vs half-sync vs original bump).
6. Multi-metric chimera characterisation (BC, SI, η, χ).
7. N-scaling chimera lifetime (N=128, 256, 512).

Generates JSON result files in ``benchmarks/results/``.

Usage:
    python benchmarks/y4q1_3_benchmarks.py
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import torch

# =========================================================================
# Setup
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Limit GPU threads to avoid OOM on 8GB cards
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)


def _save(name: str, data: dict) -> None:
    path = RESULTS_DIR / f"benchmark_y4q1_3_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


def _cleanup() -> None:
    """Free GPU/CPU memory between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =========================================================================
# Benchmark 1: Gold-standard Chimera Detection
# =========================================================================


def benchmark_gold_standard_chimera() -> None:
    """Chimera detection with literature-optimal parameters.

    Protocol:
    - N=256, k=90 (R/N ≈ 0.35), cosine kernel A=0.995
    - K=100, α=π/2−0.05≈1.521
    - RK4 integrator, dt=0.05, 10K steps (discarding first 5K as transient)
    - Gaussian-bump IC
    - 3 seeds for reproducibility
    - All metrics: BC, SI, η, χ
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        chimera_index,
        cosine_coupling_kernel,
        discontinuity_measure,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import (
        bootstrap_ci,
        gaussian_bump_ic,
    )

    print("=== Q1.3 Benchmark 1: Gold-Standard Chimera Detection ===")

    N = 256
    k = 90
    K = 100.0
    alpha = math.pi / 2 - 0.05
    A = 0.995
    dt = 0.05
    n_steps = 10_000
    transient_steps = 5_000
    n_seeds = 3

    weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
    nbr = ring_topology(N, k, device="cpu")

    results: dict = {
        "benchmark": "gold_standard_chimera",
        "params": {
            "N": N, "k": k, "K": K, "alpha": alpha, "A": A,
            "dt": dt, "n_steps": n_steps, "transient_steps": transient_steps,
            "integrator": "rk4", "n_seeds": n_seeds,
        },
        "seeds": [],
    }

    bc_values: list[float] = []
    si_values: list[float] = []
    chi_values: list[float] = []
    eta_values: list[int] = []

    for seed in range(n_seeds):
        ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_strength=K,
            coupling_mode="ring",
            k_neighbors=k,
            phase_lag=alpha,
            integrator="rk4",
            coupling_weights=weights,
            device=DEVICE,
            seed=seed,
        )

        t0 = time.perf_counter()
        result = sim.run(
            n_steps=n_steps,
            dt=dt,
            initial_phase=ic,
            record_trajectory=True,
        )
        wall_time = time.perf_counter() - t0

        # Compute metrics on post-transient final state
        phase = result.final_phase.cpu()
        r_local = local_order_parameter(phase, nbr)
        bc = bimodality_index(r_local)
        si = strength_of_incoherence(phase, window_size=10)
        mask, eta = discontinuity_measure(phase, threshold_ratio=0.01)
        chi = chimera_index(phase, nbr, threshold=0.5)

        seed_result = {
            "seed": seed,
            "bc": bc,
            "si": float(si.item()),
            "eta": eta,
            "chi": chi,
            "r_mean": float(r_local.mean().item()),
            "r_std": float(r_local.std().item()),
            "order_param_final": result.order_parameter[-1],
            "wall_time_s": wall_time,
            "throughput": result.throughput,
            "chimera_bc": bc > 0.555,
        }
        results["seeds"].append(seed_result)
        bc_values.append(bc)
        si_values.append(float(si.item()))
        chi_values.append(chi)
        eta_values.append(eta)

        tag = "CHIMERA" if bc > 0.555 else "no-chimera"
        print(
            f"  Seed {seed}: BC={bc:.4f} SI={si.item():.3f} "
            f"η={eta} χ={chi:.3f} [{tag}] ({wall_time:.1f}s)"
        )

    # Aggregate statistics
    bc_ci = bootstrap_ci(bc_values, n_bootstrap=5000, alpha=0.05, seed=42)
    results["aggregate"] = {
        "bc_mean": bc_ci["mean"],
        "bc_ci_lower": bc_ci["ci_lower"],
        "bc_ci_upper": bc_ci["ci_upper"],
        "bc_ci_width": bc_ci["ci_width"],
        "si_mean": sum(si_values) / len(si_values),
        "chi_mean": sum(chi_values) / len(chi_values),
        "eta_values": eta_values,
        "chimera_detected_any": any(bc > 0.555 for bc in bc_values),
        "chimera_detected_all": all(bc > 0.555 for bc in bc_values),
    }

    print(
        f"  Aggregate: BC={bc_ci['mean']:.4f} "
        f"[{bc_ci['ci_lower']:.4f}, {bc_ci['ci_upper']:.4f}]"
    )
    _save("gold_standard_chimera", results)


# =========================================================================
# Benchmark 2: RK4 vs Euler Chimera Accuracy
# =========================================================================


def benchmark_rk4_vs_euler() -> None:
    """Compare chimera detection accuracy between RK4 and Euler integrators.

    Uses identical parameters except integrator. Shows that RK4
    preserves chimera patterns that Euler integration may destroy.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        cosine_coupling_kernel,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic

    print("\n=== Q1.3 Benchmark 2: RK4 vs Euler Chimera Accuracy ===")

    N = 256
    k = 90
    K = 100.0
    alpha = math.pi / 2 - 0.05
    A = 0.995
    n_seeds = 3

    weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
    nbr = ring_topology(N, k, device="cpu")

    results: dict = {
        "benchmark": "rk4_vs_euler",
        "params": {"N": N, "k": k, "K": K, "alpha": alpha, "A": A},
        "comparisons": [],
    }

    for dt in [0.01, 0.05, 0.1]:
        n_steps = max(2000, int(500.0 / dt))  # ~500 time units
        for integrator in ["euler", "rk4"]:
            bcs: list[float] = []
            sis: list[float] = []
            for seed in range(n_seeds):
                ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
                sim = OscilloSim(
                    n_oscillators=N,
                    coupling_strength=K,
                    coupling_mode="ring",
                    k_neighbors=k,
                    phase_lag=alpha,
                    integrator=integrator,
                    coupling_weights=weights,
                    device=DEVICE,
                    seed=seed,
                )
                result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
                phase = result.final_phase.cpu()
                r_local = local_order_parameter(phase, nbr)
                bcs.append(bimodality_index(r_local))
                sis.append(float(strength_of_incoherence(phase).item()))

            mean_bc = sum(bcs) / len(bcs)
            mean_si = sum(sis) / len(sis)
            results["comparisons"].append({
                "integrator": integrator,
                "dt": dt,
                "n_steps": n_steps,
                "bc_mean": mean_bc,
                "bc_values": bcs,
                "si_mean": mean_si,
                "chimera_detected": mean_bc > 0.555,
            })
            tag = "C" if mean_bc > 0.555 else "."
            print(
                f"  {integrator:5s} dt={dt:.2f}: "
                f"BC={mean_bc:.4f} SI={mean_si:.3f} [{tag}]"
            )

    _save("rk4_vs_euler", results)


# =========================================================================
# Benchmark 3: Cosine Kernel vs Uniform Coupling
# =========================================================================


def benchmark_cosine_vs_uniform() -> None:
    """Compare cosine nonlocal kernel with uniform (standard) coupling.

    Demonstrates that the cosine kernel creates proper spatial structure
    enabling chimera formation, while uniform coupling does not.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        chimera_index,
        cosine_coupling_kernel,
        discontinuity_measure,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic

    print("\n=== Q1.3 Benchmark 3: Cosine Kernel vs Uniform Coupling ===")

    N = 256
    k = 90
    K = 100.0
    alpha = math.pi / 2 - 0.05
    dt = 0.05
    n_steps = 6000
    n_seeds = 3
    nbr = ring_topology(N, k, device="cpu")

    results: dict = {
        "benchmark": "cosine_vs_uniform",
        "params": {"N": N, "k": k, "K": K, "alpha": alpha, "dt": dt,
                    "n_steps": n_steps},
        "comparisons": [],
    }

    for label, A_val in [("uniform", None), ("cosine_0.9", 0.9),
                          ("cosine_0.995", 0.995)]:
        if A_val is not None:
            cw = cosine_coupling_kernel(N, k, A=A_val, device=DEVICE)
        else:
            cw = None

        bcs: list[float] = []
        sis: list[float] = []
        chis: list[float] = []
        etas: list[int] = []

        for seed in range(n_seeds):
            ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
            sim = OscilloSim(
                n_oscillators=N,
                coupling_strength=K,
                coupling_mode="ring",
                k_neighbors=k,
                phase_lag=alpha,
                integrator="rk4",
                coupling_weights=cw,
                device=DEVICE,
                seed=seed,
            )
            result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
            phase = result.final_phase.cpu()
            r_local = local_order_parameter(phase, nbr)
            bcs.append(bimodality_index(r_local))
            sis.append(float(strength_of_incoherence(phase).item()))
            _, eta = discontinuity_measure(phase)
            etas.append(eta)
            chis.append(chimera_index(phase, nbr))

        mean_bc = sum(bcs) / len(bcs)
        results["comparisons"].append({
            "kernel": label,
            "bc_mean": mean_bc,
            "bc_values": bcs,
            "si_mean": sum(sis) / len(sis),
            "chi_mean": sum(chis) / len(chis),
            "eta_values": etas,
            "chimera_detected": mean_bc > 0.555,
        })
        tag = "C" if mean_bc > 0.555 else "."
        print(f"  {label:14s}: BC={mean_bc:.4f} [{tag}]")

    _save("cosine_vs_uniform", results)


# =========================================================================
# Benchmark 4: K–α Sensitivity Sweep (Refined)
# =========================================================================


def benchmark_k_alpha_sensitivity() -> None:
    """Refined K–α sweep around the known chimera region.

    Uses cosine kernel + RK4 with higher K values than Q1.2
    to map the chimera boundary more precisely.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        cosine_coupling_kernel,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic

    print("\n=== Q1.3 Benchmark 4: K–α Sensitivity Sweep ===")

    N = 256
    k = 90
    A = 0.995
    dt = 0.05
    n_steps = 4000
    n_seeds = 2

    weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
    nbr = ring_topology(N, k, device="cpu")

    K_values = [20.0, 50.0, 75.0, 100.0, 120.0, 150.0]
    alpha_values = [1.3, 1.4, 1.47, 1.50, 1.521, 1.55, 1.57]

    results: dict = {
        "benchmark": "k_alpha_sensitivity",
        "params": {"N": N, "k": k, "A": A, "dt": dt, "n_steps": n_steps},
        "grid": [],
    }

    for K in K_values:
        for alpha in alpha_values:
            bcs: list[float] = []
            for seed in range(n_seeds):
                ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
                sim = OscilloSim(
                    n_oscillators=N,
                    coupling_strength=K,
                    coupling_mode="ring",
                    k_neighbors=k,
                    phase_lag=alpha,
                    integrator="rk4",
                    coupling_weights=weights,
                    device=DEVICE,
                    seed=seed,
                )
                result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
                phase = result.final_phase.cpu()
                r_local = local_order_parameter(phase, nbr)
                bcs.append(bimodality_index(r_local))

            mean_bc = sum(bcs) / len(bcs)
            results["grid"].append({
                "K": K,
                "alpha": alpha,
                "bc_mean": mean_bc,
                "bc_values": bcs,
                "chimera_detected": mean_bc > 0.555,
            })
            tag = "C" if mean_bc > 0.555 else "."
            print(f"  K={K:6.1f}, α={alpha:.3f}: BC={mean_bc:.4f} [{tag}]")

    _save("k_alpha_sensitivity", results)


# =========================================================================
# Benchmark 5: IC Comparison
# =========================================================================


def benchmark_ic_comparison() -> None:
    """Compare three initial condition strategies for chimera formation.

    1. Original bump (chimera_initial_condition): 6·exp(-30(x-0.5)²)·r
    2. Gaussian smooth bump (gaussian_bump_ic): φ₀ + A₀·exp(-(i-i₀)²/(2σ²)) + ε
    3. Half-sync/half-random (half_sync_half_random_ic)
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        cosine_coupling_kernel,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import (
        chimera_initial_condition,
        gaussian_bump_ic,
        half_sync_half_random_ic,
    )

    print("\n=== Q1.3 Benchmark 5: IC Comparison ===")

    N = 256
    k = 90
    K = 100.0
    alpha = math.pi / 2 - 0.05
    A = 0.995
    dt = 0.05
    n_steps = 6000
    n_seeds = 3

    weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
    nbr = ring_topology(N, k, device="cpu")

    ic_generators = {
        "original_bump": lambda s: chimera_initial_condition(N, seed=s),
        "gaussian_bump": lambda s: gaussian_bump_ic(N, seed=s),
        "half_sync_random": lambda s: half_sync_half_random_ic(N, seed=s),
    }

    results: dict = {
        "benchmark": "ic_comparison",
        "params": {"N": N, "k": k, "K": K, "alpha": alpha, "A": A,
                    "dt": dt, "n_steps": n_steps},
        "comparisons": [],
    }

    for ic_name, ic_fn in ic_generators.items():
        bcs: list[float] = []
        sis: list[float] = []
        for seed in range(n_seeds):
            ic = ic_fn(seed).to(DEVICE)
            sim = OscilloSim(
                n_oscillators=N,
                coupling_strength=K,
                coupling_mode="ring",
                k_neighbors=k,
                phase_lag=alpha,
                integrator="rk4",
                coupling_weights=weights,
                device=DEVICE,
                seed=seed,
            )
            result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
            phase = result.final_phase.cpu()
            r_local = local_order_parameter(phase, nbr)
            bcs.append(bimodality_index(r_local))
            sis.append(float(strength_of_incoherence(phase).item()))

        mean_bc = sum(bcs) / len(bcs)
        results["comparisons"].append({
            "ic_type": ic_name,
            "bc_mean": mean_bc,
            "bc_values": bcs,
            "si_mean": sum(sis) / len(sis),
            "chimera_detected": mean_bc > 0.555,
        })
        tag = "C" if mean_bc > 0.555 else "."
        print(f"  {ic_name:20s}: BC={mean_bc:.4f} [{tag}]")

    _save("ic_comparison", results)


# =========================================================================
# Benchmark 6: Multi-Metric Chimera Characterisation
# =========================================================================


def benchmark_multi_metric_chimera() -> None:
    """Full multi-metric chimera characterisation at optimal parameters.

    Records all 5 metrics (BC, SI, η, χ, r_local distribution) across
    seeds and reports comprehensive statistics with bootstrap CIs.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        chimera_index,
        cosine_coupling_kernel,
        discontinuity_measure,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import (
        bootstrap_ci,
        gaussian_bump_ic,
    )

    print("\n=== Q1.3 Benchmark 6: Multi-Metric Chimera Characterisation ===")

    N = 256
    k = 90
    K = 100.0
    alpha = math.pi / 2 - 0.05
    A = 0.995
    dt = 0.05
    n_steps = 8000
    n_seeds = 5

    weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
    nbr = ring_topology(N, k, device="cpu")

    results: dict = {
        "benchmark": "multi_metric_chimera",
        "params": {"N": N, "k": k, "K": K, "alpha": alpha, "A": A,
                    "dt": dt, "n_steps": n_steps, "n_seeds": n_seeds},
        "seeds": [],
    }

    bc_vals: list[float] = []
    si_vals: list[float] = []
    chi_vals: list[float] = []

    for seed in range(n_seeds):
        ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_strength=K,
            coupling_mode="ring",
            k_neighbors=k,
            phase_lag=alpha,
            integrator="rk4",
            coupling_weights=weights,
            device=DEVICE,
            seed=seed,
        )
        result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
        phase = result.final_phase.cpu()
        r_local = local_order_parameter(phase, nbr)

        bc = bimodality_index(r_local)
        si = float(strength_of_incoherence(phase, window_size=10).item())
        mask, eta = discontinuity_measure(phase, threshold_ratio=0.01)
        chi = chimera_index(phase, nbr, threshold=0.5)
        n_coherent = int(mask.sum().item())

        seed_data = {
            "seed": seed,
            "bc": bc,
            "si": si,
            "eta": eta,
            "chi": chi,
            "n_coherent": n_coherent,
            "n_incoherent": N - n_coherent,
            "r_local_mean": float(r_local.mean().item()),
            "r_local_std": float(r_local.std().item()),
            "r_local_min": float(r_local.min().item()),
            "r_local_max": float(r_local.max().item()),
            "order_param_final": result.order_parameter[-1],
        }
        results["seeds"].append(seed_data)
        bc_vals.append(bc)
        si_vals.append(si)
        chi_vals.append(chi)

        print(
            f"  Seed {seed}: BC={bc:.4f} SI={si:.3f} η={eta} "
            f"χ={chi:.3f} coherent={n_coherent}/{N}"
        )

    # Aggregate with bootstrap CIs
    bc_ci = bootstrap_ci(bc_vals, n_bootstrap=5000, seed=42)
    si_ci = bootstrap_ci(si_vals, n_bootstrap=5000, seed=42)
    chi_ci = bootstrap_ci(chi_vals, n_bootstrap=5000, seed=42)

    results["aggregate"] = {
        "bc": {"mean": bc_ci["mean"], "ci_lower": bc_ci["ci_lower"],
               "ci_upper": bc_ci["ci_upper"], "se": bc_ci["se"]},
        "si": {"mean": si_ci["mean"], "ci_lower": si_ci["ci_lower"],
               "ci_upper": si_ci["ci_upper"], "se": si_ci["se"]},
        "chi": {"mean": chi_ci["mean"], "ci_lower": chi_ci["ci_lower"],
               "ci_upper": chi_ci["ci_upper"], "se": chi_ci["se"]},
        "chimera_detected_bc": bc_ci["mean"] > 0.555,
        "chimera_count": sum(1 for bc in bc_vals if bc > 0.555),
    }

    print(
        f"\n  BC:  {bc_ci['mean']:.4f} [{bc_ci['ci_lower']:.4f}, "
        f"{bc_ci['ci_upper']:.4f}]"
    )
    print(
        f"  SI:  {si_ci['mean']:.3f} [{si_ci['ci_lower']:.3f}, "
        f"{si_ci['ci_upper']:.3f}]"
    )
    print(
        f"  χ:   {chi_ci['mean']:.3f} [{chi_ci['ci_lower']:.3f}, "
        f"{chi_ci['ci_upper']:.3f}]"
    )

    _save("multi_metric_chimera", results)


# =========================================================================
# Benchmark 7: N-Scaling Chimera Lifetime
# =========================================================================


def benchmark_n_scaling() -> None:
    """Chimera persistence as a function of system size N.

    Literature predicts chimera lifetime scales as exp(N²).
    We compare N=128, 256, 512 with fixed R/N ratio.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        cosine_coupling_kernel,
        local_order_parameter,
        ring_topology,
        strength_of_incoherence,
    )
    from prinet.utils.y4q1_tools import gaussian_bump_ic

    print("\n=== Q1.3 Benchmark 7: N-Scaling Chimera Lifetime ===")

    K = 100.0
    alpha = math.pi / 2 - 0.05
    A = 0.995
    dt = 0.05
    r_over_n = 0.35  # Fixed coupling range ratio
    n_seeds = 2

    # N=512 may be slow on CPU; adjust step count accordingly
    configs = [
        {"N": 128, "n_steps": 8000},
        {"N": 256, "n_steps": 6000},
        {"N": 512, "n_steps": 4000},
    ]

    results: dict = {
        "benchmark": "n_scaling",
        "params": {"K": K, "alpha": alpha, "A": A, "dt": dt,
                    "r_over_n": r_over_n},
        "configurations": [],
    }

    for cfg in configs:
        N = cfg["N"]
        n_steps = cfg["n_steps"]
        k = max(4, int(2 * r_over_n * N))  # R/N = k/(2N) → k = 2*R/N*N
        if k % 2 != 0:
            k -= 1  # Even k for symmetric ring

        weights = cosine_coupling_kernel(N, k, A=A, device=DEVICE)
        nbr = ring_topology(N, k, device="cpu")

        bcs: list[float] = []
        sis: list[float] = []

        for seed in range(n_seeds):
            ic = gaussian_bump_ic(N, seed=seed).to(DEVICE)
            sim = OscilloSim(
                n_oscillators=N,
                coupling_strength=K,
                coupling_mode="ring",
                k_neighbors=k,
                phase_lag=alpha,
                integrator="rk4",
                coupling_weights=weights,
                device=DEVICE,
                seed=seed,
            )

            t0 = time.perf_counter()
            result = sim.run(n_steps=n_steps, dt=dt, initial_phase=ic)
            wall = time.perf_counter() - t0

            phase = result.final_phase.cpu()
            r_local = local_order_parameter(phase, nbr)
            bcs.append(bimodality_index(r_local))
            sis.append(float(strength_of_incoherence(phase).item()))

        mean_bc = sum(bcs) / len(bcs)
        mean_si = sum(sis) / len(sis)
        results["configurations"].append({
            "N": N,
            "k": k,
            "n_steps": n_steps,
            "bc_mean": mean_bc,
            "bc_values": bcs,
            "si_mean": mean_si,
            "chimera_detected": mean_bc > 0.555,
            "wall_time_last_seed_s": wall,
        })
        tag = "C" if mean_bc > 0.555 else "."
        print(
            f"  N={N:4d}, k={k:3d}: BC={mean_bc:.4f} SI={mean_si:.3f} "
            f"[{tag}] ({wall:.1f}s)"
        )

    _save("n_scaling", results)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Q1.3 Chimera Deepening Benchmarks\n")

    t0 = time.perf_counter()

    benchmark_gold_standard_chimera()
    _cleanup()
    benchmark_rk4_vs_euler()
    _cleanup()
    benchmark_cosine_vs_uniform()
    _cleanup()
    benchmark_k_alpha_sensitivity()
    _cleanup()
    benchmark_ic_comparison()
    _cleanup()
    benchmark_multi_metric_chimera()
    _cleanup()
    benchmark_n_scaling()

    elapsed = time.perf_counter() - t0
    print(f"\nAll Q1.3 benchmarks completed in {elapsed:.1f}s")
    print(f"JSON files saved to: {RESULTS_DIR}")
