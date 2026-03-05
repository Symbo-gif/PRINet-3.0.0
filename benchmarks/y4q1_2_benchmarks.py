"""Year 4 Q1.2 Deepened Benchmarks — Rigorous Scientific Analysis.

Extends the 8 Q1 benchmarks with:
- Multi-seed reproducibility (3-5 seeds per configuration)
- Chimera-promoting initial conditions (Abrams & Strogatz 2006)
- K × α 2D parameter sweeps for chimera boundary mapping
- Bootstrap 95% confidence intervals on all key metrics
- Cohen's d effect sizes for variant comparisons
- Welch's t-tests with Bonferroni correction
- Spatial autocorrelation of local order parameter
- Seed stability analysis (coefficient of variation)
- Extended simulation parameters (N≥1024, 2000+ steps)
- Occlusion and noise-sensitivity analysis for MOT
- Learning curve tracking for ablation training
- Batch size scaling for FLOPs/wall-time

Generates 10+ JSON result files in ``benchmarks/results/``.

Usage:
    python benchmarks/y4q1_2_benchmarks.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import torch

# =========================================================================
# Setup
# =========================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _save(name: str, data: dict) -> None:
    path = RESULTS_DIR / f"benchmark_y4q1_2_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


# =========================================================================
# Benchmark 1: Extended Ring Chimera with Proper ICs (T.1 + T.2)
# =========================================================================

def benchmark_ring_chimera_extended() -> None:
    """Chimera emergence with Abrams & Strogatz single-hump IC.

    Key improvements over Q1:
    - Uses ``chimera_initial_condition()`` for proper IC
    - N up to 1024
    - 2000 steps (vs 500)
    - 3 seeds per configuration
    - Bootstrap CIs on bimodality coefficient
    - Spatial autocorrelation measurement
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
    )
    from prinet.utils.y4q1_tools import (
        bootstrap_ci,
        chimera_initial_condition,
        seed_stability_analysis,
        spatial_correlation,
    )

    print("=== Q1.2 Benchmark 1: Extended Ring Chimera (proper IC) ===")
    results: dict = {"benchmark": "ring_chimera_extended", "configs": []}
    N_SEEDS = 3
    N_STEPS = 2000

    for N in [256, 512, 1024]:
        k = min(20, N // 4)
        for alpha in [1.0, 1.2, 1.4, 1.457, 1.5]:
            seed_bcs: list[float] = []
            seed_r_means: list[float] = []
            seed_r_stds: list[float] = []
            seed_entries: list[dict] = []

            for seed in range(N_SEEDS):
                ic = chimera_initial_condition(N, seed=seed)
                sim = OscilloSim(
                    n_oscillators=N,
                    coupling_mode="ring",
                    k_neighbors=k,
                    coupling_strength=4.0,
                    phase_lag=alpha,
                    device=DEVICE,
                )
                result = sim.run(
                    n_steps=N_STEPS, dt=0.02,
                    initial_phase=ic,
                    record_trajectory=True,
                    record_interval=200,
                )

                nbr_idx = ring_topology(N, k, device="cpu")
                r_local = local_order_parameter(result.final_phase, nbr_idx)
                bc = bimodality_index(r_local)
                sp_corr = spatial_correlation(r_local, max_lag=min(50, N // 4))

                seed_bcs.append(bc)
                seed_r_means.append(float(r_local.mean()))
                seed_r_stds.append(float(r_local.std()))
                seed_entries.append({
                    "seed": seed,
                    "bimodality_coefficient": bc,
                    "chimera_detected": bc > 0.555,
                    "r_local_mean": float(r_local.mean()),
                    "r_local_std": float(r_local.std()),
                    "final_order_param": result.order_parameter[-1],
                    "spatial_corr_lag1": sp_corr[1] if len(sp_corr) > 1 else None,
                    "spatial_corr_lag10": sp_corr[10] if len(sp_corr) > 10 else None,
                })

            bc_ci = bootstrap_ci(seed_bcs)
            entry = {
                "N": N,
                "phase_lag": alpha,
                "k_neighbors": k,
                "n_steps": N_STEPS,
                "n_seeds": N_SEEDS,
                "bc_mean": bc_ci["mean"],
                "bc_ci_lower": bc_ci["ci_lower"],
                "bc_ci_upper": bc_ci["ci_upper"],
                "bc_ci_width": bc_ci["ci_width"],
                "r_local_mean_across_seeds": sum(seed_r_means) / len(seed_r_means),
                "r_local_std_across_seeds": sum(seed_r_stds) / len(seed_r_stds),
                "chimera_detected_any": any(bc > 0.555 for bc in seed_bcs),
                "chimera_detected_all": all(bc > 0.555 for bc in seed_bcs),
                "per_seed": seed_entries,
            }
            results["configs"].append(entry)
            chimera_str = "CHIMERA" if entry["chimera_detected_any"] else "no chimera"
            print(
                f"  N={N}, α={alpha:.3f}: BC={bc_ci['mean']:.3f} "
                f"[{bc_ci['ci_lower']:.3f}, {bc_ci['ci_upper']:.3f}] [{chimera_str}]"
            )

    _save("ring_chimera_extended", results)


# =========================================================================
# Benchmark 2: K × α 2D Chimera Boundary Sweep (T.2)
# =========================================================================

def benchmark_k_alpha_sweep() -> None:
    """2D parameter sweep of coupling strength K and phase lag α.

    Maps the chimera boundary in (K, α) space with 2 seeds per
    point for reproducibility.
    """
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
    )
    from prinet.utils.y4q1_tools import chimera_initial_condition

    print("\n=== Q1.2 Benchmark 2: K × α Chimera Boundary Sweep ===")
    N = 256
    k = 16
    results: dict = {
        "benchmark": "k_alpha_chimera_sweep",
        "N": N,
        "k": k,
        "grid": [],
    }

    K_values = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    alpha_values = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.457, 1.5, 1.57]
    N_SEEDS = 2

    for K in K_values:
        for alpha in alpha_values:
            bcs = []
            for seed in range(N_SEEDS):
                ic = chimera_initial_condition(N, seed=seed)
                sim = OscilloSim(
                    n_oscillators=N,
                    coupling_mode="ring",
                    k_neighbors=k,
                    coupling_strength=K,
                    phase_lag=alpha,
                    device=DEVICE,
                )
                result = sim.run(
                    n_steps=1500, dt=0.02,
                    initial_phase=ic,
                )
                nbr_idx = ring_topology(N, k, device="cpu")
                r_local = local_order_parameter(result.final_phase, nbr_idx)
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
            print(f"  K={K:.1f}, α={alpha:.3f}: BC={mean_bc:.3f} [{tag}]")

    _save("k_alpha_chimera_sweep", results)


# =========================================================================
# Benchmark 3: Small-World Extended with Multi-Seed (T.1 + T.2)
# =========================================================================

def benchmark_small_world_extended() -> None:
    """Extended small-world comparison with 3 seeds and CIs."""
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
        small_world_topology,
    )
    from prinet.utils.y4q1_tools import (
        bootstrap_ci,
        chimera_initial_condition,
    )

    print("\n=== Q1.2 Benchmark 3: Small-World Extended ===")
    N, k, alpha = 256, 16, 1.457
    N_SEEDS = 3
    results: dict = {
        "benchmark": "small_world_extended",
        "N": N,
        "k": k,
        "alpha": alpha,
        "comparisons": [],
    }

    for p_rewire in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        mode = "ring" if p_rewire == 0.0 else "small_world"
        bcs = []
        r_means = []

        for seed in range(N_SEEDS):
            ic = chimera_initial_condition(N, seed=seed)
            sim = OscilloSim(
                n_oscillators=N,
                coupling_mode=mode,
                k_neighbors=k,
                coupling_strength=4.0,
                phase_lag=alpha,
                p_rewire=p_rewire,
                device=DEVICE,
            )
            result = sim.run(n_steps=1500, dt=0.02, initial_phase=ic)

            if mode == "ring":
                nbr_idx = ring_topology(N, k, device="cpu")
            else:
                nbr_idx = small_world_topology(
                    N, k, p_rewire=p_rewire, device="cpu", seed=seed,
                )
            r_local = local_order_parameter(result.final_phase, nbr_idx)
            bcs.append(bimodality_index(r_local))
            r_means.append(float(r_local.mean()))

        bc_ci = bootstrap_ci(bcs)
        results["comparisons"].append({
            "p_rewire": p_rewire,
            "topology": mode,
            "n_seeds": N_SEEDS,
            "bc_mean": bc_ci["mean"],
            "bc_ci_lower": bc_ci["ci_lower"],
            "bc_ci_upper": bc_ci["ci_upper"],
            "r_local_mean": sum(r_means) / len(r_means),
            "chimera_detected": bc_ci["mean"] > 0.555,
        })
        print(
            f"  p={p_rewire:.2f}: BC={bc_ci['mean']:.3f} "
            f"[{bc_ci['ci_lower']:.3f}, {bc_ci['ci_upper']:.3f}]"
        )

    _save("small_world_extended", results)


# =========================================================================
# Benchmark 4: Extended Temporal MOT with Occlusion (T.3)
# =========================================================================

def benchmark_temporal_mot_extended() -> None:
    """MOT with occlusion simulation, noise sensitivity, scaling."""
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT

    print("\n=== Q1.2 Benchmark 4: Temporal MOT Extended ===")
    results: dict = {"benchmark": "temporal_mot_extended", "experiments": []}

    # --- Experiment A: sequence length scaling ---
    print("  [A] Sequence length scaling")
    for n_frames in [5, 10, 20, 40, 60]:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=6, slot_dim=32,
            num_iterations=3, match_threshold=0.3,
        )
        model.eval()
        torch.manual_seed(42)
        base_positions = torch.randn(6, 4)
        frames = []
        for t in range(n_frames):
            noise = torch.randn(6, 4) * 0.1
            dets = base_positions + noise + t * 0.05
            frames.append(dets)

        with torch.no_grad():
            tr = model.track_sequence(frames)
        results["experiments"].append({
            "experiment": "sequence_length_scaling",
            "n_frames": n_frames,
            "n_slots": 6,
            "identity_preservation": tr["identity_preservation"],
            "n_identity_matches": len(tr["identity_matches"]),
        })
        print(f"    frames={n_frames}: IP={tr['identity_preservation']:.3f}")

    # --- Experiment B: noise sensitivity ---
    print("  [B] Noise sensitivity")
    for noise_scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=6, slot_dim=32,
            num_iterations=3, match_threshold=0.3,
        )
        model.eval()
        torch.manual_seed(42)
        base_positions = torch.randn(6, 4)
        frames = []
        for t in range(20):
            noise = torch.randn(6, 4) * noise_scale
            dets = base_positions + noise + t * 0.05
            frames.append(dets)

        with torch.no_grad():
            tr = model.track_sequence(frames)
        results["experiments"].append({
            "experiment": "noise_sensitivity",
            "noise_scale": noise_scale,
            "n_frames": 20,
            "identity_preservation": tr["identity_preservation"],
        })
        print(f"    noise={noise_scale:.2f}: IP={tr['identity_preservation']:.3f}")

    # --- Experiment C: occlusion simulation ---
    print("  [C] Occlusion simulation")
    for occlusion_rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=6, slot_dim=32,
            num_iterations=3, match_threshold=0.3,
        )
        model.eval()
        torch.manual_seed(42)
        base_positions = torch.randn(6, 4)
        frames = []
        for t in range(20):
            noise = torch.randn(6, 4) * 0.1
            dets = base_positions + noise + t * 0.05
            # Simulate occlusion: randomly drop detections
            gen = torch.Generator().manual_seed(42 + t)
            mask = torch.rand(6, generator=gen) > occlusion_rate
            visible_dets = dets[mask]
            if visible_dets.numel() == 0:
                visible_dets = dets[:1]  # keep at least 1
            frames.append(visible_dets)

        with torch.no_grad():
            tr = model.track_sequence(frames)
        results["experiments"].append({
            "experiment": "occlusion",
            "occlusion_rate": occlusion_rate,
            "identity_preservation": tr["identity_preservation"],
        })
        print(f"    occlusion={occlusion_rate:.1%}: IP={tr['identity_preservation']:.3f}")

    # --- Experiment D: slot count scaling ---
    print("  [D] Slot count scaling")
    for n_slots in [2, 4, 6, 8, 12]:
        model = TemporalSlotAttentionMOT(
            detection_dim=4, num_slots=n_slots, slot_dim=32,
            num_iterations=3, match_threshold=0.3,
        )
        model.eval()
        torch.manual_seed(42)
        n_objects = min(n_slots, 6)
        base_positions = torch.randn(n_objects, 4)
        frames = []
        for t in range(20):
            noise = torch.randn(n_objects, 4) * 0.1
            dets = base_positions + noise + t * 0.05
            frames.append(dets)

        with torch.no_grad():
            tr = model.track_sequence(frames)
        results["experiments"].append({
            "experiment": "slot_count_scaling",
            "n_slots": n_slots,
            "n_objects": n_objects,
            "identity_preservation": tr["identity_preservation"],
        })
        print(f"    slots={n_slots}: IP={tr['identity_preservation']:.3f}")

    _save("temporal_mot_extended", results)


# =========================================================================
# Benchmark 5: Extended Ablation with Multi-Seed & Learning Curves (T.5)
# =========================================================================

def benchmark_ablation_extended() -> None:
    """Deep ablation: 50 epochs, 5 seeds, learning curves, stats."""
    from prinet.utils.y4q1_tools import (
        AblationConfig,
        bootstrap_ci,
        cohens_d,
        count_flops,
        create_ablation_model,
        welch_t_test,
    )

    print("\n=== Q1.2 Benchmark 5: Extended Ablation ===")
    N_SEEDS = 5
    N_EPOCHS = 50
    N_TRAIN = 256
    N_TEST = 64
    results: dict = {
        "benchmark": "ablation_extended",
        "n_seeds": N_SEEDS,
        "n_epochs": N_EPOCHS,
        "variants": [],
    }

    variant_accs: dict[str, list[float]] = {}

    for variant in ["full", "attention_only", "oscillator_only", "shared_phase"]:
        print(f"  Training {variant} ({N_SEEDS} seeds × {N_EPOCHS} epochs)...")
        seed_results: list[dict] = []

        for seed in range(N_SEEDS):
            torch.manual_seed(42 + seed)
            x_train = torch.randn(N_TRAIN, 64)
            y_train = torch.randint(0, 5, (N_TRAIN,))
            x_test = torch.randn(N_TEST, 64)
            y_test = torch.randint(0, 5, (N_TEST,))

            model = create_ablation_model(variant, n_input=64, n_classes=5, d_model=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            epoch_losses: list[float] = []
            epoch_accs: list[float] = []

            for epoch in range(N_EPOCHS):
                model.train()
                optimizer.zero_grad()
                out = model(x_train)
                loss = torch.nn.functional.nll_loss(out, y_train)
                loss.backward()
                optimizer.step()

                # Record learning curve
                model.eval()
                with torch.no_grad():
                    test_out = model(x_test)
                    preds = test_out.argmax(dim=-1)
                    acc = float((preds == y_test).float().mean())
                    test_loss = float(torch.nn.functional.nll_loss(test_out, y_test))
                epoch_losses.append(test_loss)
                epoch_accs.append(acc)

            seed_results.append({
                "seed": seed,
                "final_accuracy": epoch_accs[-1],
                "final_loss": epoch_losses[-1],
                "learning_curve_acc": epoch_accs[::5],  # every 5th epoch
                "learning_curve_loss": epoch_losses[::5],
                "convergence_epoch": _find_convergence(epoch_losses),
            })

        accs = [r["final_accuracy"] for r in seed_results]
        losses = [r["final_loss"] for r in seed_results]
        variant_accs[variant] = accs

        acc_ci = bootstrap_ci(accs)
        loss_ci = bootstrap_ci(losses)

        n_params = sum(
            p.numel()
            for p in create_ablation_model(
                variant, n_input=64, n_classes=5, d_model=32
            ).parameters()
        )

        entry = {
            "variant": variant,
            "n_params": n_params,
            "n_seeds": N_SEEDS,
            "n_epochs": N_EPOCHS,
            "acc_mean": acc_ci["mean"],
            "acc_ci_lower": acc_ci["ci_lower"],
            "acc_ci_upper": acc_ci["ci_upper"],
            "acc_se": acc_ci["se"],
            "loss_mean": loss_ci["mean"],
            "loss_ci_lower": loss_ci["ci_lower"],
            "loss_ci_upper": loss_ci["ci_upper"],
            "per_seed": seed_results,
        }
        results["variants"].append(entry)
        print(
            f"    {variant:20s}: acc={acc_ci['mean']:.3f} "
            f"[{acc_ci['ci_lower']:.3f}, {acc_ci['ci_upper']:.3f}], "
            f"params={n_params:,}"
        )

    # Pairwise comparisons with Bonferroni correction
    print("  Pairwise comparisons (Bonferroni-corrected):")
    comparisons: list[dict] = []
    variant_names = list(variant_accs.keys())
    n_comparisons = len(variant_names) * (len(variant_names) - 1) // 2

    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            v_a, v_b = variant_names[i], variant_names[j]
            test_result = welch_t_test(variant_accs[v_a], variant_accs[v_b])
            bonferroni_p = min(test_result["p_value"] * n_comparisons, 1.0)
            comp = {
                "comparison": f"{v_a} vs {v_b}",
                "t_stat": test_result["t_stat"],
                "p_value_raw": test_result["p_value"],
                "p_value_bonferroni": bonferroni_p,
                "cohens_d": test_result["cohens_d"],
                "mean_diff": test_result["mean_diff"],
                "significant_005": bonferroni_p < 0.05,
            }
            comparisons.append(comp)
            sig = "*" if bonferroni_p < 0.05 else ""
            print(
                f"    {v_a} vs {v_b}: d={test_result['cohens_d']:.2f}, "
                f"p={bonferroni_p:.4f}{sig}"
            )

    results["pairwise_comparisons"] = comparisons
    _save("ablation_extended", results)


def _find_convergence(losses: list[float], window: int = 5, threshold: float = 0.01) -> int:
    """Find the epoch where loss stabilises (relative change < threshold)."""
    if len(losses) < window + 1:
        return len(losses)
    for i in range(window, len(losses)):
        recent = losses[i - window:i]
        mean_recent = sum(recent) / len(recent)
        if mean_recent > 0 and abs(losses[i] - mean_recent) / mean_recent < threshold:
            return i
    return len(losses)


# =========================================================================
# Benchmark 6: FLOPs with Batch Size Scaling (T.6)
# =========================================================================

def benchmark_flops_batch_scaling() -> None:
    """FLOPs efficiency at different batch sizes and model dimensions."""
    from prinet.utils.y4q1_tools import create_ablation_model, count_flops

    print("\n=== Q1.2 Benchmark 6: FLOPs Batch Scaling ===")
    results: dict = {"benchmark": "flops_batch_scaling", "models": []}

    for d_model in [32, 64, 128]:
        for batch_size in [1, 4, 8, 16, 32]:
            for variant in ["full", "attention_only", "oscillator_only"]:
                model = create_ablation_model(
                    variant, n_input=128, n_classes=10, d_model=d_model,
                )
                flops = count_flops(model, (batch_size, 128))
                n_params = sum(p.numel() for p in model.parameters())

                results["models"].append({
                    "variant": variant,
                    "d_model": d_model,
                    "batch_size": batch_size,
                    "total_flops": flops["total_flops"],
                    "total_params": n_params,
                    "flops_per_param": flops["total_flops"] / max(n_params, 1),
                    "flops_per_sample": flops["total_flops"] / max(batch_size, 1),
                })

    # Print summary: FLOPs growth with batch size for d_model=64
    print("  d_model=64, variant=full, batch scaling:")
    for entry in results["models"]:
        if entry["d_model"] == 64 and entry["variant"] == "full":
            print(
                f"    batch={entry['batch_size']:>3}: "
                f"{entry['total_flops']:>12,} FLOPs "
                f"({entry['flops_per_sample']:>10,} per sample)"
            )

    _save("flops_batch_scaling", results)


# =========================================================================
# Benchmark 7: Wall-Time with Batch Scaling and GPU (T.6)
# =========================================================================

def benchmark_wall_time_extended() -> None:
    """Extended wall-time: batch scaling, more runs, GPU if available."""
    from prinet.utils.y4q1_tools import create_ablation_model, measure_wall_time

    print("\n=== Q1.2 Benchmark 7: Wall-Time Extended ===")
    results: dict = {"benchmark": "wall_time_extended", "timings": []}

    for batch_size in [1, 8, 16, 32]:
        for variant in ["full", "attention_only", "oscillator_only", "shared_phase"]:
            # CPU timing
            model_cpu = create_ablation_model(
                variant, n_input=128, n_classes=10, d_model=64,
            )
            model_cpu.eval()
            x_cpu = torch.randn(batch_size, 128)
            timing_cpu = measure_wall_time(model_cpu, x_cpu, n_warmup=5, n_runs=30)

            entry = {
                "variant": variant,
                "batch_size": batch_size,
                "device": "cpu",
                "mean_ms": timing_cpu["mean_ms"],
                "std_ms": timing_cpu["std_ms"],
                "min_ms": timing_cpu["min_ms"],
                "max_ms": timing_cpu["max_ms"],
            }
            results["timings"].append(entry)

            # GPU timing if available
            if DEVICE == "cuda":
                model_gpu = create_ablation_model(
                    variant, n_input=128, n_classes=10, d_model=64,
                ).to("cuda")
                model_gpu.eval()
                x_gpu = torch.randn(batch_size, 128, device="cuda")
                timing_gpu = measure_wall_time(model_gpu, x_gpu, n_warmup=10, n_runs=50)

                results["timings"].append({
                    "variant": variant,
                    "batch_size": batch_size,
                    "device": "cuda",
                    "mean_ms": timing_gpu["mean_ms"],
                    "std_ms": timing_gpu["std_ms"],
                    "min_ms": timing_gpu["min_ms"],
                    "max_ms": timing_gpu["max_ms"],
                })

    # Print summary
    print("  batch=16, CPU:")
    for t in results["timings"]:
        if t["batch_size"] == 16 and t["device"] == "cpu":
            print(f"    {t['variant']:20s}: {t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms")

    if DEVICE == "cuda":
        print("  batch=16, CUDA:")
        for t in results["timings"]:
            if t["batch_size"] == 16 and t["device"] == "cuda":
                print(f"    {t['variant']:20s}: {t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms")

    _save("wall_time_extended", results)


# =========================================================================
# Benchmark 8: Ring Scaling Extended to N=8192 (T.1)
# =========================================================================

def benchmark_ring_scaling_extended() -> None:
    """Scaling ring-topology throughput up to N=8192."""
    from prinet.utils.oscillosim import OscilloSim
    from prinet.utils.y4q1_tools import bootstrap_ci

    print("\n=== Q1.2 Benchmark 8: Ring Scaling Extended ===")
    results: dict = {"benchmark": "ring_scaling_extended", "scales": []}
    N_SEEDS = 3

    for N in [64, 256, 1024, 2048, 4096, 8192]:
        throughputs = []
        r_finals = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            sim = OscilloSim(
                n_oscillators=N,
                coupling_mode="ring",
                k_neighbors=min(20, N // 4),
                coupling_strength=4.0,
                phase_lag=1.457,
                device=DEVICE,
            )
            result = sim.run(n_steps=500, dt=0.01)
            throughputs.append(result.throughput)
            r_finals.append(result.order_parameter[-1])
            del sim, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tp_ci = bootstrap_ci(throughputs)
        results["scales"].append({
            "N": N,
            "k_neighbors": min(20, N // 4),
            "n_seeds": N_SEEDS,
            "throughput_mean": tp_ci["mean"],
            "throughput_ci_lower": tp_ci["ci_lower"],
            "throughput_ci_upper": tp_ci["ci_upper"],
            "r_final_mean": sum(r_finals) / len(r_finals),
        })
        print(
            f"  N={N:>5}: {tp_ci['mean']:>12,.0f} ± {tp_ci['se']:.0f} osc-steps/s"
        )

    _save("ring_scaling_extended", results)


# =========================================================================
# Benchmark 9: Chimera Seed Stability Analysis (T.2)
# =========================================================================

def benchmark_chimera_seed_stability() -> None:
    """Measure seed-to-seed variance in chimera detection."""
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
    )
    from prinet.utils.y4q1_tools import (
        chimera_initial_condition,
        seed_stability_analysis,
    )

    print("\n=== Q1.2 Benchmark 9: Chimera Seed Stability ===")
    N = 256
    k = 16
    alpha = 1.457
    N_SEEDS = 10
    results: dict = {
        "benchmark": "chimera_seed_stability",
        "N": N,
        "k": k,
        "alpha": alpha,
        "n_seeds": N_SEEDS,
        "per_seed": [],
    }

    for seed in range(N_SEEDS):
        ic = chimera_initial_condition(N, seed=seed)
        sim = OscilloSim(
            n_oscillators=N,
            coupling_mode="ring",
            k_neighbors=k,
            coupling_strength=4.0,
            phase_lag=alpha,
            device=DEVICE,
        )
        result = sim.run(n_steps=2000, dt=0.02, initial_phase=ic)
        nbr_idx = ring_topology(N, k, device="cpu")
        r_local = local_order_parameter(result.final_phase, nbr_idx)
        bc = bimodality_index(r_local)

        results["per_seed"].append({
            "seed": seed,
            "bimodality_coefficient": bc,
            "chimera_detected": bc > 0.555,
            "final_order_param": result.order_parameter[-1],
            "r_local_mean": float(r_local.mean()),
            "r_local_std": float(r_local.std()),
        })
        print(f"  seed={seed}: BC={bc:.3f}")

    stability = seed_stability_analysis(results["per_seed"], "bimodality_coefficient")
    results["stability"] = stability
    print(
        f"  Stability: mean={stability['mean']:.3f}, "
        f"std={stability['std']:.3f}, CV={stability['cv']:.3f}"
    )

    _save("chimera_seed_stability", results)


# =========================================================================
# Benchmark 10: Ablation Convergence Analysis (T.5)
# =========================================================================

def benchmark_ablation_convergence() -> None:
    """Track convergence speed and final performance across variants."""
    from prinet.utils.y4q1_tools import create_ablation_model

    print("\n=== Q1.2 Benchmark 10: Ablation Convergence ===")
    results: dict = {"benchmark": "ablation_convergence", "variants": []}
    N_EPOCHS = 80

    for variant in ["full", "attention_only", "oscillator_only", "shared_phase"]:
        torch.manual_seed(42)
        x_train = torch.randn(256, 64)
        y_train = torch.randint(0, 5, (256,))
        x_test = torch.randn(64, 64)
        y_test = torch.randint(0, 5, (64,))

        model = create_ablation_model(variant, n_input=64, n_classes=5, d_model=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses: list[float] = []
        test_losses: list[float] = []
        test_accs: list[float] = []
        grad_norms: list[float] = []

        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(x_train)
            loss = torch.nn.functional.nll_loss(out, y_train)
            loss.backward()

            # Track gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

            optimizer.step()
            train_losses.append(float(loss))

            model.eval()
            with torch.no_grad():
                test_out = model(x_test)
                test_loss = float(torch.nn.functional.nll_loss(test_out, y_test))
                test_acc = float((test_out.argmax(-1) == y_test).float().mean())
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        convergence_epoch = _find_convergence(test_losses)
        results["variants"].append({
            "variant": variant,
            "n_epochs": N_EPOCHS,
            "convergence_epoch": convergence_epoch,
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
            "final_test_acc": test_accs[-1],
            "train_loss_curve": train_losses[::5],
            "test_loss_curve": test_losses[::5],
            "test_acc_curve": test_accs[::5],
            "grad_norm_curve": grad_norms[::5],
            "grad_norm_final": grad_norms[-1],
        })
        print(
            f"  {variant:20s}: converge@{convergence_epoch}, "
            f"acc={test_accs[-1]:.3f}, grad_norm={grad_norms[-1]:.4f}"
        )

    _save("ablation_convergence", results)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Q1.2 Deepened Benchmarks\n")

    t0 = time.perf_counter()

    def _cleanup() -> None:
        """Free GPU/CPU memory between benchmarks to prevent OOM crashes."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    benchmark_ring_chimera_extended()
    _cleanup()
    benchmark_k_alpha_sweep()
    _cleanup()
    benchmark_small_world_extended()
    _cleanup()
    benchmark_temporal_mot_extended()
    _cleanup()
    benchmark_ablation_extended()
    _cleanup()
    benchmark_flops_batch_scaling()
    _cleanup()
    benchmark_wall_time_extended()
    _cleanup()
    benchmark_ring_scaling_extended()
    _cleanup()
    benchmark_chimera_seed_stability()
    _cleanup()
    benchmark_ablation_convergence()

    elapsed = time.perf_counter() - t0
    print(f"\nAll Q1.2 benchmarks completed in {elapsed:.1f}s")
    print(f"JSON files saved to: {RESULTS_DIR}")
