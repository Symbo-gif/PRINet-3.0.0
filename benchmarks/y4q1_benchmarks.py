"""Year 4 Q1 Benchmarks — Paper-Critical Experiments.

Generates 8+ JSON result files covering:
1. Ring chimera emergence (T.1/T.2)
2. Small-world chimera comparison (T.1/T.2)
3. Chimera phase-lag sweep (T.2)
4. Temporal MOT identity preservation (T.3)
5. Extended CLEVR-N capacity (T.4, lightweight)
6. Ablation variant comparison (T.5)
7. FLOPs efficiency comparison (T.6)
8. Wall-time comparison (T.6)

Usage:
    python benchmarks/y4q1_benchmarks.py
"""

from __future__ import annotations

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
    path = RESULTS_DIR / f"benchmark_y4q1_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")


# =========================================================================
# Benchmark 1: Ring Chimera Emergence (T.1 + T.2)
# =========================================================================

def benchmark_ring_chimera() -> None:
    """Simulate Kuramoto oscillators on a ring with phase lag and
    measure chimera emergence via local order parameter and bimodality."""
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
    )

    print("=== Benchmark 1: Ring Chimera Emergence ===")
    results: dict = {"benchmark": "ring_chimera", "configs": []}

    for N in [128, 256, 512]:
        for alpha in [0.0, 0.5, 1.0, 1.457]:
            sim = OscilloSim(
                n_oscillators=N,
                coupling_mode="ring",
                k_neighbors=min(20, N // 4),
                coupling_strength=4.0,
                phase_lag=alpha,
                device=DEVICE,
            )
            result = sim.run(
                n_steps=500, dt=0.01,
                record_trajectory=True, record_interval=50,
            )

            nbr_idx = ring_topology(N, min(20, N // 4), device="cpu")
            r_local = local_order_parameter(result.final_phase, nbr_idx)
            bc = bimodality_index(r_local)

            entry = {
                "N": N,
                "phase_lag": alpha,
                "k_neighbors": min(20, N // 4),
                "final_order_param": result.order_parameter[-1],
                "bimodality_coefficient": bc,
                "chimera_detected": bc > 0.555,
                "r_local_mean": float(r_local.mean()),
                "r_local_std": float(r_local.std()),
                "wall_time_s": result.wall_time_s,
            }
            results["configs"].append(entry)
            chimera_str = "CHIMERA" if bc > 0.555 else "no chimera"
            print(f"  N={N}, α={alpha:.3f}: r={result.order_parameter[-1]:.3f}, BC={bc:.3f} [{chimera_str}]")

    _save("ring_chimera", results)


# =========================================================================
# Benchmark 2: Small-World Chimera Comparison (T.1 + T.2)
# =========================================================================

def benchmark_small_world_chimera() -> None:
    """Compare chimera emergence on ring vs small-world topologies."""
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
        small_world_topology,
    )

    print("\n=== Benchmark 2: Small-World vs Ring Chimera ===")
    results: dict = {"benchmark": "small_world_chimera", "comparisons": []}

    N, k, alpha = 256, 16, 1.457

    for p_rewire in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        mode = "ring" if p_rewire == 0.0 else "small_world"
        sim = OscilloSim(
            n_oscillators=N,
            coupling_mode=mode,
            k_neighbors=k,
            coupling_strength=4.0,
            phase_lag=alpha,
            p_rewire=p_rewire,
            device=DEVICE,
        )
        result = sim.run(n_steps=500, dt=0.01)

        if mode == "ring":
            nbr_idx = ring_topology(N, k, device="cpu")
        else:
            nbr_idx = small_world_topology(N, k, p_rewire=p_rewire, device="cpu")

        r_local = local_order_parameter(result.final_phase, nbr_idx)
        bc = bimodality_index(r_local)

        entry = {
            "p_rewire": p_rewire,
            "topology": mode,
            "final_order_param": result.order_parameter[-1],
            "bimodality_coefficient": bc,
            "chimera_detected": bc > 0.555,
            "r_local_mean": float(r_local.mean()),
            "r_local_std": float(r_local.std()),
        }
        results["comparisons"].append(entry)
        print(f"  p={p_rewire:.2f}: BC={bc:.3f}, r={result.order_parameter[-1]:.3f}")

    _save("small_world_chimera", results)


# =========================================================================
# Benchmark 3: Phase-Lag Sweep for Chimera Boundary (T.2)
# =========================================================================

def benchmark_phase_lag_sweep() -> None:
    """Fine-grained sweep of phase lag α to map the chimera boundary."""
    from prinet.utils.oscillosim import (
        OscilloSim,
        bimodality_index,
        local_order_parameter,
        ring_topology,
    )

    print("\n=== Benchmark 3: Phase-Lag Sweep ===")
    N, k = 256, 16
    results: dict = {"benchmark": "phase_lag_sweep", "N": N, "k": k, "sweeps": []}

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.457, 1.5, 1.57]

    for alpha in alphas:
        sim = OscilloSim(
            n_oscillators=N,
            coupling_mode="ring",
            k_neighbors=k,
            coupling_strength=4.0,
            phase_lag=alpha,
            device=DEVICE,
        )
        result = sim.run(n_steps=500, dt=0.01)
        nbr_idx = ring_topology(N, k, device="cpu")
        r_local = local_order_parameter(result.final_phase, nbr_idx)
        bc = bimodality_index(r_local)

        results["sweeps"].append({
            "alpha": alpha,
            "bimodality_coefficient": bc,
            "order_parameter": result.order_parameter[-1],
            "r_local_std": float(r_local.std()),
        })
        print(f"  α={alpha:.3f}: BC={bc:.3f}")

    _save("phase_lag_sweep", results)


# =========================================================================
# Benchmark 4: Temporal MOT Identity Preservation (T.3)
# =========================================================================

def benchmark_temporal_mot() -> None:
    """Measure temporal slot attention MOT identity preservation."""
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT

    print("\n=== Benchmark 4: Temporal MOT Identity Preservation ===")
    results: dict = {"benchmark": "temporal_mot", "configs": []}

    for n_frames in [5, 10, 20]:
        for n_slots in [4, 8]:
            model = TemporalSlotAttentionMOT(
                detection_dim=4, num_slots=n_slots, slot_dim=32,
                num_iterations=3, match_threshold=0.3,
            )
            model.eval()

            # Generate synthetic trackable detections
            torch.manual_seed(42)
            frames = []
            base_positions = torch.randn(n_slots, 4)
            for t in range(n_frames):
                noise = torch.randn(n_slots, 4) * 0.1
                dets = base_positions + noise + t * 0.05
                frames.append(dets)

            with torch.no_grad():
                track_result = model.track_sequence(frames)

            entry = {
                "n_frames": n_frames,
                "n_slots": n_slots,
                "identity_preservation": track_result["identity_preservation"],
                "n_identity_matches": len(track_result["identity_matches"]),
                "per_frame_similarity_mean": float(
                    torch.tensor(track_result["per_frame_similarity"]).mean()
                ) if track_result["per_frame_similarity"] else 0.0,
            }
            results["configs"].append(entry)
            print(f"  frames={n_frames}, slots={n_slots}: IP={entry['identity_preservation']:.3f}")

    _save("temporal_mot", results)


# =========================================================================
# Benchmark 5: Ablation Variant Comparison (T.5)
# =========================================================================

def benchmark_ablation() -> None:
    """Compare 4 ablation variants on synthetic classification."""
    from prinet.utils.y4q1_tools import (
        AblationConfig,
        create_ablation_model,
        count_flops,
    )

    print("\n=== Benchmark 5: Ablation Variant Comparison ===")
    results: dict = {"benchmark": "ablation_variants", "variants": []}

    torch.manual_seed(42)
    x_train = torch.randn(128, 64)
    y_train = torch.randint(0, 5, (128,))
    x_test = torch.randn(32, 64)
    y_test = torch.randint(0, 5, (32,))

    for variant in ["full", "attention_only", "oscillator_only", "shared_phase"]:
        model = create_ablation_model(variant, n_input=64, n_classes=5, d_model=32)
        n_params = sum(p.numel() for p in model.parameters())
        flops_info = count_flops(model, (8, 64))

        # Quick train (20 epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            out = model(x_train)
            loss = torch.nn.functional.nll_loss(out, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_test)
            preds = out.argmax(dim=-1)
            acc = float((preds == y_test).float().mean())
            test_loss = float(torch.nn.functional.nll_loss(out, y_test))

        entry = {
            "variant": variant,
            "n_params": n_params,
            "total_flops": flops_info["total_flops"],
            "test_accuracy": acc,
            "test_loss": test_loss,
        }
        results["variants"].append(entry)
        print(f"  {variant:20s}: acc={acc:.3f}, params={n_params:,}, FLOPs={flops_info['total_flops']:,}")

    _save("ablation_variants", results)


# =========================================================================
# Benchmark 6: FLOPs Efficiency Comparison (T.6)
# =========================================================================

def benchmark_flops_efficiency() -> None:
    """Compare FLOPs across model sizes."""
    from prinet.utils.y4q1_tools import create_ablation_model, count_flops

    print("\n=== Benchmark 6: FLOPs Efficiency ===")
    results: dict = {"benchmark": "flops_efficiency", "models": []}

    for d_model in [32, 64, 128]:
        for variant in ["full", "attention_only", "oscillator_only"]:
            model = create_ablation_model(
                variant, n_input=128, n_classes=10, d_model=d_model,
            )
            flops = count_flops(model, (8, 128))
            n_params = sum(p.numel() for p in model.parameters())

            entry = {
                "variant": variant,
                "d_model": d_model,
                "total_flops": flops["total_flops"],
                "total_params": n_params,
                "flops_per_param": flops["total_flops"] / max(n_params, 1),
                "n_layers": len(flops["layer_flops"]),
            }
            results["models"].append(entry)
            print(f"  {variant:20s} d={d_model}: {flops['total_flops']:>12,} FLOPs, {n_params:>8,} params")

    _save("flops_efficiency", results)


# =========================================================================
# Benchmark 7: Wall-Time Comparison (T.6)
# =========================================================================

def benchmark_wall_time() -> None:
    """Measure forward-pass latency for ablation variants."""
    from prinet.utils.y4q1_tools import create_ablation_model, measure_wall_time

    print("\n=== Benchmark 7: Wall-Time Comparison ===")
    results: dict = {"benchmark": "wall_time", "timings": []}

    x = torch.randn(16, 128)

    for variant in ["full", "attention_only", "oscillator_only", "shared_phase"]:
        model = create_ablation_model(variant, n_input=128, n_classes=10, d_model=64)
        model.eval()

        timing = measure_wall_time(model, x, n_warmup=5, n_runs=20)
        timing["variant"] = variant
        results["timings"].append(timing)
        print(f"  {variant:20s}: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")

    _save("wall_time", results)


# =========================================================================
# Benchmark 8: OscilloSim Scaling with Ring Topology (T.1)
# =========================================================================

def benchmark_ring_scaling() -> None:
    """Measure ring-topology simulation throughput at different scales."""
    from prinet.utils.oscillosim import OscilloSim

    print("\n=== Benchmark 8: Ring Topology Scaling ===")
    results: dict = {"benchmark": "ring_scaling", "scales": []}

    for N in [64, 256, 1024, 4096]:
        sim = OscilloSim(
            n_oscillators=N,
            coupling_mode="ring",
            k_neighbors=min(20, N // 4),
            coupling_strength=4.0,
            phase_lag=1.457,
            device=DEVICE,
        )
        result = sim.run(n_steps=200, dt=0.01)

        entry = {
            "N": N,
            "k_neighbors": min(20, N // 4),
            "wall_time_s": result.wall_time_s,
            "throughput": result.throughput,
            "final_order_param": result.order_parameter[-1],
        }
        results["scales"].append(entry)
        print(f"  N={N:>5}: {result.throughput:>12,.0f} osc-steps/s, r={result.order_parameter[-1]:.3f}")

    _save("ring_scaling", results)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}\n")

    t0 = time.perf_counter()

    benchmark_ring_chimera()
    benchmark_small_world_chimera()
    benchmark_phase_lag_sweep()
    benchmark_temporal_mot()
    benchmark_ablation()
    benchmark_flops_efficiency()
    benchmark_wall_time()
    benchmark_ring_scaling()

    elapsed = time.perf_counter() - t0
    print(f"\nAll benchmarks completed in {elapsed:.1f}s")
    print(f"JSON files saved to: {RESULTS_DIR}")
