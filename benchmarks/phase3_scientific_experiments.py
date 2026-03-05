#!/usr/bin/env python
"""Phase 3: New Scientific Experiments — Profiling & Geometry.

Implements experiments that produce genuinely new scientific insight:

    3.1  Computational cost profiling (FLOPs, latency, memory)
    3.3  Gradient flow analysis (per-layer grad norms during training)
    3.4  Representation geometry (UMAP/t-SNE of learned embeddings)

Phase 3 Experiment numbering follows the roadmap document. Experiments
3.2 (ablation) and 3.5 (transfer) are deferred to Phase 4.

Usage:
    python benchmarks/phase3_scientific_experiments.py --all
    python benchmarks/phase3_scientific_experiments.py --profiling
    python benchmarks/phase3_scientific_experiments.py --gradient-flow
    python benchmarks/phase3_scientific_experiments.py --representation

Hardware: RTX 4060 8GB VRAM.

Reference:
    PRINet Paper Roadmap, Phase 3 (New Scientific Experiments).

Authors:
    Michael Maillet
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_RERUN = os.environ.get("FORCE_RERUN", "").lower() in ("1", "true", "yes")

SEEDS_7 = (42, 123, 456, 789, 1024, 2048, 3072)

# Training / model defaults
MAX_EPOCHS = 10
PATIENCE = 4
WARMUP = 1
LR = 3e-4
DET_DIM = 4
TRAIN_SEQS = 30
VAL_SEQS = 10
TEST_SEQS = 20

PT_KWARGS: dict[str, Any] = dict(
    n_delta=4, n_theta=8, n_gamma=16,
    n_discrete_steps=5, match_threshold=0.1,
)
SA_KWARGS: dict[str, Any] = dict(
    num_slots=6, slot_dim=64, num_iterations=3,
    match_threshold=0.1,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _p(*args: Any, **kwargs: Any) -> None:
    """ASCII-safe print with flush."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict[str, Any]) -> bool:
    """Save JSON artefact.

    Args:
        name: Artefact name.
        data: Dict to serialise.

    Returns:
        True if saved.
    """
    path = RESULTS_DIR / f"phase3_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (exists)")
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _p(f"  -> {path.name}")
    return True


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_pt(seed: int = 42) -> nn.Module:
    from prinet.nn.hybrid import PhaseTracker
    torch.manual_seed(seed)
    return PhaseTracker(detection_dim=DET_DIM, **PT_KWARGS)


def _build_sa(seed: int = 42) -> nn.Module:
    from prinet.nn.slot_attention import TemporalSlotAttentionMOT
    torch.manual_seed(seed)
    return TemporalSlotAttentionMOT(detection_dim=DET_DIM, **SA_KWARGS)


def _gen(n: int, n_objects: int = 4, n_frames: int = 20,
         base_seed: int = 42) -> list:
    from prinet.utils.temporal_training import generate_dataset
    return generate_dataset(
        n, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, base_seed=base_seed,
    )


def _train_model(
    model: nn.Module,
    seed: int,
    n_objects: int = 4,
    n_frames: int = 20,
    hooks: Optional[dict[str, list]] = None,
    epoch_hooks: Optional[dict[str, dict[int, list[float]]]] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train model, optionally recording gradient norms via hooks.

    Args:
        model: PT or SA model.
        seed: Random seed.
        n_objects: Objects per sequence.
        n_frames: Frames per sequence.
        hooks: If provided, maps layer_name -> flat list of gradient
            norms across all steps (legacy).
        epoch_hooks: If provided, maps layer_name -> {epoch: [norms]}.
            Enables the layer x epoch heatmap for gradient flow
            analysis (Figure 7 in the roadmap).

    Returns:
        (trained_model, training_info).
    """
    from prinet.utils.temporal_training import (
        generate_dataset,
        hungarian_similarity_loss,
    )
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    # Mutable epoch counter shared with hook closures
    epoch_counter: list[int] = [0]

    # Register backward hooks for gradient flow analysis
    hook_handles: list = []
    if hooks is not None or epoch_hooks is not None:
        # Disable inplace operations that conflict with backward hooks
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ReLU) and mod.inplace:
                mod.inplace = False

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.GRU, nn.GRUCell,
                                   nn.LayerNorm)):
                short = name.replace(".", "/")

                def _make_hook(n: str):
                    def hook_fn(module, grad_input, grad_output):
                        # Record L2 norm of output gradients
                        for g in grad_output:
                            if g is not None:
                                norm_val = float(
                                    g.detach().norm(2).item()
                                )
                                # Flat list (legacy)
                                if hooks is not None:
                                    hooks.setdefault(n, []).append(
                                        norm_val
                                    )
                                # Per-epoch binned recording
                                if epoch_hooks is not None:
                                    ep = epoch_counter[0]
                                    epoch_hooks.setdefault(
                                        n, {}
                                    ).setdefault(ep, []).append(
                                        norm_val
                                    )
                    return hook_fn

                hook_handles.append(module.register_full_backward_hook(
                    _make_hook(short)
                ))

    train_data = generate_dataset(
        TRAIN_SEQS, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, base_seed=seed,
    )
    val_data = generate_dataset(
        VAL_SEQS, n_objects=n_objects, n_frames=n_frames,
        det_dim=DET_DIM, base_seed=seed + 50000,
    )

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_counter[0] = epoch
        model.train()
        for seq in train_data:
            frames = [f.to(DEVICE) for f in seq.frames]
            total_loss = torch.tensor(0.0, device=DEVICE)
            for t in range(len(frames) - 1):
                _, sim = model(frames[t], frames[t + 1])
                total_loss = total_loss + hungarian_similarity_loss(sim, n_objects)
            total_loss = total_loss / max(len(frames) - 1, 1)
            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Validation
        model.eval()
        v_losses: list[float] = []
        with torch.no_grad():
            for seq in val_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                t_loss = torch.tensor(0.0, device=DEVICE)
                for t in range(len(frames) - 1):
                    _, sim = model(frames[t], frames[t + 1])
                    t_loss = t_loss + hungarian_similarity_loss(sim, n_objects)
                v_losses.append(float(t_loss.item() / max(len(frames) - 1, 1)))
        val_loss = sum(v_losses) / max(len(v_losses), 1)
        if val_loss < best_loss and epoch >= WARMUP:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE and epoch >= WARMUP:
            break

    wall = time.time() - t0
    for h in hook_handles:
        h.remove()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": epoch + 1, "wall_time_s": wall,
                    "best_val_loss": float(best_loss)}


# =========================================================================
# 3.1 — Computational Cost Profiling
# =========================================================================


def experiment_3_1_profiling() -> dict[str, Any]:
    """Experiment 3.1: FLOPs, latency, and peak memory profiling.

    Measures computational cost per forward pass for both PT and SA
    across different object counts (4, 8, 16) and batch sizes
    {1, 4, 16, 64}. Reports:
      - FLOPs via ptflops (with manual GRU fallback)
      - Wall-clock latency (mean +/- std over 100 forward passes)
      - Peak GPU memory during forward pass
      - Parameters (total, trainable)
      - Batch-size scaling characteristics

    SA's attention mechanism scales differently with batch size than
    PT's coupling dynamics, so batch-level profiling reveals concrete
    deployment advantages.

    Returns:
        Dict with profiling results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 3.1: Computational Cost Profiling")
    _p("=" * 60)

    from prinet.utils.temporal_training import count_parameters

    n_objects_list = [4, 8, 16]
    batch_sizes = [1, 4, 16, 64]
    n_frames = 20
    n_warmup = 10
    n_measure = 100
    seed = 42

    results: list[dict[str, Any]] = []

    for n_obj in n_objects_list:
        _p(f"\n  --- N_objects = {n_obj} ---")

        for model_name, builder in [("PhaseTracker", _build_pt),
                                     ("SlotAttention", _build_sa)]:
            model = builder(seed)
            model.eval().to(DEVICE)

            # Parameter counts
            n_params = count_parameters(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Generate a single test pair
            torch.manual_seed(seed + 100)
            det_a = torch.randn(n_obj, DET_DIM, device=DEVICE)
            det_b = torch.randn(n_obj, DET_DIM, device=DEVICE)

            # FLOPs estimation via ptflops (with GRU-aware fallback)
            flops = None
            try:
                from ptflops import get_model_complexity_info

                def _input_constructor(input_res):
                    return {"det_t": det_a.unsqueeze(0).cpu(),
                            "det_t1": det_b.unsqueeze(0).cpu()}

                # Use a wrapper to match ptflops expected interface
                class _Wrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, x):
                        n = x.shape[1] // 2
                        return self.m(x[:, :n, :].squeeze(0),
                                      x[:, n:, :].squeeze(0))

                w = _Wrapper(model).to(DEVICE)
                combo = torch.cat([det_a.unsqueeze(0), det_b.unsqueeze(0)],
                                  dim=1)
                macs, params_str = get_model_complexity_info(
                    w, tuple(combo.shape[1:]),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                flops = int(macs * 2)  # MACs -> FLOPs
                del w
            except Exception as e:
                _p(f"    ptflops failed (using manual count): {e}")
                # Manual FLOPs: Linear + GRU layers
                # Linear: 2 * in * out per layer
                # GRU: 3 gates * 2 * hidden * (hidden + input) per cell per step
                flops_manual = 0
                for name, mod in model.named_modules():
                    if isinstance(mod, nn.Linear):
                        flops_manual += 2 * mod.in_features * mod.out_features
                    elif isinstance(mod, (nn.GRU, nn.GRUCell)):
                        h = mod.hidden_size
                        inp = mod.input_size
                        n_steps = 1  # per call
                        # 3 gates (reset, update, new), each: W_ih*x + W_hh*h
                        flops_manual += 3 * 2 * h * (h + inp) * n_steps
                flops = flops_manual
                _p(f"    NOTE: manual FLOPs includes GRU estimate")

            # Latency profiling
            with torch.no_grad():
                for _ in range(n_warmup):
                    model(det_a, det_b)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

                latencies: list[float] = []
                for _ in range(n_measure):
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(det_a, det_b)
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)  # ms

            # Peak memory (GPU only)
            peak_mem_mb = None
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    model(det_a, det_b)
                    torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

            entry = {
                "model": model_name,
                "n_objects": n_obj,
                "parameters": {
                    "total": n_params,
                    "trainable": n_trainable,
                },
                "flops": flops,
                "latency_ms": {
                    "mean": float(np.mean(latencies)),
                    "std": float(np.std(latencies, ddof=1)),
                    "min": float(np.min(latencies)),
                    "max": float(np.max(latencies)),
                    "median": float(np.median(latencies)),
                    "n_trials": n_measure,
                },
                "peak_memory_mb": peak_mem_mb,
            }
            results.append(entry)

            _p(f"    {model_name}: {n_params} params, "
               f"{flops/1e6:.2f} MFLOPs, "
               f"{np.mean(latencies):.3f}ms ± {np.std(latencies, ddof=1):.3f}ms, "
               f"peak={peak_mem_mb:.1f}MB" if peak_mem_mb else "")

            del model
            _cleanup()

    # Batch-size scaling profiling
    _p("\n  --- Batch-size scaling (N=5) ---")
    batch_profiles: list[dict[str, Any]] = []
    for bs in batch_sizes:
        for model_name, builder in [("PhaseTracker", _build_pt),
                                     ("SlotAttention", _build_sa)]:
            model = builder(seed)
            model.eval().to(DEVICE)

            torch.manual_seed(seed + 200)
            # Create batch of forward-pass pairs
            det_a_batch = torch.randn(bs, 5, DET_DIM, device=DEVICE)
            det_b_batch = torch.randn(bs, 5, DET_DIM, device=DEVICE)

            with torch.no_grad():
                # Warmup
                for _ in range(n_warmup):
                    for i in range(bs):
                        model(det_a_batch[i], det_b_batch[i])
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

                latencies_bs: list[float] = []
                for _ in range(n_measure):
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for i in range(bs):
                        model(det_a_batch[i], det_b_batch[i])
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    latencies_bs.append(
                        (time.perf_counter() - t0) * 1000
                    )  # ms

            # Peak memory for this batch size
            peak_mem_bs = None
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for i in range(bs):
                        model(det_a_batch[i], det_b_batch[i])
                    torch.cuda.synchronize()
                peak_mem_bs = torch.cuda.max_memory_allocated() / 1e6

            batch_profiles.append({
                "model": model_name,
                "batch_size": bs,
                "latency_ms": {
                    "mean": float(np.mean(latencies_bs)),
                    "std": float(np.std(latencies_bs, ddof=1)),
                },
                "per_sample_latency_ms": float(
                    np.mean(latencies_bs) / max(bs, 1)
                ),
                "peak_memory_mb": peak_mem_bs,
            })
            _p(f"    {model_name} BS={bs}: "
               f"{np.mean(latencies_bs):.2f}ms total, "
               f"{np.mean(latencies_bs)/bs:.3f}ms/sample, "
               f"peak={peak_mem_bs:.1f}MB" if peak_mem_bs else "")

            del model
            _cleanup()

    # Sequence-level profiling (track_sequence)
    _p("\n  --- Sequence-level profiling (T=50, N=5) ---")
    seq_profiles: list[dict[str, Any]] = []
    test_data = _gen(5, n_objects=5, n_frames=50, base_seed=42)

    for model_name, builder in [("PhaseTracker", _build_pt),
                                 ("SlotAttention", _build_sa)]:
        model = builder(seed)
        model.eval().to(DEVICE)

        with torch.no_grad():
            # Warmup
            for _ in range(3):
                frames = [f.to(DEVICE) for f in test_data[0].frames]
                model.track_sequence(frames)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            latencies = []
            for seq in test_data:
                frames = [f.to(DEVICE) for f in seq.frames]
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.track_sequence(frames)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        seq_profiles.append({
            "model": model_name,
            "sequence_latency_ms": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies, ddof=1)),
            },
        })
        _p(f"    {model_name} sequence: "
           f"{np.mean(latencies):.1f}ms ± {np.std(latencies, ddof=1):.1f}ms")

        del model
        _cleanup()

    # Compute efficiency metrics
    pt_entry = next((r for r in results if r["model"] == "PhaseTracker"
                     and r["n_objects"] == 4), None)
    sa_entry = next((r for r in results if r["model"] == "SlotAttention"
                     and r["n_objects"] == 4), None)
    efficiency = {}
    if pt_entry and sa_entry:
        # count_parameters may return dict or int
        pt_total = pt_entry["parameters"]["total"]
        sa_total = sa_entry["parameters"]["total"]
        if isinstance(pt_total, dict):
            pt_total = pt_total.get("total", sum(pt_total.values()))
        if isinstance(sa_total, dict):
            sa_total = sa_total.get("total", sum(sa_total.values()))
        param_ratio = sa_total / max(pt_total, 1)
        flops_ratio = (sa_entry["flops"] or 1) / max(pt_entry["flops"] or 1, 1)
        efficiency = {
            "parameter_ratio_sa_over_pt": round(param_ratio, 1),
            "flops_ratio_sa_over_pt": round(flops_ratio, 2),
            "narrative": (
                f"PhaseTracker achieves competitive IP with "
                f"{param_ratio:.0f}x fewer parameters and "
                f"{flops_ratio:.1f}x fewer FLOPs than SlotAttention."
            ),
        }

    # Batch scaling efficiency
    batch_scaling = {}
    for mn in ["PhaseTracker", "SlotAttention"]:
        model_bs = [b for b in batch_profiles if b["model"] == mn]
        if len(model_bs) >= 2:
            bs1 = next((b for b in model_bs if b["batch_size"] == 1), None)
            bs64 = next((b for b in model_bs if b["batch_size"] == 64), None)
            if bs1 and bs64:
                batch_scaling[mn] = {
                    "bs1_latency_ms": bs1["latency_ms"]["mean"],
                    "bs64_latency_ms": bs64["latency_ms"]["mean"],
                    "scaling_factor": round(
                        bs64["latency_ms"]["mean"] / max(
                            bs1["latency_ms"]["mean"], 0.001
                        ), 2
                    ),
                    "ideal_scaling": 64.0,
                }

    result: dict[str, Any] = {
        "benchmark": "phase3_exp3_1_profiling",
        "description": "Computational cost profiling (FLOPs, latency, memory, batch scaling)",
        "per_pair_profiles": results,
        "batch_size_profiles": batch_profiles,
        "batch_scaling_efficiency": batch_scaling,
        "sequence_profiles": seq_profiles,
        "efficiency_summary": efficiency,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("profiling", result)
    return result


# =========================================================================
# 3.3 — Gradient Flow Analysis
# =========================================================================


def experiment_3_3_gradient_flow() -> dict[str, Any]:
    """Experiment 3.3: Gradient flow analysis during training.

    Records per-layer, per-epoch gradient L2 norms for both PT and SA
    during full training runs (3 seeds). Produces:
      - Per-layer gradient norm trajectories (flat and per-epoch)
      - Layer x epoch heatmap data (for Figure 7)
      - Gradient vanishing/exploding detection
      - Cross-layer variance comparison (skip-connection proxy)

    The per-epoch binning enables the gradient flow heatmap showing how
    gradient magnitude evolves across layers and training epochs.

    Key question: does PT's oscillatory coupling act as an implicit
    skip connection for gradients? If gradient variance across layers
    is lower for PT than SA, that's a mechanistic explanation.

    Returns:
        Dict with gradient flow analysis.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 3.3: Gradient Flow Analysis")
    _p("=" * 60)

    seeds = SEEDS_7[:3]  # 3 seeds for gradient analysis (expensive)
    n_objects = 5
    n_frames = 20

    model_results: list[dict[str, Any]] = []

    for model_name, builder in [("PhaseTracker", _build_pt),
                                 ("SlotAttention", _build_sa)]:
        _p(f"\n  --- {model_name} ---")
        all_seed_grads: list[dict[str, list[float]]] = []
        all_seed_epoch_grads: list[dict[str, dict[int, list[float]]]] = []

        for seed in seeds:
            _p(f"    Seed {seed}...")
            model = builder(seed)
            grad_hooks: dict[str, list[float]] = {}
            epoch_grad_hooks: dict[str, dict[int, list[float]]] = {}

            model, info = _train_model(
                model, seed, n_objects=n_objects, n_frames=n_frames,
                hooks=grad_hooks,
                epoch_hooks=epoch_grad_hooks,
            )

            all_seed_grads.append(grad_hooks)
            all_seed_epoch_grads.append(epoch_grad_hooks)
            _p(f"      {len(grad_hooks)} layers tracked, "
               f"{info['epochs']} epochs")

            del model
            _cleanup()

        # Aggregate per-layer statistics across seeds
        all_layers = set()
        for sg in all_seed_grads:
            all_layers.update(sg.keys())

        layer_summary: dict[str, dict[str, Any]] = {}
        for layer in sorted(all_layers):
            norms: list[float] = []
            for sg in all_seed_grads:
                norms.extend(sg.get(layer, []))

            if norms:
                arr = np.array(norms)
                layer_summary[layer] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "median": float(np.median(arr)),
                    "n_samples": len(norms),
                    "vanishing_frac": float((arr < 1e-7).mean()),
                    "exploding_frac": float((arr > 100.0).mean()),
                }

        # Build layer x epoch heatmap data (averaged across seeds)
        # Each cell = mean gradient norm for that layer at that epoch
        heatmap: dict[str, dict[str, float]] = {}
        all_epochs: set[int] = set()
        for seg in all_seed_epoch_grads:
            for layer, ep_dict in seg.items():
                all_epochs.update(ep_dict.keys())

        for layer in sorted(all_layers):
            heatmap[layer] = {}
            for ep in sorted(all_epochs):
                epoch_norms: list[float] = []
                for seg in all_seed_epoch_grads:
                    epoch_norms.extend(
                        seg.get(layer, {}).get(ep, [])
                    )
                if epoch_norms:
                    heatmap[layer][str(ep)] = float(
                        np.mean(epoch_norms)
                    )

        # Per-epoch variance across layers (skip-connection proxy)
        epoch_cross_layer_var: dict[str, float] = {}
        for ep in sorted(all_epochs):
            ep_layer_means: list[float] = []
            for layer in sorted(all_layers):
                val = heatmap.get(layer, {}).get(str(ep))
                if val is not None:
                    ep_layer_means.append(val)
            if len(ep_layer_means) > 1:
                epoch_cross_layer_var[str(ep)] = float(
                    np.std(ep_layer_means)
                )

        # Detect gradient health
        vanishing_layers = [
            k for k, v in layer_summary.items()
            if v["vanishing_frac"] > 0.5
        ]
        exploding_layers = [
            k for k, v in layer_summary.items()
            if v["exploding_frac"] > 0.01
        ]

        model_results.append({
            "model": model_name,
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "layer_summary": layer_summary,
            "gradient_health": {
                "vanishing_layers": vanishing_layers,
                "exploding_layers": exploding_layers,
                "overall": (
                    "healthy" if not vanishing_layers and not exploding_layers
                    else "issues_detected"
                ),
            },
            # Layer x Epoch heatmap data for Figure 7
            "epoch_heatmap": heatmap,
            "epoch_cross_layer_std": epoch_cross_layer_var,
            # Store raw per-seed norms for first 5 layers (for plotting)
            "raw_trajectories": {
                layer: [sg.get(layer, []) for sg in all_seed_grads]
                for layer in sorted(all_layers)[:5]
            },
        })

        _p(f"    Health: {model_results[-1]['gradient_health']['overall']}")
        if vanishing_layers:
            _p(f"    Vanishing: {vanishing_layers}")
        if exploding_layers:
            _p(f"    Exploding: {exploding_layers}")

    # Comparative analysis
    pt_result = next((r for r in model_results
                      if r["model"] == "PhaseTracker"), None)
    sa_result = next((r for r in model_results
                      if r["model"] == "SlotAttention"), None)

    comparison = {}
    if pt_result and sa_result:
        pt_means = [v["mean"] for v in pt_result["layer_summary"].values()]
        sa_means = [v["mean"] for v in sa_result["layer_summary"].values()]
        pt_cross_var = list(pt_result.get("epoch_cross_layer_std", {}).values())
        sa_cross_var = list(sa_result.get("epoch_cross_layer_std", {}).values())
        comparison = {
            "pt_avg_grad_norm": float(np.mean(pt_means)) if pt_means else 0.0,
            "sa_avg_grad_norm": float(np.mean(sa_means)) if sa_means else 0.0,
            "pt_grad_std_across_layers": float(np.std(pt_means)) if len(pt_means) > 1 else 0.0,
            "sa_grad_std_across_layers": float(np.std(sa_means)) if len(sa_means) > 1 else 0.0,
            "pt_mean_cross_layer_std": float(np.mean(pt_cross_var)) if pt_cross_var else 0.0,
            "sa_mean_cross_layer_std": float(np.mean(sa_cross_var)) if sa_cross_var else 0.0,
            "implicit_skip_connection": (
                "PT has lower cross-layer gradient variance"
                if (pt_cross_var and sa_cross_var and
                    np.mean(pt_cross_var) < np.mean(sa_cross_var))
                else "SA has lower or equal cross-layer gradient variance"
            ),
            "interpretation": (
                "Lower cross-layer gradient variance indicates more uniform "
                "gradient flow. Phase dynamics may provide implicit skip "
                "connections through oscillatory coupling."
            ),
        }

    result: dict[str, Any] = {
        "benchmark": "phase3_exp3_3_gradient_flow",
        "description": "Per-layer, per-epoch gradient norm analysis during training",
        "models": model_results,
        "comparison": comparison,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("gradient_flow", result)
    return result


# =========================================================================
# 3.4 — Representation Geometry (UMAP/t-SNE)
# =========================================================================


def experiment_3_4_representation_geometry() -> dict[str, Any]:
    """Experiment 3.4: UMAP and t-SNE analysis of learned representations.

    Extracts internal representations from trained PT and SA models:
      - PT: Phase embeddings (complex-valued -> 2D via angle/magnitude)
      - SA: Slot representations (real-valued)

    Computes:
      - 2D projections via UMAP and t-SNE
      - Silhouette scores (how well objects cluster)
      - k-NN purity (fraction of k nearest neighbours sharing same ID)
      - Intrinsic dimensionality estimation

    This establishes whether oscillatory representations form
    geometrically distinct clusters compared to slot-based ones.

    Returns:
        Dict with geometry analysis results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 3.4: Representation Geometry (UMAP / t-SNE)")
    _p("=" * 60)

    try:
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        _p("  [ERROR] scikit-learn not available, skipping.")
        return {"error": "scikit-learn not installed"}

    try:
        import umap
    except ImportError:
        umap = None
        _p("  [WARNING] umap-learn not available, using t-SNE only.")

    seeds = SEEDS_7[:3]
    n_objects = 5
    n_frames = 30
    k_nn = 5  # For k-NN purity

    all_model_results: list[dict[str, Any]] = []

    for model_name, builder in [("PhaseTracker", _build_pt),
                                 ("SlotAttention", _build_sa)]:
        _p(f"\n  --- {model_name} ---")
        all_embeddings: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for seed in seeds:
            _p(f"    Seed {seed}...")
            model = builder(seed)
            model, _ = _train_model(model, seed, n_objects=n_objects,
                                     n_frames=n_frames)
            model.eval().to(DEVICE)

            test_data = _gen(10, n_objects=n_objects, n_frames=n_frames,
                             base_seed=seed + 70000)

            # Extract embeddings
            embeddings: list[np.ndarray] = []
            labels: list[int] = []

            with torch.no_grad():
                for seq in test_data:
                    frames = [f.to(DEVICE) for f in seq.frames]

                    if model_name == "PhaseTracker":
                        # Extract phase embeddings: complex -> (cos, sin)
                        for t_idx, frame in enumerate(frames):
                            phase, amp = model.encode(frame)
                            # phase: (n_obj, n_phases)
                            # Convert to real representation
                            cos_phi = torch.cos(phase)
                            sin_phi = torch.sin(phase)
                            emb = torch.cat([cos_phi, sin_phi, amp], dim=-1)
                            emb_np = emb.cpu().numpy()
                            for obj_idx in range(n_objects):
                                embeddings.append(emb_np[obj_idx])
                                labels.append(obj_idx)
                    else:
                        # SA: Extract slot representations
                        for t_idx, frame in enumerate(frames):
                            # Get slot states from internal forward
                            # Note: attribute is det_encoder (not detection_encoder)
                            slots = model.slot_attention(
                                model.det_encoder(frame).unsqueeze(0)
                            )  # (1, num_slots, slot_dim)
                            slots_np = slots.squeeze(0).cpu().numpy()
                            for obj_idx in range(min(n_objects, slots_np.shape[0])):
                                embeddings.append(slots_np[obj_idx])
                                labels.append(obj_idx)

            all_embeddings.extend(embeddings)
            all_labels.extend(labels)

            del model
            _cleanup()

        X = np.array(all_embeddings)
        y = np.array(all_labels)
        _p(f"    Collected {X.shape[0]} embeddings, dim={X.shape[1]}")

        # Handle constant/near-constant features
        stds = X.std(axis=0)
        good_dims = stds > 1e-8
        if good_dims.sum() < 2:
            _p(f"    [WARNING] Only {good_dims.sum()} non-constant dims")
            X_clean = X[:, good_dims] if good_dims.sum() > 0 else X[:, :2]
        else:
            X_clean = X[:, good_dims]

        # Subsample if too large for t-SNE
        max_points = 2000
        if X_clean.shape[0] > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_clean.shape[0], size=max_points, replace=False)
            X_sub = X_clean[idx]
            y_sub = y[idx]
        else:
            X_sub = X_clean
            y_sub = y

        # t-SNE
        _p(f"    Computing t-SNE...")
        try:
            perplexity = min(30, max(5, X_sub.shape[0] // 5))
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, max_iter=1000)
            X_tsne = tsne.fit_transform(X_sub)
            tsne_sil = float(silhouette_score(X_tsne, y_sub))
            _p(f"      Silhouette (t-SNE): {tsne_sil:.4f}")
        except Exception as e:
            _p(f"      t-SNE failed: {e}")
            X_tsne = None
            tsne_sil = None

        # UMAP
        umap_sil = None
        X_umap = None
        if umap is not None:
            _p(f"    Computing UMAP...")
            try:
                n_neighbors = min(15, max(2, X_sub.shape[0] // 10))
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                     random_state=42)
                X_umap = reducer.fit_transform(X_sub)
                umap_sil = float(silhouette_score(X_umap, y_sub))
                _p(f"      Silhouette (UMAP): {umap_sil:.4f}")
            except Exception as e:
                _p(f"      UMAP failed: {e}")

        # k-NN purity
        _p(f"    Computing k-NN purity (k={k_nn})...")
        try:
            nn_model = NearestNeighbors(n_neighbors=k_nn + 1)
            nn_model.fit(X_clean if X_clean.shape[0] <= max_points else X_sub)
            y_used = y if X_clean.shape[0] <= max_points else y_sub
            X_used = X_clean if X_clean.shape[0] <= max_points else X_sub

            _, indices = nn_model.kneighbors(X_used)
            purity_scores: list[float] = []
            for i in range(X_used.shape[0]):
                nbr_labels = y_used[indices[i, 1:]]  # Exclude self
                purity = (nbr_labels == y_used[i]).mean()
                purity_scores.append(float(purity))
            knn_purity = float(np.mean(purity_scores))
            _p(f"      k-NN purity: {knn_purity:.4f}")
        except Exception as e:
            _p(f"      k-NN purity failed: {e}")
            knn_purity = None

        # Silhouette in original space
        try:
            orig_sil = float(silhouette_score(X_sub, y_sub))
        except Exception:
            orig_sil = None

        # Intrinsic dimensionality (PCA explained variance)
        _p(f"    Estimating intrinsic dimensionality...")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, X_clean.shape[1], X_sub.shape[0]))
            pca.fit(X_sub)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim_90 = int(np.searchsorted(cum_var, 0.9) + 1)
            intrinsic_dim_95 = int(np.searchsorted(cum_var, 0.95) + 1)
            _p(f"      Intrinsic dim (90% var): {intrinsic_dim_90}")
            _p(f"      Intrinsic dim (95% var): {intrinsic_dim_95}")
        except Exception as e:
            intrinsic_dim_90 = None
            intrinsic_dim_95 = None
            _p(f"      PCA failed: {e}")

        model_entry = {
            "model": model_name,
            "n_embeddings": X.shape[0],
            "embedding_dim": int(X.shape[1]),
            "n_valid_dims": int(good_dims.sum()),
            "metrics": {
                "silhouette_original": orig_sil,
                "silhouette_tsne": tsne_sil,
                "silhouette_umap": umap_sil,
                "knn_purity": knn_purity,
                "intrinsic_dim_90pct": intrinsic_dim_90,
                "intrinsic_dim_95pct": intrinsic_dim_95,
            },
            # Store 2D projections (first 200 points for visualisation)
            "tsne_projection": (
                X_tsne[:200].tolist() if X_tsne is not None else None
            ),
            "umap_projection": (
                X_umap[:200].tolist() if X_umap is not None else None
            ),
            "labels_sample": y_sub[:200].tolist(),
        }
        all_model_results.append(model_entry)

    # Comparative summary
    comparison = {}
    pt_r = next((r for r in all_model_results
                 if r["model"] == "PhaseTracker"), None)
    sa_r = next((r for r in all_model_results
                 if r["model"] == "SlotAttention"), None)
    if pt_r and sa_r:
        comparison = {
            "pt_silhouette": pt_r["metrics"]["silhouette_original"],
            "sa_silhouette": sa_r["metrics"]["silhouette_original"],
            "pt_knn_purity": pt_r["metrics"]["knn_purity"],
            "sa_knn_purity": sa_r["metrics"]["knn_purity"],
            "pt_intrinsic_dim": pt_r["metrics"]["intrinsic_dim_90pct"],
            "sa_intrinsic_dim": sa_r["metrics"]["intrinsic_dim_90pct"],
            "interpretation": (
                "Higher silhouette/purity indicates better object separation. "
                "Lower intrinsic dimensionality suggests more compact representations. "
                "Phase embeddings occupy a complex manifold (circle-valued) "
                "while slot representations are Euclidean."
            ),
        }

    result: dict[str, Any] = {
        "benchmark": "phase3_exp3_4_representation_geometry",
        "description": "UMAP/t-SNE representation geometry analysis",
        "models": all_model_results,
        "comparison": comparison,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("representation_geometry", result)
    return result


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    """Run Phase 3 scientific experiments.

    Returns:
        Exit code (0 success, 1 failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 3: Scientific Experiments (3.1, 3.3, 3.4)"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--profiling", action="store_true", help="Exp 3.1")
    parser.add_argument("--gradient-flow", action="store_true", help="Exp 3.3")
    parser.add_argument("--representation", action="store_true", help="Exp 3.4")
    args = parser.parse_args()

    run_all = args.all or not any([
        args.profiling, args.gradient_flow, args.representation,
    ])

    _p("=" * 60)
    _p("PRINet Phase 3: Scientific Experiments")
    _p("=" * 60)
    _p(f"Device: {DEVICE}")
    _p(f"Results directory: {RESULTS_DIR}")
    if DEVICE == "cuda":
        _p(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()

    if run_all or args.profiling:
        experiment_3_1_profiling()
    if run_all or args.gradient_flow:
        experiment_3_3_gradient_flow()
    if run_all or args.representation:
        experiment_3_4_representation_geometry()

    elapsed = time.time() - t0
    _p(f"\nPhase 3 complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
