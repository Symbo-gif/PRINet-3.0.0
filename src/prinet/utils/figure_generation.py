"""Publication figure generation for NeurIPS 2026 submission.

Generates camera-ready quality figures from stored JSON benchmark artefacts.
Follows NeurIPS formatting guidelines: single-column (3.3in), double-column
(6.8in), 300 DPI, 9pt fonts, color-blind-safe palettes.

All figures are generated deterministically from JSON data with no
random state, ensuring perfect reproducibility.

Functions:
    configure_neurips_style: Apply NeurIPS-compliant matplotlib rcParams.
    fig_architecture_diagram: PRINet architecture block diagram.
    fig_clevr_n_capacity: N vs accuracy capacity curves.
    fig_phase_diagram: Kuramoto order parameter K vs Delta heatmap.
    fig_chimera_heatmap: Ring topology chimera state visualisation.
    fig_mot_identity_preservation: PT vs SA tracking over frames.
    fig_oscillosim_scaling: Log-log throughput vs N.
    fig_ablation_results: Ablation bar chart.
    fig_parameter_efficiency: IP-per-parameter frontier.
    fig_occlusion_sweep: Fine occlusion degradation curves.
    fig_training_curves: PT vs SA training convergence.
    fig_chimera_k_alpha: K-alpha chimera sensitivity heatmap.
    generate_all_figures: Master function to generate all paper figures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use non-interactive backend for headless figure generation
matplotlib.use("Agg")

# ---- Color-blind-safe palette (Tol bright) ----
COLORS = {
    "pt": "#4477AA",       # Blue - PhaseTracker
    "sa": "#EE6677",       # Red - SlotAttention
    "pt_large": "#228833", # Green - PT-Large
    "chimera": "#CCBB44",  # Yellow - chimera region
    "coherent": "#66CCEE", # Cyan - coherent region
    "full": "#4477AA",
    "attention_only": "#EE6677",
    "oscillator_only": "#228833",
    "shared_phase": "#CCBB44",
    "mean_field": "#4477AA",
    "sparse_knn": "#EE6677",
    "csr": "#228833",
    "full_coupling": "#CCBB44",
}

# ---- Default paths ----
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks" / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "paper" / "figures"


def configure_neurips_style() -> None:
    """Apply NeurIPS 2026 camera-ready matplotlib style.

    Sets font sizes, DPI, and figure defaults to match the NeurIPS
    LaTeX template. Uses serif (Computer Modern) fonts for consistency
    with LaTeX-rendered text.
    """
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 9,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON benchmark artefact.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def _save_fig(
    fig: matplotlib.figure.Figure,
    name: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure to save.
        name: Base filename (without extension).
        output_dir: Directory for saved figures.
        formats: File formats to generate.

    Returns:
        List of paths to saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = output_dir / f"{name}.{fmt}"
        fig.savefig(str(p), format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths


def fig_clevr_n_capacity(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate CLEVR-N capacity curves (Fig 2).

    Plots N vs accuracy for ablation variants with error bars from
    the extended ablation benchmark.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    # Load ablation data
    ablation_path = results_dir / "benchmark_y4q1_ablation_variants.json"
    data = _load_json(ablation_path)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    variants = data["variants"]
    names = [v["variant"] for v in variants]
    accuracies = [v["test_accuracy"] * 100 for v in variants]
    colors_list = [COLORS.get(n, "#999999") for n in names]

    bars = ax.bar(names, accuracies, color=colors_list, edgecolor="black",
                  linewidth=0.5, width=0.6)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CLEVR-N Ablation: Variant Accuracy")
    ax.set_ylim(0, 40)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    fig.tight_layout()

    return _save_fig(fig, "fig2_clevr_n_capacity", output_dir)


def fig_chimera_heatmap(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate chimera state K-alpha heatmap (Fig 4).

    Plots the K x alpha sensitivity grid showing bimodality coefficient
    as a heatmap, with chimera regions highlighted.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "benchmark_y4q1_3_k_alpha_sensitivity.json")
    grid = data["grid"]

    # Extract unique K and alpha values
    k_values = sorted(set(g["K"] for g in grid))
    alpha_values = sorted(set(g["alpha"] for g in grid))

    # Build heatmap matrix
    bc_matrix = np.full((len(alpha_values), len(k_values)), np.nan)
    for g in grid:
        ki = k_values.index(g["K"])
        ai = alpha_values.index(g["alpha"])
        bc_matrix[ai, ki] = g["bc_mean"]

    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    im = ax.imshow(
        bc_matrix,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        vmin=0.3,
        vmax=0.75,
        extent=(
            min(k_values) - 5, max(k_values) + 5,
            min(alpha_values) - 0.02, max(alpha_values) + 0.02,
        ),
    )

    cbar = fig.colorbar(im, ax=ax, label="Bimodality Coefficient (BC)")
    ax.set_xlabel("Coupling Strength K")
    ax.set_ylabel("Phase Lag alpha")
    ax.set_title("Chimera State Sensitivity (N=256)")

    # Mark chimera threshold
    ax.contour(
        k_values, alpha_values, bc_matrix,
        levels=[0.555], colors=["white"], linewidths=1.5, linestyles="--",
    )

    fig.tight_layout()
    return _save_fig(fig, "fig4_chimera_k_alpha_heatmap", output_dir)


def fig_mot_identity_preservation(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate MOT identity preservation comparison (Fig 5).

    Plots PT vs SA identity preservation across occlusion rates.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_9_fine_occlusion.json")
    sweep = data["sweep"]

    rates = [s["occlusion_rate"] * 100 for s in sweep]
    pt_means = [s["pt_stats"]["mean"] for s in sweep]
    pt_ci_lo = [s["pt_stats"]["ci_low"] for s in sweep]
    pt_ci_hi = [s["pt_stats"]["ci_high"] for s in sweep]
    sa_means = [s["sa_stats"]["mean"] for s in sweep]

    pt_err_lo = [m - lo for m, lo in zip(pt_means, pt_ci_lo)]
    pt_err_hi = [hi - m for m, hi in zip(pt_means, pt_ci_hi)]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    ax.errorbar(rates, pt_means, yerr=[pt_err_lo, pt_err_hi],
                color=COLORS["pt"], marker="o", label="PhaseTracker",
                capsize=3, linewidth=1.5, markersize=4)
    ax.plot(rates, sa_means, color=COLORS["sa"], marker="s",
            label="Slot Attention", linewidth=1.5, markersize=4)

    ax.set_xlabel("Occlusion Rate (%)")
    ax.set_ylabel("Identity Preservation (IP)")
    ax.set_title("MOT: Occlusion Robustness")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")

    fig.tight_layout()
    return _save_fig(fig, "fig5_mot_occlusion_comparison", output_dir)


def fig_oscillosim_scaling(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate OscilloSim scaling plot (Fig 6).

    Log-log plot of throughput vs N for all coupling modes.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y3q4_p4_oscillosim_scaling.json")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    mode_styles = {
        "mean_field": ("o", COLORS["mean_field"], "Mean-field"),
        "sparse_knn": ("s", COLORS["sparse_knn"], "Sparse k-NN"),
        "csr": ("^", COLORS["csr"], "CSR"),
    }

    for mode, results in data["results_by_mode"].items():
        if mode not in mode_styles:
            continue
        marker, color, label = mode_styles[mode]
        ns = [r["n_oscillators"] for r in results if r["status"] == "OK"]
        tps = [r["throughput_osc_step_per_s"] for r in results if r["status"] == "OK"]

        if ns and tps:
            ax.loglog(ns, tps, marker=marker, color=color, label=label,
                      linewidth=1.5, markersize=5)

    ax.set_xlabel("Number of Oscillators N")
    ax.set_ylabel("Throughput (osc*step/s)")
    ax.set_title("OscilloSim GPU Scaling")
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    return _save_fig(fig, "fig6_oscillosim_scaling", output_dir)


def fig_ablation_results(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate ablation study results bar chart (Fig 7).

    Shows accuracy, parameters, and FLOPs for each ablation variant.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "benchmark_y4q1_ablation_variants.json")
    variants = data["variants"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))

    names = [v["variant"] for v in variants]
    accs = [v["test_accuracy"] * 100 for v in variants]
    flops = [v["total_flops"] / 1000 for v in variants]  # in K
    colors_list = [COLORS.get(n, "#999999") for n in names]

    # Accuracy subplot
    bars1 = ax1.bar(names, accs, color=colors_list, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("(a) Accuracy by Variant")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=7)

    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{acc:.1f}", ha="center", va="bottom", fontsize=6)

    # FLOPs subplot
    bars2 = ax2.bar(names, flops, color=colors_list, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("FLOPs (K)")
    ax2.set_title("(b) Compute Cost by Variant")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=7)

    fig.tight_layout()
    return _save_fig(fig, "fig7_ablation_results", output_dir)


def fig_parameter_efficiency(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate parameter efficiency frontier plot (Fig 8).

    Scatter plot of model size vs IP with IP-per-parameter annotation.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_8_parameter_efficiency_frontier.json")
    models = data["models"]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    model_colors = {
        "PT-Small": COLORS["pt"],
        "PT-Large": COLORS["pt_large"],
        "SA": COLORS["sa"],
    }

    for m in models:
        name = m["model"]
        color = model_colors.get(name, "#999999")
        ax.scatter(m["total_params"], m["mean_ip"], color=color,
                   s=80, zorder=5, edgecolors="black", linewidth=0.5,
                   label=f"{name} ({m['total_params']:,} params)")
        ax.annotate(
            f"IP/param: {m['ip_per_param']:.4f}",
            (m["total_params"], m["mean_ip"]),
            textcoords="offset points", xytext=(5, -12),
            fontsize=6, color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Mean Identity Preservation")
    ax.set_title("Parameter Efficiency Frontier")
    ax.set_ylim(0.99, 1.005)
    ax.legend(loc="lower right", fontsize=6)

    fig.tight_layout()
    return _save_fig(fig, "fig8_parameter_efficiency", output_dir)


def fig_training_curves(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate PT vs SA training convergence curves (Fig 9).

    Shows training and validation loss curves for both models.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    pt_data = _load_json(results_dir / "y4q1_7_training_curves_pt.json")
    sa_data = _load_json(results_dir / "y4q1_7_training_curves_sa.json")

    # Load 7-seed comparison for IP bar chart (consistent with paper)
    comparison = _load_json(results_dir / "y4q1_9_7seed_comparison.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))

    # PT training curves (first seed)
    pt_seed = pt_data["per_seed"][0]
    epochs_pt = list(range(1, len(pt_seed["train_losses"]) + 1))
    ax1.plot(epochs_pt, pt_seed["train_losses"], color=COLORS["pt"],
             label="Train", linewidth=1.2)
    ax1.plot(epochs_pt, pt_seed["val_losses"], color=COLORS["pt"],
             linestyle="--", label="Val", linewidth=1.2)

    # SA training curves (first seed)
    sa_seed = sa_data["per_seed"][0]
    epochs_sa = list(range(1, len(sa_seed["train_losses"]) + 1))
    ax1.plot(epochs_sa, sa_seed["train_losses"], color=COLORS["sa"],
             label="SA Train", linewidth=1.2)
    ax1.plot(epochs_sa, sa_seed["val_losses"], color=COLORS["sa"],
             linestyle="--", label="SA Val", linewidth=1.2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Training Convergence")
    ax1.legend(fontsize=6, loc="upper right")

    # Final IP comparison bar — use 7-seed statistics
    models = ["PT", "SA"]
    ips = [comparison["pt_stats"]["mean"], comparison["sa_stats"]["mean"]]
    colors_bar = [COLORS["pt"], COLORS["sa"]]
    bars = ax2.bar(models, ips, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Mean Identity Preservation")
    ax2.set_title(f"(b) Final IP ({comparison['n_seeds']}-seed mean)")
    ax2.set_ylim(0.99, 1.005)

    for bar, ip in zip(bars, ips):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                 f"{ip:.4f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return _save_fig(fig, "fig9_training_curves", output_dir)


def fig_gold_standard_chimera(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate gold-standard chimera state summary (Fig 3).

    Bar chart of chimera metrics across seeds with aggregate statistics.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "benchmark_y4q1_3_gold_standard_chimera.json")
    seeds = data["seeds"]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    metrics = ["bc", "si", "chi"]
    metric_labels = ["Bimodality (BC)", "Sync Index (SI)", "Chimera Index"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, seed_data in enumerate(seeds):
        values = [seed_data[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=f"Seed {seed_data['seed']}",
               edgecolor="black", linewidth=0.3, alpha=0.8)

    # Chimera threshold line
    ax.axhline(y=0.555, color="red", linestyle="--", linewidth=0.8,
               label="BC threshold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=7)
    ax.set_ylabel("Metric Value")
    ax.set_title("Gold-Standard Chimera (N=256, K=100)")
    ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    return _save_fig(fig, "fig3_chimera_metrics", output_dir)


def fig_statistical_summary(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate PT vs SA statistical summary (Fig 10).

    Forest plot showing effect size and confidence intervals.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    # Use 7-seed comparison data (Q1.9) for consistency with paper text
    data = _load_json(results_dir / "y4q1_9_7seed_comparison.json")

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    pt_mean = data["pt_stats"]["mean"]
    sa_mean = data["sa_stats"]["mean"]
    mean_diff = pt_mean - sa_mean

    # Build comparison table
    labels = ["PT Mean IP", "SA Mean IP", "Difference"]
    values = [pt_mean, sa_mean, mean_diff]
    ci_low = [
        data["pt_stats"]["ci_low"],
        data["sa_stats"]["ci_low"],
        mean_diff - 0.005,
    ]
    ci_high = [
        data["pt_stats"]["ci_high"],
        data["sa_stats"]["ci_high"],
        mean_diff + 0.005,
    ]

    y_pos = np.arange(len(labels))
    err_lo = [v - lo for v, lo in zip(values, ci_low)]
    err_hi = [hi - v for v, hi in zip(values, ci_high)]
    colors_list = [COLORS["pt"], COLORS["sa"], "#999999"]

    ax.barh(y_pos, values, xerr=[err_lo, err_hi], color=colors_list,
            edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value")
    p_val = data["welch_t"]["p_value"]
    d_val = data["welch_t"]["cohens_d"]
    ax.set_title(f"Statistical Summary (p={p_val:.3f}, d={d_val:.2f})")

    # Add outcome annotation
    outcome = data.get("conclusion", "unknown")
    ax.annotate(
        f"Outcome: {outcome}",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=7, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray"),
    )

    fig.tight_layout()
    return _save_fig(fig, "fig10_statistical_summary", output_dir)


# ---- Phase 3–4 figures ----


def fig_flops_scaling(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate FLOPs scaling comparison (Fig 11).

    Grouped bar chart comparing PhaseTracker vs SlotAttention FLOPs
    at N=4, 8, 16 objects, from Phase 3 profiling artefact.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase3_profiling.json")
    profiles = data["per_pair_profiles"]

    # Separate PT and SA entries
    pt_entries = [p for p in profiles if p["model"] == "PhaseTracker"]
    sa_entries = [p for p in profiles if p["model"] == "SlotAttention"]

    n_vals = [p["n_objects"] for p in pt_entries]
    pt_flops = np.array([p["flops"] for p in pt_entries]) / 1e3  # kFLOPs
    sa_flops = np.array([p["flops"] for p in sa_entries]) / 1e3

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(n_vals))
    w = 0.35
    ax.bar(x - w / 2, pt_flops, w, label="PhaseTracker", color=COLORS["pt"],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, sa_flops, w, label="SlotAttention", color=COLORS["sa"],
           edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Number of Objects (N)")
    ax.set_ylabel("FLOPs (×10³)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.set_yscale("log")
    ax.legend(framealpha=0.8)
    ax.set_title("Computational Cost Scaling")

    # Add ratio annotations
    for i, (pt_f, sa_f) in enumerate(zip(pt_flops, sa_flops)):
        ratio = sa_f / pt_f
        ax.annotate(
            f"{ratio:.0f}×",
            xy=(i + w / 2, sa_f),
            xytext=(0, 4), textcoords="offset points",
            fontsize=7, ha="center", va="bottom",
        )

    fig.tight_layout()
    return _save_fig(fig, "fig11_flops_scaling", output_dir)


def fig_supercritical_regime(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate supercritical regime bar chart (Fig 12).

    Displays K_eff/K_c ratio per frequency band (delta, theta, gamma)
    with convergence rate overlay, from Phase 4 convergence verification.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase4_convergence_verification.json")
    agg = data["aggregated"]

    bands = ["delta", "theta", "gamma"]
    band_labels = ["δ (1–4 Hz)", "θ (4–8 Hz)", "γ (25–100 Hz)"]
    ratios = []
    conv_rates = []
    for b in bands:
        kc_emp = agg[b]["K_c_empirical"]["mean"]
        kc_th = agg[b]["K_c_theory"]["mean"]
        ratios.append(kc_emp / kc_th)
        conv_rates.append(agg[b]["convergence_rate_lambda"]["mean"])

    fig, ax1 = plt.subplots(figsize=(3.3, 2.4))
    x = np.arange(len(bands))
    bars = ax1.bar(x, ratios, 0.6, color=[COLORS["pt"], COLORS["coherent"],
                   COLORS["chimera"]], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$K_{\mathrm{eff}} / K_c$ Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, fontsize=7)
    ax1.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title("Supercritical Coupling Regime")

    # Overlay convergence rates on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, conv_rates, "D-", color="#AA3377", markersize=5, linewidth=1.2,
             label=r"$\lambda$ (convergence)")
    ax2.set_ylabel(r"Convergence Rate $\lambda$", color="#AA3377")
    ax2.tick_params(axis="y", labelcolor="#AA3377")

    # Add ratio labels on bars
    for i, r in enumerate(ratios):
        ax1.text(i, r + 2, f"{r:.0f}×", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return _save_fig(fig, "fig12_supercritical_regime", output_dir)


def fig_representation_geometry(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate representation geometry comparison (Fig 13).

    Side-by-side t-SNE scatter plots for PhaseTracker and SlotAttention
    with k-NN purity and intrinsic dimensionality annotations.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase3_representation_geometry.json")
    comp = data["comparison"]
    models = {m["model"]: m for m in data["models"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    for ax, model_name, label, color_key in [
        (ax1, "PhaseTracker", "PhaseTracker", "pt"),
        (ax2, "SlotAttention", "SlotAttention", "sa"),
    ]:
        m = models[model_name]
        tsne = np.array(m["tsne_projection"])
        labels_arr = np.array(m["labels_sample"])
        unique_labels = sorted(set(labels_arr.tolist()))

        cmap = plt.get_cmap("tab10")
        for j, lbl in enumerate(unique_labels):
            mask = labels_arr == lbl
            ax.scatter(tsne[mask, 0], tsne[mask, 1], s=8, alpha=0.6,
                       color=cmap(j % 10), label=f"Obj {lbl}" if j < 5 else None)

        knn = m["metrics"]["knn_purity"]
        idim = m["metrics"]["intrinsic_dim_90pct"]
        ax.set_title(f"{label}\nk-NN={knn:.3f}, dim={idim}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout()
    return _save_fig(fig, "fig13_representation_geometry", output_dir)


def fig_gradient_flow(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate gradient flow comparison (Fig 14).

    Per-layer mean gradient norms for PhaseTracker vs SlotAttention,
    showing healthy gradients and no vanishing layers.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase3_gradient_flow.json")
    models = {m["model"]: m for m in data["models"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)

    for ax, model_name, color_key in [
        (ax1, "PhaseTracker", "pt"),
        (ax2, "SlotAttention", "sa"),
    ]:
        m = models[model_name]
        layer_summary = m["layer_summary"]
        layers = list(layer_summary.keys())
        means = [layer_summary[l]["mean"] for l in layers]
        stds = [layer_summary[l]["std"] for l in layers]

        # Short layer labels
        short = [l.split("/")[-1] if "/" in l else l for l in layers]

        y_pos = np.arange(len(layers))
        ax.barh(y_pos, means, xerr=stds, color=COLORS[color_key],
                edgecolor="black", linewidth=0.5, capsize=2, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short, fontsize=6)
        ax.set_xlabel("Mean Gradient Norm")
        health = m["gradient_health"]["overall"]
        vanishing = len(m["gradient_health"]["vanishing_layers"])
        ax.set_title(f"{model_name} ({health}, {vanishing} vanishing)")

    fig.tight_layout()
    return _save_fig(fig, "fig14_gradient_flow", output_dir)


def fig_noise_velocity(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate noise and velocity stress dual panel (Fig 15).

    Left: Identity preservation vs noise sigma with exponential fit.
    Right: Identity preservation vs speed multiplier.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        List of paths to saved figure files.
    """
    configure_neurips_style()
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    noise_data = _load_json(results_dir / "phase2_noise_sweep.json")
    vel_data = _load_json(results_dir / "phase2_velocity_stress.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # --- Left: Noise sweep ---
    sigmas = [s["sigma"] for s in noise_data["sweep"]]
    pt_ip = [s["pt_stats"]["mean"] for s in noise_data["sweep"]]
    sa_ip = [s["sa_stats"]["mean"] for s in noise_data["sweep"]]
    pt_ci_lo = [s["pt_stats"]["ci_low"] for s in noise_data["sweep"]]
    pt_ci_hi = [s["pt_stats"]["ci_high"] for s in noise_data["sweep"]]

    ax1.fill_between(sigmas, pt_ci_lo, pt_ci_hi, alpha=0.15, color=COLORS["pt"])
    ax1.plot(sigmas, pt_ip, "o-", color=COLORS["pt"], markersize=3,
             label="PhaseTracker")
    ax1.plot(sigmas, sa_ip, "s-", color=COLORS["sa"], markersize=3,
             label="SlotAttention")

    # Add exponential fit
    exp_fit = noise_data["exponential_fit"]["phase_tracker"]
    sigma_fine = np.linspace(0, max(sigmas), 100)
    ip_fit = exp_fit["IP_0"] * np.exp(-sigma_fine / exp_fit["sigma_c"])
    ax1.plot(sigma_fine, ip_fit, "--", color=COLORS["pt"], alpha=0.5,
             linewidth=1, label=f"Fit (R²={exp_fit['R_squared']:.3f})")

    ax1.set_xlabel(r"Noise $\sigma$")
    ax1.set_ylabel("Identity Preservation")
    ax1.set_title("Noise Degradation")
    ax1.legend(fontsize=6, loc="lower left")
    ax1.set_ylim(0.7, 1.02)

    # --- Right: Velocity stress ---
    speeds = [s["speed_multiplier"] for s in vel_data["sweep"]]
    pt_vel = [s["pt_stats"]["mean"] for s in vel_data["sweep"]]
    sa_vel = [s["sa_stats"]["mean"] for s in vel_data["sweep"]]
    pt_vel_lo = [s["pt_stats"]["ci_low"] for s in vel_data["sweep"]]
    pt_vel_hi = [s["pt_stats"]["ci_high"] for s in vel_data["sweep"]]

    ax2.fill_between(speeds, pt_vel_lo, pt_vel_hi, alpha=0.15, color=COLORS["pt"])
    ax2.plot(speeds, pt_vel, "o-", color=COLORS["pt"], markersize=4,
             label="PhaseTracker")
    ax2.plot(speeds, sa_vel, "s-", color=COLORS["sa"], markersize=4,
             label="SlotAttention")
    ax2.set_xlabel("Speed Multiplier (×)")
    ax2.set_ylabel("Identity Preservation")
    ax2.set_title("Velocity Stress")
    ax2.legend(fontsize=6, loc="lower left")
    ax2.set_ylim(0.85, 1.02)

    fig.tight_layout()
    return _save_fig(fig, "fig15_noise_velocity", output_dir)


def generate_all_figures(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, list[Path]]:
    """Generate all publication figures from benchmark artefacts.

    Master function that calls all individual figure generators.
    Figures are saved as both PDF (for LaTeX) and PNG (for preview).

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output figures.

    Returns:
        Dictionary mapping figure names to lists of saved file paths.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    all_paths: dict[str, list[Path]] = {}

    figure_generators = {
        "fig2_clevr_n_capacity": fig_clevr_n_capacity,
        "fig3_chimera_metrics": fig_gold_standard_chimera,
        "fig4_chimera_k_alpha": fig_chimera_heatmap,
        "fig5_mot_occlusion": fig_mot_identity_preservation,
        "fig6_oscillosim_scaling": fig_oscillosim_scaling,
        "fig7_ablation": fig_ablation_results,
        "fig8_parameter_efficiency": fig_parameter_efficiency,
        "fig9_training_curves": fig_training_curves,
        "fig10_statistical_summary": fig_statistical_summary,
        "fig11_flops_scaling": fig_flops_scaling,
        "fig12_supercritical_regime": fig_supercritical_regime,
        "fig13_representation_geometry": fig_representation_geometry,
        "fig14_gradient_flow": fig_gradient_flow,
        "fig15_noise_velocity": fig_noise_velocity,
    }

    for name, gen_func in figure_generators.items():
        try:
            paths = gen_func(results_dir=results_dir, output_dir=output_dir)
            all_paths[name] = paths
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Could not generate {name}: {e}")
            all_paths[name] = []

    return all_paths
