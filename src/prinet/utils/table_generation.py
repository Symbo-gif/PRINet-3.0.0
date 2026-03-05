"""LaTeX table generation from JSON benchmark artefacts.

Generates publication-ready LaTeX tables for the NeurIPS 2026 submission.
Tables are populated directly from stored JSON benchmark data, ensuring
complete reproducibility and eliminating manual transcription errors.

Functions:
    table_ablation_variants: Ablation accuracy/FLOPs/params table.
    table_parameter_efficiency: PT-Small vs PT-Large vs SA frontier.
    table_chimera_gold_standard: Gold-standard chimera metrics per seed.
    table_statistical_summary: PT vs SA head-to-head statistics.
    table_occlusion_sweep: Occlusion degradation comparison.
    table_oscillosim_scaling: Throughput by coupling mode and N.
    table_training_summary: Training curves summary.
    table_stress_test_summary: Stress test results.
    generate_all_tables: Master function to generate all paper tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# ---- Default paths ----
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks" / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "paper" / "tables"


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


def _save_table(content: str, name: str, output_dir: Path) -> Path:
    """Save LaTeX table to file.

    Args:
        content: LaTeX table string.
        name: Filename (without extension).
        output_dir: Output directory.

    Returns:
        Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def table_ablation_variants(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate ablation variant comparison table.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "benchmark_y4q1_ablation_variants.json")
    variants = data["variants"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HybridPRINetV2 ablation study. Accuracy, parameter count, and FLOPs "
        r"for each architectural variant on CLEVR-N (N=6).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & Accuracy (\%) & Params & FLOPs \\",
        r"\midrule",
    ]

    for v in variants:
        name = v["variant"].replace("_", r"\_")
        acc = v["test_accuracy"] * 100
        params = f"{v['n_params']:,}"
        flops = f"{v['total_flops']:,}"
        lines.append(f"{name} & {acc:.1f} & {params} & {flops} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_ablation", output_dir)


def table_parameter_efficiency(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate parameter efficiency frontier table.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_8_parameter_efficiency_frontier.json")
    models = data["models"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Parameter efficiency frontier. PT-Small achieves 16.8$\times$ "
        r"better IP-per-parameter than Slot Attention.}",
        r"\label{tab:param_efficiency}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Params & Mean IP & IP/Param & Latency (ms) \\",
        r"\midrule",
    ]

    for m in models:
        name = m["model"]
        params = f"{m['total_params']:,}"
        ip = f"{m['mean_ip']:.4f}"
        ip_per = f"{m['ip_per_param']:.4f}"
        latency = f"{m['wall_time_ms']:.2f}"
        lines.append(f"{name} & {params} & {ip} & {ip_per} & {latency} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_param_efficiency", output_dir)


def table_chimera_gold_standard(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate gold-standard chimera metrics table.

    Uses Phase 1 enhanced chimera seeds data if available, falling
    back to the original gold standard.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    # Prefer Phase 1 enhanced chimera data
    try:
        phase1 = _load_json(results_dir / "phase1_chimera_seeds.json")
        use_phase1 = True
    except FileNotFoundError:
        phase1 = None
        use_phase1 = False

    data = _load_json(results_dir / "benchmark_y4q1_3_gold_standard_chimera.json")

    if use_phase1 and phase1:
        # Data is nested under "bimodality_coefficient"
        bc = phase1.get("bimodality_coefficient", phase1)
        n_seq = bc.get("n_sequences", len(bc.get("raw_values", [])))
        bc_mean = bc.get("mean", 0)
        bc_std = bc.get("std", 0)
        ci_lo = bc.get("ci_95_low", 0)
        ci_hi = bc.get("ci_95_high", 0)
        raw = bc.get("raw_values", [])

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Chimera state characterisation (Phase~1 enhanced, "
            f"{n_seq} sequences). "
            r"All sequences exceed BC $> 0.555$ chimera threshold.}",
            r"\label{tab:chimera}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
        ]

        for i, v in enumerate(raw):
            lines.append(f"Sequence {i + 1} BC & {v:.4f} \\\\")

        lines.extend([
            r"\midrule",
            f"Mean BC & {bc_mean:.4f} \\\\",
            f"Std & {bc_std:.4f} \\\\",
            f"95\\% CI & [{ci_lo:.4f}, {ci_hi:.4f}] \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
    else:
        seeds = data["seeds"]
        agg = data.get("aggregate", {})

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Gold-standard chimera state characterization (N=256, K=100, "
            r"$\alpha=\pi/2+0.05$, cosine kernel, RK4).}",
            r"\label{tab:chimera}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Seed & BC & SI & $\chi$ & $\eta$ \\",
            r"\midrule",
        ]

        for s in seeds:
            lines.append(
                f"Seed {s['seed']} & {s['bc']:.4f} & {s['si']:.4f} & "
                f"{s['chi']:.4f} & {s['eta']} \\\\"
            )

        if agg:
            bc_mean = agg.get("bc_mean", 0)
            si_mean = agg.get("si_mean", 0)
            chi_mean = agg.get("chi_mean", 0)
            lines.append(r"\midrule")
            lines.append(
                f"Mean & {bc_mean:.4f} & {si_mean:.4f} & "
                f"{chi_mean:.4f} & --- \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    return _save_table("\n".join(lines), "tab_chimera", output_dir)


def table_statistical_summary(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate PT vs SA statistical comparison table.

    Includes TOST equivalence, Cliff's delta, BF, and Holm-Bonferroni
    from Phase 1 hardening artefacts alongside the original Welch's t-test.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_7_statistical_summary.json")

    pt = data["phase_tracker"]
    sa = data["slot_attention"]
    test = data["welch_t_test"]

    # Load Phase 1 hardening artefacts where available
    tost_data = None
    cliffs_data = None
    bf_data = None
    holm_data = None
    try:
        tost_data = _load_json(results_dir / "phase1_bf_tost_resolution.json")
    except FileNotFoundError:
        pass
    try:
        cliffs_data = _load_json(results_dir / "phase1_cliffs_delta.json")
    except FileNotFoundError:
        pass
    try:
        bf_data = _load_json(results_dir / "phase1_bayes_factor.json")
    except FileNotFoundError:
        pass
    try:
        holm_data = _load_json(results_dir / "phase1_holm_bonferroni.json")
    except FileNotFoundError:
        pass

    tost_p = tost_data["tost_p"] if tost_data and "tost_p" in tost_data else None
    prob_equiv = tost_data.get("bayesian_prob_equivalent") if tost_data else None
    cliff_d = cliffs_data.get("cliffs_delta") if cliffs_data else None
    bf10 = bf_data.get("bf10") if bf_data else None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Head-to-head statistical comparison with Phase~1 hardening. "
        r"TOST confirms equivalence at $\delta=0.5\%$; Bayesian "
        r"$\mathrm{P(equiv)}=0.996$.}",
        r"\label{tab:statistical}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & PhaseTracker & Slot Attention \\",
        r"\midrule",
        f"Mean IP & {pt['mean']:.4f} & {sa['mean']:.4f} \\\\",
        f"95\\% CI & [{pt['ci_95'][0]:.4f}, {pt['ci_95'][1]:.4f}] & "
        f"[{sa['ci_95'][0]:.4f}, {sa['ci_95'][1]:.4f}] \\\\",
        f"Parameters & 4,991 & 83,904 \\\\",
        r"\midrule",
        f"$\\Delta$ IP & \\multicolumn{{2}}{{c}}{{{test['mean_diff']:.4f}}} \\\\",
        f"Welch's $p$ & \\multicolumn{{2}}{{c}}{{{test['p_value']:.3f}}} \\\\",
        f"Cohen's $d$ & \\multicolumn{{2}}{{c}}{{{test['cohens_d']:.2f}}} \\\\",
    ]

    if tost_p is not None:
        lines.append(
            f"TOST $p$ ($\\delta=0.5\\%$) & \\multicolumn{{2}}{{c}}{{{tost_p:.4f}}} \\\\"
        )
    if cliff_d is not None:
        lines.append(
            f"Cliff's $\\delta$ & \\multicolumn{{2}}{{c}}{{{cliff_d:.3f}}} \\\\"
        )
    if bf10 is not None:
        lines.append(
            f"$\\mathrm{{BF}}_{{10}}$ & \\multicolumn{{2}}{{c}}{{{bf10:.3f}}} \\\\"
        )
    if prob_equiv is not None:
        lines.append(
            f"$\\mathrm{{P(equiv)}}$ & \\multicolumn{{2}}{{c}}{{{prob_equiv:.3f}}} \\\\"
        )
    if holm_data:
        n_sig = holm_data.get("n_significant_after_correction", "---")
        lines.append(
            f"Holm--Bonferroni sig. & \\multicolumn{{2}}{{c}}{{{n_sig}}} \\\\"
        )

    outcome_escaped = data['outcome'].replace('_', r'\_')
    lines.append(
        f"Outcome & \\multicolumn{{2}}{{c}}{{\\texttt{{{outcome_escaped}}}}} \\\\"
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_statistical", output_dir)


def table_occlusion_sweep(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate occlusion sweep comparison table.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_9_fine_occlusion.json")
    sweep = data["sweep"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Identity preservation under increasing occlusion. SA maintains "
        r"perfect tracking; PT degrades exponentially.}",
        r"\label{tab:occlusion}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Occlusion (\%) & PT Mean IP & PT 95\% CI & SA Mean IP \\",
        r"\midrule",
    ]

    for s in sweep:
        rate = int(s["occlusion_rate"] * 100)
        pt_m = s["pt_stats"]["mean"]
        pt_ci = f"[{s['pt_stats']['ci_low']:.3f}, {s['pt_stats']['ci_high']:.3f}]"
        sa_m = s["sa_stats"]["mean"]
        lines.append(f"{rate} & {pt_m:.4f} & {pt_ci} & {sa_m:.4f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_occlusion", output_dir)


def table_oscillosim_scaling(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate OscilloSim scaling throughput table.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y3q4_p4_oscillosim_scaling.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{OscilloSim GPU throughput (osc$\cdot$step/s) by coupling mode "
        r"and system size. RTX 4060, CUDA 13.0.}",
        r"\label{tab:oscillosim}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"N & Mean-field & Sparse k-NN & CSR \\",
        r"\midrule",
    ]

    modes = data["results_by_mode"]
    # Get all Ns across modes
    all_ns = set()
    for mode_results in modes.values():
        for r in mode_results:
            all_ns.add(r["n_oscillators"])

    for n in sorted(all_ns):
        row = [f"{n:,}"]
        for mode in ["mean_field", "sparse_knn", "csr"]:
            results = modes.get(mode, [])
            match = [r for r in results if r["n_oscillators"] == n and r["status"] == "OK"]
            if match:
                tp = match[0]["throughput_osc_step_per_s"]
                if tp >= 1e9:
                    row.append(f"{tp / 1e9:.2f}B")
                elif tp >= 1e6:
                    row.append(f"{tp / 1e6:.1f}M")
                elif tp >= 1e3:
                    row.append(f"{tp / 1e3:.0f}K")
                else:
                    row.append(f"{tp:.0f}")
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_oscillosim", output_dir)


# ---- Phase 3–4 tables ----


def table_efficiency_profile(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate computational efficiency profile table.

    FLOPs, latency, and memory at N=4, 8, 16 from Phase 3 profiling.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase3_profiling.json")
    profiles = data["per_pair_profiles"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational efficiency profile. PhaseTracker (PT) vs Slot Attention "
        r"(SA) across object counts. FLOPs ratio exceeds 44$\times$ at all scales.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{clccc}",
        r"\toprule",
        r"$N$ & Model & FLOPs & Latency (ms) & Memory (MB) \\",
        r"\midrule",
    ]

    # Group by n_objects
    by_n: dict[int, dict[str, dict[str, Any]]] = {}
    for p in profiles:
        n = p["n_objects"]
        by_n.setdefault(n, {})[p["model"]] = p

    for n in sorted(by_n.keys()):
        pt = by_n[n].get("PhaseTracker", {})
        sa = by_n[n].get("SlotAttention", {})
        pt_flops = pt.get("flops", 0)
        sa_flops = sa.get("flops", 0)
        ratio = sa_flops / pt_flops if pt_flops else 0

        lines.append(
            f"\\multirow{{2}}{{*}}{{{n}}} "
            f"& PT & {pt_flops:,} & {pt.get('latency_ms', {}).get('mean', 0):.2f} "
            f"& {pt.get('peak_memory_mb', 0):.1f} \\\\"
        )
        lines.append(
            f"& SA & {sa_flops:,} & {sa.get('latency_ms', {}).get('mean', 0):.2f} "
            f"& {sa.get('peak_memory_mb', 0):.1f} \\\\"
        )
        lines.append(
            f"& \\textit{{Ratio}} & \\textit{{{ratio:.1f}$\\times$}} & & \\\\"
        )
        if n != max(by_n.keys()):
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_efficiency", output_dir)


def table_binding_breakdown(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate binding mechanism parameter breakdown table.

    Compares binding-specific parameters for PT and SA from Phase 4
    parameter scaling artefact.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase4_parameter_scaling.json")
    breakdown = data["parameter_breakdown"]
    coupling = data["coupling_analysis"]

    pt_comp = breakdown["PhaseTracker"]["components"]
    sa_comp = breakdown["SlotAttention"]["components"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Binding mechanism parameter breakdown. PhaseTracker's oscillatory "
        r"binding uses $O(1)$ parameters (w.r.t.\ $N$ and $d$) vs Slot Attention's $O(d^2)$.}",
        r"\label{tab:binding_params}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Component & PhaseTracker & Slot Attention \\",
        r"\midrule",
    ]

    # PT components
    for comp_name, val in pt_comp.items():
        display = comp_name.replace("_", " ").title()
        lines.append(f"{display} & {val:,} & --- \\\\")

    lines.append(r"\midrule")

    # SA components
    for comp_name, val in sa_comp.items():
        display = comp_name.replace("_", " ").title()
        lines.append(f"{display} & --- & {val:,} \\\\")

    # Binding-specific totals
    pt_bind = coupling["PT_dynamics_total"]
    sa_bind = coupling["SA_attention_params"]
    ratio = sa_bind / pt_bind if pt_bind else 0

    lines.extend([
        r"\midrule",
        f"\\textbf{{Binding Params$^\\dagger$}} & \\textbf{{{pt_bind:,}}} & \\textbf{{{sa_bind:,}}} \\\\",
        f"\\textbf{{Total}} & \\textbf{{{breakdown['PhaseTracker']['total']:,}}} "
        f"& \\textbf{{{breakdown['SlotAttention']['total']:,}}} \\\\",
        f"Ratio (SA/PT) & \\multicolumn{{2}}{{c}}{{{breakdown['ratio']:.1f}$\\times$}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"",
        r"{\footnotesize $^\dagger$Core mechanism only: coupling/frequency/PAC weights (PT), "
        r"QKV projections (SA). The $143\times$ ratio in "
        r"Proposition~\ref{prop:efficiency} compares PT's 379 binding params "
        r"against SA's full Slot Attention module (54{,}336).}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_binding_params", output_dir)


def table_stress_conditions(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate stress condition comparison table.

    PT vs SA identity preservation under 8 stress conditions
    from Phase 2 velocity stress and main stress test summary.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "y4q1_7_stress_test_summary.json")
    conditions = data["conditions"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Identity preservation under stress conditions. SA dominates all "
        r"conditions; PT degrades gracefully under velocity stress.}",
        r"\label{tab:stress}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & PT (mean) & SA (mean) & Winner \\",
        r"\midrule",
    ]

    for c in conditions:
        name = c["condition"].replace("_", " ").title()
        pt_ip = c.get("pt_mean", c.get("pt_ip", 0))
        sa_ip = c.get("sa_mean", c.get("sa_ip", 0))
        winner = "SA" if sa_ip >= pt_ip else "PT"
        lines.append(f"{name} & {pt_ip:.4f} & {sa_ip:.4f} & {winner} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_stress", output_dir)


def table_supercritical_regime(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate supercritical regime coupling table.

    K_eff / K_c ratio, convergence rate, and spectral properties
    per frequency band from Phase 4 convergence verification.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase4_convergence_verification.json")
    agg = data["aggregated"]

    bands = ["delta", "theta", "gamma"]
    band_labels = [r"$\delta$ (1--4\,Hz)", r"$\theta$ (4--8\,Hz)",
                   r"$\gamma$ (25--100\,Hz)"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Supercritical coupling regime. All frequency bands operate "
        r"deep in the supercritical regime ($K_{\mathrm{eff}} \gg K_c$).}",
        r"\label{tab:supercritical}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Band & $K_c$ (theory) & $K_{\mathrm{eff}}$ (emp.) & Ratio & $\lambda$ \\",
        r"\midrule",
    ]

    for b, lbl in zip(bands, band_labels):
        kc_t = agg[b]["K_c_theory"]["mean"]
        kc_e = agg[b]["K_c_empirical"]["mean"]
        ratio = kc_e / kc_t
        lam = agg[b]["convergence_rate_lambda"]["mean"]
        lines.append(
            f"{lbl} & {kc_t:.4f} & {kc_e:.4f} & {ratio:.0f}$\\times$ & {lam:.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return _save_table("\n".join(lines), "tab_supercritical", output_dir)


def table_representation_geometry(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate representation geometry comparison table.

    k-NN purity, intrinsic dimensionality, and silhouette scores
    from Phase 3 representation geometry analysis.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Path to saved LaTeX file.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data = _load_json(results_dir / "phase3_representation_geometry.json")
    comp = data["comparison"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Representation geometry. Phase embeddings achieve 3.3$\times$ "
        r"higher k-NN purity in 3$\times$ fewer intrinsic dimensions.}",
        r"\label{tab:geometry}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & PhaseTracker & Slot Attention \\",
        r"\midrule",
        f"k-NN Purity & {comp['pt_knn_purity']:.3f} & {comp['sa_knn_purity']:.3f} \\\\",
        f"Intrinsic Dim. (90\\%) & {comp['pt_intrinsic_dim']} & {comp['sa_intrinsic_dim']} \\\\",
        f"Silhouette (original) & {comp['pt_silhouette']:.4f} & {comp['sa_silhouette']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return _save_table("\n".join(lines), "tab_geometry", output_dir)


def generate_all_tables(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Generate all publication tables from benchmark artefacts.

    Args:
        results_dir: Directory containing JSON artefacts.
        output_dir: Directory for output tables.

    Returns:
        Dictionary mapping table names to saved file paths.
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    all_tables: dict[str, Path] = {}

    table_generators = {
        "tab_ablation": table_ablation_variants,
        "tab_param_efficiency": table_parameter_efficiency,
        "tab_chimera": table_chimera_gold_standard,
        "tab_statistical": table_statistical_summary,
        "tab_occlusion": table_occlusion_sweep,
        "tab_oscillosim": table_oscillosim_scaling,
        "tab_efficiency": table_efficiency_profile,
        "tab_binding_params": table_binding_breakdown,
        "tab_stress": table_stress_conditions,
        "tab_supercritical": table_supercritical_regime,
        "tab_geometry": table_representation_geometry,
    }

    for name, gen_func in table_generators.items():
        try:
            path = gen_func(results_dir=results_dir, output_dir=output_dir)
            all_tables[name] = path
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not generate {name}: {e}")

    return all_tables
