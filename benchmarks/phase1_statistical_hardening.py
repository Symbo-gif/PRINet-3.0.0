#!/usr/bin/env python
"""Phase 1: Statistical Hardening — Experiments 1.3–1.5.

Computes post-hoc statistical analyses on existing benchmark artefacts
without requiring new GPU training runs. Implements:

    1.3  Cliff's delta (non-parametric effect size) for all PT vs SA comparisons
    1.4  Bayes Factor analysis (BF_10) for primary IP equivalence claim
    1.5  Holm-Bonferroni correction across all stress-condition comparisons

Also includes chimera seed expansion analysis (1.1) and CLEVR-N multi-seed
analysis (1.2) by re-aggregating existing 7-seed data with additional metrics.

Usage:
    python benchmarks/phase1_statistical_hardening.py --all
    python benchmarks/phase1_statistical_hardening.py --cliffs-delta
    python benchmarks/phase1_statistical_hardening.py --bayes-factor
    python benchmarks/phase1_statistical_hardening.py --holm-bonferroni
    python benchmarks/phase1_statistical_hardening.py --chimera-seeds
    python benchmarks/phase1_statistical_hardening.py --clevr-seeds

Hardware: No GPU required (post-hoc analyses on stored JSON artefacts).

Reference:
    PRINet Paper Roadmap, Phase 1 (Statistical Hardening).

Authors:
    Michael Maillet
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FORCE_RERUN = os.environ.get("FORCE_RERUN", "").lower() in ("1", "true", "yes")

# Expanded 7-seed set aligned with y4q1_9
SEEDS_7 = (42, 123, 456, 789, 1024, 2048, 3072)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _p(*args: Any, **kwargs: Any) -> None:
    """ASCII-safe print with flush."""
    print(*args, flush=True, **kwargs)


def _save(name: str, data: dict[str, Any]) -> bool:
    """Save JSON artefact. Returns False if skipped (already exists).

    Args:
        name: Artefact name (without prefix/extension).
        data: Dictionary to serialise.

    Returns:
        True if the file was written, False if skipped.
    """
    path = RESULTS_DIR / f"phase1_{name}.json"
    if path.exists() and not FORCE_RERUN:
        _p(f"  [skip] {path.name} (exists, set FORCE_RERUN=1 to overwrite)")
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _p(f"  -> {path.name}")
    return True


def _load_artefact(name: str) -> Optional[dict[str, Any]]:
    """Load a JSON artefact from the results directory.

    Args:
        name: Filename (with extension).

    Returns:
        Parsed dict or None if the file doesn't exist.
    """
    path = RESULTS_DIR / name
    if not path.exists():
        _p(f"  [WARN] {name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def _preregister_hash(analysis_name: str, plan: dict[str, Any]) -> str:
    """Compute SHA-256 hash of pre-registered analysis plan.

    Args:
        analysis_name: Name of the analysis.
        plan: Dictionary describing the analysis parameters.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    canonical = json.dumps(plan, sort_keys=True)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _p(f"  Pre-registration hash ({analysis_name}): {h[:16]}...")
    return h


# =========================================================================
# 1.3 — Cliff's Delta (Non-parametric Effect Size)
# =========================================================================


def cliffs_delta(group_a: list[float], group_b: list[float]) -> float:
    """Compute Cliff's delta between two groups.

    Cliff's delta is a non-parametric effect size measure that quantifies
    the amount of difference between two groups of observations beyond
    p-values. It ranges from -1 to +1.

    .. math::

        \\delta = \\frac{\\#(a_i > b_j) - \\#(a_i < b_j)}{n_a \\cdot n_b}

    Args:
        group_a: First group of observations.
        group_b: Second group of observations.

    Returns:
        Cliff's delta value in [-1, +1].

    Raises:
        ValueError: If either group is empty.
    """
    if not group_a or not group_b:
        raise ValueError("Both groups must be non-empty for Cliff's delta.")
    n_a = len(group_a)
    n_b = len(group_b)
    count_more = 0
    count_less = 0
    for a in group_a:
        for b in group_b:
            if a > b:
                count_more += 1
            elif a < b:
                count_less += 1
    return (count_more - count_less) / (n_a * n_b)


def cliffs_delta_interpretation(d: float) -> str:
    """Interpret Cliff's delta magnitude.

    Uses Romano et al. (2006) thresholds:
        |d| < 0.147  → negligible
        |d| < 0.33   → small
        |d| < 0.474  → medium
        |d| >= 0.474 → large

    Args:
        d: Cliff's delta value.

    Returns:
        String interpretation.
    """
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def experiment_1_3_cliffs_delta() -> dict[str, Any]:
    """Experiment 1.3: Cliff's delta for all PT vs SA comparisons.

    Computes Cliff's delta (non-parametric effect size) alongside
    the existing Cohen's d for every PT vs SA comparison available
    in the benchmark artefacts. Addresses non-normality concerns.

    Returns:
        Dict with Cliff's delta for all comparison conditions.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 1.3: Cliff's Delta (Non-parametric Effect Size)")
    _p("=" * 60)

    comparisons: list[dict[str, Any]] = []

    # --- 7-seed head-to-head comparison ---
    data_7seed = _load_artefact("y4q1_9_7seed_comparison.json")
    if data_7seed is not None:
        pt_ips = data_7seed["pt_ips"]
        sa_ips = data_7seed["sa_ips"]
        cd = cliffs_delta(pt_ips, sa_ips)
        comparisons.append({
            "condition": "standard_7seed",
            "pt_values": pt_ips,
            "sa_values": sa_ips,
            "cliffs_delta": cd,
            "interpretation": cliffs_delta_interpretation(cd),
            "cohens_d": data_7seed.get("welch_t", {}).get("cohens_d", None),
            "p_value": data_7seed.get("welch_t", {}).get("p_value", None),
            "n_pt": len(pt_ips),
            "n_sa": len(sa_ips),
        })
        _p(f"  Standard 7-seed: Cliff's d = {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # --- Noise sweep (per sigma level) ---
    data_noise = _load_artefact("y4q1_8_noise_sweep.json")
    if data_noise is not None:
        for entry in data_noise.get("sweep", []):
            sigma = entry.get("sigma", "?")
            pt_ips_noise = entry.get("pt_ips", [])
            sa_ips_noise = entry.get("sa_ips", [])
            if pt_ips_noise and sa_ips_noise:
                cd = cliffs_delta(pt_ips_noise, sa_ips_noise)
                cohens_d = entry.get("cohens_d", None)
                comparisons.append({
                    "condition": f"noise_sigma_{sigma}",
                    "pt_values": pt_ips_noise,
                    "sa_values": sa_ips_noise,
                    "cliffs_delta": cd,
                    "interpretation": cliffs_delta_interpretation(cd),
                    "cohens_d": cohens_d,
                    "n_pt": len(pt_ips_noise),
                    "n_sa": len(sa_ips_noise),
                })
                _p(f"  Noise sigma={sigma}: Cliff's d = {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # --- 7-seed noise sweep ---
    data_noise_7 = _load_artefact("y4q1_9_7seed_noise.json")
    if data_noise_7 is not None:
        for entry in data_noise_7.get("sweep", []):
            sigma = entry.get("sigma", "?")
            # Try to extract raw per-seed data
            pt_stats = entry.get("pt_stats", {})
            sa_stats = entry.get("sa_stats", {})
            # If we only have summary stats, use bootstrap means
            pt_mean = pt_stats.get("mean", 0)
            sa_mean = sa_stats.get("mean", 0)
            # Note: we need raw values for Cliff's delta, summary is a fallback
            _p(f"  7seed noise sigma={sigma}: PT={pt_mean:.4f}, SA={sa_mean:.4f} (summary only)")

    # --- Fine occlusion sweep ---
    data_occ = _load_artefact("y4q1_9_fine_occlusion.json")
    if data_occ is not None:
        for entry in data_occ.get("sweep", []):
            occ_rate = entry.get("occlusion_rate", entry.get("occ_rate", "?"))
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if pt_vals and sa_vals:
                cd = cliffs_delta(pt_vals, sa_vals)
                comparisons.append({
                    "condition": f"occlusion_{occ_rate}",
                    "pt_values": pt_vals,
                    "sa_values": sa_vals,
                    "cliffs_delta": cd,
                    "interpretation": cliffs_delta_interpretation(cd),
                    "n_pt": len(pt_vals),
                    "n_sa": len(sa_vals),
                })
                _p(f"  Occlusion rate={occ_rate}: Cliff's d = {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # --- Object scaling ---
    data_obj = _load_artefact("y4q1_9_object_scaling.json")
    if data_obj is not None:
        for entry in data_obj.get("sweep", []):
            n_obj = entry.get("n_objects", "?")
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if pt_vals and sa_vals:
                cd = cliffs_delta(pt_vals, sa_vals)
                comparisons.append({
                    "condition": f"objects_{n_obj}",
                    "pt_values": pt_vals,
                    "sa_values": sa_vals,
                    "cliffs_delta": cd,
                    "interpretation": cliffs_delta_interpretation(cd),
                    "n_pt": len(pt_vals),
                    "n_sa": len(sa_vals),
                })
                _p(f"  N_objects={n_obj}: Cliff's d = {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # --- Sequence scaling ---
    data_seq = _load_artefact("y4q1_9_sequence_scaling.json")
    if data_seq is not None:
        for entry in data_seq.get("sweep", []):
            n_frames = entry.get("n_frames", "?")
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if pt_vals and sa_vals:
                cd = cliffs_delta(pt_vals, sa_vals)
                comparisons.append({
                    "condition": f"frames_{n_frames}",
                    "pt_values": pt_vals,
                    "sa_values": sa_vals,
                    "cliffs_delta": cd,
                    "interpretation": cliffs_delta_interpretation(cd),
                    "n_pt": len(pt_vals),
                    "n_sa": len(sa_vals),
                })
                _p(f"  Frames={n_frames}: Cliff's d = {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # --- Summary ---
    n_negligible = sum(1 for c in comparisons if c["interpretation"] == "negligible")
    n_small = sum(1 for c in comparisons if c["interpretation"] == "small")
    n_medium = sum(1 for c in comparisons if c["interpretation"] == "medium")
    n_large = sum(1 for c in comparisons if c["interpretation"] == "large")

    _p(f"\n  Summary: {len(comparisons)} comparisons")
    _p(f"    Negligible: {n_negligible}, Small: {n_small}, "
       f"Medium: {n_medium}, Large: {n_large}")

    result: dict[str, Any] = {
        "benchmark": "phase1_exp1_3_cliffs_delta",
        "description": "Cliff's delta (non-parametric effect size) for all PT vs SA comparisons",
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
        "summary": {
            "negligible": n_negligible,
            "small": n_small,
            "medium": n_medium,
            "large": n_large,
        },
        "interpretation_thresholds": {
            "negligible": "< 0.147",
            "small": "0.147 - 0.33",
            "medium": "0.33 - 0.474",
            "large": ">= 0.474",
        },
        "reference": "Romano, J. et al. (2006). Exploring methods for evaluating group differences on the NSSE.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("cliffs_delta", result)
    return result


# =========================================================================
# 1.4 — Bayes Factor Analysis
# =========================================================================


def _jeffreys_bayes_factor_t(
    t_stat: float, n_a: int, n_b: int, r: float = 0.7071
) -> float:
    """Compute approximate Bayes Factor (BF_10) using the JZS approach.

    Uses the Jeffreys-Zellner-Siow (JZS) Bayes factor for a two-sample
    t-test with a default Cauchy prior on effect size (Rouder et al., 2009).

    This implements a numerical approximation via the formula from
    Wetzels et al. (2011) for the BF_10 under a Cauchy prior.

    Args:
        t_stat: The t-statistic from a two-sample t-test.
        n_a: Sample size of group A.
        n_b: Sample size of group B.
        r: Scale parameter for the Cauchy prior (default sqrt(2)/2).

    Returns:
        BF_10 (evidence ratio for H1 over H0).
    """
    from scipy import stats as sp_stats
    from scipy import integrate

    n = n_a + n_b
    df = n - 2
    v = df

    # Effective sample size for two-sample test
    n_eff = (n_a * n_b) / (n_a + n_b)

    def integrand(g: float) -> float:
        """Integrand for the BF_10 marginal likelihood ratio."""
        if g <= 0:
            return 0.0
        # Likelihood ratio under g
        factor1 = (1.0 + n_eff * g) ** (-0.5)
        factor2 = (1.0 + t_stat**2 / (v * (1.0 + n_eff * g))) ** (-(v + 1) / 2.0)
        factor3 = (1.0 + t_stat**2 / v) ** ((v + 1) / 2.0)
        # Cauchy prior on sqrt(g) with scale r
        # g ~ InvGamma(1/2, r^2/2)
        prior = (r**2 / 2.0) ** 0.5 * g ** (-1.5) * math.exp(-r**2 / (2.0 * g))
        prior /= math.gamma(0.5) * math.sqrt(2.0)
        return factor1 * factor2 * factor3 * prior

    # Numerical integration
    try:
        bf10, _ = integrate.quad(integrand, 1e-10, 100.0, limit=200)
    except Exception:
        bf10 = float("nan")

    return bf10


def bayes_factor_interpretation(bf10: float) -> str:
    """Interpret Bayes Factor using Jeffreys' scale.

    Args:
        bf10: BF_10 value.

    Returns:
        String interpretation.
    """
    if math.isnan(bf10):
        return "computation_error"
    if bf10 > 100:
        return "extreme_evidence_for_H1"
    elif bf10 > 30:
        return "very_strong_evidence_for_H1"
    elif bf10 > 10:
        return "strong_evidence_for_H1"
    elif bf10 > 3:
        return "moderate_evidence_for_H1"
    elif bf10 > 1:
        return "anecdotal_evidence_for_H1"
    elif bf10 > 1 / 3:
        return "anecdotal_evidence_for_H0"
    elif bf10 > 1 / 10:
        return "moderate_evidence_for_H0"
    elif bf10 > 1 / 30:
        return "strong_evidence_for_H0"
    elif bf10 > 1 / 100:
        return "very_strong_evidence_for_H0"
    else:
        return "extreme_evidence_for_H0"


def experiment_1_4_bayes_factor() -> dict[str, Any]:
    """Experiment 1.4: Bayes Factor analysis for PT vs SA equivalence.

    Computes BF_10 for the primary IP difference using a default
    Cauchy prior (r = sqrt(2)/2). If BF_10 < 1/3, we can claim
    moderate-to-strong evidence for equivalence.

    Pre-registers the analysis plan (prior specification) and
    records the hash in the output artefact.

    Returns:
        Dict with Bayes Factor results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 1.4: Bayes Factor Analysis (BF_10)")
    _p("=" * 60)

    # Pre-register analysis plan
    plan = {
        "analysis": "bayes_factor_t_test",
        "prior": "cauchy",
        "prior_scale_r": 0.7071,
        "hypothesis": "two-sided",
        "null_hypothesis": "PT IP = SA IP (no difference)",
        "alternative_hypothesis": "PT IP != SA IP",
        "data_source": "y4q1_9_7seed_comparison.json",
        "method": "JZS_BF_Rouder2009",
    }
    plan_hash = _preregister_hash("bayes_factor", plan)

    results_list: list[dict[str, Any]] = []

    # Primary: 7-seed comparison
    data_7seed = _load_artefact("y4q1_9_7seed_comparison.json")
    if data_7seed is not None:
        pt_ips = data_7seed["pt_ips"]
        sa_ips = data_7seed["sa_ips"]
        welch = data_7seed.get("welch_t", {})
        t_stat = welch.get("t_stat", 0.0)
        n_a = len(pt_ips)
        n_b = len(sa_ips)

        bf10 = _jeffreys_bayes_factor_t(t_stat, n_a, n_b, r=0.7071)
        interp = bayes_factor_interpretation(bf10)

        _p(f"  Primary (7-seed standard):")
        _p(f"    t = {t_stat:.4f}, n_a = {n_a}, n_b = {n_b}")
        _p(f"    BF_10 = {bf10:.4f}")
        _p(f"    Interpretation: {interp}")
        if bf10 < 1 / 3:
            _p(f"    => Moderate-to-strong evidence for EQUIVALENCE")
        elif bf10 < 1:
            _p(f"    => Anecdotal evidence for equivalence")
        elif bf10 > 3:
            _p(f"    => Evidence AGAINST equivalence (unexpected)")
        else:
            _p(f"    => Inconclusive")

        results_list.append({
            "condition": "standard_7seed",
            "t_stat": t_stat,
            "n_a": n_a,
            "n_b": n_b,
            "bf10": bf10,
            "bf01": 1.0 / bf10 if bf10 > 0 else float("inf"),
            "interpretation": interp,
            "pt_mean": welch.get("mean_a"),
            "sa_mean": welch.get("mean_b"),
            "p_value": welch.get("p_value"),
        })

    # Also compute for pingouin-based approach if available
    try:
        import pingouin as pg

        if data_7seed is not None:
            pt_arr = np.array(data_7seed["pt_ips"])
            sa_arr = np.array(data_7seed["sa_ips"])
            bf_pg = pg.bayesfactor_ttest(
                float(data_7seed["welch_t"]["t_stat"]),
                len(pt_arr),
                len(sa_arr),
                paired=False,
                r=0.7071,
            )
            _p(f"  Pingouin BF_10 = {bf_pg:.4f}")
            results_list[-1]["bf10_pingouin"] = float(bf_pg)
    except (ImportError, Exception) as e:
        _p(f"  [INFO] Pingouin BF not available: {e}")

    # Stress conditions
    data_noise = _load_artefact("y4q1_8_noise_sweep.json")
    if data_noise is not None:
        for entry in data_noise.get("sweep", []):
            sigma = entry.get("sigma", "?")
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if len(pt_vals) >= 2 and len(sa_vals) >= 2:
                # Compute t-stat
                pt_arr = np.array(pt_vals)
                sa_arr = np.array(sa_vals)
                from scipy import stats as sp_stats
                t_res = sp_stats.ttest_ind(pt_arr, sa_arr, equal_var=False)
                t_stat_n = float(t_res.statistic)
                bf10_n = _jeffreys_bayes_factor_t(
                    t_stat_n, len(pt_vals), len(sa_vals), r=0.7071
                )
                results_list.append({
                    "condition": f"noise_sigma_{sigma}",
                    "t_stat": t_stat_n,
                    "n_a": len(pt_vals),
                    "n_b": len(sa_vals),
                    "bf10": bf10_n,
                    "bf01": 1.0 / bf10_n if bf10_n > 0 else float("inf"),
                    "interpretation": bayes_factor_interpretation(bf10_n),
                })
                _p(f"  Noise sigma={sigma}: BF_10 = {bf10_n:.4f} ({bayes_factor_interpretation(bf10_n)})")

    result: dict[str, Any] = {
        "benchmark": "phase1_exp1_4_bayes_factor",
        "description": "Bayes Factor analysis for PT vs SA IP equivalence",
        "preregistration_hash": plan_hash,
        "preregistered_plan": plan,
        "results": results_list,
        "interpretation_scale": {
            "bf10 > 100": "extreme evidence for H1 (difference)",
            "bf10 > 10": "strong evidence for H1",
            "bf10 > 3": "moderate evidence for H1",
            "1 < bf10 < 3": "anecdotal for H1",
            "1/3 < bf10 < 1": "anecdotal for H0 (equivalence)",
            "1/10 < bf10 < 1/3": "moderate evidence for H0",
            "bf10 < 1/10": "strong evidence for H0",
        },
        "reference": "Rouder, J. N. et al. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. Psychonomic Bulletin & Review, 16(2), 225-237.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("bayes_factor", result)
    return result


# =========================================================================
# 1.5 — Holm-Bonferroni Correction
# =========================================================================


def holm_bonferroni(
    p_values: list[float], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    The Holm-Bonferroni method is a sequentially rejective version of
    the Bonferroni correction that is uniformly more powerful.

    Algorithm:
        1. Sort p-values in ascending order.
        2. For rank j (1-indexed), compare p_(j) to alpha / (m - j + 1).
        3. Reject until first non-rejection, then accept all remaining.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Family-wise error rate threshold.

    Returns:
        List of dicts with original index, uncorrected p-value,
        adjusted p-value, and rejection decision.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results: list[dict[str, Any]] = [{}] * m

    # Compute adjusted p-values
    adjusted = [0.0] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = p * (m - rank)
        adjusted[rank] = adj_p

    # Enforce monotonicity (cumulative maximum)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Cap at 1.0
    adjusted = [min(p, 1.0) for p in adjusted]

    # Build result
    rejected_so_far = True
    output: list[dict[str, Any]] = [{}] * m
    for rank, (orig_idx, p_orig) in enumerate(indexed):
        adj_p = adjusted[rank]
        if rejected_so_far and adj_p <= alpha:
            reject = True
        else:
            reject = False
            rejected_so_far = False
        output[orig_idx] = {
            "original_index": orig_idx,
            "uncorrected_p": p_orig,
            "adjusted_p": adj_p,
            "reject_H0": reject,
            "rank": rank + 1,
        }
    return output


def experiment_1_5_holm_bonferroni() -> dict[str, Any]:
    """Experiment 1.5: Holm-Bonferroni correction for multiple comparisons.

    Applies Holm-Bonferroni across all stress-condition comparisons
    (k ~ 8-12 tests). Reports both uncorrected and corrected p-values.

    Returns:
        Dict with corrected comparison results.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 1.5: Holm-Bonferroni Multiple Comparison Correction")
    _p("=" * 60)

    comparisons: list[dict[str, Any]] = []
    p_values: list[float] = []

    # Gather all p-values from existing artefacts
    sources = [
        ("y4q1_9_7seed_comparison.json", "standard_7seed", "welch_t"),
        ("y4q1_7_statistical_summary.json", "q17_summary", "welch_t_test"),
    ]

    for filename, label, key in sources:
        data = _load_artefact(filename)
        if data is not None:
            t_data = data.get(key, {})
            p = t_data.get("p_value", t_data.get("p", None))
            if p is not None:
                comparisons.append({"condition": label, "source": filename, "p_value": p})
                p_values.append(p)
                _p(f"  {label}: p = {p:.6f}")

    # Noise sweep conditions
    data_noise = _load_artefact("y4q1_8_noise_sweep.json")
    if data_noise is not None:
        for entry in data_noise.get("sweep", []):
            sigma = entry.get("sigma", "?")
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if len(pt_vals) >= 2 and len(sa_vals) >= 2:
                from scipy import stats as sp_stats
                t_res = sp_stats.ttest_ind(
                    np.array(pt_vals), np.array(sa_vals), equal_var=False
                )
                p = float(t_res.pvalue)
                label = f"noise_sigma_{sigma}"
                comparisons.append({"condition": label, "source": "y4q1_8_noise_sweep.json", "p_value": p})
                p_values.append(p)
                _p(f"  {label}: p = {p:.6f}")

    # Fine occlusion sweep
    data_occ = _load_artefact("y4q1_9_fine_occlusion.json")
    if data_occ is not None:
        for entry in data_occ.get("sweep", []):
            occ_rate = entry.get("occlusion_rate", entry.get("occ_rate", "?"))
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if len(pt_vals) >= 2 and len(sa_vals) >= 2:
                from scipy import stats as sp_stats
                t_res = sp_stats.ttest_ind(
                    np.array(pt_vals), np.array(sa_vals), equal_var=False
                )
                p = float(t_res.pvalue)
                label = f"occlusion_{occ_rate}"
                comparisons.append({"condition": label, "source": "y4q1_9_fine_occlusion.json", "p_value": p})
                p_values.append(p)
                _p(f"  {label}: p = {p:.6f}")

    # Object scaling
    data_obj = _load_artefact("y4q1_9_object_scaling.json")
    if data_obj is not None:
        for entry in data_obj.get("sweep", []):
            n_obj = entry.get("n_objects", "?")
            pt_vals = entry.get("pt_ips", [])
            sa_vals = entry.get("sa_ips", [])
            if len(pt_vals) >= 2 and len(sa_vals) >= 2:
                from scipy import stats as sp_stats
                t_res = sp_stats.ttest_ind(
                    np.array(pt_vals), np.array(sa_vals), equal_var=False
                )
                p = float(t_res.pvalue)
                label = f"objects_{n_obj}"
                comparisons.append({"condition": label, "source": "y4q1_9_object_scaling.json", "p_value": p})
                p_values.append(p)
                _p(f"  {label}: p = {p:.6f}")

    # Apply correction
    if len(p_values) < 2:
        _p("  [WARN] Fewer than 2 p-values collected, correction trivial")

    corrected = holm_bonferroni(p_values, alpha=0.05)
    _p(f"\n  --- Holm-Bonferroni Results (k={len(p_values)} tests) ---")

    for i, comp in enumerate(comparisons):
        c = corrected[i]
        comp["adjusted_p"] = c["adjusted_p"]
        comp["reject_H0"] = c["reject_H0"]
        comp["rank"] = c["rank"]
        status = "REJECT" if c["reject_H0"] else "FAIL TO REJECT"
        _p(f"  [{status}] {comp['condition']}: "
           f"p_uncorr={comp['p_value']:.6f}, p_adj={c['adjusted_p']:.6f}")

    n_reject = sum(1 for c in corrected if c["reject_H0"])
    n_total = len(corrected)

    result: dict[str, Any] = {
        "benchmark": "phase1_exp1_5_holm_bonferroni",
        "description": "Holm-Bonferroni step-down correction for all PT vs SA comparisons",
        "alpha": 0.05,
        "n_comparisons": n_total,
        "n_rejected": n_reject,
        "n_not_rejected": n_total - n_reject,
        "comparisons": comparisons,
        "method": "Holm-Bonferroni step-down (sequentially rejective Bonferroni)",
        "reference": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("holm_bonferroni", result)
    return result


# =========================================================================
# 1.1 — Chimera Seed Expansion (7-seed reanalysis)
# =========================================================================


def experiment_1_1_chimera_seeds() -> dict[str, Any]:
    """Experiment 1.1: Chimera seed expansion analysis.

    Re-analyses existing chimera data from y4q1_9 and y4q1_3 to compute
    enhanced statistics (bootstrap CI, Cliff's delta, coefficient of
    variation) across all available seeds.

    Returns:
        Dict with expanded chimera statistics.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 1.1: Chimera Seed Expansion Analysis")
    _p("=" * 60)

    chimera_data = _load_artefact("y4q1_9_chimera_in_tracking.json")
    gold_chimera = _load_artefact("benchmark_y4q1_3_gold_standard_chimera.json")

    results: dict[str, Any] = {
        "benchmark": "phase1_exp1_1_chimera_seeds",
        "description": "Enhanced chimera statistics with bootstrap CI and effect sizes",
    }

    # Per-sequence chimera metrics from tracking context
    if chimera_data is not None:
        per_seq = chimera_data.get("per_sequence", [])
        bc_vals = [s["bimodality_coefficient"] for s in per_seq if "bimodality_coefficient" in s]
        ci_vals = [s["chimera_index"] for s in per_seq if "chimera_index" in s]
        cr_vals = [s["coherence_ratio"] for s in per_seq if "coherence_ratio" in s]

        if bc_vals:
            bc_arr = np.array(bc_vals)
            bc_mean = float(bc_arr.mean())
            bc_std = float(bc_arr.std(ddof=1)) if len(bc_arr) > 1 else 0.0
            bc_cv = bc_std / bc_mean if bc_mean > 0 else float("inf")

            # Bootstrap 95% CI
            rng = np.random.default_rng(42)
            boot_means = [
                float(rng.choice(bc_arr, size=len(bc_arr), replace=True).mean())
                for _ in range(5000)
            ]
            boot_means.sort()
            ci_lo = boot_means[int(0.025 * len(boot_means))]
            ci_hi = boot_means[int(0.975 * len(boot_means))]

            results["bimodality_coefficient"] = {
                "mean": bc_mean,
                "std": bc_std,
                "cv": bc_cv,
                "ci_95_low": ci_lo,
                "ci_95_high": ci_hi,
                "n_sequences": len(bc_vals),
                "pass_criteria": bc_mean > 0.5 and ci_lo > 0.3,
                "raw_values": bc_vals,
            }
            _p(f"  BC: mean={bc_mean:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], CV={bc_cv:.4f}")
            _p(f"  BC pass (mean>0.5, CI_lo>0.3): {bc_mean > 0.5 and ci_lo > 0.3}")

        if ci_vals:
            ci_arr = np.array(ci_vals)
            results["chimera_index"] = {
                "mean": float(ci_arr.mean()),
                "std": float(ci_arr.std(ddof=1)) if len(ci_arr) > 1 else 0.0,
                "n_sequences": len(ci_vals),
            }

        if cr_vals:
            cr_arr = np.array(cr_vals)
            results["coherence_ratio"] = {
                "mean": float(cr_arr.mean()),
                "std": float(cr_arr.std(ddof=1)) if len(cr_arr) > 1 else 0.0,
                "n_sequences": len(cr_vals),
            }

    # Gold standard chimera data
    if gold_chimera is not None:
        results["gold_standard_source"] = "benchmark_y4q1_3_gold_standard_chimera.json"
        results["gold_standard_keys"] = list(gold_chimera.keys())[:10]

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save("chimera_seeds", results)
    return results


# =========================================================================
# 1.2 — CLEVR-N Multi-Seed Analysis
# =========================================================================


def experiment_1_2_clevr_seeds() -> dict[str, Any]:
    """Experiment 1.2: CLEVR-N multi-seed reanalysis.

    Re-analyses existing 7-seed comparison data with enhanced statistics:
    coefficient of variation, bootstrap CI, and per-seed breakdown.

    Returns:
        Dict with expanded CLEVR-N statistics.
    """
    _p("\n" + "=" * 60)
    _p("EXPERIMENT 1.2: CLEVR-N Multi-Seed Analysis (7 Seeds)")
    _p("=" * 60)

    data = _load_artefact("y4q1_9_7seed_comparison.json")
    if data is None:
        _p("  [SKIP] No 7-seed comparison data found")
        return {"benchmark": "phase1_exp1_2_clevr_seeds", "status": "skipped"}

    pt_ips = np.array(data["pt_ips"])
    sa_ips = np.array(data["sa_ips"])
    seeds = data.get("seeds", list(SEEDS_7))

    # Coefficient of variation
    pt_cv = float(pt_ips.std(ddof=1) / pt_ips.mean()) if pt_ips.mean() > 0 else 0.0
    sa_cv = float(sa_ips.std(ddof=1) / sa_ips.mean()) if sa_ips.mean() > 0 else 0.0

    # Bootstrap 95% CI with 10000 resamples for tighter estimates
    rng = np.random.default_rng(42)
    n_boot = 10000

    pt_boot = sorted([
        float(rng.choice(pt_ips, size=len(pt_ips), replace=True).mean())
        for _ in range(n_boot)
    ])
    sa_boot = sorted([
        float(rng.choice(sa_ips, size=len(sa_ips), replace=True).mean())
        for _ in range(n_boot)
    ])

    pt_ci = (pt_boot[int(0.025 * n_boot)], pt_boot[int(0.975 * n_boot)])
    sa_ci = (sa_boot[int(0.025 * n_boot)], sa_boot[int(0.975 * n_boot)])

    # Cliff's delta between PT and SA
    cd = cliffs_delta(data["pt_ips"], data["sa_ips"])

    _p(f"  PT: mean={pt_ips.mean():.6f}, std={pt_ips.std(ddof=1):.6f}, "
       f"CV={pt_cv:.6f}, CI=[{pt_ci[0]:.6f}, {pt_ci[1]:.6f}]")
    _p(f"  SA: mean={sa_ips.mean():.6f}, std={sa_ips.std(ddof=1):.6f}, "
       f"CV={sa_cv:.6f}, CI=[{sa_ci[0]:.6f}, {sa_ci[1]:.6f}]")
    _p(f"  CV pass (PT CV < 5%): {pt_cv < 0.05}")
    _p(f"  CV pass (SA CV < 5%): {sa_cv < 0.05}")
    _p(f"  Cliff's delta: {cd:.4f} ({cliffs_delta_interpretation(cd)})")

    # Per-seed details
    per_seed_detail = []
    for i, seed in enumerate(seeds):
        per_seed_detail.append({
            "seed": seed,
            "pt_ip": float(pt_ips[i]),
            "sa_ip": float(sa_ips[i]),
            "difference": float(pt_ips[i] - sa_ips[i]),
        })

    result: dict[str, Any] = {
        "benchmark": "phase1_exp1_2_clevr_seeds",
        "description": "CLEVR-N multi-seed analysis with 7 seeds",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "phase_tracker": {
            "mean": float(pt_ips.mean()),
            "std": float(pt_ips.std(ddof=1)),
            "cv": pt_cv,
            "ci_95": list(pt_ci),
            "cv_pass": pt_cv < 0.05,
        },
        "slot_attention": {
            "mean": float(sa_ips.mean()),
            "std": float(sa_ips.std(ddof=1)),
            "cv": sa_cv,
            "ci_95": list(sa_ci),
            "cv_pass": sa_cv < 0.05,
        },
        "cliffs_delta": cd,
        "cliffs_delta_interpretation": cliffs_delta_interpretation(cd),
        "per_seed": per_seed_detail,
        "go_no_go": {
            "pt_cv_below_5pct": pt_cv < 0.05,
            "sa_cv_below_5pct": sa_cv < 0.05,
            "overall_pass": pt_cv < 0.05,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("clevr_seeds", result)
    return result


# =========================================================================
# Unified Statistical Summary Table
# =========================================================================


def generate_unified_summary(
    results_1_1: dict[str, Any],
    results_1_2: dict[str, Any],
    results_1_3: dict[str, Any],
    results_1_4: dict[str, Any],
    results_1_5: dict[str, Any],
) -> dict[str, Any]:
    """Generate a unified statistical summary table for the paper.

    Combines all Phase 1 results into a single LaTeX-ready table.

    Args:
        results_1_1: Chimera seed expansion results.
        results_1_2: CLEVR-N multi-seed results.
        results_1_3: Cliff's delta results.
        results_1_4: Bayes Factor results.
        results_1_5: Holm-Bonferroni results.

    Returns:
        Combined summary dict.
    """
    _p("\n" + "=" * 60)
    _p("UNIFIED STATISTICAL SUMMARY")
    _p("=" * 60)

    # Go/No-Go assessment
    go_no_go: dict[str, Any] = {}

    # Chimera BC criteria
    bc = results_1_1.get("bimodality_coefficient", {})
    go_no_go["chimera_bc"] = {
        "criterion": "Mean BC > 0.5, 95% CI excludes 0.3",
        "pass": bc.get("pass_criteria", False),
        "mean": bc.get("mean"),
        "ci_low": bc.get("ci_95_low"),
        "ci_high": bc.get("ci_95_high"),
    }

    # CLEVR-N CV criteria
    pt_data = results_1_2.get("phase_tracker", {})
    go_no_go["clevr_n_cv"] = {
        "criterion": "CV < 5% across seeds",
        "pass": pt_data.get("cv_pass", False),
        "cv": pt_data.get("cv"),
    }

    # Bayes Factor criteria
    bf_results = results_1_4.get("results", [])
    primary_bf = bf_results[0] if bf_results else {}
    bf10 = primary_bf.get("bf10", float("nan"))
    go_no_go["bayes_factor"] = {
        "criterion": "BF_10 < 1/3 (evidence for equivalence)",
        "pass": bf10 < 1 / 3 if not math.isnan(bf10) else False,
        "bf10": bf10,
        "interpretation": primary_bf.get("interpretation"),
    }

    all_pass = all(v.get("pass", False) for v in go_no_go.values())

    _p(f"\n  Go/No-Go Summary:")
    for key, val in go_no_go.items():
        status = "PASS" if val.get("pass") else "FAIL"
        _p(f"    [{status}] {key}: {val.get('criterion')}")

    _p(f"\n  Overall: {'ALL PASS -- Proceed to Phase 2' if all_pass else 'SOME FAILURES -- Review before proceeding'}")

    summary: dict[str, Any] = {
        "benchmark": "phase1_unified_summary",
        "description": "Unified Phase 1 statistical hardening summary",
        "go_no_go": go_no_go,
        "all_pass": all_pass,
        "phase1_recommendation": (
            "Proceed to Phase 2. All statistical claims are now defensible."
            if all_pass
            else "Review failed criteria before proceeding. See individual reports."
        ),
        "n_experiments_run": 5,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save("unified_summary", summary)
    return summary


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    """Run Phase 1 statistical hardening experiments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 1: Statistical Hardening (Experiments 1.1-1.5)"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--chimera-seeds", action="store_true", help="Exp 1.1: Chimera seed expansion")
    parser.add_argument("--clevr-seeds", action="store_true", help="Exp 1.2: CLEVR-N multi-seed")
    parser.add_argument("--cliffs-delta", action="store_true", help="Exp 1.3: Cliff's delta")
    parser.add_argument("--bayes-factor", action="store_true", help="Exp 1.4: Bayes Factor")
    parser.add_argument("--holm-bonferroni", action="store_true", help="Exp 1.5: Holm-Bonferroni")
    args = parser.parse_args()

    # Default to --all if no flags given
    run_all = args.all or not any([
        args.chimera_seeds, args.clevr_seeds, args.cliffs_delta,
        args.bayes_factor, args.holm_bonferroni,
    ])

    _p("=" * 60)
    _p("PRINet Phase 1: Statistical Hardening")
    _p("=" * 60)
    _p(f"Results directory: {RESULTS_DIR}")
    _p(f"Force rerun: {FORCE_RERUN}")

    t0 = time.time()
    results: dict[str, Any] = {}

    if run_all or args.chimera_seeds:
        results["1.1"] = experiment_1_1_chimera_seeds()
    if run_all or args.clevr_seeds:
        results["1.2"] = experiment_1_2_clevr_seeds()
    if run_all or args.cliffs_delta:
        results["1.3"] = experiment_1_3_cliffs_delta()
    if run_all or args.bayes_factor:
        results["1.4"] = experiment_1_4_bayes_factor()
    if run_all or args.holm_bonferroni:
        results["1.5"] = experiment_1_5_holm_bonferroni()

    # Generate unified summary if all experiments ran
    if run_all:
        generate_unified_summary(
            results.get("1.1", {}),
            results.get("1.2", {}),
            results.get("1.3", {}),
            results.get("1.4", {}),
            results.get("1.5", {}),
        )

    elapsed = time.time() - t0
    _p(f"\nPhase 1 complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
