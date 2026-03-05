"""Benchmark Reporting Pipeline & Leaderboard Generator.

Provides utilities for aggregating benchmark JSON results into
structured Markdown reports, leaderboard tables, and SCALR metrics
summaries.

Q3 Tasks:
    * **Task 4.3a** — ``generate_benchmark_report()``
    * **Task 4.4** — ``generate_scalr_metrics_report()``
    * **Leaderboard** — ``generate_leaderboard()``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---- Report Generation -------------------------------------------------


def generate_benchmark_report(
    results_dir: str | Path,
    output_path: str | Path | None = None,
    title: str = "PRINet Benchmark Report",
) -> str:
    """Aggregate JSON benchmark outputs into a structured Markdown report.

    Scans ``results_dir`` for ``*.json`` files, extracts key metrics,
    and produces a Markdown document with comparison tables.

    Args:
        results_dir: Directory containing JSON benchmark result files.
        output_path: Optional path to write the Markdown report.
            If ``None``, only returns the string.
        title: Report title.

    Returns:
        Markdown report string.
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    sections: list[str] = []
    sections.append(f"# {title}\n")
    sections.append(
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"
    )

    if not json_files:
        sections.append("**No benchmark JSON files found.**\n")
        report = "\n".join(sections)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report, encoding="utf-8")
        return report

    # Summary table
    sections.append("## Summary\n")
    sections.append("| File | Status | Key Metrics |")
    sections.append("|------|--------|-------------|")

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            status = _extract_status(data)
            metrics = _extract_key_metrics(data)
            sections.append(f"| `{jf.name}` | {status} | {metrics} |")
        except (json.JSONDecodeError, KeyError) as exc:
            sections.append(f"| `{jf.name}` | ERROR | Parse error: {exc} |")

    sections.append("")

    # Per-file detail sections
    sections.append("## Detailed Results\n")
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            sections.append(f"### {jf.stem}\n")
            sections.append(_format_detail(data))
        except (json.JSONDecodeError, KeyError):
            sections.append(f"### {jf.stem}\n")
            sections.append("_Error reading file._\n")

    report = "\n".join(sections)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def generate_leaderboard(
    results_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate a ranked model × metric leaderboard from benchmark JSONs.

    Scans for JSON files containing per-model accuracy or loss data
    (e.g., CLEVR-N results, OscilloBench results) and produces a
    ranked comparison table in Markdown format.

    Args:
        results_dir: Directory with benchmark JSON files.
        output_path: Optional path to write the leaderboard Markdown.

    Returns:
        Markdown leaderboard string.
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    # Collect (model_name, benchmark, metric_name, value) tuples
    rows: list[dict[str, Any]] = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            rows.extend(_extract_leaderboard_rows(jf.stem, data))
        except (json.JSONDecodeError, KeyError):
            pass

    sections: list[str] = []
    sections.append("# PRINet Leaderboard\n")
    sections.append(
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"
    )

    if not rows:
        sections.append("No leaderboard data found.\n")
    else:
        # Sort by accuracy descending (or loss ascending)
        rows.sort(key=lambda r: -r.get("value", 0.0))

        sections.append(
            "| Rank | Model | Benchmark | Metric | Value |"
        )
        sections.append(
            "|------|-------|-----------|--------|-------|"
        )
        for i, row in enumerate(rows, 1):
            sections.append(
                f"| {i} | {row['model']} | {row['benchmark']} "
                f"| {row['metric']} | {row['value']:.4f} |"
            )

    sections.append("")
    report = "\n".join(sections)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def generate_scalr_metrics_report(
    r_history: list[float],
    window: int = 50,
    output_path: str | Path | None = None,
) -> str:
    """Generate SCALR synchronization stability report.

    Computes windowed coefficient of variation (CV) of order parameter
    ``r(t)`` and flags desynchronization events.

    Args:
        r_history: Time series of order parameter values.
        window: Window size for CV computation.
        output_path: Optional path to save report.

    Returns:
        Markdown report string.
    """
    import math

    sections: list[str] = []
    sections.append("# SCALR Metrics Report\n")
    sections.append(f"- Total epochs: {len(r_history)}")
    sections.append(f"- Analysis window: {window}\n")

    if len(r_history) < window:
        sections.append("_Insufficient data for windowed analysis._\n")
        report = "\n".join(sections)
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
        return report

    # Compute windowed CV
    cvs: list[float] = []
    means: list[float] = []
    for i in range(window, len(r_history) + 1):
        w = r_history[i - window : i]
        mu = sum(w) / len(w)
        var = sum((x - mu) ** 2 for x in w) / len(w)
        std = math.sqrt(var)
        cv = std / max(mu, 1e-8)
        cvs.append(cv)
        means.append(mu)

    # Desync events: CV > 0.1
    desync_events = [(i + window, cv) for i, cv in enumerate(cvs) if cv > 0.1]

    sections.append("## Order Parameter Summary\n")
    sections.append(f"- Final r(t): {r_history[-1]:.4f}")
    sections.append(f"- Mean r(t): {sum(r_history) / len(r_history):.4f}")
    sections.append(f"- Mean CV: {sum(cvs) / len(cvs):.4f}")
    sections.append(f"- Max CV: {max(cvs):.4f}")
    sections.append(f"- Desync events (CV > 0.1): {len(desync_events)}\n")

    if desync_events:
        sections.append("### Desynchronization Events\n")
        sections.append("| Epoch | CV |")
        sections.append("|-------|----|")
        for epoch, cv in desync_events[:20]:  # Cap at 20
            sections.append(f"| {epoch} | {cv:.4f} |")
        sections.append("")

    report = "\n".join(sections)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")

    return report


# ---- Internal Helpers --------------------------------------------------


def _extract_status(data: dict[str, Any]) -> str:
    """Extract status from benchmark JSON."""
    if "status" in data:
        return str(data["status"])
    if "benchmarks" in data:
        statuses = [
            b.get("status", "UNKNOWN")
            for b in data["benchmarks"]
            if isinstance(b, dict)
        ]
        if all(s == "PASS" for s in statuses):
            return "ALL PASS"
        fails = sum(1 for s in statuses if s in ("FAIL", "ERROR"))
        return f"{len(statuses) - fails}/{len(statuses)} PASS"
    return "OK"


def _extract_key_metrics(data: dict[str, Any]) -> str:
    """Extract key metrics summary from benchmark JSON."""
    parts: list[str] = []

    # Look for common metric patterns
    for key in ("test_acc", "accuracy", "final_accuracy"):
        if key in data:
            parts.append(f"{key}={data[key]:.4f}")
    for key in ("wall_time_s", "elapsed_s"):
        if key in data:
            parts.append(f"time={data[key]:.1f}s")
    if "param_count" in data:
        parts.append(f"params={data['param_count']}")

    # Nested benchmark results
    if "benchmarks" in data and isinstance(data["benchmarks"], list):
        n = len(data["benchmarks"])
        parts.append(f"{n} sub-benchmarks")

    return ", ".join(parts) if parts else "—"


def _format_detail(data: dict[str, Any]) -> str:
    """Format a benchmark JSON as Markdown detail block."""
    lines: list[str] = []

    # Top-level scalars
    skip_keys = {"benchmarks", "results", "raw_data"}
    for key, val in data.items():
        if key in skip_keys:
            continue
        if isinstance(val, (str, int, float, bool)):
            lines.append(f"- **{key}**: {val}")

    # Nested benchmarks
    if "benchmarks" in data and isinstance(data["benchmarks"], list):
        lines.append("\n**Sub-benchmarks:**\n")
        for bench in data["benchmarks"]:
            if isinstance(bench, dict):
                name = bench.get("name", bench.get("benchmark", "?"))
                status = bench.get("status", "?")
                lines.append(f"- {name}: {status}")

    # Per-model results (e.g., CLEVR-N)
    if "results" in data and isinstance(data["results"], dict):
        lines.append("\n**Model Results:**\n")
        lines.append("| Model | Metric | Value |")
        lines.append("|-------|--------|-------|")
        for model_name, model_results in data["results"].items():
            if isinstance(model_results, list):
                for r in model_results:
                    if isinstance(r, dict) and "test_acc" in r:
                        n = r.get("n_items", "?")
                        lines.append(
                            f"| {model_name} (N={n}) "
                            f"| test_acc | {r['test_acc']:.4f} |"
                        )
            elif isinstance(model_results, (int, float)):
                lines.append(
                    f"| {model_name} | value | {model_results} |"
                )

    lines.append("")
    return "\n".join(lines)


def _extract_leaderboard_rows(
    benchmark_name: str, data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Extract (model, benchmark, metric, value) rows for leaderboard."""
    rows: list[dict[str, Any]] = []

    # CLEVR-N style: {model_name: [{n_items, test_acc, ...}, ...]}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list):
                for entry in val:
                    if isinstance(entry, dict) and "test_acc" in entry:
                        n = entry.get("n_items", "?")
                        rows.append(
                            {
                                "model": f"{key}",
                                "benchmark": f"{benchmark_name} (N={n})",
                                "metric": "test_acc",
                                "value": float(entry["test_acc"]),
                            }
                        )

        # OscilloBench style: {benchmarks: [{name, ..., test_acc}]}
        if "benchmarks" in data and isinstance(data["benchmarks"], list):
            for bench in data["benchmarks"]:
                if not isinstance(bench, dict):
                    continue
                bname = bench.get("name", benchmark_name)
                # Look for per-model results
                if "results" in bench and isinstance(bench["results"], dict):
                    for model, metrics in bench["results"].items():
                        if isinstance(metrics, dict):
                            for mk, mv in metrics.items():
                                if isinstance(mv, (int, float)):
                                    rows.append(
                                        {
                                            "model": model,
                                            "benchmark": bname,
                                            "metric": mk,
                                            "value": float(mv),
                                        }
                                    )
                # Top-level metrics
                for mk in ("test_acc", "accuracy"):
                    if mk in bench:
                        rows.append(
                            {
                                "model": bench.get("model", "PRINet"),
                                "benchmark": bname,
                                "metric": mk,
                                "value": float(bench[mk]),
                            }
                        )

    return rows
