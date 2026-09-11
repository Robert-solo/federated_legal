"""Evaluation table writers."""

from __future__ import annotations

import csv
from pathlib import Path

from fedlegal.evaluation.schemas import MetricResult


def write_metric_tables(metrics: dict[str, MetricResult], output_dir: str | Path) -> dict[str, str]:
    """Write CSV and Markdown metric tables."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "metrics_table.csv"
    md_path = directory / "metrics_table.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric in metrics.values():
            writer.writerow([metric.name, f"{metric.value:.6f}"])

    lines = ["| Metric | Value |", "|---|---:|"]
    for metric in metrics.values():
        lines.append(f"| {metric.name} | {metric.value:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path)}
