"""Dependency-free evaluation plot generation."""

from __future__ import annotations

from pathlib import Path

from fedlegal.evaluation.schemas import MetricResult


def write_metric_bar_plot(metrics: dict[str, MetricResult], output_dir: str | Path) -> str:
    """Write an SVG bar plot for normalized and cost metrics."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics_bar.svg"
    values = list(metrics.values())
    width = 820
    row_height = 42
    height = 58 + row_height * len(values)
    rows = []
    max_value = max((metric.value for metric in values), default=1.0)
    scale_max = max(max_value, 1.0)
    for index, metric in enumerate(values):
        y = 42 + index * row_height
        normalized = metric.value / scale_max if scale_max else 0.0
        bar_width = int(500 * max(0.0, min(1.0, normalized)))
        rows.append(
            f'<text x="24" y="{y + 18}" font-size="14">{metric.name}</text>'
            f'<rect x="250" y="{y}" width="{bar_width}" height="24" fill="#376d8a" />'
            f'<text x="{260 + bar_width}" y="{y + 18}" font-size="13">{metric.value:.4f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        '<text x="24" y="26" font-size="18" font-weight="600">Evaluation Metrics</text>'
        + "".join(rows)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")
    return str(path)
