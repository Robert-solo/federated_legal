"""Visualization helpers for aggregation diagnostics."""

from __future__ import annotations

from pathlib import Path

from fedlegal.aggregation.legal_conflict_metrics import AggregationConflictReport


def write_conflict_summary_svg(
    report: AggregationConflictReport,
    output_path: str | Path,
) -> Path:
    """Write a small dependency-free SVG bar chart for conflict components."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [
        ("Citation", report.citation_conflict),
        ("Reasoning", report.reasoning_conflict),
        ("Verdict", report.verdict_conflict),
        ("Rule", report.rule_alignment_distance),
        ("Contradiction", report.contradiction_penalty),
    ]
    width = 720
    row_height = 38
    height = 60 + row_height * len(values)
    bars = []
    for index, (label, value) in enumerate(values):
        y = 40 + index * row_height
        bar_width = int(460 * max(0.0, min(1.0, value)))
        bars.append(
            f'<text x="24" y="{y + 18}" font-size="14">{label}</text>'
            f'<rect x="150" y="{y}" width="{bar_width}" height="22" fill="#2f6f73" />'
            f'<text x="{160 + bar_width}" y="{y + 17}" font-size="13">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        '<text x="24" y="24" font-size="18" font-weight="600">'
        "Conflict-Aware Aggregation Diagnostics"
        "</text>"
        + "".join(bars)
        + "</svg>"
    )
    output.write_text(svg, encoding="utf-8")
    return output
