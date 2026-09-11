"""Summarize real experiment and baseline outputs into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/evaluation/paper_summary")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary(root)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_csv_tables(summary, output_dir)
    write_svg_plot(summary, output_dir)


def collect_summary(root: Path) -> dict[str, Any]:
    api_runs = []
    for report_path in sorted(root.glob("api_baselines/*/report.json")):
        api_runs.append(json.loads(report_path.read_text(encoding="utf-8")))

    fl_runs = []
    for report_path in sorted(root.glob("real_flower_peft/*/real_experiment_report.json")):
        fl_runs.append(json.loads(report_path.read_text(encoding="utf-8")))

    evaluation_runs = []
    for report_path in sorted(root.glob("evaluation/*/metrics.json")):
        evaluation_runs.append(
            {
                "run_name": report_path.parent.name,
                "metrics": json.loads(report_path.read_text(encoding="utf-8")),
            }
        )

    return {
        "api_baselines": api_runs,
        "real_flower_runs": fl_runs,
        "evaluation_runs": evaluation_runs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Experiment Summary", ""]
    lines.append("## API Baselines")
    lines.append("")
    lines.append("| Run | Task | Accuracy | Parse Rate | API Error Rate | Evaluated |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for run in summary["api_baselines"]:
        lines.append(
            f"| {run.get('run_name')} | {run.get('task', 'classification')} | "
            f"{float(run.get('accuracy', 0.0)):.4f} | "
            f"{float(run.get('parse_rate', 0.0)):.4f} | "
            f"{float(run.get('api_error_rate', 0.0)):.4f} | {int(run.get('evaluated', 0))} |"
        )
    lines.append("")
    lines.append("## Real Flower Runs")
    lines.append("")
    lines.append("| Run | Dataset | Final Acc | Comm Bytes | Drift |")
    lines.append("|---|---|---:|---:|---:|")
    for run in summary["real_flower_runs"]:
        lines.append(
            f"| {run.get('run_name')} | {run.get('dataset')} | "
            f"{_fmt(run.get('final_accuracy'))} | {_fmt(run.get('communication_bytes'))} | "
            f"{_fmt(run.get('mean_client_drift_l2'))} |"
        )
    lines.append("")
    lines.append("## Metric Coverage")
    lines.append("")
    lines.append("| Metric | Source | Status |")
    lines.append("|---|---|---|")
    lines.append("| legal accuracy | API + Flower eval | covered |")
    lines.append("| citation accuracy/consistency | API predictions | covered for inference baselines |")
    lines.append("| hallucination rate | API generated text | covered heuristically |")
    lines.append("| communication cost | Flower communication logs | covered |")
    lines.append("| client drift | Flower communication logs | covered |")
    lines.append("| cross-jurisdiction generalization | jurisdiction metadata | partially covered; current CaseHOLD eval has one test jurisdiction |")
    lines.append("| aggregation stability | communication/client drift logs | covered heuristically |")
    lines.append("| privacy leakage risk | sensitive-token metadata | partially covered; needs richer sensitive-token annotations |")
    return "\n".join(lines) + "\n"


def write_csv_tables(summary: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "api_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "task",
                "model",
                "accuracy",
                "parse_rate",
                "api_error_rate",
                "evaluated",
                "num_examples",
                "elapsed_sec",
            ]
        )
        for run in summary["api_baselines"]:
            writer.writerow(
                [
                    run.get("run_name"),
                    run.get("task", "classification"),
                    run.get("model"),
                    run.get("accuracy"),
                    run.get("parse_rate"),
                    run.get("api_error_rate"),
                    run.get("evaluated"),
                    run.get("num_examples"),
                    run.get("elapsed_sec"),
                ]
            )

    with (output_dir / "real_flower_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "model_name",
                "dataset",
                "final_accuracy",
                "best_accuracy",
                "communication_bytes",
                "uplink_bytes",
                "downlink_bytes",
                "mean_client_drift_l2",
                "num_rounds",
                "num_clients",
            ]
        )
        for run in summary["real_flower_runs"]:
            writer.writerow(
                [
                    run.get("run_name"),
                    run.get("model_name"),
                    run.get("dataset"),
                    run.get("final_accuracy"),
                    run.get("best_accuracy"),
                    run.get("communication_bytes"),
                    run.get("uplink_bytes"),
                    run.get("downlink_bytes"),
                    run.get("mean_client_drift_l2"),
                    run.get("num_rounds"),
                    run.get("num_clients"),
                ]
            )


def write_svg_plot(summary: dict[str, Any], output_dir: Path) -> None:
    rows = [
        (run.get("run_name", ""), float(run.get("final_accuracy") or 0.0))
        for run in summary["real_flower_runs"]
        if str(run.get("run_name", "")).startswith("paper_")
    ]
    width = 980
    row_height = 36
    height = 70 + row_height * max(len(rows), 1)
    max_value = max((value for _, value in rows), default=1.0)
    max_value = max(max_value, 0.01)
    body = []
    for index, (name, value) in enumerate(rows):
        y = 48 + index * row_height
        bar_width = int(460 * value / max_value)
        label = _short_label(name)
        body.append(
            f'<text x="24" y="{y + 17}" font-size="12">{_escape(label)}</text>'
            f'<rect x="380" y="{y}" width="{bar_width}" height="22" fill="#2d6f8f" />'
            f'<text x="{390 + bar_width}" y="{y + 16}" font-size="12">{value:.4f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white" />'
        '<text x="24" y="28" font-size="18" font-weight="600">Real Flower/PEFT CaseHOLD Accuracy</text>'
        + "".join(body)
        + "</svg>"
    )
    (output_dir / "real_flower_accuracy.svg").write_text(svg, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _short_label(value: str) -> str:
    return (
        value.removeprefix("paper_")
        .replace("_casehold_", " / ")
        .replace("_jurisdiction_", " / jurisdiction / ")
        .replace("_label_dirichlet_", " / label_dirichlet / ")
        .replace("_iid_", " / iid / ")
    )[:82]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
