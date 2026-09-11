"""Generate classification-specific analysis for an API baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    predictions_path = args.run_dir / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    comparable = [
        row
        for row in rows
        if row.get("gold_label") is not None
        and row.get("predicted_label") is not None
        and not row.get("error")
    ]
    labels = sorted(
        {str(row["gold_label"]) for row in comparable}
        | {str(row["predicted_label"]) for row in comparable},
        key=_label_sort_key,
    )
    confusion = {
        gold: {predicted: 0 for predicted in labels}
        for gold in labels
    }
    for row in comparable:
        confusion[str(row["gold_label"])][str(row["predicted_label"])] += 1

    class_metrics = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[gold][label] for gold in labels if gold != label)
        false_negative = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        support = sum(confusion[label].values())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        class_metrics.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    correct = sum(int(row.get("correct", False)) for row in comparable)
    accuracy = _safe_divide(correct, len(comparable))
    ci_low, ci_high = wilson_interval(correct, len(comparable))
    analysis = {
        "run_name": args.run_dir.name,
        "num_predictions": len(rows),
        "num_comparable": len(comparable),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_ci95_wilson": [ci_low, ci_high],
        "macro_precision": _mean(item["precision"] for item in class_metrics),
        "macro_recall": _mean(item["recall"] for item in class_metrics),
        "macro_f1": _mean(item["f1"] for item in class_metrics),
        "labels": labels,
        "not_applicable_metrics": {
            "citation_metrics": "CaseHOLD multiple-choice labels do not require generated citations.",
            "hallucination_rate": "Label-only output has no free-form factual generation to assess.",
            "reasoning_coherence": "Provider-side thinking is disabled and no rationale is generated.",
            "cross_jurisdiction_generalization": "All examples currently have one unknown jurisdiction group.",
        },
    }

    (args.run_dir / "classification_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_class_metrics(args.run_dir / "class_metrics.csv", class_metrics)
    write_confusion_csv(args.run_dir / "confusion_matrix.csv", labels, confusion)
    write_confusion_svg(args.run_dir / "confusion_matrix.svg", labels, confusion)
    write_report(args.run_dir / "classification_analysis.md", analysis, class_metrics)
    print(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True))


def wilson_interval(successes: int, total: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z_score**2 / total
    center = (proportion + z_score**2 / (2 * total)) / denominator
    margin = (
        z_score
        * math.sqrt(proportion * (1 - proportion) / total + z_score**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def write_class_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, labels: list[str], confusion: dict[str, dict[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\predicted", *labels])
        for gold in labels:
            writer.writerow([gold, *(confusion[gold][predicted] for predicted in labels)])


def write_confusion_svg(path: Path, labels: list[str], confusion: dict[str, dict[str, int]]) -> None:
    cell = 70
    margin_left = 130
    margin_top = 100
    width = margin_left + cell * len(labels) + 30
    height = margin_top + cell * len(labels) + 50
    maximum = max((value for row in confusion.values() for value in row.values()), default=1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="Arial" font-size="18" font-weight="600">CaseHOLD confusion matrix</text>',
        f'<text x="{margin_left + cell * len(labels) / 2}" y="58" text-anchor="middle" font-family="Arial" font-size="13">Predicted label</text>',
        f'<text x="22" y="{margin_top + cell * len(labels) / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 22 {margin_top + cell * len(labels) / 2})">Gold label</text>',
    ]
    for index, label in enumerate(labels):
        x = margin_left + index * cell + cell / 2
        y = margin_top + index * cell + cell / 2 + 5
        elements.append(f'<text x="{x}" y="{margin_top - 16}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
        elements.append(f'<text x="{margin_left - 20}" y="{y}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    for row_index, gold in enumerate(labels):
        for column_index, predicted in enumerate(labels):
            value = confusion[gold][predicted]
            intensity = value / maximum if maximum else 0.0
            shade = int(245 - 155 * intensity)
            fill = f"rgb({shade},{shade + 5},{255})"
            x = margin_left + column_index * cell
            y = margin_top + row_index * cell
            text_color = "white" if intensity > 0.55 else "#111827"
            elements.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d1d5db"/>')
            elements.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-family="Arial" font-size="15" fill="{text_color}">{value}</text>')
    elements.append("</svg>")
    path.write_text("".join(elements), encoding="utf-8")


def write_report(path: Path, analysis: dict[str, Any], class_metrics: list[dict[str, Any]]) -> None:
    low, high = analysis["accuracy_ci95_wilson"]
    lines = [
        f"# Classification Analysis: {analysis['run_name']}",
        "",
        f"- Accuracy: {analysis['accuracy']:.4f} ({analysis['correct']}/{analysis['num_comparable']})",
        f"- Wilson 95% CI: [{low:.4f}, {high:.4f}]",
        f"- Macro-F1: {analysis['macro_f1']:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in class_metrics:
        lines.append(
            f"| {item['label']} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )
    lines.extend(["", "## Metric Scope", ""])
    for metric, reason in analysis["not_applicable_metrics"].items():
        lines.append(f"- `{metric}`: not applicable; {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _label_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.lstrip("-").isdigit() else (1, value)


if __name__ == "__main__":
    main()
