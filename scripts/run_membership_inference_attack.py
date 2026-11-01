"""Evaluate a loss-based black-box membership inference attack.

The input contains only record identifiers, membership labels, and model-derived
scores. Raw legal text is not copied into attack artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_features(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or {bool(row["member"]) for row in rows} != {False, True}:
        raise ValueError("Membership features must contain both members and non-members.")
    return rows


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("AUC requires both classes.")
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positives = labels == 1
    negatives = ~positives
    true_positive_rate = float(predictions[positives].mean())
    true_negative_rate = float((~predictions[negatives]).mean())
    return (true_positive_rate + true_negative_rate) / 2.0


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    return float(
        max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy(labels, scores >= threshold),
                -abs(float(threshold) - float(np.median(scores))),
            ),
        )
    )


def roc_points(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1], [-np.inf]))
    positives = labels == 1
    negatives = ~positives
    return [
        {
            "threshold": float(threshold),
            "false_positive_rate": float((scores[negatives] >= threshold).mean()),
            "true_positive_rate": float((scores[positives] >= threshold).mean()),
        }
        for threshold in thresholds
    ]


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, seed: int, repetitions: int
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    member_indices = np.flatnonzero(labels == 1)
    nonmember_indices = np.flatnonzero(labels == 0)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate(
            (
                generator.choice(member_indices, len(member_indices), replace=True),
                generator.choice(nonmember_indices, len(nonmember_indices), replace=True),
            )
        )
        values.append(rank_auc(labels[sample], scores[sample]))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def write_roc_svg(path: Path, points: list[dict[str, float]], auc: float) -> None:
    size = 520
    margin = 58
    plot = size - 2 * margin
    coordinates = " ".join(
        f"{margin + point['false_positive_rate'] * plot:.1f},"
        f"{size - margin - point['true_positive_rate'] * plot:.1f}"
        for point in points
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{size-margin}" x2="{size-margin}" y2="{margin}" stroke="#9aa0a6" stroke-dasharray="6 5"/>
<line x1="{margin}" y1="{size-margin}" x2="{size-margin}" y2="{size-margin}" stroke="black"/>
<line x1="{margin}" y1="{size-margin}" x2="{margin}" y2="{margin}" stroke="black"/>
<polyline points="{coordinates}" fill="none" stroke="#276FBF" stroke-width="3"/>
<text x="{size/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18">Membership inference ROC (AUC={auc:.3f})</text>
<text x="{size/2}" y="{size-14}" text-anchor="middle" font-family="Arial" font-size="15">False-positive rate</text>
<text x="17" y="{size/2}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 17 {size/2})">True-positive rate</text>
</svg>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_features(args.features)
    random.Random(args.seed).shuffle(rows)
    calibration = rows[::2]
    evaluation = rows[1::2]
    calibration_labels = np.asarray([int(row["member"]) for row in calibration])
    calibration_scores = np.asarray([-float(row["loss"]) for row in calibration])
    labels = np.asarray([int(row["member"]) for row in evaluation])
    scores = np.asarray([-float(row["loss"]) for row in evaluation])
    attack_direction = 1.0 if rank_auc(calibration_labels, calibration_scores) >= 0.5 else -1.0
    calibration_scores *= attack_direction
    scores *= attack_direction
    threshold = select_threshold(calibration_labels, calibration_scores)
    predictions = scores >= threshold
    auc = rank_auc(labels, scores)
    ci_low, ci_high = bootstrap_auc(
        labels, scores, args.seed, args.bootstrap_repetitions
    )
    points = roc_points(labels, scores)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "attack": "black_box_loss_membership_inference",
        "attack_auc": auc,
        "attack_auc_ci95": [ci_low, ci_high],
        "attack_direction_selected_on_calibration_half": (
            "lower_loss_is_member" if attack_direction > 0 else "higher_loss_is_member"
        ),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "threshold_selected_on_calibration_half": threshold,
        "calibration_examples": len(calibration),
        "evaluation_examples": len(evaluation),
        "member_examples": int(labels.sum()),
        "nonmember_examples": int((labels == 0).sum()),
        "raw_text_in_attack_artifacts": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "attack_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, score, prediction in zip(evaluation, scores, predictions, strict=True):
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "language": row["language"],
                        "member": bool(row["member"]),
                        "attack_score": float(score),
                        "predicted_member": bool(prediction),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (args.output / "roc_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    write_roc_svg(args.output / "membership_roc.svg", points, auc)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
