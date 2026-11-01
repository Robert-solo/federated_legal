"""Summarize MultiEURLEX client-update privacy and membership attacks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_multieurlex_comparison import interval


RUN_PATTERN = re.compile(
    r"^multieurlex_privacy_strict_(no_privacy|gaussian_0_5|gaussian_1_0)_en-pl_seed(\d+)$"
)


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for directory in sorted(root.glob("multieurlex_privacy_strict_*_en-pl_seed*")):
        match = RUN_PATTERN.match(directory.name)
        if not match:
            continue
        acceptance_path = directory / "acceptance.json"
        attack_path = directory / "privacy_attack" / "metrics.json"
        if not acceptance_path.exists() or not attack_path.exists():
            continue
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        metrics = json.loads((directory / "final_metrics.json").read_text(encoding="utf-8"))
        attack = json.loads(attack_path.read_text(encoding="utf-8"))
        condition, seed = match.groups()
        runs.append(
            {
                "condition": condition,
                "seed": int(seed),
                "acceptance_status": str(acceptance["status"]),
                "micro_f1": float(metrics["en-pl_micro_f1"]),
                "macro_f1": float(metrics["en-pl_macro_f1"]),
                "loss": float(metrics["en-pl_loss"]),
                "epsilon": (
                    None if metrics.get("dp_epsilon") is None else float(metrics["dp_epsilon"])
                ),
                "delta": float(json.loads((directory / "run_config.json").read_text(encoding="utf-8"))["dp_delta"]),
                "noise_multiplier": float(metrics["dp_noise_multiplier"]),
                "attack_auc": float(attack["attack_auc"]),
                "attack_auc_ci95_low": float(attack["attack_auc_ci95"][0]),
                "attack_auc_ci95_high": float(attack["attack_auc_ci95"][1]),
                "attack_balanced_accuracy": float(attack["balanced_accuracy"]),
                "communication_bytes_total": int(metrics["logical_communication_bytes"])
                * int(metrics["round"]),
                "path": str(directory),
            }
        )
    return runs


def analyze(
    runs: list[dict[str, Any]], conditions: list[str], seeds: list[int]
) -> dict[str, Any]:
    indexed = {(row["condition"], row["seed"]): row for row in runs}
    missing = [
        {"condition": condition, "seed": seed}
        for condition in conditions
        for seed in seeds
        if (condition, seed) not in indexed
    ]
    summaries = []
    diagnostic_summaries = []
    for condition in conditions:
        cohort = [indexed[(condition, seed)] for seed in seeds if (condition, seed) in indexed]
        accepted = [row for row in cohort if row["acceptance_status"] == "passed"]
        for metric in ("micro_f1", "macro_f1", "loss", "attack_auc", "attack_balanced_accuracy"):
            if cohort:
                diagnostic_summaries.append(
                    {"condition": condition, "metric": metric}
                    | interval([float(row[metric]) for row in cohort])
                )
            if accepted:
                summaries.append(
                    {"condition": condition, "metric": metric}
                    | interval([float(row[metric]) for row in accepted])
                )
    return {
        "claim_scope": "multilingual_eu_law_client_update_privacy",
        "matrix_complete": not missing,
        "acceptance_matrix_complete": not missing
        and all(row["acceptance_status"] == "passed" for row in runs),
        "missing_runs": missing,
        "completed_run_count": len(runs),
        "accepted_run_count": sum(
            row["acceptance_status"] == "passed" for row in runs
        ),
        "acceptance_failures": [
            {"condition": row["condition"], "seed": row["seed"]}
            for row in runs
            if row["acceptance_status"] != "passed"
        ],
        "runs": runs,
        "summaries": summaries,
        "diagnostic_summaries": diagnostic_summaries,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, summaries: list[dict[str, Any]], conditions: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["No update DP", "Gaussian 0.5*", "Gaussian 1.0*"]
    colors = ["#777777", "#D97941", "#3B6FB6"]
    indexed = {(row["condition"], row["metric"]): row for row in summaries}
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    positions = np.arange(len(conditions))
    for axis, metric, ylabel in (
        (axes[0], "loss", "Held-out BCE loss (lower is better)"),
        (axes[1], "attack_auc", "Membership attack AUC"),
    ):
        rows = [indexed[(condition, metric)] for condition in conditions]
        axis.bar(
            positions,
            [row["mean"] for row in rows],
            yerr=[row["standard_deviation"] for row in rows],
            color=colors,
            capsize=3,
            width=0.68,
        )
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0.5, color="#444444", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1)
    figure.text(
        0.5,
        0.01,
        "* Gaussian conditions did not pass calibration-threshold acceptance across all seeds.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["no_privacy", "gaussian_0_5", "gaussian_1_0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.input)
    result = analyze(runs, args.conditions, args.seeds)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "completed_runs.csv", runs)
    write_csv(args.output / "summary.csv", result["summaries"])
    write_csv(
        args.output / "diagnostic_summary.csv", result["diagnostic_summaries"]
    )
    if result["matrix_complete"]:
        plot(
            args.output / "privacy_utility_attack.pdf",
            result["diagnostic_summaries"],
            args.conditions,
        )
    print(
        json.dumps(
            {
                "accepted_run_count": result["accepted_run_count"],
                "completed_run_count": result["completed_run_count"],
                "matrix_complete": result["matrix_complete"],
                "acceptance_matrix_complete": result["acceptance_matrix_complete"],
                "missing_runs": result["missing_runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
