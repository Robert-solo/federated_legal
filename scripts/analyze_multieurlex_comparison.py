"""Summarize accepted MultiEURLEX multilingual-transfer experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RUN_PATTERN = re.compile(
    r"^multieurlex_(fedavg|fedprox|lambda0|proxy_flen|flen)_(en-[a-z]+)_seed(\d+)$"
)
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def interval(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half_width = critical * standard_deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def load_accepted_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for directory in sorted(root.glob("multieurlex_*")):
        match = RUN_PATTERN.match(directory.name)
        if not match:
            continue
        acceptance_path = directory / "acceptance.json"
        metrics_path = directory / "final_metrics.json"
        if not acceptance_path.exists() or not metrics_path.exists():
            continue
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if acceptance.get("status") != "passed":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        threshold_audit = json.loads(
            (directory / "threshold_calibration.json").read_text(encoding="utf-8")
        )
        method, holdout, seed = match.groups()
        if metrics.get("method") != method or metrics.get("holdout") != holdout:
            raise ValueError(f"Directory and metrics disagree for {directory}")
        if int(metrics.get("seed")) != int(seed):
            raise ValueError(f"Directory and seed disagree for {directory}")
        if metrics.get("claim_scope") != "multilingual_eu_law_transfer":
            raise ValueError(f"Invalid claim scope for {directory}")
        if run_config.get("threshold_objective") != "micro_f1":
            raise ValueError(f"Invalid threshold objective for {directory}")
        final_threshold = threshold_audit[-1]
        if final_threshold.get("selection_objective") != "micro_f1":
            raise ValueError(f"Threshold audit objective mismatch for {directory}")
        if final_threshold.get("at_search_boundary"):
            raise ValueError(f"Boundary threshold cannot be accepted for {directory}")
        if holdout in final_threshold.get("selected_on_languages", []):
            raise ValueError(f"Held-out language leaked into threshold selection for {directory}")
        prediction_count = validate_predictions(
            directory / "predictions.jsonl", float(metrics["threshold"])
        )
        if prediction_count != int(acceptance.get("predictions_written", -1)):
            raise ValueError(f"Prediction count mismatch for {directory}")
        fedprox_penalty = float(metrics["mean_fedprox_penalty"])
        if method == "fedprox" and fedprox_penalty <= 0:
            raise ValueError(f"FedProx penalty was not active for {directory}")
        if method != "fedprox" and fedprox_penalty != 0:
            raise ValueError(f"Unexpected FedProx penalty for {directory}")
        communication_per_round = int(metrics["logical_communication_bytes"])
        runs.append(
            {
                "method": method,
                "holdout": holdout,
                "seed": int(seed),
                "micro_f1": float(metrics[f"{holdout}_micro_f1"]),
                "macro_f1": float(metrics[f"{holdout}_macro_f1"]),
                "loss": float(metrics[f"{holdout}_loss"]),
                "threshold": float(metrics["threshold"]),
                "communication_bytes_per_round": communication_per_round,
                "communication_bytes_total": communication_per_round * int(metrics["round"]),
                "mean_update_l2": float(metrics["mean_update_l2"]),
                "mean_pairwise_update_l2": float(metrics["mean_pairwise_update_l2"]),
                "mean_fedprox_penalty": float(metrics["mean_fedprox_penalty"]),
                "path": str(directory),
            }
        )
    return runs


def validate_predictions(path: Path, threshold: float) -> int:
    seen: set[tuple[str, str]] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            probabilities = row.get("probabilities")
            if not isinstance(probabilities, list) or len(probabilities) != 21:
                raise ValueError(f"Invalid probability vector in {path}")
            if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in probabilities):
                raise ValueError(f"Invalid probability value in {path}")
            expected = [index for index, value in enumerate(probabilities) if value >= threshold]
            if row.get("predicted_labels") != expected:
                raise ValueError(f"Prediction/threshold mismatch in {path}")
            key = (str(row.get("language")), str(row.get("id")))
            if not row.get("id") or key in seen:
                raise ValueError(f"Missing or duplicate prediction identifier in {path}")
            seen.add(key)
            count += 1
    return count


def analyze(
    runs: list[dict[str, Any]],
    methods: list[str],
    holdouts: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    indexed = {(run["method"], run["holdout"], run["seed"]): run for run in runs}
    missing = [
        {"method": method, "holdout": holdout, "seed": seed}
        for method in methods
        for holdout in holdouts
        for seed in seeds
        if (method, holdout, seed) not in indexed
    ]
    summaries = []
    for method in methods:
        for holdout in holdouts:
            cohort = [indexed[(method, holdout, seed)] for seed in seeds if (method, holdout, seed) in indexed]
            if not cohort:
                continue
            for metric in ("micro_f1", "macro_f1", "loss"):
                summaries.append(
                    {"method": method, "holdout": holdout, "metric": metric}
                    | interval([run[metric] for run in cohort])
                )
        complete_seeds = [
            seed
            for seed in seeds
            if all((method, holdout, seed) in indexed for holdout in holdouts)
        ]
        for metric in ("micro_f1", "macro_f1", "loss"):
            seed_means = [
                statistics.fmean(indexed[(method, holdout, seed)][metric] for holdout in holdouts)
                for seed in complete_seeds
            ]
            if seed_means:
                summaries.append(
                    {"method": method, "holdout": "mean_across_holdouts", "metric": metric}
                    | interval(seed_means)
                )
    comparisons = []
    for method in methods:
        if method == "fedavg":
            continue
        for holdout in holdouts:
            common = [
                seed
                for seed in seeds
                if (method, holdout, seed) in indexed and ("fedavg", holdout, seed) in indexed
            ]
            for metric in ("micro_f1", "macro_f1"):
                differences = [
                    indexed[(method, holdout, seed)][metric]
                    - indexed[("fedavg", holdout, seed)][metric]
                    for seed in common
                ]
                if not differences:
                    continue
                result = interval(differences)
                standard_deviation = float(result["standard_deviation"])
                comparisons.append(
                    {
                        "method": method,
                        "baseline": "fedavg",
                        "holdout": holdout,
                        "metric": metric,
                        "paired_seeds": common,
                        "effect_size_dz": (
                            float(result["mean"]) / standard_deviation
                            if standard_deviation > 0
                            else None
                        ),
                        "superiority_gate_passed": (
                            len(common) >= 5 and float(result["ci95_low"]) > 0
                        ),
                    }
                    | result,
                )
        common = [
            seed
            for seed in seeds
            if all(
                (method, holdout, seed) in indexed
                and ("fedavg", holdout, seed) in indexed
                for holdout in holdouts
            )
        ]
        for metric in ("micro_f1", "macro_f1"):
            differences = [
                statistics.fmean(
                    indexed[(method, holdout, seed)][metric]
                    - indexed[("fedavg", holdout, seed)][metric]
                    for holdout in holdouts
                )
                for seed in common
            ]
            if differences:
                result = interval(differences)
                standard_deviation = float(result["standard_deviation"])
                comparisons.append(
                    {
                        "method": method,
                        "baseline": "fedavg",
                        "holdout": "mean_across_holdouts",
                        "metric": metric,
                        "paired_seeds": common,
                        "effect_size_dz": (
                            float(result["mean"]) / standard_deviation
                            if standard_deviation > 0
                            else None
                        ),
                        "superiority_gate_passed": (
                            len(common) >= 5 and float(result["ci95_low"]) > 0
                        ),
                    }
                    | result,
                )
    return {
        "claim_scope": "multilingual_eu_law_transfer",
        "national_jurisdiction_claim_permitted": False,
        "matrix_complete": not missing,
        "missing_runs": missing,
        "accepted_run_count": len(runs),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "flen_superiority_evaluated": "flen" in methods,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_micro_f1(path: Path, summaries: list[dict[str, Any]], methods: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holdouts = ["en-de", "en-fr", "en-es", "en-pl", "mean_across_holdouts"]
    labels = ["DE", "FR", "ES", "PL", "Mean"]
    width = 0.8 / len(methods)
    positions = np.arange(len(holdouts), dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    colors = {"fedavg": "#3B6FB6", "fedprox": "#D97941", "lambda0": "#4E9A68"}
    display_names = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "lambda0": r"$\lambda=0$ control",
    }
    for method_index, method in enumerate(methods):
        rows = {
            row["holdout"]: row
            for row in summaries
            if row["method"] == method and row["metric"] == "micro_f1"
        }
        means = [float(rows[holdout]["mean"]) for holdout in holdouts]
        deviations = [float(rows[holdout]["standard_deviation"]) for holdout in holdouts]
        offset = (method_index - (len(methods) - 1) / 2) * width
        axis.bar(
            positions + offset,
            means,
            width=width,
            yerr=deviations,
            capsize=3,
            color=colors.get(method),
            label=display_names.get(method, method),
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Held-out micro-F1")
    axis.set_ylim(0, 0.5)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=len(methods), loc="upper center")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["fedavg", "fedprox", "lambda0"])
    parser.add_argument("--holdouts", nargs="+", default=["en-de", "en-fr", "en-es", "en-pl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_accepted_runs(args.input)
    analysis = analyze(runs, args.methods, args.holdouts, args.seeds)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "accepted_runs.csv", runs)
    write_csv(args.output / "summary.csv", analysis["summaries"])
    write_csv(args.output / "paired_comparisons.csv", analysis["paired_comparisons"])
    plot_micro_f1(args.output / "heldout_micro_f1.pdf", analysis["summaries"], args.methods)
    print(json.dumps({key: analysis[key] for key in ("accepted_run_count", "matrix_complete", "missing_runs")}, indent=2))


if __name__ == "__main__":
    main()
