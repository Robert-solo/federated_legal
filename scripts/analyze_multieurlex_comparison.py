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
        method, holdout, seed = match.groups()
        if metrics.get("method") != method or metrics.get("holdout") != holdout:
            raise ValueError(f"Directory and metrics disagree for {directory}")
        if int(metrics.get("seed")) != int(seed):
            raise ValueError(f"Directory and seed disagree for {directory}")
        if metrics.get("claim_scope") != "multilingual_eu_law_transfer":
            raise ValueError(f"Invalid claim scope for {directory}")
        runs.append(
            {
                "method": method,
                "holdout": holdout,
                "seed": int(seed),
                "micro_f1": float(metrics[f"{holdout}_micro_f1"]),
                "macro_f1": float(metrics[f"{holdout}_macro_f1"]),
                "loss": float(metrics[f"{holdout}_loss"]),
                "threshold": float(metrics["threshold"]),
                "communication_bytes": int(metrics["logical_communication_bytes"]),
                "mean_update_l2": float(metrics["mean_update_l2"]),
                "mean_pairwise_update_l2": float(metrics["mean_pairwise_update_l2"]),
                "mean_fedprox_penalty": float(metrics["mean_fedprox_penalty"]),
                "path": str(directory),
            }
        )
    return runs


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
    print(json.dumps({key: analysis[key] for key in ("accepted_run_count", "matrix_complete", "missing_runs")}, indent=2))


if __name__ == "__main__":
    main()
