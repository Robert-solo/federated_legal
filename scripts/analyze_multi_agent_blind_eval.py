"""Analyze paired automatic outcomes from the blinded multi-agent evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def exact_mcnemar_pvalue(single_only: int, multi_only: int) -> float:
    discordant = single_only + multi_only
    if discordant == 0:
        return 1.0
    smaller = min(single_only, multi_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def analyze(
    rows: list[dict[str, Any]], seed: int = 41, repetitions: int = 10_000
) -> dict[str, Any]:
    single = np.asarray([bool(row["single_correct"]) for row in rows], dtype=float)
    multi = np.asarray([bool(row["multi_correct"]) for row in rows], dtype=float)
    differences = multi - single
    generator = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            differences[generator.integers(0, len(rows), len(rows))].mean()
            for _ in range(repetitions)
        ]
    )
    single_only = int(np.logical_and(single == 1, multi == 0).sum())
    multi_only = int(np.logical_and(single == 0, multi == 1).sum())
    return {
        "cases": len(rows),
        "single_accuracy": float(single.mean()),
        "multi_accuracy": float(multi.mean()),
        "paired_accuracy_difference": float(differences.mean()),
        "paired_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "single_only_correct": single_only,
        "multi_only_correct": multi_only,
        "both_correct": int(np.logical_and(single == 1, multi == 1).sum()),
        "both_incorrect": int(np.logical_and(single == 0, multi == 0).sum()),
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(single_only, multi_only),
        "human_review_status": "awaiting_two_blinded_reviewers",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = analyze(rows, args.seed, args.bootstrap_repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
