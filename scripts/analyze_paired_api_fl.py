"""Compare API and federated predictions on identical CaseHOLD examples."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-predictions", type=Path, required=True)
    parser.add_argument("--fl-predictions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api_rows = _read_jsonl(args.api_predictions)[: args.limit]
    fl_rows = _read_jsonl(args.fl_predictions)[: args.limit]
    if len(api_rows) != len(fl_rows):
        raise ValueError(f"Prediction count mismatch: API={len(api_rows)}, FL={len(fl_rows)}")

    pairs = []
    for index, (api, federated) in enumerate(zip(api_rows, fl_rows, strict=True)):
        if int(api["gold_label"]) != int(federated["gold_label"]):
            raise ValueError(f"Gold label mismatch at index {index}")
        pairs.append(
            {
                "index": index,
                "gold_label": int(api["gold_label"]),
                "api_correct": bool(api["correct"]),
                "fl_correct": bool(federated["correct"]),
                "api_prediction": int(api["predicted_label"]),
                "fl_prediction": int(federated["predicted_label"]),
            }
        )

    api_correct = sum(item["api_correct"] for item in pairs)
    fl_correct = sum(item["fl_correct"] for item in pairs)
    api_only = sum(item["api_correct"] and not item["fl_correct"] for item in pairs)
    fl_only = sum(item["fl_correct"] and not item["api_correct"] for item in pairs)
    difference = (api_correct - fl_correct) / len(pairs) if pairs else 0.0
    bootstrap_low, bootstrap_high = paired_bootstrap_ci(
        pairs, args.seed, args.bootstrap_iterations
    )
    result = {
        "num_examples": len(pairs),
        "api_correct": api_correct,
        "api_accuracy": api_correct / len(pairs) if pairs else 0.0,
        "fl_correct": fl_correct,
        "fl_accuracy": fl_correct / len(pairs) if pairs else 0.0,
        "api_minus_fl_accuracy_difference": difference,
        "paired_bootstrap_ci95": [bootstrap_low, bootstrap_high],
        "mcnemar": {
            "api_only_correct": api_only,
            "fl_only_correct": fl_only,
            "exact_two_sided_p": exact_mcnemar_p(api_only, fl_only),
        },
        "paired_by_index": pairs,
        "interpretation": (
            "Paired comparison on identical validation examples; the federated result is one "
            "FedAvg checkpoint/seed and does not establish a multi-seed superiority claim."
        ),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "paired_by_index"}, ensure_ascii=False, indent=2, sort_keys=True))


def paired_bootstrap_ci(pairs: list[dict[str, object]], seed: int, iterations: int) -> tuple[float, float]:
    rng = random.Random(seed)
    if not pairs:
        return 0.0, 0.0
    differences = [int(item["api_correct"]) - int(item["fl_correct"]) for item in pairs]
    samples = []
    for _ in range(iterations):
        samples.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def exact_mcnemar_p(api_only: int, fl_only: int) -> float:
    discordant = api_only + fl_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(api_only, fl_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
