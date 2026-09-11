"""Run evaluation and generate report artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.config import load_experiment_config
from fedlegal.evaluation import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Experiment config YAML.")
    parser.add_argument("--predictions", help="Prediction JSON/JSONL file.")
    parser.add_argument("--communication-log", help="Federated communication JSONL file.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    report = EvaluationRunner(config).run(
        predictions_path=args.predictions,
        communication_log_path=args.communication_log,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
