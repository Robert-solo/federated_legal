"""Run federated legal LLM orchestration.

The default mode is a dry run that validates Flower-compatible orchestration,
communication logging, and checkpointing without model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.config import load_experiment_config
from fedlegal.federated import FlowerServerOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML.")
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Use runtime Flower simulation. Full LLM training is not implemented yet.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    orchestrator = FlowerServerOrchestrator(config, dry_run=not args.runtime)
    result = (
        orchestrator.run_flower_simulation()
        if args.runtime
        else orchestrator.run_dry()
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
