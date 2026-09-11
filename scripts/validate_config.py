"""Validate an experiment YAML file against the typed config schema."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to an experiment YAML config.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    print(json.dumps(config.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
