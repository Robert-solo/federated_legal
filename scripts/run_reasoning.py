"""Run federated multi-agent legal reasoning on a case JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.agents import (
    FederatedLegalReasoningGraph,
    FederatedReasoningPacket,
    ReasoningTraceLogger,
)
from fedlegal.config import load_experiment_config
from fedlegal.reasoning import LegalCase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Experiment config with agents section.")
    parser.add_argument("case", help="Case JSON file.")
    parser.add_argument("--packets", help="Optional federated packet JSON file.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    case = LegalCase.model_validate(json.loads(Path(args.case).read_text(encoding="utf-8")))
    packets = []
    if args.packets:
        payload = json.loads(Path(args.packets).read_text(encoding="utf-8"))
        packets = [FederatedReasoningPacket.model_validate(item) for item in payload]

    graph = FederatedLegalReasoningGraph(
        config.agents,
        conflict_config=config.conflict,
        trace_logger=ReasoningTraceLogger(config.log_dir, config.name),
    )
    state = graph.run(case, packets)
    print(json.dumps(state.verdict.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
