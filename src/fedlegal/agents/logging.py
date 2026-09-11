"""Reasoning trace logging."""

from __future__ import annotations

import json
from pathlib import Path

from fedlegal.agents.state import LegalReasoningState


class ReasoningTraceLogger:
    """Write agent traces and final verdicts to JSONL."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{run_name}_reasoning_traces.jsonl"

    def log_state(self, state: LegalReasoningState) -> None:
        """Append all traces for a completed workflow."""

        with self.path.open("a", encoding="utf-8") as handle:
            for trace in state.memory.traces:
                handle.write(
                    json.dumps(
                        {
                            "case_id": state.case.case_id,
                            **trace.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            if state.verdict is not None:
                handle.write(
                    json.dumps(
                        {
                            "case_id": state.case.case_id,
                            "agent": "JudgeAgent",
                            "action": "final_verdict",
                            "verdict": state.verdict.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
