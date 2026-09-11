"""Reasoning memory utilities."""

from __future__ import annotations

from fedlegal.agents.state import AgentMemory, ReasoningTrace


class ReasoningMemoryStore:
    """In-memory trace store keyed by case ID."""

    def __init__(self) -> None:
        self._items: dict[str, AgentMemory] = {}

    def get_or_create(self, case_id: str) -> AgentMemory:
        """Return memory for a case."""

        if case_id not in self._items:
            self._items[case_id] = AgentMemory(case_id=case_id)
        return self._items[case_id]

    def append(self, case_id: str, trace: ReasoningTrace) -> None:
        """Append one trace."""

        self.get_or_create(case_id).add_trace(trace)
