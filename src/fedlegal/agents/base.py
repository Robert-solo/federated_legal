"""Base classes for deterministic legal reasoning agents."""

from __future__ import annotations

from dataclasses import dataclass

from fedlegal.agents.state import LegalReasoningState, ReasoningTrace


@dataclass
class BaseLegalAgent:
    """Base interface for legal reasoning agents."""

    name: str

    def add_trace(
        self,
        state: LegalReasoningState,
        action: str,
        content: str,
        citations: list[str] | None = None,
    ) -> None:
        state.memory.add_trace(
            ReasoningTrace(
                agent=self.name,
                action=action,
                content=content,
                citations=citations or [],
            )
        )

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        raise NotImplementedError
