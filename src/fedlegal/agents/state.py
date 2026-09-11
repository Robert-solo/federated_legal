"""State, memory, and trace schemas for federated legal agents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from fedlegal.reasoning.case_schema import LegalCase

AgentRole = Literal["prosecutor", "defense", "judge", "citation", "conflict"]


class ReasoningTrace(BaseModel):
    """One auditable reasoning step.

    Traces should contain abstract reasoning and citations only. Raw local
    client data should not be exchanged across federated clients.
    """

    agent: str
    action: str
    content: str
    citations: list[str] = Field(default_factory=list)
    client_id: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebateTurn(BaseModel):
    """One prosecutor or defense debate turn."""

    role: Literal["prosecutor", "defense"]
    argument: str
    cited_authorities: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class AgentMemory(BaseModel):
    """Reasoning memory retained during one case workflow."""

    case_id: str
    traces: list[ReasoningTrace] = Field(default_factory=list)
    debate: list[DebateTurn] = Field(default_factory=list)
    verified_citations: dict[str, bool] = Field(default_factory=dict)
    conflict_flags: list[str] = Field(default_factory=list)

    def add_trace(self, trace: ReasoningTrace) -> None:
        self.traces.append(trace)

    def add_debate_turn(self, turn: DebateTurn) -> None:
        self.debate.append(turn)


class FederatedReasoningPacket(BaseModel):
    """Federated-safe client reasoning exchange object."""

    client_id: str
    jurisdiction: str
    reasoning_embedding: list[float] = Field(default_factory=list)
    verdict_distribution: list[float] = Field(default_factory=list)
    compressed_legal_representation: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    """Final judicial output from JudgeAgent."""

    label: str
    rationale: str
    confidence: float
    supporting_citations: list[str] = Field(default_factory=list)
    client_weights: dict[str, float] = Field(default_factory=dict)


class LegalReasoningState(BaseModel):
    """State passed through the LangGraph workflow."""

    case: LegalCase
    memory: AgentMemory
    federated_packets: list[FederatedReasoningPacket] = Field(default_factory=list)
    prosecutor_argument: DebateTurn | None = None
    defense_argument: DebateTurn | None = None
    citation_report: dict[str, bool] = Field(default_factory=dict)
    conflict_report: dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict | None = None
