"""LangGraph multi-agent legal reasoning skeleton."""

from fedlegal.agents.graph import FederatedLegalReasoningGraph, build_agent_graph_plan
from fedlegal.agents.legal_agents import (
    CitationAgent,
    ConflictAgent,
    DefenseAgent,
    JudgeAgent,
    ProsecutorAgent,
)
from fedlegal.agents.logging import ReasoningTraceLogger
from fedlegal.agents.state import (
    AgentMemory,
    DebateTurn,
    FederatedReasoningPacket,
    LegalReasoningState,
    ReasoningTrace,
    Verdict,
)

__all__ = [
    "AgentMemory",
    "CitationAgent",
    "ConflictAgent",
    "DebateTurn",
    "DefenseAgent",
    "FederatedLegalReasoningGraph",
    "FederatedReasoningPacket",
    "JudgeAgent",
    "LegalReasoningState",
    "ProsecutorAgent",
    "ReasoningTrace",
    "ReasoningTraceLogger",
    "Verdict",
    "build_agent_graph_plan",
]
