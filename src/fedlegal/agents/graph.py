"""LangGraph orchestration for federated multi-agent judicial reasoning."""

from __future__ import annotations

from dataclasses import dataclass

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
    FederatedReasoningPacket,
    LegalReasoningState,
)
from fedlegal.config.schemas import AgentConfig, ConflictConfig
from fedlegal.reasoning.case_schema import LegalCase


@dataclass(frozen=True)
class AgentGraphPlan:
    """Role and exchange contract for the future LangGraph workflow."""

    framework: str
    roles: tuple[str, ...]
    exchange_format: str


def build_agent_graph_plan(config: AgentConfig) -> AgentGraphPlan:
    """Create a LangGraph plan without constructing runtime nodes."""

    return AgentGraphPlan(
        framework=config.framework,
        roles=tuple(config.roles),
        exchange_format=config.exchange_format,
    )


class FederatedLegalReasoningGraph:
    """Federated multi-agent legal reasoning workflow.

    LangGraph orchestration is used when available. A deterministic fallback is
    provided so tests and dry runs work in minimal environments.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        conflict_config: ConflictConfig | None = None,
        trace_logger: ReasoningTraceLogger | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.conflict_config = conflict_config or ConflictConfig()
        self.trace_logger = trace_logger
        self.prosecutor = ProsecutorAgent()
        self.defense = DefenseAgent()
        self.citation = CitationAgent()
        self.conflict = ConflictAgent(self.conflict_config)
        self.judge = JudgeAgent()
        self._compiled_graph = self._try_compile_langgraph()

    def run(
        self,
        case: LegalCase,
        federated_packets: list[FederatedReasoningPacket] | None = None,
    ) -> LegalReasoningState:
        """Run debate, citation verification, conflict detection, and verdict aggregation."""

        state = LegalReasoningState(
            case=case,
            memory=AgentMemory(case_id=case.case_id),
            federated_packets=federated_packets or [],
        )
        if self._compiled_graph is not None:
            result = self._compiled_graph.invoke(state)
            state = result if isinstance(result, LegalReasoningState) else LegalReasoningState(**result)
        else:
            state = self._run_fallback(state)
        if self.trace_logger is not None:
            self.trace_logger.log_state(state)
        return state

    def _run_fallback(self, state: LegalReasoningState) -> LegalReasoningState:
        for node in (self.prosecutor, self.defense, self.citation, self.conflict, self.judge):
            state = node.run(state)
        return state

    def _try_compile_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(LegalReasoningState)
        graph.add_node("prosecutor", self.prosecutor.run)
        graph.add_node("defense", self.defense.run)
        graph.add_node("citation", self.citation.run)
        graph.add_node("conflict", self.conflict.run)
        graph.add_node("judge", self.judge.run)
        graph.set_entry_point("prosecutor")
        graph.add_edge("prosecutor", "defense")
        graph.add_edge("defense", "citation")
        graph.add_edge("citation", "conflict")
        graph.add_edge("conflict", "judge")
        graph.add_edge("judge", END)
        return graph.compile()
