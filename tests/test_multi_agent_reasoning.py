from pathlib import Path
from tempfile import TemporaryDirectory

from fedlegal.agents import (
    CitationAgent,
    ConflictAgent,
    DefenseAgent,
    FederatedLegalReasoningGraph,
    FederatedReasoningPacket,
    JudgeAgent,
    ProsecutorAgent,
    ReasoningTraceLogger,
)
from fedlegal.config import load_experiment_config
from fedlegal.reasoning import LegalCase


def test_required_agents_exist() -> None:
    assert ProsecutorAgent().name == "ProsecutorAgent"
    assert DefenseAgent().name == "DefenseAgent"
    assert JudgeAgent().name == "JudgeAgent"
    assert CitationAgent().name == "CitationAgent"
    assert ConflictAgent().name == "ConflictAgent"


def test_federated_reasoning_graph_runs_debate_and_verdict() -> None:
    config = load_experiment_config("configs/experiments/federated_multi_agent_reasoning.yaml")
    case = LegalCase(
        case_id="case-001",
        jurisdiction="us_common_law",
        facts=["The plaintiff alleges a rights violation under 42 U.S.C. § 1983."],
        legal_issues=["civil rights liability"],
        evidence=["The record includes Article 6 compliance objections."],
    )
    packets = [
        FederatedReasoningPacket(
            client_id="court_us_01",
            jurisdiction="us_common_law",
            reasoning_embedding=[0.8, 0.2, 0.1],
            verdict_distribution=[0.75, 0.25],
            compressed_legal_representation={
                "citations": ["42 U.S.C. § 1983"],
                "legal_tradition": "common_law",
            },
        ),
        FederatedReasoningPacket(
            client_id="institute_eu_01",
            jurisdiction="european_regulatory_law",
            reasoning_embedding=[0.2, 0.8, 0.1],
            verdict_distribution=[0.35, 0.65],
            compressed_legal_representation={
                "citations": ["Article 6"],
                "legal_tradition": "civil_law_regulatory",
                "contradiction_score": 0.4,
            },
        ),
    ]

    state = FederatedLegalReasoningGraph(config.agents, config.conflict).run(case, packets)

    assert state.verdict is not None
    assert state.prosecutor_argument is not None
    assert state.defense_argument is not None
    assert "42 U.S.C. § 1983" in state.citation_report
    assert state.conflict_report["total_conflict"] > 0
    assert len(state.memory.traces) >= 5


def test_reasoning_trace_logger_writes_jsonl() -> None:
    config = load_experiment_config("configs/experiments/federated_multi_agent_reasoning.yaml")
    case = LegalCase(
        case_id="case-002",
        jurisdiction="contract_law",
        facts=["Clause 12 restricts assignment."],
        legal_issues=["contract assignment"],
    )
    with TemporaryDirectory() as tmp:
        logger = ReasoningTraceLogger(tmp, "reasoning_test")
        state = FederatedLegalReasoningGraph(config.agents, config.conflict, logger).run(case)
        path = Path(tmp) / "reasoning_test_reasoning_traces.jsonl"

        assert state.verdict is not None
        assert path.exists()
        assert "final_verdict" in path.read_text(encoding="utf-8")
