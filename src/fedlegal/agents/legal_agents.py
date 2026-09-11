"""Federated multi-agent legal reasoning roles."""

from __future__ import annotations

from collections import Counter

from fedlegal.agents.base import BaseLegalAgent
from fedlegal.agents.state import DebateTurn, LegalReasoningState, Verdict
from fedlegal.aggregation.legal_conflict_metrics import LegalConflictProfile, build_conflict_report
from fedlegal.config.schemas import ConflictConfig
from fedlegal.data.citations import extract_citations


class ProsecutorAgent(BaseLegalAgent):
    """Build accusation-side legal reasoning."""

    def __init__(self) -> None:
        super().__init__("ProsecutorAgent")

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        issue = _first_or_default(state.case.legal_issues, "legal liability")
        facts = "; ".join(state.case.facts[:3]) or "submitted facts"
        citations = extract_citations(" ".join(state.case.facts + state.case.evidence))
        argument = (
            f"The prosecution position is that the facts support {issue}. "
            f"Key factual basis: {facts}."
        )
        turn = DebateTurn(
            role="prosecutor",
            argument=argument,
            cited_authorities=citations,
            confidence=0.65 if citations else 0.55,
        )
        state.prosecutor_argument = turn
        state.memory.add_debate_turn(turn)
        self.add_trace(state, "accusation_reasoning", argument, citations)
        return state


class DefenseAgent(BaseLegalAgent):
    """Build rebuttal-side legal reasoning."""

    def __init__(self) -> None:
        super().__init__("DefenseAgent")

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        issue = _first_or_default(state.case.legal_issues, "legal liability")
        evidence = "; ".join(state.case.evidence[:3]) or "available evidence"
        citations = extract_citations(" ".join(state.case.evidence + state.case.facts))
        argument = (
            f"The defense position contests {issue} by emphasizing uncertainty, "
            f"alternative interpretation, and evidentiary limits: {evidence}."
        )
        turn = DebateTurn(
            role="defense",
            argument=argument,
            cited_authorities=citations,
            confidence=0.62 if evidence else 0.5,
        )
        state.defense_argument = turn
        state.memory.add_debate_turn(turn)
        self.add_trace(state, "rebuttal_reasoning", argument, citations)
        return state


class CitationAgent(BaseLegalAgent):
    """Verify citations used by debate agents."""

    def __init__(self) -> None:
        super().__init__("CitationAgent")

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        cited = []
        for turn in state.memory.debate:
            cited.extend(turn.cited_authorities)
        report = {citation: _looks_like_legal_citation(citation) for citation in sorted(set(cited))}
        state.citation_report = report
        state.memory.verified_citations.update(report)
        invalid = [citation for citation, ok in report.items() if not ok]
        content = (
            f"Verified {len(report)} citations; invalid citations: {invalid or 'none'}."
        )
        self.add_trace(state, "citation_verification", content, list(report))
        return state


class ConflictAgent(BaseLegalAgent):
    """Detect debate and federated reasoning conflicts."""

    def __init__(self, conflict_config: ConflictConfig | None = None) -> None:
        super().__init__("ConflictAgent")
        self.conflict_config = conflict_config or ConflictConfig()

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        flags: list[str] = []
        if state.prosecutor_argument and state.defense_argument:
            confidence_gap = abs(
                state.prosecutor_argument.confidence - state.defense_argument.confidence
            )
            if confidence_gap < 0.1:
                flags.append("balanced_adversarial_positions")
            if set(state.prosecutor_argument.cited_authorities).isdisjoint(
                set(state.defense_argument.cited_authorities)
            ) and state.prosecutor_argument.cited_authorities and state.defense_argument.cited_authorities:
                flags.append("citation_divergence_between_parties")

        profiles = [
            LegalConflictProfile(
                client_id=packet.client_id,
                jurisdiction=packet.jurisdiction,
                reasoning_embedding=tuple(packet.reasoning_embedding),
                verdict_distribution=tuple(packet.verdict_distribution),
                citations=tuple(packet.compressed_legal_representation.get("citations", ())),
                legal_tradition=str(packet.compressed_legal_representation.get("legal_tradition", "")),
                contradiction_score=float(
                    packet.compressed_legal_representation.get("contradiction_score", 0.0)
                ),
            )
            for packet in state.federated_packets
        ]
        report = build_conflict_report(profiles, self.conflict_config)
        if report.total_conflict > 0.5:
            flags.append("high_cross_jurisdiction_conflict")
        state.memory.conflict_flags.extend(flags)
        state.conflict_report = {
            "flags": flags,
            "total_conflict": report.total_conflict,
            "citation_conflict": report.citation_conflict,
            "verdict_conflict": report.verdict_conflict,
            "rule_alignment_distance": report.rule_alignment_distance,
        }
        self.add_trace(state, "conflict_detection", str(state.conflict_report))
        return state


class JudgeAgent(BaseLegalAgent):
    """Aggregate debate turns and federated verdict distributions."""

    def __init__(self) -> None:
        super().__init__("JudgeAgent")

    def run(self, state: LegalReasoningState) -> LegalReasoningState:
        prosecutor_score = state.prosecutor_argument.confidence if state.prosecutor_argument else 0.0
        defense_score = state.defense_argument.confidence if state.defense_argument else 0.0
        federated_scores = _aggregate_federated_verdicts(state)
        if federated_scores:
            prosecutor_score += federated_scores.get("liability", 0.0)
            defense_score += federated_scores.get("no_liability", 0.0)

        label = "liability_supported" if prosecutor_score >= defense_score else "liability_not_established"
        citations = [
            citation
            for citation, valid in state.citation_report.items()
            if valid
        ]
        confidence = max(prosecutor_score, defense_score) / max(
            prosecutor_score + defense_score,
            1.0,
        )
        rationale = (
            "The judge aggregates adversarial debate, citation verification, "
            "cross-client verdict distributions, and conflict flags before issuing "
            f"the final result: {label}."
        )
        state.verdict = Verdict(
            label=label,
            rationale=rationale,
            confidence=round(confidence, 4),
            supporting_citations=citations,
            client_weights=_client_weights(state),
        )
        self.add_trace(state, "verdict_aggregation", rationale, citations)
        return state


def _aggregate_federated_verdicts(state: LegalReasoningState) -> dict[str, float]:
    totals = Counter()
    for packet in state.federated_packets:
        if len(packet.verdict_distribution) >= 2:
            totals["liability"] += packet.verdict_distribution[0]
            totals["no_liability"] += packet.verdict_distribution[1]
    count = max(len(state.federated_packets), 1)
    return {key: value / count for key, value in totals.items()}


def _client_weights(state: LegalReasoningState) -> dict[str, float]:
    if not state.federated_packets:
        return {}
    weight = 1.0 / len(state.federated_packets)
    return {packet.client_id: weight for packet in state.federated_packets}


def _first_or_default(values: list[str], default: str) -> str:
    return values[0] if values else default


def _looks_like_legal_citation(citation: str) -> bool:
    return bool(extract_citations(citation, [citation]))
