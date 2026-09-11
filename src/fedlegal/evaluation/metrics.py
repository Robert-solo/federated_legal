"""Evaluation metrics for federated legal LLM experiments."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from fedlegal.aggregation.citation_divergence import normalize_citation_set
from fedlegal.evaluation.schemas import EvaluationExample, MetricResult


def citation_accuracy(examples: list[EvaluationExample]) -> MetricResult:
    """Measure citation precision against gold citations or citation syntax."""

    correct = 0
    total = 0
    for example in examples:
        predicted = normalize_citation_set(tuple(example.predicted_citations))
        gold = normalize_citation_set(tuple(example.gold_citations))
        if gold:
            correct += len(predicted & gold)
            total += len(predicted)
        else:
            correct += sum(1 for citation in predicted if citation)
            total += len(predicted)
    value = correct / total if total else 0.0
    return MetricResult(
        name="citation_accuracy",
        value=value,
        details={"correct_citations": correct, "predicted_citations": total},
    )


def citation_consistency(examples: list[EvaluationExample]) -> MetricResult:
    """Measure citation agreement between predicted and gold authorities."""

    similarities: list[float] = []
    for example in examples:
        predicted = normalize_citation_set(tuple(example.predicted_citations))
        gold = normalize_citation_set(tuple(example.gold_citations))
        if not predicted and not gold:
            continue
        union = predicted | gold
        if not union:
            continue
        similarities.append(len(predicted & gold) / len(union))
    value = sum(similarities) / len(similarities) if similarities else 0.0
    return MetricResult(
        name="citation_consistency",
        value=value,
        details={"num_examples": len(similarities)},
    )


def legal_consistency(examples: list[EvaluationExample]) -> MetricResult:
    """Measure legal consistency using explicit scores or label agreement."""

    explicit = [item.legal_consistency_score for item in examples if item.legal_consistency_score is not None]
    if explicit:
        value = sum(explicit) / len(explicit)
        return MetricResult(name="legal_consistency", value=value, details={"source": "explicit"})

    comparable = [
        item for item in examples if item.predicted_label is not None and item.gold_label is not None
    ]
    correct = sum(1 for item in comparable if _labels_match(item.predicted_label, item.gold_label))
    value = correct / len(comparable) if comparable else 0.0
    return MetricResult(
        name="legal_consistency",
        value=value,
        details={"correct_labels": correct, "labeled_examples": len(comparable)},
    )


def legal_accuracy(examples: list[EvaluationExample]) -> MetricResult:
    """Measure exact legal answer accuracy."""

    comparable = [
        item for item in examples if item.predicted_label is not None and item.gold_label is not None
    ]
    correct = sum(1 for item in comparable if _labels_match(item.predicted_label, item.gold_label))
    value = correct / len(comparable) if comparable else 0.0
    return MetricResult(
        name="legal_accuracy",
        value=value,
        details={"correct_labels": correct, "labeled_examples": len(comparable)},
    )


def micro_f1(examples: list[EvaluationExample]) -> MetricResult:
    """Compute micro-F1 for multi-label legal classification tasks."""

    true_positive = 0
    false_positive = 0
    false_negative = 0
    evaluated = 0
    for example in examples:
        if example.predicted_label is None or example.gold_label is None:
            continue
        predicted = set(_label_values(example.predicted_label))
        gold = set(_label_values(example.gold_label))
        if not predicted and not gold:
            continue
        evaluated += 1
        true_positive += len(predicted & gold)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    value = (2 * true_positive / denominator) if denominator else 0.0
    return MetricResult(
        name="micro_f1",
        value=value,
        details={
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "evaluated_examples": evaluated,
        },
    )


def exact_match_f1(examples: list[EvaluationExample]) -> MetricResult:
    """Compute normalized exact match plus token F1 for extractive legal QA."""

    exact = 0
    token_f1_sum = 0.0
    evaluated = 0
    for example in examples:
        if example.predicted_label is None or example.gold_label is None:
            continue
        prediction = _normalize_text_answer(str(example.predicted_label))
        gold = _normalize_text_answer(str(example.gold_label))
        if not gold:
            continue
        evaluated += 1
        exact += int(prediction == gold)
        token_f1_sum += _token_f1(prediction, gold)
    exact_match = exact / evaluated if evaluated else 0.0
    mean_token_f1 = token_f1_sum / evaluated if evaluated else 0.0
    return MetricResult(
        name="exact_match_f1",
        value=mean_token_f1,
        details={
            "exact_match": exact_match,
            "token_f1": mean_token_f1,
            "evaluated_examples": evaluated,
        },
    )


def hallucination_rate(examples: list[EvaluationExample]) -> MetricResult:
    """Estimate unsupported generation rate.

    A generated answer is treated as supported when it either cites a gold
    authority or overlaps with provided supported facts. This is intentionally
    conservative and dependency-light until model-based factuality checks are
    added.
    """

    hallucinated = 0
    evaluated = 0
    for example in examples:
        if not example.generated_text:
            continue
        evaluated += 1
        text = example.generated_text.lower()
        gold_citations = normalize_citation_set(tuple(example.gold_citations))
        predicted_citations = normalize_citation_set(tuple(example.predicted_citations))
        has_supported_citation = bool(gold_citations and predicted_citations & gold_citations)
        has_fact_overlap = any(fact.lower() in text for fact in example.supported_facts if fact)
        if not has_supported_citation and not has_fact_overlap:
            hallucinated += 1
    value = hallucinated / evaluated if evaluated else 0.0
    return MetricResult(
        name="hallucination_rate",
        value=value,
        details={"hallucinated_examples": hallucinated, "evaluated_examples": evaluated},
    )


def reasoning_coherence(examples: list[EvaluationExample]) -> MetricResult:
    """Estimate reasoning coherence from explicit scores or generation quality."""

    explicit = [item.metadata.get("reasoning_score") for item in examples if item.metadata.get("reasoning_score") is not None]
    if explicit:
        scores = [float(value) for value in explicit]
        return MetricResult(
            name="reasoning_coherence",
            value=sum(scores) / len(scores),
            details={"source": "explicit", "count": len(scores)},
        )

    supported = []
    for example in examples:
        if not example.generated_text:
            continue
        text = example.generated_text.strip()
        supported.append(1.0 if text else 0.0)
    value = sum(supported) / len(supported) if supported else 0.0
    return MetricResult(
        name="reasoning_coherence",
        value=value,
        details={"source": "generation_presence", "count": len(supported)},
    )


def communication_cost(log_path: str | Path | None) -> MetricResult:
    """Sum federated communication bytes from JSONL communication logs."""

    if not log_path or not Path(log_path).exists():
        return MetricResult(name="communication_cost", value=0.0, details={"num_events": 0})
    total_bytes = 0
    events = 0
    for event in _read_jsonl(log_path):
        total_bytes += int(event.get("num_bytes", 0))
        events += 1
    return MetricResult(
        name="communication_cost",
        value=float(total_bytes),
        details={"num_events": events, "unit": "bytes"},
    )


def client_drift(log_path: str | Path | None) -> MetricResult:
    """Estimate client drift from logged drift metrics or update norms."""

    if not log_path or not Path(log_path).exists():
        return MetricResult(name="client_drift", value=0.0, details={"source": "missing_log"})

    explicit: list[float] = []
    round_norms: dict[int, list[float]] = {}
    for event in _read_jsonl(log_path):
        metrics = event.get("metrics", {})
        if "client_drift" in metrics:
            explicit.append(float(metrics["client_drift"]))
        if "update_norm" in metrics:
            round_norms.setdefault(int(event.get("round_id", 0)), []).append(float(metrics["update_norm"]))

    if explicit:
        return MetricResult(
            name="client_drift",
            value=sum(explicit) / len(explicit),
            details={"source": "explicit_client_drift", "count": len(explicit)},
        )

    pairwise_distances: list[float] = []
    for norms in round_norms.values():
        for left, right in combinations(norms, 2):
            pairwise_distances.append(abs(left - right))
    value = sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0.0
    return MetricResult(
        name="client_drift",
        value=value,
        details={"source": "update_norm_pairwise_distance", "count": len(pairwise_distances)},
    )


def cross_jurisdiction_generalization(examples: list[EvaluationExample]) -> MetricResult:
    """Average legal accuracy across jurisdiction groups."""

    grouped: dict[str, list[EvaluationExample]] = {}
    for example in examples:
        jurisdiction = str(example.metadata.get("jurisdiction", "unknown"))
        grouped.setdefault(jurisdiction, []).append(example)

    scores: list[float] = []
    for jurisdiction, items in grouped.items():
        comparable = [
            item for item in items if item.predicted_label is not None and item.gold_label is not None
        ]
        if not comparable:
            continue
        correct = sum(1 for item in comparable if item.predicted_label == item.gold_label)
        scores.append(correct / len(comparable))
    value = sum(scores) / len(scores) if scores else 0.0
    return MetricResult(
        name="cross_jurisdiction_generalization",
        value=value,
        details={"jurisdictions": len(scores)},
    )


def privacy_leakage_risk(examples: list[EvaluationExample]) -> MetricResult:
    """Heuristic privacy leakage risk from sensitive-token repetition."""

    leaked = 0
    evaluated = 0
    for example in examples:
        tokens = [str(item).lower() for item in example.metadata.get("privacy_sensitive_tokens", []) if item]
        if not tokens:
            continue
        evaluated += 1
        text = example.generated_text.lower()
        if any(token in text for token in tokens):
            leaked += 1
    value = leaked / evaluated if evaluated else 0.0
    return MetricResult(
        name="privacy_leakage_risk",
        value=value,
        details={"evaluated_examples": evaluated, "leaked_examples": leaked},
    )


def aggregation_stability(log_path: str | Path | None) -> MetricResult:
    """Estimate aggregation stability from round-wise update statistics."""

    if not log_path or not Path(log_path).exists():
        return MetricResult(name="aggregation_stability", value=0.0, details={"source": "missing_log"})

    round_norms: dict[int, list[float]] = {}
    round_drift: list[float] = []
    for event in _read_jsonl(log_path):
        round_id = int(event.get("round_id", event.get("round", 0)))
        metrics = event.get("metrics", {})
        if "update_norm" in metrics:
            round_norms.setdefault(round_id, []).append(float(metrics["update_norm"]))
        if "client_drift_l2" in metrics:
            round_drift.append(float(metrics["client_drift_l2"]))

    std_terms = []
    for norms in round_norms.values():
        if len(norms) > 1:
            mean = sum(norms) / len(norms)
            variance = sum((value - mean) ** 2 for value in norms) / len(norms)
            std_terms.append(variance ** 0.5)
    if std_terms:
        mean_std = sum(std_terms) / len(std_terms)
        value = 1.0 / (1.0 + mean_std)
        return MetricResult(
            name="aggregation_stability",
            value=value,
            details={"source": "round_update_std", "rounds": len(std_terms)},
        )

    if round_drift:
        mean_drift = sum(round_drift) / len(round_drift)
        value = 1.0 / (1.0 + mean_drift)
        return MetricResult(
            name="aggregation_stability",
            value=value,
            details={"source": "client_drift_l2", "count": len(round_drift)},
        )

    return MetricResult(name="aggregation_stability", value=0.0, details={"source": "empty_log"})


def load_examples(path: str | Path | None) -> list[EvaluationExample]:
    """Load evaluation examples from JSONL or JSON."""

    if not path:
        return []
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation prediction file not found: {input_path}")
    if input_path.suffix.lower() == ".jsonl":
        return [EvaluationExample.model_validate(item) for item in _read_jsonl(input_path)]
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [EvaluationExample.model_validate(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return [EvaluationExample.model_validate(item) for item in payload["examples"]]
    raise ValueError(f"Unsupported evaluation prediction format: {input_path}")


def mean_prediction_confidence(examples: list[EvaluationExample]) -> MetricResult:
    """Placeholder for completeness when confidence scores are provided."""

    values = [float(item.metadata["confidence"]) for item in examples if item.metadata.get("confidence") is not None]
    value = sum(values) / len(values) if values else 0.0
    return MetricResult(name="mean_prediction_confidence", value=value, details={"count": len(values)})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _labels_match(predicted: Any, gold: Any) -> bool:
    """Compare scalar or list labels after stable normalization."""

    predicted_values = _label_values(predicted)
    gold_values = _label_values(gold)
    if len(predicted_values) > 1 or len(gold_values) > 1:
        return set(predicted_values) == set(gold_values)
    return predicted_values == gold_values


def _label_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({_normalize_label(item) for item in value if item is not None})
    if isinstance(value, str) and "|" in value:
        return sorted({_normalize_label(item) for item in value.split("|") if item.strip()})
    return [_normalize_label(value)]


def _normalize_label(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.lower()


def _normalize_text_answer(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _token_f1(prediction: str, gold: str) -> float:
    predicted_tokens = prediction.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 1.0 if predicted_tokens == gold_tokens else 0.0
    common = 0
    remaining = gold_tokens.copy()
    for token in predicted_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
