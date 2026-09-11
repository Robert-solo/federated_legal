from pathlib import Path
from tempfile import TemporaryDirectory

from fedlegal.aggregation import (
    ClientUpdate,
    ConflictAwareFedAvg,
    LegalConflictProfile,
    citation_divergence,
    jurisdiction_distance,
)
from fedlegal.config import load_experiment_config
from fedlegal.federated import FlowerServerOrchestrator


def test_citation_divergence_uses_jaccard_distance() -> None:
    value = citation_divergence(("Article 6", "42 U.S.C. § 1983"), ("Article 6", "Clause 12"))

    assert round(value, 3) == 0.667


def test_jurisdiction_distance_reflects_legal_tradition() -> None:
    left = LegalConflictProfile(
        client_id="a",
        jurisdiction="us_common_law",
        legal_tradition="common_law",
    )
    right = LegalConflictProfile(
        client_id="b",
        jurisdiction="singapore_hybrid_law",
        legal_tradition="hybrid",
    )

    assert round(jurisdiction_distance(left, right), 2) == 0.3


def test_conflict_aware_fedavg_changes_weights_with_citations() -> None:
    config = load_experiment_config("configs/experiments/conflict_aware_fedavg.yaml")
    aggregator = ConflictAwareFedAvg(config.conflict)
    result = aggregator.aggregate(
        [
            ClientUpdate(
                client_id="cited",
                parameters=[[1.0]],
                num_examples=10,
                profile=LegalConflictProfile(
                    client_id="cited",
                    citations=("Article 6", "Clause 12", "42 U.S.C. § 1983"),
                    verdict_distribution=(0.8, 0.2),
                    jurisdiction="us_common_law",
                    legal_tradition="common_law",
                ),
            ),
            ClientUpdate(
                client_id="uncited",
                parameters=[[3.0]],
                num_examples=10,
                profile=LegalConflictProfile(
                    client_id="uncited",
                    citations=(),
                    verdict_distribution=(0.2, 0.8),
                    jurisdiction="chinese_civil_law",
                    legal_tradition="civil_law",
                ),
            ),
        ]
    )

    assert result.weights["cited"] > result.weights["uncited"]
    assert result.parameters[0][0] < 2.0
    assert result.conflict_report.total_conflict > 0


def test_conflict_aware_dry_run_writes_diagnostics() -> None:
    config = load_experiment_config("configs/experiments/conflict_aware_fedavg.yaml")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config.federated.num_rounds = 1
        config.federated.checkpoint_dir = root / "checkpoints"
        config.federated.communication_log_dir = root / "communication"
        config.conflict.log_dir = root / "aggregation"
        config.figure_dir = root / "figures"

        result = FlowerServerOrchestrator(config).run_dry()

        assert result.metrics["total_conflict"] > 0
        assert (root / "aggregation" / "conflict_aware_fedavg_aggregation.jsonl").exists()
        assert (root / "figures" / "conflict_aware_fedavg" / "conflict_round_0001.svg").exists()
