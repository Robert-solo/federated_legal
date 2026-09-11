from fedlegal.aggregation import ConflictScores, weighted_conflict_penalty
from fedlegal.agents import build_agent_graph_plan
from fedlegal.config import load_data_config, load_experiment_config
from fedlegal.data import extract_citations
from fedlegal.data import build_partition_plan
from fedlegal.evaluation import REQUIRED_METRICS
from fedlegal.federated import build_server_plan
from fedlegal.privacy import build_privacy_plan
from fedlegal.training import build_lora_plan


def test_baseline_config_loads() -> None:
    config = load_experiment_config("configs/experiments/baseline_fedlora.yaml")

    assert config.name == "baseline_fedlora"
    assert config.federated.framework == "flower"
    assert config.model.adaptation == "lora"
    assert len(config.data.jurisdictions) == 4


def test_architecture_plans_are_derivable() -> None:
    config = load_experiment_config("configs/experiments/fedavg_qwen_lora.yaml")

    partition_plan = build_partition_plan(config.data)
    server_plan = build_server_plan(config)
    lora_plan = build_lora_plan(config.model)
    graph_plan = build_agent_graph_plan(config.agents)
    privacy_plan = build_privacy_plan(config.privacy)

    assert partition_plan.strategy == "dirichlet"
    assert server_plan.strategy == "fedavg"
    assert lora_plan.rank == 16
    assert "judge" in graph_plan.roles
    assert privacy_plan.secure_aggregation is True


def test_required_metrics_are_configured() -> None:
    config = load_experiment_config("configs/experiments/langgraph_reasoning.yaml")

    assert set(REQUIRED_METRICS).issubset(set(config.evaluation.metrics))


def test_conflict_penalty_matches_weighted_contract() -> None:
    config = load_experiment_config("configs/experiments/conflict_aware.yaml")
    scores = ConflictScores(
        citation_conflict=0.1,
        reasoning_conflict=0.2,
        verdict_conflict=0.3,
        rule_alignment_distance=0.4,
    )

    penalty = weighted_conflict_penalty(scores, config.conflict)

    assert round(penalty, 4) == 0.048


def test_dataset_config_loads() -> None:
    config = load_data_config("configs/datasets/legal_supported.yaml")

    assert [source.name for source in config.datasets] == ["CAIL", "LexGLUE", "CaseHOLD", "CUAD"]
    assert config.partition_strategy == "dirichlet"


def test_citation_extraction_handles_statutes_and_contracts() -> None:
    text = "The court cited 42 U.S.C. § 1983, Article 6, and Clause 12. See 《民法典》第五百条."
    citations = extract_citations(text)

    assert "42 U.S.C. § 1983" in citations
    assert "Article 6" in citations
    assert "Clause 12" in citations
