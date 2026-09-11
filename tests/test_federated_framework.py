from pathlib import Path
from tempfile import TemporaryDirectory

from fedlegal.config import load_experiment_config
import pytest

from fedlegal.federated import (
    FlowerServerOrchestrator,
    build_flower_strategy,
    build_server_plan,
    build_strategy_plan,
)
from fedlegal.training import build_lora_plan


def test_fedavg_qwen_plan_loads() -> None:
    config = load_experiment_config("configs/experiments/fedavg_qwen_lora.yaml")
    plan = build_server_plan(config)
    lora = build_lora_plan(config.model)

    assert plan.strategy == "fedavg"
    assert plan.strategy_plan.name == "fedavg"
    assert lora.model_family == "qwen2.5"
    assert lora.adaptation == "lora"


def test_fedprox_and_scaffold_strategy_plans() -> None:
    fedprox = load_experiment_config("configs/experiments/fedprox_llama3_adapter.yaml")
    scaffold = load_experiment_config("configs/experiments/scaffold_mistral_peft.yaml")

    fedprox_plan = build_strategy_plan(fedprox.federated)
    scaffold_plan = build_strategy_plan(scaffold.federated)

    assert fedprox_plan.name == "fedprox"
    assert fedprox_plan.fedprox_mu == 0.05
    assert scaffold_plan.name == "scaffold"
    assert scaffold_plan.scaffold_server_lr == 1.0
    assert build_lora_plan(fedprox.model).model_family == "llama3"
    assert build_lora_plan(scaffold.model).model_family == "mistral"


def test_conflict_aware_runtime_cannot_fall_back_to_fedavg() -> None:
    config = load_experiment_config("configs/experiments/conflict_aware.yaml")

    with pytest.raises(ValueError, match="FedAvg fallback is prohibited"):
        build_flower_strategy(config.federated)


def test_dry_orchestration_writes_checkpoints_and_logs() -> None:
    config = load_experiment_config("configs/experiments/fedavg_qwen_lora.yaml")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config.federated.num_rounds = 2
        config.federated.checkpoint_dir = root / "checkpoints"
        config.federated.communication_log_dir = root / "communication"

        result = FlowerServerOrchestrator(config).run_dry()

        assert result.rounds == 2
        assert Path(result.checkpoint_dir, "round_0002.npz").exists()
        assert Path(result.communication_log).exists()
