"""HuggingFace and PEFT local training pipeline skeleton."""

from fedlegal.training.local_trainer import LocalAdapterTrainer, LocalTrainingResult
from fedlegal.training.peft_pipeline import (
    AdapterTrainingPlan,
    build_adapter_tuning_plan,
    build_lora_plan,
    build_peft_lora_config,
)

__all__ = [
    "AdapterTrainingPlan",
    "LocalAdapterTrainer",
    "LocalTrainingResult",
    "build_adapter_tuning_plan",
    "build_lora_plan",
    "build_peft_lora_config",
]
