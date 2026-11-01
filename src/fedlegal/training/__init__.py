"""HuggingFace and PEFT local training pipeline skeleton."""

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


def __getattr__(name):
    if name in {"LocalAdapterTrainer", "LocalTrainingResult"}:
        from fedlegal.training import local_trainer
        return getattr(local_trainer, name)
    raise AttributeError(name)
