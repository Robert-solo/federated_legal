"""PEFT/LoRA adapter training plan utilities."""

from __future__ import annotations

from dataclasses import dataclass

from fedlegal.config.schemas import ModelConfig

SUPPORTED_MODEL_FAMILIES = {
    "qwen2.5": ("Qwen/Qwen2.5", "qwen"),
    "llama3": ("meta-llama/Meta-Llama-3", "llama"),
    "mistral": ("mistralai/Mistral", "mistral"),
    "custom": ("", ""),
}


@dataclass(frozen=True)
class AdapterTrainingPlan:
    """Configuration needed to initialize a local PEFT or adapter tuning job."""

    base_model: str
    model_family: str
    tokenizer: str
    adaptation: str
    adapter_name: str
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    max_seq_length: int
    use_deepspeed: bool


LoRAAdapterPlan = AdapterTrainingPlan


def build_lora_plan(config: ModelConfig) -> AdapterTrainingPlan:
    """Build a PEFT plan without downloading or initializing model weights."""

    validate_model_compatibility(config)
    tokenizer = config.tokenizer or config.base_model
    return AdapterTrainingPlan(
        base_model=config.base_model,
        model_family=config.model_family,
        tokenizer=tokenizer,
        adaptation=config.adaptation,
        adapter_name=config.adapter_name,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=tuple(config.lora.target_modules),
        max_seq_length=config.max_seq_length,
        use_deepspeed=config.use_deepspeed,
    )


def validate_model_compatibility(config: ModelConfig) -> None:
    """Check supported model family names without contacting model registries."""

    if config.model_family not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise ValueError(f"Unsupported model family {config.model_family!r}. Expected: {supported}")


def build_peft_lora_config(plan: AdapterTrainingPlan):
    """Build a PEFT `LoraConfig` when PEFT is installed."""

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("PEFT is required to build a LoRA config.") from exc

    return LoraConfig(
        r=plan.rank,
        lora_alpha=plan.alpha,
        lora_dropout=plan.dropout,
        target_modules=list(plan.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def build_adapter_tuning_plan(config: ModelConfig) -> AdapterTrainingPlan:
    """Build an adapter-tuning plan using the same model compatibility contract."""

    if config.adaptation not in {"adapter", "adapter_tuning", "peft", "lora"}:
        raise ValueError(f"Unsupported adaptation strategy: {config.adaptation}")
    return build_lora_plan(config)
