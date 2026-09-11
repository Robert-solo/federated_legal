"""Typed configuration schemas for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class JurisdictionConfig(BaseModel):
    """Metadata for one simulated legal institution or jurisdiction."""

    client_id: str
    jurisdiction: str
    legal_tradition: str
    institution_type: str
    dataset_names: list[str] = Field(default_factory=list)
    sample_weight: float = 1.0


class DatasetSourceConfig(BaseModel):
    """Raw legal dataset source and normalization metadata."""

    name: Literal[
        "CAIL",
        "LexGLUE",
        "CaseHOLD",
        "CUAD",
        "LeCaRDv2",
        "LegalBench",
        "MultiEURLEX",
    ]
    task: str | None = None
    raw_path: Path | None = None
    split: str = "train"
    text_fields: list[str] = Field(default_factory=list)
    label_field: str | None = None
    citation_fields: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    jurisdiction_field: str | None = None
    authority_scope_field: str | None = None
    language_fields: list[str] = Field(default_factory=list)
    require_explicit_jurisdiction: bool = False
    institution_id: str | None = None
    institution_type: str | None = None


class TokenizerConfig(BaseModel):
    """Tokenizer settings for processed legal records."""

    name_or_path: str | None = None
    backend: Literal["auto", "huggingface", "simple"] = "auto"
    max_length: int = 2048
    add_special_tokens: bool = True
    save_text_tokens: bool = False


class DataConfig(BaseModel):
    """Dataset and non-IID partition settings."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    partition_dir: Path = Path("data/partitions")
    partition_strategy: Literal[
        "dirichlet",
        "label_skew",
        "jurisdiction_holdout",
        "jurisdiction_based",
        "institution_based",
    ] = "dirichlet"
    dirichlet_alpha: float = 0.3
    seed: int = 42
    datasets: list[DatasetSourceConfig] = Field(default_factory=list)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    jurisdictions: list[JurisdictionConfig]


class FederatedConfig(BaseModel):
    """Flower simulation and federated optimization settings."""

    framework: Literal["flower"] = "flower"
    strategy: Literal["fedavg", "fedprox", "scaffold", "fednova", "conflict_aware"] = "fedavg"
    num_clients: int = 4
    num_rounds: int = 100
    local_epochs: int = 1
    client_fraction: float = 1.0
    min_available_clients: int = 4
    communication_payload: Literal["lora_adapters", "gradients", "reasoning_embeddings"] = (
        "lora_adapters"
    )
    fedprox_mu: float = 0.01
    scaffold_server_lr: float = 1.0
    scaffold_client_lr: float = 1.0
    checkpoint_dir: Path = Path("outputs/checkpoints")
    communication_log_dir: Path = Path("outputs/logs/communication")


class LoRAConfig(BaseModel):
    """PEFT LoRA adapter settings."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class ModelConfig(BaseModel):
    """HuggingFace model and local adaptation settings."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer: str | None = None
    model_family: Literal["qwen2.5", "llama3", "mistral", "custom"] = "qwen2.5"
    adaptation: Literal["lora", "peft", "adapter", "adapter_tuning", "prompt_tuning"] = "lora"
    max_seq_length: int = 4096
    use_deepspeed: bool = False
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    adapter_name: str = "legal_adapter"


class ConflictConfig(BaseModel):
    """Conflict-aware aggregation controls."""

    enabled: bool = True
    penalty_lambda: float = 0.2
    citation_weight: float = 0.25
    reasoning_weight: float = 0.25
    verdict_weight: float = 0.25
    rule_alignment_weight: float = 0.25
    citation_weighted_aggregation: bool = True
    contradiction_penalty: bool = True
    sample_count_cap: int = Field(default=10_000, gt=0)
    diversity_floor: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_smoothing: float = Field(default=1.0, ge=0.0, le=1.0)
    missing_component_policy: Literal["omit_and_renormalize", "exclude_client", "error"] = (
        "omit_and_renormalize"
    )
    cohort_policy: Literal["target_compatible", "transferable_only"] = "target_compatible"
    calibration_manifest: Path | None = None
    routing_thresholds: dict[str, float] = Field(default_factory=dict)
    visualization: bool = True
    log_dir: Path = Path("outputs/logs/aggregation")


class PrivacyConfig(BaseModel):
    """Privacy-preserving communication controls."""

    differential_privacy: bool = True
    noise_multiplier: float = 0.8
    secure_aggregation: bool = True
    gradient_compression: bool = True
    sanitize_legal_entities: bool = True


class AgentConfig(BaseModel):
    """LangGraph multi-agent reasoning settings."""

    framework: Literal["langgraph"] = "langgraph"
    roles: list[str] = Field(
        default_factory=lambda: [
            "jurisdiction_alignment",
            "prosecutor",
            "defense",
            "citation_verification",
            "conflict_detection",
            "privacy_auditor",
            "judge",
        ]
    )
    exchange_format: Literal[
        "reasoning_embeddings",
        "compressed_legal_representations",
        "verdict_distributions",
    ] = "reasoning_embeddings"


class EvaluationConfig(BaseModel):
    """Metrics required by the paper and AGENTS instructions."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "legal_accuracy",
            "citation_consistency",
            "reasoning_coherence",
            "hallucination_rate",
            "communication_cost",
            "client_drift",
            "cross_jurisdiction_generalization",
            "privacy_leakage_risk",
            "aggregation_stability",
        ]
    )
    output_dir: Path = Path("outputs/evaluation")
    predictions_path: Path | None = None
    communication_log_path: Path | None = None
    aggregation_log_path: Path | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    seed: int = 42
    description: str
    data: DataConfig
    federated: FederatedConfig
    model: ModelConfig
    conflict: ConflictConfig = Field(default_factory=ConflictConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    log_dir: Path = Path("outputs/logs")
    figure_dir: Path = Path("outputs/figures")
