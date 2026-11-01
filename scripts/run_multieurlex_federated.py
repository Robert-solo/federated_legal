"""Run LoRA federated multilabel classification on MultiEURLEX.

Clients are language partitions under one EU regulatory authority. Results from
this runner measure multilingual EU-law transfer, never national-jurisdiction
transfer. The ``flen`` method is gated on an expert-adjudicated calibration
manifest. ``proxy_flen`` uses prediction disagreement on disjoint source-language
validation probes and is reported as mechanism evidence, not legal-conflict evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.aggregation import RoutingState, TargetClientUpdate, TargetConditionedAggregator
from fedlegal.config.schemas import ConflictConfig

LABEL_COUNT = 21
THRESHOLD_GRID = (0.001, 0.002, 0.005) + tuple(
    round(float(value), 2)
    for value in np.concatenate((np.arange(0.01, 0.51, 0.01), np.arange(0.55, 0.91, 0.05)))
)


class MultiEURLEXDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        text = str(row.get("parallel_text") or row.get("text") or "")
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(LABEL_COUNT, dtype=torch.float32)
        for label in row.get("label") or []:
            value = int(label)
            if 0 <= value < LABEL_COUNT:
                labels[value] = 1.0
        return {key: value.squeeze(0) for key, value in encoded.items()} | {
            "labels": labels,
            "row_index": torch.tensor(index),
        }


def load_rows(
    root: Path,
    language: str,
    split: str,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    path = root / language / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: str(row.get("celex_id") or row.get("id")))
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
        rows.sort(key=lambda row: str(row.get("celex_id") or row.get("id")))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Binary label-probability calibration, pooled over records and labels."""
    p, y = probabilities.reshape(-1), labels.reshape(-1)
    bin_ids = np.minimum((p * 15).astype(int), 14)
    ece = 0.0
    for index in range(15):
        mask = bin_ids == index
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"brier": float(np.mean((p-y)**2)), "ece_15": ece}


class PeftMultilabelModel(torch.nn.Module):
    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Linear(hidden_size, LABEL_COUNT)

    @property
    def config(self):
        return self.backbone.config

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})()


def build_model(model_name: str, tokenizer, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_name)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["q_proj", "v_proj"],
    )
    model = PeftMultilabelModel(backbone, int(backbone.config.hidden_size))
    model.backbone = get_peft_model(model.backbone, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device)


def adapter_state(model) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model.backbone)
    state.update(
        {
            f"classifier.{key}": value.detach().cpu().clone()
            for key, value in model.classifier.state_dict().items()
        }
    )
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def load_adapter(model, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    backbone_state = {
        key: value for key, value in state.items() if not key.startswith("classifier.")
    }
    set_peft_model_state_dict(model.backbone, backbone_state)
    classifier_state = {
        key.removeprefix("classifier."): value
        for key, value in state.items()
        if key.startswith("classifier.")
    }
    model.classifier.load_state_dict(classifier_state, strict=True)


def trainable_parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def proximal_penalty(model, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = [
        (parameter - reference[name].to(parameter.device)).pow(2).sum()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in reference
    ]
    if not terms:
        raise RuntimeError("FedProx found no trainable parameters matching its global snapshot.")
    return torch.stack(terms).sum()


def aggregate(
    states: list[dict[str, torch.Tensor]],
    sizes: list[int],
    method: str,
) -> dict[str, torch.Tensor]:
    if method not in {"fedavg", "fedprox", "lambda0", "unadapted", "local_only"}:
        raise ValueError(f"Method {method!r} requires target-conditioned aggregation.")
    weights = np.asarray(sizes, dtype=np.float64)
    weights /= weights.sum()
    return {
        key: sum(
            state[key] * float(weight)
            for state, weight in zip(states, weights, strict=True)
        )
        for key in states[0]
    }


def load_flen_manifest(path: Path, target: str, client_ids: list[str]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "calibration_level": "expert_adjudicated",
        "target_jurisdiction": target,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in required.items()
        if manifest.get(key) != value
    }
    clients = manifest.get("clients")
    if mismatches or not isinstance(clients, dict) or set(clients) != set(client_ids):
        raise ValueError(
            "FLEN requires a passed expert-adjudicated target manifest with exactly "
            f"the participating clients; mismatches={mismatches}."
        )
    return manifest


def aggregate_flen(
    states: list[dict[str, torch.Tensor]],
    sizes: list[int],
    client_ids: list[str],
    manifest: dict[str, Any],
    penalty_lambda: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    keys = list(states[0])
    updates = []
    for client_id, state, size in zip(client_ids, states, sizes, strict=True):
        metadata = manifest["clients"][client_id]
        updates.append(
            TargetClientUpdate(
                client_id=client_id,
                target_jurisdiction=manifest["target_jurisdiction"],
                shared_parameters=[state[key] for key in keys],
                num_examples=size,
                authority_compatibility=float(metadata["authority_compatibility"]),
                routing_state=RoutingState(metadata["routing_state"]),
                conflict_exposure=float(metadata["conflict_exposure"]),
            )
        )
    config = ConflictConfig(
        penalty_lambda=penalty_lambda,
        diversity_floor=float(manifest.get("diversity_floor", 0.1)),
        weight_smoothing=1.0,
        sample_count_cap=int(manifest.get("sample_count_cap", 10_000)),
    )
    result = TargetConditionedAggregator(config).aggregate(updates)
    return dict(zip(keys, result.shared_parameters, strict=True)), {
        "weights": result.weights,
        "conflict_exposures": result.diagnostics.conflict_exposures,
        "excluded_clients": result.diagnostics.excluded_clients,
    }


def multilabel_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    gold = labels.astype(bool)
    true_positive = np.logical_and(predictions, gold).sum(axis=0)
    false_positive = np.logical_and(predictions, np.logical_not(gold)).sum(axis=0)
    false_negative = np.logical_and(np.logical_not(predictions), gold).sum(axis=0)
    precision_by_label = true_positive / np.maximum(true_positive + false_positive, 1)
    recall_by_label = true_positive / np.maximum(true_positive + false_negative, 1)
    f1_by_label = 2 * precision_by_label * recall_by_label / np.maximum(
        precision_by_label + recall_by_label, 1e-12
    )
    micro_precision = float(true_positive.sum() / max(true_positive.sum() + false_positive.sum(), 1))
    micro_recall = float(true_positive.sum() / max(true_positive.sum() + false_negative.sum(), 1))
    return {
        "micro_f1": 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12),
        "macro_f1": float(f1_by_label.mean()),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
    }


@torch.no_grad()
def predict(model, rows, tokenizer, max_length, batch_size, device):
    loader = DataLoader(MultiEURLEXDataset(rows, tokenizer, max_length), batch_size=batch_size)
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    indices: list[int] = []
    for batch in loader:
        indices.extend(int(index) for index in batch.pop("row_index").tolist())
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
        probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels), float(np.mean(losses)), indices


def threshold_search(
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
    objective: str = "micro_f1",
) -> tuple[float, list[dict[str, float]]]:
    if objective not in {"micro_f1", "macro_f1"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    probabilities = np.concatenate([values[0] for values in validation.values()])
    labels = np.concatenate([values[1] for values in validation.values()])
    scores = []
    for threshold in THRESHOLD_GRID:
        metrics = multilabel_metrics(probabilities, labels, threshold)
        scores.append(
            {
                "threshold": threshold,
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    selected = max(
        scores,
        key=lambda item: (item[objective], -abs(item["threshold"] - 0.5)),
    )["threshold"]
    return selected, scores


def select_threshold(validation: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    return threshold_search(validation)[0]


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in state.values())


def client_drift(
    before: dict[str, torch.Tensor],
    after_states: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    deltas = [
        torch.cat([(state[key] - before[key]).float().reshape(-1) for key in before])
        for state in after_states
    ]
    norms = [float(delta.norm()) for delta in deltas]
    pairwise = [float((left - right).norm()) for left, right in combinations(deltas, 2)]
    return {
        "mean_update_l2": float(np.mean(norms)),
        "mean_pairwise_update_l2": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def clip_client_states(
    reference: dict[str, torch.Tensor],
    states: list[dict[str, torch.Tensor]],
    clip_norm: float,
) -> tuple[list[dict[str, torch.Tensor]], list[float], list[float]]:
    """Clip each complete client update to an L2 client-level sensitivity bound."""

    clipped: list[dict[str, torch.Tensor]] = []
    norms: list[float] = []
    scales: list[float] = []
    for state in states:
        delta = torch.cat(
            [(state[key] - reference[key]).float().reshape(-1) for key in reference]
        )
        norm = float(delta.norm())
        scale = min(1.0, clip_norm / max(norm, 1e-12))
        clipped.append(
            {
                key: reference[key] + (state[key] - reference[key]) * scale
                for key in reference
            }
        )
        norms.append(norm)
        scales.append(scale)
    return clipped, norms, scales


def add_gaussian_update_noise(
    reference: dict[str, torch.Tensor],
    aggregate_state: dict[str, torch.Tensor],
    sensitivity: float,
    noise_multiplier: float,
) -> dict[str, torch.Tensor]:
    if noise_multiplier <= 0:
        return aggregate_state
    return {
        key: reference[key]
        + (aggregate_state[key] - reference[key])
        + torch.randn_like(aggregate_state[key]) * (sensitivity * noise_multiplier)
        for key in reference
    }


def replace_one_client_sensitivity(clip_norm: float, weights: list[float]) -> float:
    return 2.0 * clip_norm * max(weights)


def gaussian_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float | None:
    """RDP accountant for full-participation Gaussian client-update releases."""

    if noise_multiplier <= 0:
        return None
    orders = np.arange(2.0, 257.0)
    rdp = rounds * orders / (2.0 * noise_multiplier**2)
    epsilons = rdp + math.log(1.0 / delta) / (orders - 1.0)
    return float(epsilons.min())


@torch.no_grad()
def prediction_disagreement_exposure(
    model,
    states: list[dict[str, torch.Tensor]],
    probe_rows: dict[str, list[dict[str, Any]]],
    tokenizer,
    args,
    device,
) -> list[float]:
    """Gold-error-gated prediction disagreement on disjoint validation probes."""

    client_probabilities: list[np.ndarray] = []
    probe_labels: np.ndarray | None = None
    for state in states:
        load_adapter(model, state)
        probe_predictions = [
            predict(model, rows, tokenizer, args.max_length, args.batch_size, device)
            for rows in probe_rows.values()
        ]
        probabilities = [prediction[0] for prediction in probe_predictions]
        client_probabilities.append(np.concatenate(probabilities, axis=0))
        if probe_labels is None:
            probe_labels = np.concatenate(
                [prediction[1] for prediction in probe_predictions], axis=0
            )

    assert probe_labels is not None

    pair_scores: dict[tuple[int, int], float] = {}
    for left, right in combinations(range(len(client_probabilities)), 2):
        p = client_probabilities[left]
        q = client_probabilities[right]
        disagreement = np.abs(p - q)
        error_exposure = np.maximum(np.abs(p - probe_labels), np.abs(q - probe_labels))
        pair_scores[(left, right)] = float(np.mean(disagreement * error_exposure))
    return [
        float(
            np.mean(
                [
                    score
                    for (left, right), score in pair_scores.items()
                    if client_index in {left, right}
                ]
            )
        )
        for client_index in range(len(states))
    ]


def aggregate_proxy_flen(
    states: list[dict[str, torch.Tensor]],
    sizes: list[int],
    client_ids: list[str],
    exposures: list[float],
    target: str,
    penalty_lambda: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_exposures = [float(exposure) for exposure in exposures]
    low, high = min(raw_exposures), max(raw_exposures)
    if high > low:
        calibrated_exposures = [
            (exposure - low) / (high - low) for exposure in raw_exposures
        ]
    else:
        calibrated_exposures = [0.0 for _ in raw_exposures]
    manifest = {
        "target_jurisdiction": target,
        "diversity_floor": 0.1,
        "clients": {
            client_id: {
                "authority_compatibility": 1.0,
                "routing_state": RoutingState.TRANSFERABLE.value,
                "conflict_exposure": exposure,
            }
            for client_id, exposure in zip(
                client_ids, calibrated_exposures, strict=True
            )
        },
    }
    state, audit = aggregate_flen(states, sizes, client_ids, manifest, penalty_lambda)
    audit["calibration_level"] = "automatic_prediction_disagreement"
    audit["formal_legal_claims_allowed"] = False
    audit["raw_prediction_disagreement"] = dict(
        zip(client_ids, raw_exposures, strict=True)
    )
    audit["exposure_calibration"] = "within_round_min_max"
    return state, audit


def train_client(model, rows, tokenizer, args, device, global_state):
    load_adapter(model, global_state)
    reference = trainable_parameter_snapshot(model)
    model.train()
    loader = DataLoader(
        MultiEURLEXDataset(rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    trainable_parameters = (
        list(model.classifier.parameters())
        if args.method == "unadapted"
        else [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    proximal_total = 0.0
    steps = 0
    for _ in range(args.local_epochs):
        for batch in loader:
            batch.pop("row_index")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            if args.method == "fedprox":
                proximal = proximal_penalty(model, reference)
                proximal_total += float(proximal.detach().cpu())
                loss = loss + args.fedprox_mu * proximal / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
    return adapter_state(model), proximal_total / max(steps, 1)


@torch.no_grad()
def membership_features(
    model,
    rows_by_language: dict[str, list[dict[str, Any]]],
    tokenizer,
    max_length: int,
    batch_size: int,
    device,
    member: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    for language, rows in rows_by_language.items():
        loader = DataLoader(MultiEURLEXDataset(rows, tokenizer, max_length), batch_size=batch_size)
        for batch in loader:
            row_indices = [int(value) for value in batch.pop("row_index").tolist()]
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                output.logits, batch["labels"], reduction="none"
            ).mean(dim=1)
            probabilities = torch.sigmoid(output.logits)
            confidence = torch.maximum(probabilities, 1 - probabilities).mean(dim=1)
            for offset, row_index in enumerate(row_indices):
                row = rows[row_index]
                records.append(
                    {
                        "id": row.get("celex_id") or row.get("id"),
                        "language": language,
                        "member": member,
                        "loss": float(losses[offset].cpu()),
                        "confidence": float(confidence[offset].cpu()),
                    }
                )
    return records


def write_predictions(path: Path, rows_by_language, predictions, threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for language, (probabilities, labels, _, indices) in predictions.items():
            for offset, row_index in enumerate(indices):
                row = rows_by_language[language][row_index]
                record = {
                    "id": row.get("celex_id") or row.get("id"),
                    "language": language,
                    "gold_labels": np.flatnonzero(labels[offset] >= 0.5).tolist(),
                    "predicted_labels": np.flatnonzero(probabilities[offset] >= threshold).tolist(),
                    "probabilities": probabilities[offset].round(7).tolist(),
                    "threshold": threshold,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", default="en-pl")
    parser.add_argument("--languages", nargs="+", default=["en-de", "en-fr", "en-es", "en-pl"])
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--limit-per-client", type=int, default=500)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--penalty-lambda", type=float, default=0.2)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--clip-updates", action="store_true")
    parser.add_argument("--probe-limit", type=int, default=64)
    parser.add_argument("--test-final-only", action="store_true")
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--write-membership-features", action="store_true")
    parser.add_argument(
        "--threshold-objective",
        choices=["micro_f1", "macro_f1"],
        default="micro_f1",
    )
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--method",
        choices=[
            "fedavg",
            "fedprox",
            "lambda0",
            "flen",
            "proxy_flen",
            "local_only",
            "unadapted",
        ],
        default="fedavg",
    )
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.holdout not in args.languages:
        raise SystemExit("holdout must be in languages")
    if args.method == "flen" and args.calibration_manifest is None:
        raise SystemExit("flen requires --calibration-manifest with expert-adjudicated evidence")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("CUDA is required for this run but is unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    source_languages = [language for language in args.languages if language != args.holdout]
    client_languages = [args.holdout] if args.method == "local_only" else source_languages
    training_pool = {
        language: load_rows(args.raw_root, language, "train", args.limit_per_client + args.probe_limit, args.seed)
        for language in client_languages
    }
    train_rows = {language: rows[:args.limit_per_client] for language, rows in training_pool.items()}
    probe_rows = {language: rows[args.limit_per_client:] for language, rows in training_pool.items()}
    validation_rows = {
        language: load_rows(args.raw_root, language, "validation", args.validation_limit, args.seed)
        for language in args.languages
    }
    test_rows = {
        language: load_rows(args.raw_root, language, "test", args.eval_limit,
                            args.seed if args.evaluation_seed is None else args.evaluation_seed)
        for language in args.languages
    }
    manifest = None
    if args.method == "flen":
        manifest = load_flen_manifest(args.calibration_manifest, args.holdout, client_languages)
    model = build_model(args.model, tokenizer, device)
    if args.method == "unadapted":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    global_state = adapter_state(model)
    records: list[dict[str, Any]] = []
    aggregation_audit: list[dict[str, Any]] = []
    threshold_audit: list[dict[str, Any]] = []
    started = time.time()
    final_predictions = {}
    for round_id in range(1, args.rounds + 1):
        round_reference = {
            key: value.detach().cpu().clone() for key, value in global_state.items()
        }
        client_states = []
        proximal_values = []
        for language in client_languages:
            state, proximal_value = train_client(
                model, train_rows[language], tokenizer, args, device, round_reference
            )
            client_states.append(state)
            proximal_values.append(proximal_value)
        sizes = [len(train_rows[language]) for language in client_languages]
        drift = client_drift(round_reference, client_states)
        if args.method == "flen":
            global_state, audit = aggregate_flen(
                client_states, sizes, client_languages, manifest, args.penalty_lambda
            )
            aggregation_weights = list(audit["weights"].values())
        elif args.method == "proxy_flen":
            exposures = prediction_disagreement_exposure(
                model,
                client_states,
                probe_rows,
                tokenizer,
                args,
                device,
            )
            global_state, audit = aggregate_proxy_flen(
                client_states,
                sizes,
                client_languages,
                exposures,
                args.holdout,
                args.penalty_lambda,
            )
            audit["evidence_level"] = "automatic_prediction_disagreement"
            aggregation_weights = list(audit["weights"].values())
        else:
            global_state = aggregate(client_states, sizes, args.method)
            base_weights = (np.asarray(sizes, dtype=np.float64) / sum(sizes)).tolist()
            audit = {"weights": dict(zip(client_languages, base_weights, strict=True))}
            aggregation_weights = base_weights

        if args.clip_updates or args.dp_noise_multiplier > 0:
            clipped_states, unclipped_norms, clip_scales = clip_client_states(
                round_reference, client_states, args.dp_clip_norm
            )
            if args.method == "flen":
                clean_state, _ = aggregate_flen(
                    clipped_states, sizes, client_languages, manifest, args.penalty_lambda
                )
            elif args.method == "proxy_flen":
                clean_state, _ = aggregate_proxy_flen(
                    clipped_states,
                    sizes,
                    client_languages,
                    exposures,
                    args.holdout,
                    args.penalty_lambda,
                )
            else:
                clean_state = aggregate(clipped_states, sizes, args.method)
            sensitivity = replace_one_client_sensitivity(
                args.dp_clip_norm, aggregation_weights
            )
            global_state = add_gaussian_update_noise(
                round_reference,
                clean_state,
                sensitivity=sensitivity,
                noise_multiplier=args.dp_noise_multiplier,
            )
            audit["privacy"] = {
                "mechanism": "central_client_update_gaussian",
                "adjacency": "replace_one_client_update",
                "clip_norm": args.dp_clip_norm,
                "aggregate_l2_sensitivity": sensitivity,
                "noise_multiplier": args.dp_noise_multiplier,
                "delta": args.dp_delta,
                "epsilon": gaussian_epsilon(
                    args.dp_noise_multiplier, round_id, args.dp_delta
                ),
                "unclipped_update_norms": unclipped_norms,
                "clip_scales": clip_scales,
                "trusted_aggregator": True,
                "protection_scope": "released_weighted_update_only",
                "attack_scope": "record_membership_black_box_diagnostic",
            }
        aggregation_audit.append({"round": round_id} | audit)
        load_adapter(model, global_state)
        validation_predictions = {
            language: predict(model, rows, tokenizer, args.max_length, args.batch_size, device)
            for language, rows in validation_rows.items()
        }
        threshold, threshold_scores = threshold_search(
            {
                language: (validation_predictions[language][0], validation_predictions[language][1])
                for language in client_languages
            },
            objective=args.threshold_objective,
        )
        threshold_at_boundary = threshold in {THRESHOLD_GRID[0], THRESHOLD_GRID[-1]}
        threshold_audit.append(
            {
                "round": round_id,
                "selected_threshold": threshold,
                "selection_objective": args.threshold_objective,
                "selected_on_languages": client_languages,
                "held_out_language_excluded": args.holdout not in client_languages,
                "at_search_boundary": threshold_at_boundary,
                "scores": threshold_scores,
            }
        )
        final_predictions = {
            language: predict(model, rows, tokenizer, args.max_length, args.batch_size, device)
            for language, rows in test_rows.items()
        } if not args.test_final_only or round_id == args.rounds else {}
        record = {
            "round": round_id,
            "method": args.method,
            "seed": args.seed,
            "holdout": args.holdout,
            "claim_scope": "multilingual_eu_law_transfer",
            "device": str(device),
            "threshold": threshold,
            "threshold_at_search_boundary": threshold_at_boundary,
            "elapsed_seconds": time.time() - started,
            "logical_communication_bytes": (
                0
                if args.method == "local_only"
                else 2 * len(client_languages) * state_bytes(global_state)
            ),
            "mean_fedprox_penalty": float(np.mean(proximal_values)),
            "backbone_adapted": args.method != "unadapted",
            "dp_noise_multiplier": args.dp_noise_multiplier,
            "dp_epsilon": gaussian_epsilon(
                args.dp_noise_multiplier, round_id, args.dp_delta
            ),
        } | drift
        for language, values in final_predictions.items():
            metrics = multilabel_metrics(values[0], values[1], threshold)
            metrics["loss"] = values[2]
            metrics.update(calibration_metrics(values[0], values[1]))
            for key, value in metrics.items():
                record[f"{language}_{key}"] = value
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for record in records for key in record}))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "final_metrics.json").write_text(
        json.dumps(records[-1], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "aggregation_audit.json").write_text(
        json.dumps(aggregation_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "threshold_calibration.json").write_text(
        json.dumps(threshold_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_predictions(
        args.output / "predictions.jsonl",
        test_rows,
        final_predictions,
        float(records[-1]["threshold"]),
    )
    if args.write_membership_features:
        load_adapter(model, global_state)
        member_rows = {
            language: train_rows[language][: args.eval_limit]
            for language in client_languages
        }
        nonmember_rows = {
            language: test_rows[language][: args.eval_limit]
            for language in client_languages
        }
        features = membership_features(
            model,
            member_rows,
            tokenizer,
            args.max_length,
            args.batch_size,
            device,
            member=True,
        ) + membership_features(
            model,
            nonmember_rows,
            tokenizer,
            args.max_length,
            args.batch_size,
            device,
            member=False,
        )
        with (args.output / "membership_features.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for feature in features:
                handle.write(json.dumps(feature, ensure_ascii=False) + "\n")
    acceptance = {
        "status": "passed",
        "completed_rounds": len(records),
        "expected_rounds": args.rounds,
        "predictions_written": sum(len(rows) for rows in test_rows.values()),
        "claim_scope": "multilingual_eu_law_transfer",
        "national_jurisdiction_claim_permitted": False,
        "flen_claim_permitted": args.method == "flen" and manifest is not None,
        "finite_metrics": all(
            math.isfinite(float(value))
            for key, value in records[-1].items()
            if isinstance(value, (float, int)) and key != "seed"
        ),
        "threshold_search_interior": not bool(records[-1]["threshold_at_search_boundary"]),
        "reference_baseline_boundary_allowed": (
            args.method == "unadapted"
        ),
        "membership_features_written": (
            len(client_languages) * min(args.eval_limit, args.limit_per_client) * 2
            if args.write_membership_features
            else 0
        ),
    }
    acceptance["threshold_boundary_role"] = "diagnostic_not_a_validity_test"
    acceptance["probe_source"] = "reserved_training_split_records"
    acceptance["membership_nonmembers_source"] = "independent_test_split"
    if not acceptance["finite_metrics"]:
        acceptance["status"] = "blocked"
    (args.output / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
