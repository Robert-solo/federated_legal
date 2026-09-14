"""Run LoRA federated multilabel classification on MultiEURLEX.

Clients are language partitions under one EU regulatory authority. Results from
this runner measure multilingual EU-law transfer, never national-jurisdiction
transfer. The ``flen`` method is gated on an expert-adjudicated calibration
manifest; uncalibrated norm weighting is named ``proxy_flen``.
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
THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.10, 0.91, 0.05))


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
    if method not in {"fedavg", "fedprox", "lambda0", "proxy_flen"}:
        raise ValueError(f"Method {method!r} requires target-conditioned aggregation.")
    weights = np.asarray(sizes, dtype=np.float64)
    weights /= weights.sum()
    if method == "proxy_flen":
        norms = np.asarray(
            [sum(float(value.float().norm()) for value in state.values()) for state in states]
        )
        weights *= 1.0 / np.maximum(norms, 1e-8)
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


def select_threshold(validation: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    probabilities = np.concatenate([values[0] for values in validation.values()])
    labels = np.concatenate([values[1] for values in validation.values()])
    scores = [
        (multilabel_metrics(probabilities, labels, threshold)["macro_f1"], threshold)
        for threshold in THRESHOLD_GRID
    ]
    return max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]


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


def train_client(model, rows, tokenizer, args, device, global_state):
    load_adapter(model, global_state)
    reference = trainable_parameter_snapshot(model)
    model.train()
    loader = DataLoader(
        MultiEURLEXDataset(rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
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
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument(
        "--method",
        choices=["fedavg", "fedprox", "lambda0", "flen", "proxy_flen"],
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    client_languages = [language for language in args.languages if language != args.holdout]
    train_rows = {
        language: load_rows(args.raw_root, language, "train", args.limit_per_client, args.seed)
        for language in client_languages
    }
    validation_rows = {
        language: load_rows(args.raw_root, language, "validation", args.validation_limit, args.seed)
        for language in args.languages
    }
    test_rows = {
        language: load_rows(args.raw_root, language, "test", args.eval_limit, args.seed)
        for language in args.languages
    }
    manifest = None
    if args.method == "flen":
        manifest = load_flen_manifest(args.calibration_manifest, args.holdout, client_languages)
    model = build_model(args.model, tokenizer, device)
    global_state = adapter_state(model)
    records: list[dict[str, Any]] = []
    aggregation_audit: list[dict[str, Any]] = []
    started = time.time()
    final_predictions = {}
    for round_id in range(1, args.rounds + 1):
        client_states = []
        proximal_values = []
        for language in client_languages:
            state, proximal_value = train_client(
                model, train_rows[language], tokenizer, args, device, global_state
            )
            client_states.append(state)
            proximal_values.append(proximal_value)
        sizes = [len(train_rows[language]) for language in client_languages]
        drift = client_drift(global_state, client_states)
        if args.method == "flen":
            global_state, audit = aggregate_flen(
                client_states, sizes, client_languages, manifest, args.penalty_lambda
            )
        else:
            global_state = aggregate(client_states, sizes, args.method)
            base_weights = (np.asarray(sizes, dtype=np.float64) / sum(sizes)).tolist()
            audit = {"weights": dict(zip(client_languages, base_weights, strict=True))}
        aggregation_audit.append({"round": round_id} | audit)
        load_adapter(model, global_state)
        validation_predictions = {
            language: predict(model, rows, tokenizer, args.max_length, args.batch_size, device)
            for language, rows in validation_rows.items()
        }
        threshold = select_threshold(
            {
                language: (validation_predictions[language][0], validation_predictions[language][1])
                for language in client_languages
            }
        )
        final_predictions = {
            language: predict(model, rows, tokenizer, args.max_length, args.batch_size, device)
            for language, rows in test_rows.items()
        }
        record = {
            "round": round_id,
            "method": args.method,
            "seed": args.seed,
            "holdout": args.holdout,
            "claim_scope": "multilingual_eu_law_transfer",
            "device": str(device),
            "threshold": threshold,
            "elapsed_seconds": time.time() - started,
            "logical_communication_bytes": 2 * len(client_languages) * state_bytes(global_state),
            "mean_fedprox_penalty": float(np.mean(proximal_values)),
        } | drift
        for language, values in final_predictions.items():
            metrics = multilabel_metrics(values[0], values[1], threshold)
            metrics["loss"] = values[2]
            for key, value in metrics.items():
                record[f"{language}_{key}"] = value
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "final_metrics.json").write_text(
        json.dumps(records[-1], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "aggregation_audit.json").write_text(
        json.dumps(aggregation_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_predictions(
        args.output / "predictions.jsonl",
        test_rows,
        final_predictions,
        float(records[-1]["threshold"]),
    )
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
    }
    if not acceptance["finite_metrics"]:
        acceptance["status"] = "blocked"
    (args.output / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
