"""Run real LoRA federated multilabel classification on MultiEURLEX.

The clients are language partitions under one EU regulatory authority. This
script reports multilingual transfer, not national-jurisdiction transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_COUNT = 21


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
        return {key: value.squeeze(0) for key, value in encoded.items()} | {"labels": labels}


def load_rows(root: Path, language: str, split: str, limit: int | None, seed: int) -> list[dict[str, Any]]:
    path = root / language / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: str(row.get("celex_id") or row.get("id")))
    if limit is not None and len(rows) > limit:
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
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
    state.update({f"classifier.{key}": value.detach().cpu().clone() for key, value in model.classifier.state_dict().items()})
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def load_adapter(model, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    backbone_state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
    set_peft_model_state_dict(model.backbone, backbone_state)
    classifier_state = {key.removeprefix("classifier."): value for key, value in state.items() if key.startswith("classifier.")}
    model.classifier.load_state_dict(classifier_state, strict=True)


def aggregate(states: list[dict[str, torch.Tensor]], sizes: list[int], method: str) -> dict[str, torch.Tensor]:
    weights = np.asarray(sizes, dtype=np.float64)
    weights /= weights.sum()
    if method == "flen":
        norms = np.asarray([sum(float(value.float().norm()) for value in state.values()) for state in states])
        scores = 1.0 / np.maximum(norms, 1e-8)
        weights = weights * scores
        weights /= weights.sum()
    result: dict[str, torch.Tensor] = {}
    for key in states[0]:
        result[key] = sum(state[key] * float(weight) for state, weight in zip(states, weights, strict=True))
    return result


@torch.no_grad()
def evaluate(model, rows: list[dict[str, Any]], tokenizer, max_length: int, batch_size: int, device: torch.device) -> dict[str, float]:
    loader = DataLoader(MultiEURLEXDataset(rows, tokenizer, max_length), batch_size=batch_size)
    model.eval()
    true_positive = false_positive = false_negative = 0
    losses: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
        predictions = (torch.sigmoid(output.logits) >= 0.5).int()
        labels = batch["labels"].int()
        true_positive += int(((predictions == 1) & (labels == 1)).sum())
        false_positive += int(((predictions == 1) & (labels == 0)).sum())
        false_negative += int(((predictions == 0) & (labels == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "micro_precision": precision,
        "micro_recall": recall,
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


def train_client(model, rows, tokenizer, args, device, global_state):
    load_adapter(model, global_state)
    model.train()
    loader = DataLoader(MultiEURLEXDataset(rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    for _ in range(args.local_epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            if args.method == "fedprox":
                proximal = sum((parameter - global_state[name].to(device)).pow(2).sum() for name, parameter in model.named_parameters() if name in global_state)
                loss = loss + args.fedprox_mu * proximal / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return adapter_state(model)


def main() -> None:
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
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--method", choices=["fedavg", "fedprox", "flen"], default="fedavg")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.holdout not in args.languages:
        raise SystemExit("holdout must be in languages")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    client_languages = [language for language in args.languages if language != args.holdout]
    train_rows = {language: load_rows(args.raw_root, language, "train", args.limit_per_client, args.seed) for language in client_languages}
    eval_rows = {language: load_rows(args.raw_root, language, "test", args.eval_limit, args.seed) for language in args.languages}
    model = build_model(args.model, tokenizer, device)
    global_state = adapter_state(model)
    records: list[dict[str, Any]] = []
    started = time.time()
    for round_id in range(1, args.rounds + 1):
        client_states = []
        for language in client_languages:
            client_states.append(train_client(model, train_rows[language], tokenizer, args, device, global_state))
        global_state = aggregate(client_states, [len(train_rows[language]) for language in client_languages], args.method)
        load_adapter(model, global_state)
        metrics = {language: evaluate(model, eval_rows[language], tokenizer, args.max_length, args.batch_size, device) for language in args.languages}
        record = {"round": round_id, "method": args.method, "seed": args.seed, "holdout": args.holdout, "device": str(device), "elapsed_seconds": time.time() - started}
        for language, values in metrics.items():
            for key, value in values.items():
                record[f"{language}_{key}"] = value
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "final_metrics.json").write_text(json.dumps(records[-1], indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
