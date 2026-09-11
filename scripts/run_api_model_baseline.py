"""Run an OpenAI-compatible API baseline on legal benchmark samples."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fedlegal.data.citations import extract_citations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--dataset-path", default="data/raw/public/CaseHOLD/case_hold/validation.jsonl")
    parser.add_argument(
        "--task",
        choices=[
            "auto",
            "casehold",
            "scotus",
            "cail",
            "cuad_qa",
            "classification",
            "ecthr_a",
            "ecthr_b",
            "eurlex",
            "ledgar",
            "unfair_tos",
        ],
        default="auto",
    )
    parser.add_argument("--output-dir", default="outputs/api_baselines")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable provider-side thinking through chat_template_kwargs.",
    )
    args = parser.parse_args()

    task = infer_task(args.task, args.dataset_path)
    run_name = args.run_name or f"{safe_name(args.model)}_{task}_{args.max_samples}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.dataset_path))[: args.max_samples]
    predictions_path = output_dir / "predictions.jsonl"
    correct = 0
    evaluated = 0
    events: list[dict[str, Any]] = []
    started = time.time()

    run_config = {
        "base_url": args.base_url,
        "model": args.model,
        "dataset_path": args.dataset_path,
        "task": task,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "enable_thinking": args.enable_thinking,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prompt = build_prompt(task, row)
            valid_choices = valid_choice_count(task, row)
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a legal benchmark evaluator. Answer only with the requested "
                            "label or option. Do not include explanations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "stream": False,
            }
            if args.enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
            try:
                response = call_chat_completion(args.base_url, args.api_key, payload, args.timeout)
                message = response["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning") or ""
                error = None
            except Exception as exc:
                response = {}
                content = ""
                reasoning = ""
                error = f"{type(exc).__name__}: {exc}"
            prediction = extract_prediction(task, content, reasoning, valid_choices)
            gold = normalize_gold(task, row)
            is_correct = prediction is not None and labels_match(task, prediction, gold)
            correct += int(is_correct)
            evaluated += int(prediction is not None and error is None)
            event = {
                "index": index,
                "example_id": row.get("id") or row.get("example_id"),
                "gold_label": gold,
                "predicted_label": prediction,
                "correct": is_correct,
                "predicted_citations": extract_citations(content or reasoning),
                "gold_citations": extract_citations(row.get("context", row.get("text", ""))),
                "generated_text": content or reasoning,
                "supported_facts": supported_facts(row),
                "metadata": {
                    "task": task,
                    "jurisdiction": row.get("jurisdiction", "unknown"),
                    "source_task": row.get("source_task"),
                    "privacy_sensitive_tokens": privacy_sensitive_tokens(row),
                    "reasoning_score": 1.0 if (content or reasoning) else 0.0,
                },
                "error": error,
                "content": content,
                "reasoning": reasoning,
                "raw_response": response,
            }
            events.append(event)
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[{index + 1}/{len(rows)}] gold={gold} pred={prediction} correct={is_correct} error={error}",
                flush=True,
            )

    accuracy = correct / len(events) if events else 0.0
    api_errors = sum(1 for event in events if event.get("error"))
    parsed = sum(
        1 for event in events if event.get("predicted_label") is not None and not event.get("error")
    )
    conditional_correct = sum(
        1
        for event in events
        if event.get("predicted_label") is not None and not event.get("error") and event.get("correct")
    )
    report = {
        "run_name": run_name,
        "model": args.model,
        "base_url": args.base_url,
        "dataset_path": args.dataset_path,
        "task": task,
        "max_samples": args.max_samples,
        "num_examples": len(events),
        "evaluated": evaluated,
        "parsed": parsed,
        "parse_rate": parsed / len(events) if events else 0.0,
        "api_errors": api_errors,
        "api_error_rate": api_errors / len(events) if events else 0.0,
        "correct": correct,
        "accuracy": accuracy,
        "conditional_correct": conditional_correct,
        "conditional_accuracy": conditional_correct / parsed if parsed else 0.0,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "elapsed_sec": time.time() - started,
        "predictions_path": str(predictions_path),
        "run_config_path": str(output_dir / "run_config.json"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    acceptance = {
        "passed": (
            len(events) == args.max_samples
            and api_errors == 0
            and report["parse_rate"] >= 0.95
            and predictions_path.exists()
            and (output_dir / "run_config.json").exists()
            and all("raw_response" in event for event in events)
        ),
        "criteria": {
            "expected_predictions": args.max_samples,
            "actual_predictions": len(events),
            "api_error_rate_max": 0.0,
            "actual_api_error_rate": report["api_error_rate"],
            "parse_rate_min": 0.95,
            "actual_parse_rate": report["parse_rate"],
            "raw_responses_recorded": all("raw_response" in event for event in events),
            "run_config_recorded": (output_dir / "run_config.json").exists(),
        },
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def build_casehold_prompt(row: dict[str, Any]) -> str:
    endings = row.get("endings") or []
    options = "\n".join(f"{index}. {ending}" for index, ending in enumerate(endings))
    return (
        "Select the correct legal holding for the case context.\n\n"
        f"Context:\n{row.get('context', row.get('text', ''))}\n\n"
        f"Options:\n{options}\n\n"
        "Answer with only one digit from 0 to 4."
    )


def build_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "casehold":
        return build_casehold_prompt(row)
    if task == "scotus":
        return (
            "Classify this U.S. Supreme Court case into the correct SCOTUS issue-area label. "
            "Answer only with an integer label from 0 to 12.\n\n"
            f"Case text:\n{truncate(row.get('text', ''), 6000)}"
        )
    if task == "cail":
        return (
            "Predict the primary Chinese criminal accusation for this CAIL fact pattern. "
            "Answer only with the Chinese charge name.\n\n"
            f"Facts:\n{truncate(row.get('fact', row.get('text', '')), 4000)}"
        )
    if task in {"ecthr_a", "ecthr_b", "eurlex", "unfair_tos"}:
        return (
            "Predict all applicable legal topic labels for this legal text. "
            "Answer only with comma-separated integer label IDs. "
            "If no label applies, answer EMPTY.\n\n"
            f"Text:\n{truncate(row.get('text', ''), 6000)}"
        )
    if task == "ledgar":
        return (
            "Predict the single contract clause label ID for this text. "
            "Answer only with one integer label ID.\n\n"
            f"Text:\n{truncate(row.get('text', ''), 6000)}"
        )
    if task == "cuad_qa":
        return (
            "Answer the contract review question using only the contract context. "
            "If no answer is present, answer EMPTY.\n\n"
            f"Question:\n{row.get('question', '')}\n\n"
            f"Contract context:\n{truncate(row.get('contract_text', row.get('text', '')), 5000)}"
        )
    return (
        "Classify the following legal text. Answer only with the correct label.\n\n"
        f"Text:\n{truncate(row.get('text', ''), 5000)}"
    )


def valid_choice_count(task: str, row: dict[str, Any]) -> int | None:
    if task == "casehold":
        return len(row.get("endings") or [])
    if task == "scotus":
        return 13
    return None


def extract_prediction(
    task: str,
    content: str,
    reasoning: str,
    valid_choices: int | None,
) -> str | int | None:
    text = content or reasoning
    if task in {"casehold", "scotus"}:
        return extract_choice(text, valid_choices or 5)
    if task == "cail":
        stripped = text.strip()
        if not stripped:
            return None
        return normalize_cail_label(re.split(r"[\s,，。；;:：]", stripped, maxsplit=1)[0].strip("\"'`"))
    if task in {"ecthr_a", "ecthr_b", "eurlex", "unfair_tos"}:
        stripped = text.strip()
        if not stripped:
            return None
        if re.search(r"\bempty\b", stripped, flags=re.IGNORECASE):
            return []
        values = [int(match) for match in re.findall(r"\d+", stripped)]
        return values if values else None
    if task == "ledgar":
        stripped = text.strip()
        if not stripped:
            return None
        match = re.search(r"\d+", stripped)
        return int(match.group(0)) if match else None
    if task == "cuad_qa":
        stripped = text.strip()
        return normalize_text_answer(stripped) if stripped else None
    return text.strip() or None


def normalize_gold(task: str, row: dict[str, Any]) -> str | int | list[int] | list[str] | None:
    label = row.get("label")
    if task in {"casehold", "scotus"}:
        try:
            return int(label)
        except Exception:
            return None
    if task == "cail":
        accusation = row.get("accusation", label)
        if isinstance(accusation, list):
            return normalize_cail_label(str(accusation[0])) if accusation else None
        return normalize_cail_label(str(accusation)) if accusation is not None else None
    if task in {"ecthr_a", "ecthr_b", "eurlex", "unfair_tos"}:
        if isinstance(label, list):
            return [int(item) for item in label]
        if label is None:
            return None
        return [int(label)]
    if task == "ledgar":
        try:
            return int(label)
        except Exception:
            return None
    if task == "cuad_qa":
        answer = row.get("answer") or label
        return normalize_text_answer(str(answer)) if answer is not None else None
    if isinstance(label, list):
        return "|".join(str(item) for item in label)
    return str(label) if label is not None else None


def call_chat_completion(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def extract_choice(content: str, num_choices: int = 5) -> int | None:
    stripped = content.strip()
    max_choice = max(0, num_choices - 1)
    if re.fullmatch(r"\d+", stripped):
        value = int(stripped)
        return value if 0 <= value <= max_choice else None
    if re.fullmatch(r"[0-4]", stripped):
        return int(stripped)
    match = re.search(r"(?:answer|option|label|选择|答案|类别)\D*(\d+)", stripped, flags=re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= max_choice else None
    match = re.search(r"\b(\d+)\b", stripped)
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= max_choice else None
    return None


def infer_task(task: str, dataset_path: str) -> str:
    if task != "auto":
        return task
    lowered = dataset_path.lower()
    if "casehold" in lowered or "case_hold" in lowered:
        return "casehold"
    if "scotus" in lowered:
        return "scotus"
    if "ecthr_a" in lowered:
        return "ecthr_a"
    if "ecthr_b" in lowered:
        return "ecthr_b"
    if "eurlex" in lowered:
        return "eurlex"
    if "ledgar" in lowered:
        return "ledgar"
    if "unfair_tos" in lowered:
        return "unfair_tos"
    if "cail" in lowered:
        return "cail"
    if "cuad_qa" in lowered:
        return "cuad_qa"
    return "classification"


def truncate(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit]


def supported_facts(row: dict[str, Any]) -> list[str]:
    facts = []
    for field in ("context", "fact", "question"):
        value = row.get(field)
        if value:
            facts.append(truncate(value, 240))
    return facts


def privacy_sensitive_tokens(row: dict[str, Any]) -> list[str]:
    tokens = []
    for field in ("criminals",):
        value = row.get(field)
        if isinstance(value, list):
            tokens.extend(str(item) for item in value[:5])
        elif value:
            tokens.append(str(value))
    return tokens


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def labels_match(task: str, predicted: Any, gold: Any) -> bool:
    if predicted is None or gold is None:
        return False
    if task == "cail":
        return normalize_cail_label(str(predicted)) == normalize_cail_label(str(gold))
    if task in {"ecthr_a", "ecthr_b", "eurlex", "unfair_tos"}:
        return sorted(normalize_label_list(predicted)) == sorted(normalize_label_list(gold))
    if task == "ledgar":
        return str(predicted) == str(gold)
    if task == "cuad_qa":
        return normalize_text_answer(str(predicted)) == normalize_text_answer(str(gold))
    return str(predicted) == str(gold)


def normalize_cail_label(value: str) -> str:
    text = normalize_text_answer(value)
    if text.endswith("罪") and len(text) > 1:
        text = text[:-1]
    text = text.replace("盗窃盗窃", "盗窃")
    text = text.replace("诈骗诈骗", "诈骗")
    return text


def normalize_text_answer(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def normalize_label_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({int(item) for item in value if str(item).strip().lstrip("-").isdigit()})
    text = str(value).strip()
    if not text:
        return []
    if text.upper() == "EMPTY":
        return []
    return sorted({int(item) for item in re.findall(r"\d+", text)})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
