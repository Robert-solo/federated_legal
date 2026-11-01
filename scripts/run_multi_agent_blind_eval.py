"""Generate paired single-agent and typed-deliberation outputs for blinded review."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch


def load_cases(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: str(row.get("id")))
    if len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
        rows.sort(key=lambda row: str(row.get("id")))
    return rows


def format_case(row: dict[str, Any]) -> str:
    options = row.get("endings") or row.get("choices") or []
    rendered_options = "\n".join(
        f"{index}. {option}" for index, option in enumerate(options)
    )
    return f"Case context:\n{row.get('context') or row.get('text') or ''}\n\nOptions:\n{rendered_options}"


def parse_decision(text: str) -> tuple[int | None, str]:
    objects = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(objects):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        label = payload.get("label")
        if isinstance(label, int) and 0 <= label <= 4:
            return label, str(payload.get("rationale") or text).strip()
    match = re.search(
        r'(?:"?label"?|answer|option)\s*[:=]?\s*([0-4])\b', text, re.I
    )
    if match:
        return int(match.group(1)), text.strip()
    match = re.search(
        r"(?:correct\s+answer\s+is\s*:?[\s\n]*|^\s*)([0-4])(?:\s*[.):-]|\s*$)",
        text,
        re.I,
    )
    if match:
        return int(match.group(1)), text.strip()
    return None, text.strip()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    started = time.perf_counter()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    usage = getattr(model, "_fedlegal_generation_usage", None)
    if usage is not None:
        usage.append({"prompt_tokens":int(inputs["input_ids"].shape[1]),
                      "generated_tokens":int(generated.shape[0]),
                      "elapsed_seconds":time.perf_counter()-started})
    return text


def single_agent_output(model, tokenizer, case_text: str, max_new_tokens: int) -> str:
    return generate(
        model,
        tokenizer,
        "Select the legally supported holding. Analyze only the supplied record and do not "
        "invent authorities or facts. Return JSON with integer label 0-4 and a concise rationale.\n\n"
        + case_text,
        max_new_tokens,
    )


def deliberation_output(model, tokenizer, case_text: str, max_new_tokens: int) -> str:
    analyst = generate(
        model,
        tokenizer,
        "Act as the primary legal analyst. Compare all five holdings against the supplied "
        "context and identify the strongest option with reasons. Do not add external facts.\n\n"
        + case_text,
        max_new_tokens,
    )
    challenger = generate(
        model,
        tokenizer,
        "Act as an adversarial legal reviewer. Challenge the analysis below, identify its "
        "weakest inference, and state the strongest alternative. Use only the supplied case.\n\n"
        + case_text
        + "\n\nPrimary analysis:\n"
        + analyst,
        max_new_tokens,
    )
    verifier = generate(
        model,
        tokenizer,
        "Act as an evidence verifier. Check both analyses against the case context and options. "
        "List unsupported claims and say which option has direct textual support.\n\n"
        + case_text
        + "\n\nPrimary analysis:\n"
        + analyst
        + "\n\nChallenge:\n"
        + challenger,
        max_new_tokens,
    )
    return generate(
        model,
        tokenizer,
        "Act as the final judge. Resolve the disagreement using the verifier report and only "
        "the supplied record. Return JSON with integer label 0-4 and a concise rationale.\n\n"
        + case_text
        + "\n\nPrimary analysis:\n"
        + analyst
        + "\n\nChallenge:\n"
        + challenger
        + "\n\nVerifier report:\n"
        + verifier,
        max_new_tokens,
    )


def format_review_output(label: int | None, options: list[str], explanation: str) -> str:
    if label is None:
        return "No valid holding selected.\nReasoning: " + explanation.strip()
    return (
        f"Selected holding {label}: {options[label]}\n"
        f"Reasoning: {explanation.strip()}"
    )


def render_review_output(
    model,
    tokenizer,
    case_text: str,
    options: list[str],
    label: int | None,
    draft: str,
    max_new_tokens: int,
) -> str:
    if label is None:
        return format_review_output(label, options, draft)
    explanation = generate(
        model,
        tokenizer,
        "Write exactly two concise sentences explaining why the selected holding best "
        "completes the missing legal holding. Use only the supplied case. Start with the "
        "explanation, not an option number, heading, or JSON.\n\n"
        + case_text
        + f"\n\nSelected holding: {label}. {options[label]}"
        + "\n\nPrior draft reasoning:\n"
        + draft,
        max_new_tokens,
    )
    return format_review_output(label, options, explanation)


def blind_pair(
    case_id: str, single_output: str, multi_output: str, seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    if random.Random(f"{seed}:{case_id}").random() < 0.5:
        return (
            {"output_A": single_output, "output_B": multi_output},
            {"A": "single_agent", "B": "typed_multi_agent"},
        )
    return (
        {"output_A": multi_output, "output_B": single_output},
        {"A": "typed_multi_agent", "B": "single_agent"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--single-passes", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this run but is unavailable")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    model._fedlegal_generation_usage = []

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    cases = load_cases(args.dataset, args.limit, args.seed)
    predictions = []
    review_rows = []
    condition_key: dict[str, dict[str, str]] = {}
    for index, row in enumerate(cases):
        case_id = str(row.get("id") or f"case-{index:04d}")
        case_text = format_case(row)
        usage_start = len(model._fedlegal_generation_usage)
        single_raw = single_agent_output(
            model, tokenizer, case_text, args.max_new_tokens
        )
        for _ in range(args.single_passes - 1):
            single_raw = generate(
                model, tokenizer,
                "Recheck your previous answer against every holding. Correct any unsupported "
                "inference. Return JSON with integer label 0-4 and concise rationale.\n\n"
                + case_text + "\n\nPrevious answer:\n" + single_raw,
                args.max_new_tokens,
            )
        single_usage = model._fedlegal_generation_usage[usage_start:]
        usage_start = len(model._fedlegal_generation_usage)
        multi_raw = deliberation_output(
            model, tokenizer, case_text, args.max_new_tokens
        )
        multi_usage = model._fedlegal_generation_usage[usage_start:]
        single_label, single_rationale = parse_decision(single_raw)
        multi_label, multi_rationale = parse_decision(multi_raw)
        options = row.get("endings") or row.get("choices") or []
        single_review_output = render_review_output(
            model,
            tokenizer,
            case_text,
            options,
            single_label,
            single_rationale,
            args.max_new_tokens,
        )
        multi_review_output = render_review_output(
            model,
            tokenizer,
            case_text,
            options,
            multi_label,
            multi_rationale,
            args.max_new_tokens,
        )
        gold = int(row["label"])
        predictions.append(
            {
                "case_id": case_id,
                "gold_label": gold,
                "single_label": single_label,
                "single_correct": single_label == gold,
                "single_raw_output": single_raw,
                "single_output": single_review_output,
                "multi_label": multi_label,
                "multi_correct": multi_label == gold,
                "multi_raw_output": multi_raw,
                "multi_output": multi_review_output,
                "single_analysis_usage": {key:sum(item[key] for item in single_usage)
                    for key in ("prompt_tokens","generated_tokens","elapsed_seconds")},
                "multi_analysis_usage": {key:sum(item[key] for item in multi_usage)
                    for key in ("prompt_tokens","generated_tokens","elapsed_seconds")},
            }
        )
        pair, key = blind_pair(
            case_id, single_review_output, multi_review_output, args.seed
        )
        condition_key[case_id] = key
        review_rows.append(
            {
                "case_id": case_id,
                "case_and_options": case_text,
                **pair,
                "preferred_output": "",
                "legal_soundness_A_1_to_5": "",
                "legal_soundness_B_1_to_5": "",
                "reasoning_coherence_A_1_to_5": "",
                "reasoning_coherence_B_1_to_5": "",
                "unsupported_claim_A_yes_no": "",
                "unsupported_claim_B_yes_no": "",
                "reviewer_notes": "",
            }
        )
        print(json.dumps({"completed": index + 1, "case_id": case_id}), flush=True)

    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with (args.output / "review_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    (args.output / "condition_key.json").write_text(
        json.dumps(condition_key, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {
        "cases": len(predictions),
        "single_parse_rate": sum(row["single_label"] is not None for row in predictions)
        / len(predictions),
        "multi_parse_rate": sum(row["multi_label"] is not None for row in predictions)
        / len(predictions),
        "single_accuracy": sum(row["single_correct"] for row in predictions)
        / len(predictions),
        "multi_accuracy": sum(row["multi_correct"] for row in predictions)
        / len(predictions),
        "presentation_protocol": "shared_deterministic_two_sentence_renderer",
        "human_review_status": "awaiting_two_blinded_reviewers",
        "adapter_loaded": args.adapter is not None,
        "single_analysis_calls": args.single_passes,
        "multi_analysis_calls": 4,
    }
    (args.output / "automatic_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "REVIEW_PROTOCOL.md").write_text(
        """# Blinded legal review protocol

Reviewers receive `review_sheet.csv` without `condition_key.json`. For each pair,
select the stronger output and give holistic 1--5 ratings for legal soundness and
reasoning coherence. Mark whether either output contains an unsupported factual or
legal claim. Both conditions use the same deterministic two-sentence presentation
step before A/B randomization. Two legally trained reviewers complete the sheet
independently; they then adjudicate disagreements. Method identities are revealed only after both sheets
are frozen. Report sample size, reviewer qualifications, agreement, adjudication,
method-level means, preference rate, and unsupported-claim rate.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
