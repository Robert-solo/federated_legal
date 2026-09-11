"""Download and export public legal datasets for cross-jurisdiction experiments.

The script writes normalized raw JSONL files under `data/raw/public` and a
manifest that records dataset source, split counts, feature schema, and license
metadata. It is intentionally separate from training so dataset acquisition can
run on a login/storage node instead of a GPU allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi


@dataclass(frozen=True)
class PublicDatasetSpec:
    """One public legal dataset split group."""

    key: str
    repo_id: str
    dataset_name: str
    config: str | None
    splits: tuple[str, ...]
    jurisdiction: str
    legal_tradition: str
    institution_id: str
    institution_type: str
    task_type: str
    license_hint: str
    trust_remote_code: bool = False
    fallback_repo_id: str | None = None
    fallback_splits: tuple[str, ...] = ()
    revision: str | None = None
    converted_revision: str | None = None
    evaluation_role: str = "training_and_evaluation"
    license_note: str = ""


DEFAULT_SPECS: tuple[PublicDatasetSpec, ...] = (
    PublicDatasetSpec(
        key="lexglue_case_hold",
        repo_id="coastalcph/lex_glue",
        dataset_name="CaseHOLD",
        config="case_hold",
        splits=("train", "validation", "test"),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_casehold",
        institution_type="court",
        task_type="multiple_choice_holding_prediction",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_scotus",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="scotus",
        splits=("train", "validation", "test"),
        jurisdiction="us_supreme_court",
        legal_tradition="common_law",
        institution_id="court_us_scotus",
        institution_type="court",
        task_type="issue_area_classification",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_ecthr_a",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="ecthr_a",
        splits=("train", "validation", "test"),
        jurisdiction="european_human_rights",
        legal_tradition="civil_law_regulatory",
        institution_id="court_eu_ecthr",
        institution_type="supranational_court",
        task_type="echr_article_violation_prediction",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_ecthr_b",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="ecthr_b",
        splits=("train", "validation", "test"),
        jurisdiction="european_human_rights",
        legal_tradition="civil_law_regulatory",
        institution_id="court_eu_ecthr",
        institution_type="supranational_court",
        task_type="echr_article_violation_prediction",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_eurlex",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="eurlex",
        splits=("train", "validation", "test"),
        jurisdiction="european_regulatory_law",
        legal_tradition="civil_law_regulatory",
        institution_id="institute_eu_legislation",
        institution_type="legal_research_institute",
        task_type="eu_legislation_multilabel_classification",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_ledgar",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="ledgar",
        splits=("train", "validation", "test"),
        jurisdiction="us_contract_law",
        legal_tradition="common_law_commercial",
        institution_id="firm_us_contract_ledgar",
        institution_type="law_firm",
        task_type="contract_clause_classification",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="lexglue_unfair_tos",
        repo_id="coastalcph/lex_glue",
        dataset_name="LexGLUE",
        config="unfair_tos",
        splits=("train", "validation", "test"),
        jurisdiction="consumer_contract_law",
        legal_tradition="mixed_commercial",
        institution_id="consumer_contract_lab",
        institution_type="legal_research_institute",
        task_type="unfair_terms_multilabel_classification",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="cuad_contracts",
        repo_id="theatticusproject/cuad",
        dataset_name="CUAD",
        config=None,
        splits=("train",),
        jurisdiction="contract_law",
        legal_tradition="mixed_commercial",
        institution_id="firm_contract_cuad",
        institution_type="law_firm",
        task_type="contract_text_corpus",
        license_hint="cc-by-4.0",
    ),
    PublicDatasetSpec(
        key="cuad_qa",
        repo_id="theatticusproject/cuad-qa",
        dataset_name="CUAD",
        config=None,
        splits=("train", "test"),
        jurisdiction="contract_law",
        legal_tradition="mixed_commercial",
        institution_id="firm_contract_cuad_qa",
        institution_type="law_firm",
        task_type="contract_clause_extract_qa",
        license_hint="cc-by-4.0",
        trust_remote_code=True,
        fallback_repo_id="theatticusproject/cuad",
        fallback_splits=("train",),
    ),
)

CAIL_SPEC = PublicDatasetSpec(
    key="cail2018",
    repo_id="china-ai-law-challenge/cail2018",
    dataset_name="CAIL",
    config=None,
    splits=(
        "exercise_contest_train",
        "exercise_contest_valid",
        "exercise_contest_test",
        "first_stage_train",
        "first_stage_test",
        "final_test",
    ),
    jurisdiction="chinese_criminal_law",
    legal_tradition="civil_law",
    institution_id="court_cn_cail2018",
    institution_type="court",
    task_type="criminal_judgment_prediction",
    license_hint="unknown",
    revision="775098da3ba75f033781f8061900b62503e9bea0",
    license_note="Hugging Face metadata reports an unknown license; verify terms before training.",
)

EXTENDED_SPECS: tuple[PublicDatasetSpec, ...] = (
    PublicDatasetSpec(
        key="lecardv2_queries",
        repo_id="mteb/LeCaRDv2",
        dataset_name="LeCaRDv2",
        config="queries",
        splits=("queries",),
        jurisdiction="chinese_criminal_law",
        legal_tradition="civil_law",
        institution_id="court_cn_retrieval",
        institution_type="court",
        task_type="case_retrieval_query",
        license_hint="mit",
        revision="566865fc50a566db1635307496e46c5a1a118cb2",
        evaluation_role="retrieval_and_citation_evaluation",
    ),
    PublicDatasetSpec(
        key="lecardv2_corpus",
        repo_id="mteb/LeCaRDv2",
        dataset_name="LeCaRDv2",
        config="corpus",
        splits=("corpus",),
        jurisdiction="chinese_criminal_law",
        legal_tradition="civil_law",
        institution_id="court_cn_retrieval",
        institution_type="court",
        task_type="case_retrieval_corpus",
        license_hint="mit",
        revision="566865fc50a566db1635307496e46c5a1a118cb2",
        evaluation_role="retrieval_and_citation_evaluation",
    ),
    PublicDatasetSpec(
        key="lecardv2_qrels",
        repo_id="mteb/LeCaRDv2",
        dataset_name="LeCaRDv2",
        config=None,
        splits=("test",),
        jurisdiction="chinese_criminal_law",
        legal_tradition="civil_law",
        institution_id="court_cn_retrieval",
        institution_type="court",
        task_type="case_retrieval_relevance",
        license_hint="mit",
        revision="566865fc50a566db1635307496e46c5a1a118cb2",
        evaluation_role="retrieval_and_citation_evaluation",
    ),
    PublicDatasetSpec(
        key="legalbench_citation_prediction_classification",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="citation_prediction_classification",
        splits=("train", "test"),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="citation_prediction_classification",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="citation_consistency_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_citation_prediction_open",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="citation_prediction_open",
        splits=("train", "test"),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="citation_prediction_generation",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="citation_consistency_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_ucc_v_common_law",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="ucc_v_common_law",
        splits=("train", "test"),
        jurisdiction="us_mixed_commercial_law",
        legal_tradition="common_law_commercial",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="authority_regime_classification",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="conflict_gate_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_canada_tax_court_outcomes",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="canada_tax_court_outcomes",
        splits=("train", "test"),
        jurisdiction="canadian_tax_law",
        legal_tradition="common_law_bijural",
        institution_id="court_ca_tax",
        institution_type="court",
        task_type="judgment_outcome_classification",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="cross_jurisdiction_reasoning_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_personal_jurisdiction",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="personal_jurisdiction",
        splits=("train", "test"),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="personal_jurisdiction_reasoning",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="reasoning_and_abstention_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_rule_qa",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="rule_qa",
        splits=("test",),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="legal_rule_question_answering",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="reasoning_coherence_stress_test",
    ),
    PublicDatasetSpec(
        key="legalbench_overruling",
        repo_id="nguha/legalbench",
        dataset_name="LegalBench",
        config="overruling",
        splits=("train", "test"),
        jurisdiction="us_common_law",
        legal_tradition="common_law",
        institution_id="court_us_legalbench",
        institution_type="legal_research_institute",
        task_type="precedent_status_classification",
        license_hint="cc-by-4.0",
        revision="daec8237410aa23e3faf4bc41ad8b3a7e1696826",
        evaluation_role="citation_and_authority_stress_test",
    ),
    *tuple(
        PublicDatasetSpec(
            key=f"multieurlex_{config.replace('-', '_')}",
            repo_id="Muennighoff/multi_eurlex",
            dataset_name="MultiEURLEX",
            config=config,
            splits=("train", "validation", "test"),
            jurisdiction="european_regulatory_law",
            legal_tradition="civil_law_regulatory",
            institution_id=f"institute_eu_{config}",
            institution_type="legal_research_institute",
            task_type="parallel_multilingual_eu_law_classification",
            license_hint="cc-by-sa-4.0",
            revision="1ce71aa47b2ef72cda45cca1fa8a01803495d4dc",
            converted_revision="0a7406cafd5f3d1cc860fc0b7901772e0516e2c6",
            evaluation_role="cross_language_transfer_stress_test",
            license_note="Converted mirror of the CC-BY-SA-4.0 MultiEURLEX dataset.",
        )
        for config in ("en-de", "en-fr", "en-es", "en-pl")
    ),
)

TASK_ALIASES = {
    spec.key: spec for spec in DEFAULT_SPECS + (CAIL_SPEC,) + EXTENDED_SPECS
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/public")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task keys to download.")
    parser.add_argument("--include-cail", action="store_true", help="Include CAIL2018.")
    parser.add_argument("--include-extended", action="store_true", help="Include reviewer-requested extension datasets.")
    parser.add_argument("--extended-only", action="store_true", help="Download only reviewer-requested extension datasets.")
    parser.add_argument("--max-records-per-split", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", help="Stream records during export.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = select_specs(
        args.tasks,
        args.include_cail,
        args.include_extended,
        args.extended_only,
    )
    manifest: dict[str, Any] = {
        "created_at_unix": time.time(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "hf_home": os.environ.get("HF_HOME"),
        "output_dir": str(output_dir),
        "max_records_per_split": args.max_records_per_split,
        "datasets": [],
    }

    for spec in specs:
        print(f"[download] {spec.key} from {spec.repo_id}/{spec.config or 'default'}", flush=True)
        manifest["datasets"].append(export_spec(spec, output_dir, args))

    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def select_specs(
    tasks: list[str] | None,
    include_cail: bool,
    include_extended: bool,
    extended_only: bool,
) -> list[PublicDatasetSpec]:
    if tasks:
        unknown = sorted(set(tasks) - set(TASK_ALIASES))
        if unknown:
            valid = ", ".join(sorted(TASK_ALIASES))
            raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {valid}")
        return [TASK_ALIASES[task] for task in tasks]
    specs = list(EXTENDED_SPECS if extended_only else DEFAULT_SPECS)
    if include_cail:
        specs.append(CAIL_SPEC)
    if include_extended and not extended_only:
        specs.extend(EXTENDED_SPECS)
    return specs


def export_spec(
    spec: PublicDatasetSpec,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = dataset_metadata(spec)
    task_dir = output_dir / spec.dataset_name / task_name(spec)
    split_entries: list[dict[str, Any]] = []
    first_schema: dict[str, str] | None = None

    for split in spec.splits:
        output_path = task_dir / f"{split}.jsonl"
        if output_path.exists() and args.skip_existing:
            stats = count_jsonl(output_path)
            schema = infer_jsonl_schema(output_path)
        else:
            dataset = load_public_split(spec, split, args)
            stats, schema = write_split(spec, split, dataset, output_path, args.max_records_per_split)
        if first_schema is None:
            first_schema = schema
        split_entry = {
            "split": split,
            "output_path": str(output_path),
            "num_records": stats,
            "num_bytes": output_path.stat().st_size,
        }
        if spec.converted_revision:
            split_entry["source_urls"] = multieurlex_parquet_urls(spec, split)
        split_entries.append(split_entry)

    if spec.fallback_repo_id:
        fallback_dir = output_dir / spec.dataset_name / f"{task_name(spec)}_fallback"
        for split in spec.fallback_splits:
            output_path = fallback_dir / f"{split}.jsonl"
            if output_path.exists() and args.skip_existing:
                stats = count_jsonl(output_path)
                schema = infer_jsonl_schema(output_path)
            else:
                dataset = load_dataset(
                    spec.fallback_repo_id,
                    split=split,
                    streaming=args.streaming,
                    trust_remote_code=False,
                )
                stats, schema = write_split(spec, split, dataset, output_path, args.max_records_per_split)
            split_entries.append(
                {
                    "split": f"fallback_{split}",
                    "output_path": str(output_path),
                    "num_records": stats,
                    "features": schema,
                }
            )

    return {
        "key": spec.key,
        "dataset_name": spec.dataset_name,
        "task": task_name(spec),
        "repo_id": spec.repo_id,
        "requested_revision": spec.revision,
        "resolved_revision": metadata.get("sha"),
        "converted_revision": spec.converted_revision,
        "config": spec.config,
        "task_type": spec.task_type,
        "jurisdiction": spec.jurisdiction,
        "legal_tradition": spec.legal_tradition,
        "institution_id": spec.institution_id,
        "institution_type": spec.institution_type,
        "license": metadata.get("license") or spec.license_hint,
        "license_note": spec.license_note or (
            "Hub metadata reports unknown license." if spec.license_hint == "unknown" else ""
        ),
        "evaluation_role": spec.evaluation_role,
        "features": first_schema or {},
        "splits": split_entries,
    }


def dataset_metadata(spec: PublicDatasetSpec) -> dict[str, Any]:
    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    try:
        info = api.dataset_info(spec.repo_id, revision=spec.revision)
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}
    card_data = getattr(info, "cardData", None) or {}
    license_value = _card_value(card_data, "license")
    return {
        "license": license_value,
        "tags": list(getattr(info, "tags", []) or []),
        "sha": getattr(info, "sha", None),
    }


def load_public_split(
    spec: PublicDatasetSpec,
    split: str,
    args: argparse.Namespace,
) -> Dataset | IterableDataset:
    if spec.converted_revision:
        data_files = {split: multieurlex_parquet_urls(spec, split)}
        return load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            streaming=args.streaming,
        )
    trust_remote_code = args.trust_remote_code or spec.trust_remote_code
    try:
        return load_dataset(
            spec.repo_id,
            spec.config,
            split=split,
            streaming=args.streaming,
            trust_remote_code=trust_remote_code,
            revision=spec.revision,
        )
    except Exception as exc:
        if spec.key != "cuad_qa":
            raise
        print(
            f"[warn] {spec.key}/{split} failed via dataset loader: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print("[warn] Falling back to theatticusproject/cuad text corpus for CUAD coverage.", flush=True)
        return load_dataset(
            "theatticusproject/cuad",
            split="train",
            streaming=args.streaming,
            trust_remote_code=False,
        )


def multieurlex_parquet_urls(spec: PublicDatasetSpec, split: str) -> list[str]:
    if not spec.config or not spec.converted_revision:
        raise ValueError(f"Converted parquet metadata is incomplete for {spec.key}")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    shard_count = 2 if split == "train" else 1
    return [
        f"{endpoint}/datasets/{spec.repo_id}/resolve/{spec.converted_revision}/"
        f"{spec.config}/{split}/{shard_index:04d}.parquet"
        for shard_index in range(shard_count)
    ]


def write_split(
    spec: PublicDatasetSpec,
    split: str,
    dataset: Dataset | IterableDataset,
    output_path: Path,
    max_records: int | None,
) -> tuple[int, dict[str, str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    schema: dict[str, str] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(dataset):
            if max_records is not None and index >= max_records:
                break
            row = normalize_record(spec, split, index, dict(raw))
            if not schema:
                schema = {key: type(value).__name__ for key, value in row.items()}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count, schema


def normalize_record(
    spec: PublicDatasetSpec,
    split: str,
    index: int,
    raw: dict[str, Any],
) -> dict[str, Any]:
    if spec.key == "lexglue_case_hold":
        context = str(raw.get("context", ""))
        endings = [str(item) for item in raw.get("endings", [])]
        label = as_json_value(raw.get("label"))
        label_index = int(label) if str(label).isdigit() else -1
        answer = endings[label_index] if 0 <= label_index < len(endings) else None
        text = (
            "Legal case context:\n"
            f"{context}\n\nCandidate holdings:\n"
            + "\n".join(f"{choice}. {ending}" for choice, ending in enumerate(endings))
        )
        if answer is not None:
            text += f"\n\nCorrect legal holding:\n{answer}"
        payload = {"text": text, "context": context, "endings": endings, "label": label}
    elif spec.key.startswith("lexglue_ecthr"):
        paragraphs = [str(item) for item in raw.get("text", [])]
        payload = {"text": "\n".join(paragraphs), "paragraphs": paragraphs, "label": as_json_value(raw.get("labels"))}
    elif spec.key in {"lexglue_eurlex", "lexglue_unfair_tos"}:
        payload = {"text": str(raw.get("text", "")), "label": as_json_value(raw.get("labels"))}
    elif spec.key == "lexglue_scotus":
        payload = {"text": str(raw.get("text", "")), "label": as_json_value(raw.get("label"))}
    elif spec.key == "lexglue_ledgar":
        payload = {
            "text": str(raw.get("text", "")),
            "label": as_json_value(raw.get("label")),
            "clause_type": as_json_value(raw.get("label")),
        }
    elif spec.key == "cuad_qa":
        if "context" not in raw and "text" in raw:
            contract_text = str(raw.get("text", ""))
            payload = {
                "text": contract_text,
                "contract_text": contract_text,
                "question": "",
                "answer": "",
                "label": "contract_text",
                "answers": {"text": [], "answer_start": []},
                "metadata_note": "fallback_from_cuad_contract_text",
            }
        else:
            answers = as_json_value(raw.get("answers"))
            answer_text = ""
            if isinstance(answers, dict):
                texts = answers.get("text") or []
                answer_text = str(texts[0]) if texts else ""
            payload = {
                "text": str(raw.get("context", "")),
                "contract_text": str(raw.get("context", "")),
                "question": str(raw.get("question", "")),
                "answer": answer_text,
                "label": answer_text,
                "answers": answers,
            }
    elif spec.key == "cuad_contracts":
        payload = {"text": str(raw.get("text", "")), "contract_text": str(raw.get("text", "")), "label": "contract_text"}
    elif spec.key == "cail2018":
        accusation = as_json_value(raw.get("accusation"))
        payload = {
            "text": str(raw.get("fact", "")),
            "fact": str(raw.get("fact", "")),
            "label": accusation,
            "accusation": accusation,
            "relevant_articles": as_json_value(raw.get("relevant_articles")),
            "punish_of_money": as_json_value(raw.get("punish_of_money")),
            "death_penalty": as_json_value(raw.get("death_penalty")),
            "imprisonment": as_json_value(raw.get("imprisonment")),
            "life_imprisonment": as_json_value(raw.get("life_imprisonment")),
        }
    elif spec.key.startswith("lecardv2_"):
        if spec.key == "lecardv2_qrels":
            query_id = str(raw.get("query-id", ""))
            corpus_id = str(raw.get("corpus-id", ""))
            payload = {
                "text": f"query={query_id}\ncorpus={corpus_id}",
                "query_id": query_id,
                "corpus_id": corpus_id,
                "label": as_json_value(raw.get("score")),
            }
        else:
            payload = {
                "text": str(raw.get("text", "")),
                "title": str(raw.get("title", "")),
                "label": None,
            }
    elif spec.key.startswith("legalbench_"):
        prompt = raw.get("text") or raw.get("contract") or raw.get("question") or ""
        payload = {
            "text": str(prompt),
            "label": as_json_value(raw.get("answer")),
            "citation": as_json_value(raw.get("citation")),
            "doctrine": as_json_value(raw.get("doctrine")),
            "slice": as_json_value(raw.get("slice")),
        }
    elif spec.key.startswith("multieurlex_"):
        payload = {
            "text": str(raw.get("l1", "")),
            "parallel_text": str(raw.get("l2", "")),
            "language_primary": str(raw.get("l1_name", "")),
            "language_secondary": str(raw.get("l2_name", "")),
            "celex_id": str(raw.get("celex_id", "")),
            "label": as_json_value(raw.get("labels")),
        }
    else:
        payload = {"text": str(raw.get("text", "")), "label": as_json_value(raw.get("label"))}

    payload.update(
        {
            "id": str(raw.get("id") or f"{spec.key}-{split}-{index:08d}"),
            "source_dataset": spec.dataset_name,
            "source_task": task_name(spec),
            "source_repo": spec.repo_id,
            "source_split": split,
            "jurisdiction": spec.jurisdiction,
            "legal_tradition": spec.legal_tradition,
            "institution_id": spec.institution_id,
            "institution_type": spec.institution_type,
            "task_type": spec.task_type,
        }
    )
    return payload


def as_json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [as_json_value(item) for item in value]
    if isinstance(value, list):
        return [as_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_json_value(item) for key, item in value.items()}
    return value


def task_name(spec: PublicDatasetSpec) -> str:
    return spec.config or spec.key


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def infer_jsonl_schema(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                return {key: type(value).__name__ for key, value in row.items()}
    return {}


def _card_value(card_data: Any, key: str) -> str | None:
    if isinstance(card_data, dict):
        value = card_data.get(key)
    else:
        value = getattr(card_data, key, None)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
