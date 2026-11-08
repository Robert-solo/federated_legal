"""Freeze new reasoning cases before evaluating any confirmation checkpoint."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def select_cases(rows, limit=200):
    training_test=list(rows)
    random.Random(20261101).shuffle(training_test)
    old_inference=sorted(rows,key=lambda row:str(row.get("id")))
    old_inference=random.Random(41).sample(old_inference,60)
    excluded={str(row["id"]) for row in training_test[:1000]+old_inference}
    candidates=sorted((row for row in rows if str(row["id"]) not in excluded),key=lambda row:str(row["id"]))
    selected=random.Random(20261108).sample(candidates,limit)
    return sorted(selected,key=lambda row:str(row["id"])),excluded


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,default=Path("data/raw/public/CaseHOLD/case_hold/test.jsonl"))
    parser.add_argument("--output",type=Path,default=Path("outputs/revision_v2/reasoning_confirmation"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected,excluded=select_cases(rows)
    (args.output/"cases.jsonl").write_text("\n".join(json.dumps(row) for row in selected)+"\n",encoding="utf-8")
    config={"study":"independent_case_call_matched_reasoning_confirmation","sample_seed":20261108,
            "cases":200,"selected_case_ids":[str(row["id"]) for row in selected],
            "excluded_case_ids":sorted(excluded),"excluded_sources":["frozen_training_test_1000","earlier_reasoning_60"],
            "training_seeds":[41,42,43,44,45],"single_calls":4,"multi_calls":4,
            "primary_contrast":"four_stage_minus_four_call_self_refinement_per_training_seed",
            "primary_inference_unit":"training_seed; same fixed cases; case variation reported separately",
            "frozen_settings":"existing prompts, parser, decoding, 160 generated tokens per analysis call",
            "secondary_outcomes":["parse_rate","conditional_accuracy","prompt_tokens","generated_tokens","elapsed_seconds"],
            "human_review_status":"separate_pending_study","results_used_for_tuning":False}
    (args.output/"protocol.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"selected":len(selected),"excluded_unique_cases":len(excluded)}))


if __name__=="__main__":
    main()
