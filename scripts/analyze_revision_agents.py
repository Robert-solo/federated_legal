"""Paired case-level role and adapter contrasts with analysis-token accounting."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_multi_agent_blind_eval import analyze
from scripts.analyze_revision_v2 import read_json, read_lines, write_csv


def compare(first, second):
    if set(first)!=set(second):
        raise ValueError("Paired conditions must contain the same cases.")
    return analyze([{"single_correct":first[key],"multi_correct":second[key]} for key in sorted(first)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    cells={}
    missing=[]
    gold_reference=None
    for backbone in ("base","federated","flen"):
        for passes in (1,4):
            condition=f"{backbone}_{passes}pass"
            path=args.input/condition
            if not (path/"automatic_metrics.json").exists():
                missing.append(condition)
                continue
            rows=read_lines(path/"predictions.jsonl")
            gold={r["case_id"]:r["gold_label"] for r in rows}
            if gold_reference is not None and gold!=gold_reference:
                raise ValueError(f"Case pairing differs: {path}")
            gold_reference=gold
            cells[(backbone,passes)]={role:{r["case_id"]:r[f"{role}_correct"] for r in rows} for role in ("single","multi")}
            result=analyze(rows)
            result.update({"backbone":backbone,"single_calls":passes,"multi_calls":4,
                           "single_parse_rate":sum(r["single_label"] is not None for r in rows)/len(rows),
                           "multi_parse_rate":sum(r["multi_label"] is not None for r in rows)/len(rows)})
            for role in ("single","multi"):
                for metric in ("prompt_tokens","generated_tokens","elapsed_seconds"):
                    if all(f"{role}_analysis_usage" in r for r in rows):
                        result[f"{role}_mean_{metric}"]=float(np.mean([r[f"{role}_analysis_usage"][metric] for r in rows]))
            summaries.append(result)
    contrasts=[]
    for backbone in ("base","federated","flen"):
        if (backbone,1) in cells and (backbone,4) in cells:
            # Both packages regenerate deterministic deliberation; do not pool these repeats.
            if cells[(backbone,1)]["multi"]!=cells[(backbone,4)]["multi"]:
                raise ValueError(f"Deterministic multi-agent repeat differs: {backbone}")
            contrasts.append({"contrast":f"{backbone}:four_vs_one_call_single_agent",
                              **compare(cells[(backbone,1)]["single"],cells[(backbone,4)]["single"])})
    for backbone in ("federated","flen"):
        for passes in (1,4):
            if (backbone,passes) in cells and ("base",passes) in cells:
                for role in ("single","multi"):
                    if role=="multi" and passes==4:
                        continue
                    contrasts.append({"contrast":f"{backbone}_vs_base:{role}_{passes}pass",
                                      **compare(cells[("base",passes)][role],cells[(backbone,passes)][role])})
    result={"matrix_complete":not missing,"missing_cells":missing,"summaries":summaries,"contrasts":contrasts,
            "inference_unit":"paired_case_with_one_training_checkpoint_per_backbone",
            "human_review_status":"pending_actual_independent_reviewers",
            "repeated_multi_agent_generations_pooled":False,
            "cost_scope":"analysis_only; shared_presentation_step_excluded"}
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    write_csv(args.output/"automatic_summary.csv",summaries)
    write_csv(args.output/"paired_contrasts.csv",contrasts)
    if summaries:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(8,4))
        names=[f"{r['backbone']} ({r['single_calls']} calls)" for r in summaries]
        x=np.arange(len(names))
        axis.bar(x-.18,[r["single_accuracy"] for r in summaries],width=.36,label="Single agent")
        axis.bar(x+.18,[r["multi_accuracy"] for r in summaries],width=.36,label="Four-stage")
        axis.set_xticks(x,names,rotation=25,ha="right")
        axis.set_ylabel("Accuracy on paired cases")
        axis.legend(frameon=False)
        axis.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("pdf","png"):
            fig.savefig(args.output/f"agent_revision.{ext}",dpi=160,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"completed_cells":len(summaries),"missing":missing}))


if __name__=="__main__":
    main()
