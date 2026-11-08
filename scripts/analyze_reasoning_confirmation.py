"""Analyze new-case reasoning with training seeds as the primary replication unit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_revision_v2 import read_json, read_lines, write_csv, sign_flip_pvalue
from scripts.analyze_multieurlex_comparison import interval
from scripts.analyze_multi_agent_blind_eval import analyze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("outputs/revision_v2/reasoning_confirmation"))
    parser.add_argument("--output",type=Path,default=Path("outputs/revision_v2/analysis/reasoning_confirmation"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=read_json(args.input/"protocol.json")
    expected=["base"]+[f"{group}_seed{seed}" for group in ("fedavg","flen") for seed in range(41,46)]
    summaries=[]
    missing=[]
    for condition in expected:
        directory=args.input/"runs"/condition
        if not (directory/"automatic_metrics.json").exists():
            missing.append(condition)
            continue
        config=read_json(directory/"run_config.json")
        if config["single_passes"]!=4 or config["max_new_tokens"]!=160:
            raise ValueError(f"Frozen call/token protocol differs: {condition}")
        rows=read_lines(directory/"predictions.jsonl")
        if {str(r["case_id"]) for r in rows}!=set(protocol["selected_case_ids"]) or len(rows)!=200:
            raise ValueError(f"Confirmation cases differ: {condition}")
        both=[r for r in rows if r["single_label"] is not None and r["multi_label"] is not None]
        result={"condition":condition,"group":condition.split("_seed")[0],
                "seed":int(condition.split("_seed")[1]) if "_seed" in condition else None,
                **analyze(rows),"both_parsed_cases":len(both)}
        for role in ("single","multi"):
            result[f"{role}_parse_rate"]=sum(r[f"{role}_label"] is not None for r in rows)/len(rows)
            result[f"{role}_conditional_accuracy"]=(float(np.mean([r[f"{role}_correct"] for r in both])) if both else None)
            for metric in ("prompt_tokens","generated_tokens","elapsed_seconds"):
                result[f"{role}_{metric}"]=float(np.mean([r[f"{role}_analysis_usage"][metric] for r in rows]))
        summaries.append(result)
    effects=[]
    for group in ("fedavg","flen"):
        cohort=[r for r in summaries if r["group"]==group]
        if len(cohort)==5:
            differences=[r["paired_accuracy_difference"] for r in cohort]
            effects.append({"contrast":f"{group}:four_stage_minus_self_refinement",
                            "paired_sign_flip_pvalue":sign_flip_pvalue(differences),**interval(differences)})
    if len(effects)==2:
        ordered=sorted(effects,key=lambda r:r["paired_sign_flip_pvalue"])
        ordered[0]["holm_adjusted_pvalue"]=min(1,2*ordered[0]["paired_sign_flip_pvalue"])
        ordered[1]["holm_adjusted_pvalue"]=max(ordered[0]["holm_adjusted_pvalue"],ordered[1]["paired_sign_flip_pvalue"])
    result={"matrix_complete":not missing,"missing_conditions":missing,"summaries":summaries,
            "seed_effects":effects,"inference_unit":"five_training_seeds_per_adapter_group_on_200_fixed_new_cases",
            "case_intervals":"conditional_on_checkpoint; same cases across seeds, not 1000 independent cases",
            "human_review_status":"not_completed_by_automatic_confirmation","protocol":protocol}
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    write_csv(args.output/"checkpoint_summary.csv",summaries)
    write_csv(args.output/"seed_effects.csv",effects)
    if summaries:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        for group,color in (("fedavg","#356c92"),("flen","#4e8860")):
            cohort=[r for r in summaries if r["group"]==group]
            if cohort:
                axes[0].plot([r["seed"] for r in cohort],[100*r["paired_accuracy_difference"] for r in cohort],"o-",label=group,color=color)
                axes[1].scatter([r["single_generated_tokens"] for r in cohort],[r["multi_generated_tokens"] for r in cohort],label=group,color=color)
        axes[0].axhline(0,color="gray",linestyle="--",linewidth=1)
        axes[0].set(xlabel="Training seed",ylabel="Four-stage minus self-refinement (points)")
        axes[1].set(xlabel="Self-refinement generated tokens",ylabel="Four-stage generated tokens")
        for axis in axes:
            axis.spines[["top","right"]].set_visible(False)
            axis.legend(frameon=False)
        fig.tight_layout()
        for ext in ("pdf","png"):
            fig.savefig(args.output/f"reasoning_confirmation.{ext}",dpi=160,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"completed":len(summaries),"missing":missing}))


if __name__=="__main__":
    main()
