"""Summarize frozen published-label diagnostics and matched adapter-seed effects."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from fedlegal.evaluation.legalbench_diagnostic import classification_metrics
from scripts.analyze_revision_v2 import read_json, read_lines, write_csv
from scripts.analyze_multieurlex_comparison import interval
from scripts.analyze_multi_agent_blind_eval import analyze

TASKS=["overruling","ucc_v_common_law","citation_prediction_classification","personal_jurisdiction"]
METRICS=["accuracy","macro_f1","balanced_accuracy","brier_multiclass","ece_10"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    expected=["qwen_base","mistral_7b_base"]+[f"qwen_{group}_seed{s}" for group in ("fedavg","flen") for s in range(41,46)]
    runs={}
    predictions={}
    gold_reference=None
    diagnostics=[]
    for condition in expected:
        path=args.input/condition
        if not (path/"metrics.json").exists():
            continue
        config=read_json(path/"run_config.json")
        if config["sample_seed"]!=20261101 or config["limit"]!=240:
            raise ValueError(f"Formal diagnostic protocol differs: {path}")
        rows=read_lines(path/"predictions.jsonl")
        gold={(r["task"],r["id"]):r["gold"] for r in rows}
        if gold_reference is not None and gold!=gold_reference:
            raise ValueError(f"Matched benchmark cases differ: {path}")
        gold_reference=gold
        predictions[condition]={(r["task"],r["id"]):r for r in rows}
        runs[condition]=read_json(path/"metrics.json")
        for task in TASKS:
            cohort=[row for row in rows if row["task"]==task]
            reconstructed=classification_metrics(cohort,tuple(cohort[0]["labels"]))
            for metric in METRICS:
                if abs(reconstructed[metric]-runs[condition]["tasks"][task][metric])>1e-8:
                    raise ValueError(f"Prediction-derived {metric} differs: {condition}/{task}")
            diagnostics.append({"condition":condition,"task":task,"cases":len(cohort),
                "prompt_truncated_cases":sum(row["prompt_truncated"] for row in cohort),
                "majority_class_rate":reconstructed["majority_class_rate"],
                "prediction_counts":dict(Counter(row["prediction"] for row in cohort))})
    summaries=[]
    seed_contrasts=[]
    case_contrasts=[]
    for task in TASKS:
        for group in ("qwen_base","mistral_7b_base","qwen_fedavg","qwen_flen"):
            names=[group] if group in ("qwen_base","mistral_7b_base") else [f"{group}_seed{s}" for s in range(41,46)]
            available=[name for name in names if name in runs]
            for metric in METRICS:
                if available:
                    summaries.append({"condition":group,"task":task,"metric":metric,
                                      "replication":"training_seed" if len(names)>1 else "single_pretrained_checkpoint",
                                      **interval([runs[name]["tasks"][task][metric] for name in available])})
            if group in ("qwen_fedavg","qwen_flen") and "qwen_base" in runs:
                for metric in METRICS:
                    if available:
                        seed_contrasts.append({"treatment":group,"control":"qwen_base","task":task,"metric":metric,
                            **interval([runs[name]["tasks"][task][metric]-runs["qwen_base"]["tasks"][task][metric] for name in available])})
        if "qwen_base" in runs and "mistral_7b_base" in runs:
            keys=[key for key in predictions["qwen_base"] if key[0]==task]
            paired=[{"single_correct":predictions["qwen_base"][key]["prediction"]==predictions["qwen_base"][key]["gold"],
                     "multi_correct":predictions["mistral_7b_base"][key]["prediction"]==predictions["mistral_7b_base"][key]["gold"]} for key in keys]
            case_contrasts.append({"treatment":"mistral_7b_base","control":"qwen_base","task":task,**analyze(paired)})
        seeds=[s for s in range(41,46) if f"qwen_flen_seed{s}" in runs and f"qwen_fedavg_seed{s}" in runs]
        for metric in METRICS:
            if seeds:
                seed_contrasts.append({"treatment":"qwen_flen","control":"qwen_fedavg","task":task,"metric":metric,
                    **interval([runs[f"qwen_flen_seed{s}"]["tasks"][task][metric]-runs[f"qwen_fedavg_seed{s}"]["tasks"][task][metric] for s in seeds])})
    result={"matrix_complete":len(runs)==len(expected),"completed_conditions":list(runs),
            "missing_conditions":[c for c in expected if c not in runs],"summaries":summaries,
            "adapter_seed_contrasts":seed_contrasts,"pretrained_case_contrasts":case_contrasts,
            "prediction_diagnostics":diagnostics,
            "scope":"published_benchmark_component_diagnostics",
            "expert_conflict_annotation_replaced":False,
            "larger_model_reference":"Mistral 7B pretrained; different model family and parameter budget"}
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for name,data in (("summary",summaries),("adapter_seed_contrasts",seed_contrasts),("pretrained_case_contrasts",case_contrasts),("prediction_diagnostics",diagnostics)):
        write_csv(args.output/f"{name}.csv",data)
    if summaries:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        colors={"qwen_base":"#888888","mistral_7b_base":"#b25b43","qwen_fedavg":"#356c92","qwen_flen":"#4e8860"}
        for ax,metric in zip(axes,["accuracy","macro_f1"]):
            for group,color in colors.items():
                points={r["task"]:r for r in summaries if r["condition"]==group and r["metric"]==metric}
                tasks=[t for t in TASKS if t in points]
                if tasks:
                    ax.plot([TASKS.index(t) for t in tasks],[points[t]["mean"] for t in tasks],marker="o",label=group,color=color)
            ax.set_xticks(range(4),["Overrule","Regime","Citation","Personal jur."],rotation=25)
            ax.set_ylabel(metric.replace("_"," "))
            ax.spines[["top","right"]].set_visible(False)
        axes[1].legend(frameon=False,fontsize=8)
        fig.tight_layout()
        for ext in ("pdf","png"):
            fig.savefig(args.output/f"legalbench_revision.{ext}",dpi=160,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"completed_conditions":len(runs),"matrix_complete":result["matrix_complete"]}))


if __name__=="__main__":
    main()
