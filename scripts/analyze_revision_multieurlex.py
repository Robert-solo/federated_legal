"""Analyze the new 120-run matrix with training-seed, not language, replication."""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_revision_v2 import read_json, read_lines, write_csv, summarize

METHODS=["fedavg","fedprox","lambda0","proxy_flen","local_only","unadapted"]
HOLDOUTS=["en-de","en-fr","en-es","en-pl"]
METRICS=["micro_f1","macro_f1","loss","brier","ece_15","communication_bytes",
         "mean_update_l2","mean_pairwise_update_l2","elapsed_seconds"]


def load_runs(root: Path):
    rows=[]
    reference_ids={}
    for method in METHODS:
        for holdout in HOLDOUTS:
            for seed in range(41,46):
                path=root/f"revision_v2_multieurlex_{method}_{holdout}_seed{seed}"
                if not (path/"acceptance.json").exists():
                    continue
                accepted=read_json(path/"acceptance.json")
                if accepted["status"]!="passed":
                    continue
                config=read_json(path/"run_config.json")
                final=read_json(path/"final_metrics.json")
                if not (config["test_final_only"] and config["evaluation_seed"]==20261101
                        and final["round"]==3 and accepted["finite_metrics"]):
                    raise ValueError(f"Frozen protocol differs: {path}")
                calibration=read_json(path/"threshold_calibration.json")[-1]
                expected=[holdout] if method=="local_only" else [h for h in HOLDOUTS if h!=holdout]
                if sorted(calibration["selected_on_languages"])!=sorted(expected):
                    raise ValueError(f"Threshold data differs: {path}")
                predictions=read_lines(path/"predictions.jsonl")
                for language in HOLDOUTS:
                    cohort=[r for r in predictions if r["language"]==language]
                    ids=sorted((str(r["id"]),tuple(r["gold_labels"])) for r in cohort)
                    if len(ids)!=500 or (language in reference_ids and reference_ids[language]!=ids):
                        raise ValueError(f"Fixed test records differ: {path}, {language}")
                    reference_ids[language]=ids
                target=[r for r in predictions if r["language"]==holdout]
                tp=sum(len(set(r["gold_labels"])&set(r["predicted_labels"])) for r in target)
                denominator=sum(len(r["gold_labels"])+len(r["predicted_labels"]) for r in target)
                if abs(2*tp/max(denominator,1)-final[f"{holdout}_micro_f1"])>1e-7:
                    raise ValueError(f"Test F1 reconstruction differs: {path}")
                rows.append({"method":method,"holdout":holdout,"seed":seed,"path":str(path),
                    **{key:final[f"{holdout}_{key}"] for key in ("micro_f1","macro_f1","loss","brier","ece_15")},
                    **{key:final[key] for key in ("mean_update_l2","mean_pairwise_update_l2","elapsed_seconds")},
                    "communication_bytes":final["logical_communication_bytes"]*final["round"],
                    "threshold_at_search_boundary":final["threshold_at_search_boundary"]})
    return rows


def aggregate_holdouts(rows, methods=METHODS, holdouts=HOLDOUTS, seeds=range(41,46)):
    indexed={(r["method"],r["holdout"],r["seed"]):r for r in rows}
    combined=[]
    for method in methods:
        for seed in seeds:
            if not all((method,h,seed) in indexed for h in holdouts):
                continue
            group=[indexed[(method,h,seed)] for h in holdouts]
            item={"condition":method,"seed":seed}
            for metric in METRICS:
                item[metric]=statistics.fmean(r[metric] for r in group)
            for metric in ("micro_f1","macro_f1"):
                item[f"worst_holdout_{metric}"]=min(r[metric] for r in group)
            combined.append(item)
    return combined


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=load_runs(args.input)
    combined=aggregate_holdouts(rows)
    metrics=METRICS+["worst_holdout_micro_f1","worst_holdout_macro_f1"]
    result=summarize(combined,METHODS,list(range(41,46)),metrics,
                     [(m,"fedavg") for m in METHODS if m!="fedavg"])
    result["completed_training_runs"]=len(rows)
    result["completed_method_seed_blocks"]=result.pop("completed_runs")
    result["scope"]="source_to_target_language_transfer_under_shared_EU_authority"
    result["worst_holdout_definition"]="minimum of four independently held-out target language scores within seed"
    result["local_only_uses_target_supervision"]=True
    result["runs"]=rows
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for name,data in (("completed_runs",rows),("seed_blocks",combined),("summary",result["summaries"]),
                      ("paired_contrasts",result["paired_contrasts"])):
        write_csv(args.output/f"{name}.csv",data)
    if combined:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4.3))
        for ax,metric,title in zip(axes,["micro_f1","worst_holdout_micro_f1"],["Mean held-out language","Worst held-out language"]):
            points={r["condition"]:r for r in result["summaries"] if r["metric"]==metric}
            names=[m for m in METHODS if m in points]
            ax.bar(names,[points[m]["mean"] for m in names],yerr=[points[m]["standard_deviation"] for m in names],capsize=3,color="#356c92")
            ax.tick_params(axis="x",rotation=45)
            ax.set(title=title,ylabel="Micro F1")
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Five matched seeds; bars: seed SD"+(" (partial matrix)" if not result["matrix_complete"] else ""))
        fig.tight_layout()
        for ext in ("pdf","png"):
            fig.savefig(args.output/f"multieurlex_revision.{ext}",dpi=160,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"completed_runs":len(rows),"matrix_complete":result["matrix_complete"]}))


if __name__=="__main__":
    main()
