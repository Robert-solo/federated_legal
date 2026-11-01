"""Generate CAS result sections only from completed frozen study matrices."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

NAMES={"fedavg_10":"FedAvg (10)","fedprox_10":"FedProx (10)","ditto_8_2":"Ditto (8+2)",
       "unguarded_8_2":"Unguarded (8+2)","guarded_8_2":"Guarded (8+2)",
       "unguarded_10_2":"Unguarded (10+2)","guarded_10_2":"Guarded (10+2)",
       "flen_guarded_8_2":"Proxy-FLEN (8+2)","fedadam_10":"FedAdam (10)",
       "fedavg":"FedAvg","fedprox":"FedProx","lambda0":r"$\lambda=0$",
       "proxy_flen":"Proxy-FLEN","local_only":"Local only (target labels)","unadapted":"Frozen backbone + head"}


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.rstrip()+"\n",encoding="utf-8")


def table(path,caption,label,header,rows):
    columns="l"+"r"*(len(header)-1)
    text=("\\begin{table}[t]\n\\caption{"+caption+"}\n\\label{"+label+"}\n\\centering\n"
          "\\small\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{"+columns+"}\n\\toprule\n"+
          " & ".join(header)+" \\\\\n\\midrule\n"+
          "\n".join(" & ".join(row)+" \\\\" for row in rows)+
          "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")
    write(path,text)


def lookup(data,condition,metric):
    return next(row for row in data["summaries"] if row["condition"]==condition and row["metric"]==metric)


def formatted(data,condition,metric,scale=1,digits=2):
    row=lookup(data,condition,metric)
    return f"{scale*row['mean']:.{digits}f} $\\pm$ {scale*row['standard_deviation']:.{digits}f}"


def contrast(data,treatment,control,metric,scale=100):
    row=next(r for r in data["paired_contrasts"] if r["treatment"]==treatment and r["control"]==control and r["metric"]==metric)
    return (f"${row['mean']*scale:+.2f}$ points (95\\% paired interval "
            f"$[{row['ci95_low']*scale:.2f},{row['ci95_high']*scale:.2f}]$)")


def figure(paper,analysis,source,name,caption,label):
    shutil.copy2(analysis/source,paper/"figures"/f"{name}.pdf")
    return ("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{"+name+".pdf}\n"
            "\\caption{"+caption+"}\n\\label{"+label+"}\n\\end{figure}\n")


def casehold(data,analysis,paper):
    conditions=[c for c in NAMES if any(r["condition"]==c for r in data["summaries"]) and c.endswith(("_10","_8_2","_10_2"))]
    rows=[[NAMES[c],formatted(data,c,"accuracy",100),formatted(data,c,"worst_client_accuracy",100),
           f"{lookup(data,c,'communication_bytes')['mean']/2**20:.2f}",
           f"{lookup(data,c,'elapsed_seconds')['mean']/60:.1f}"] for c in conditions]
    table(paper/"tables/revision_casehold_table.tex",
          "Independent dual-LoRA revision: five seeds, twenty rounds and 1,000 fixed test records. Accuracy entries are percent mean and seed SD; time is observed mean wall minutes.",
          "tab:revision-casehold",["Method (shared+private)","Accuracy","Worst client","Fit payload (MiB)","Minutes"],rows)
    gate_rows=[]
    for c in ["unguarded_8_2","guarded_8_2","unguarded_10_2","guarded_10_2","flen_guarded_8_2"]:
        gate_rows.append([NAMES[c],formatted(data,c,"gate_acceptance_rate",100),
                          formatted(data,c,"test_personalization_accuracy_gain",100)])
    table(paper/"tables/revision_gate_diagnostics.tex",
          "Private-branch diagnostics. Acceptance is the fraction of client-round candidates retained; the final test gain compares the selected composition with the shared branch, in percentage points.",
          "tab:revision-gate",["Method","Accepted (percent)","Private minus shared"],gate_rows)
    text=("\\subsection{Matched-budget dual-branch revision}\n"
          "The completed revision evaluates forty original configurations and five additional FedAdam runs. "
          "All runs use independent final-test predictions and the frozen training schedule. "
          "Table~\\ref{tab:revision-casehold} reports aggregate and worst-client outcomes.\n\n"
          "\\input{tables/revision_casehold_table}\n\n"
          "At the $8+2$ budget, gating changes accuracy by "+contrast(data,"guarded_8_2","unguarded_8_2","accuracy")+". "
          "At $10+2$, its effect is "+contrast(data,"guarded_10_2","unguarded_10_2","accuracy")+". "
          "Adding proxy conflict weights to guarded $8+2$ changes accuracy by "+contrast(data,"flen_guarded_8_2","guarded_8_2","accuracy")+". "
          "These fixed-budget contrasts separate private selection from additional shared optimization.\n\n"
          "\\input{tables/revision_gate_diagnostics}\n\n")
    text+=figure(paper,analysis,"casehold_revision.pdf","revision_casehold",
                 "Completed dual-branch revision. Bars show five-seed means and seed SD on fixed test records. Worst client denotes a fixed test shard.","fig:revision-casehold")
    write(paper/"sections/revision_casehold_results.tex",text)


def multieurlex(data,analysis,paper):
    conditions=["fedavg","fedprox","lambda0","proxy_flen","local_only","unadapted"]
    rows=[[NAMES[c],formatted(data,c,"micro_f1",digits=3),formatted(data,c,"worst_holdout_micro_f1",digits=3),
           formatted(data,c,"macro_f1",digits=3),formatted(data,c,"brier",digits=3)] for c in conditions]
    table(paper/"tables/revision_multieurlex_table.tex",
          "Completed 120-run language-transfer revision. Each seed contributes a mean and minimum across four independently held-out target languages; entries report mean and seed SD.",
          "tab:revision-multieurlex",["Method","Mean micro F1","Worst micro F1","Macro F1","Brier"],rows)
    text=("\\subsection{Disjoint-probe multilingual revision}\n"
          "The completed six-method, four-holdout, five-seed revision uses fixed test records and separate training probes and threshold-validation examples. "
          "Table~\\ref{tab:revision-multieurlex} reports language-average and worst-language outcomes.\n\n"
          "\\input{tables/revision_multieurlex_table}\n\n"
          "Relative to FedAvg, proxy-FLEN changes average micro-$F_1$ by "+contrast(data,"proxy_flen","fedavg","micro_f1")+" and worst-language micro-$F_1$ by "+contrast(data,"proxy_flen","fedavg","worst_holdout_micro_f1")+". "
          "The frozen-backbone reference changes average micro-$F_1$ by "+contrast(data,"unadapted","fedavg","micro_f1")+"; target-supervised local-only adaptation changes it by "+contrast(data,"local_only","fedavg","micro_f1")+".\n\n")
    text+=figure(paper,analysis,"multieurlex_revision.pdf","revision_multieurlex",
                 "Mean and worst held-out-language performance in the completed revision; error bars show five-seed SD.","fig:revision-multieurlex")
    write(paper/"sections/revision_multieurlex_results.tex",text)


def legalbench(data,analysis,paper):
    task_names={"overruling":"Overruling","ucc_v_common_law":"UCC/common law",
                "citation_prediction_classification":"Citation support","personal_jurisdiction":"Personal jurisdiction"}
    groups=["qwen_base","qwen_fedavg","qwen_flen","mistral_7b_base"]
    display=["Qwen base","FedAvg shared","FLEN shared","Mistral 7B base"]
    rows=[]
    for task,title in task_names.items():
        for group,name in zip(groups,display):
            values=[]
            for metric in ("accuracy","macro_f1","balanced_accuracy"):
                r=next(r for r in data["summaries"] if r["condition"]==group and r["task"]==task and r["metric"]==metric)
                values.append(f"{100*r['mean']:.1f}")
            rows.append([title,name,*values])
    table(paper/"tables/revision_legalbench_table.tex",
          "Published-label LegalBench component diagnostics on 492 fixed cases. Entries are percent; adapted rows average five shared checkpoints, pretrained rows evaluate one checkpoint.",
          "tab:revision-legalbench",["Task","Checkpoint","Accuracy","Macro F1","Balanced accuracy"],rows)
    text=("\\subsection{Published-label legal component results}\n"
          "All twelve checkpoint conditions complete the four-task diagnostic. "
          "Table~\\ref{tab:revision-legalbench} separates accuracy from class-balanced outcomes. "
          "The larger pretrained checkpoint differs in both model family and size; it provides a reference for component difficulty. "
          "The released seed and case contrasts include uncertainty, and probability metrics accompany each task.\n\n"
          "\\input{tables/revision_legalbench_table}\n\n")
    text+=figure(paper,analysis,"legalbench_revision.pdf","revision_legalbench",
                 "Fixed-case component diagnostics; adapted values average five training seeds. Tasks retain their published label spaces.","fig:revision-legalbench")
    write(paper/"sections/revision_legalbench_results.tex",text)


def agents(data,analysis,paper):
    rows=[]
    for r in data["summaries"]:
        low,high=r["paired_bootstrap_ci95"]
        rows.append([r["backbone"],str(r["single_calls"]),f"{100*r['single_accuracy']:.1f}",
                     f"{100*r['multi_accuracy']:.1f}",f"${100*r['paired_accuracy_difference']:+.1f}$",
                     f"$[{100*low:.1f},{100*high:.1f}]$",f"{r['mcnemar_exact_pvalue']:.3f}"])
    table(paper/"tables/revision_agents_table.tex",
          "Sixty paired reasoning cases per checkpoint. Differences are four-stage minus single-agent accuracy in points; intervals use paired case bootstrap. The four-stage comparator is regenerated deterministically across call-budget packages and counted once.",
          "tab:revision-agents",["Checkpoint","Single calls","Single (percent)","Four-stage (percent)","Difference","95\\% CI","McNemar $p$"],rows)
    text=("\\subsection{Connected shared-adapter reasoning results}\n"
          "The completed automatic comparison links exported FedAvg and FLEN shared adapters to the same inference protocol as the base checkpoint. "
          "Table~\\ref{tab:revision-agents} compares role prompting with one-call prediction and four-call self-refinement. "
          "These are paired case outcomes for seed-41 checkpoints. Prompt and generation costs are available with the source runs. "
          "Legal soundness, coherence, preference and unsupported-claim rates await completed independent blinded review sheets.\n\n"
          "\\input{tables/revision_agents_table}\n\n")
    text+=figure(paper,analysis,"agent_revision.pdf","revision_agents",
                 "Automatic paired accuracy with base, FedAvg shared and FLEN shared checkpoints; call budgets label the single-agent controls.","fig:revision-agents")
    write(paper/"sections/revision_agents_results.tex",text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--analysis-root",type=Path,required=True)
    parser.add_argument("--paper-root",type=Path,default=Path("papers/ipm/Federated Legal elsevier"))
    args=parser.parse_args()
    published=[]
    for name,function in [("casehold_extended",casehold),("multieurlex",multieurlex),("legalbench",legalbench),("agents",agents)]:
        directory=args.analysis_root/name
        path=directory/"analysis.json"
        if path.exists():
            data=json.loads(path.read_text(encoding="utf-8"))
            if data["matrix_complete"]:
                (args.paper_root/"figures").mkdir(parents=True,exist_ok=True)
                function(data,directory,args.paper_root)
                published.append(name)
    print(json.dumps({"completed_studies_written_to_cas":published}))


if __name__=="__main__":
    main()
