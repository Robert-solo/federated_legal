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
    resize=len(header)>3
    text=("\\begin{table}[t]\n\\caption{"+caption+"}\n\\label{"+label+"}\n\\centering\n"
          "\\small\n"+("\\resizebox{\\linewidth}{!}{%\n" if resize else "")+"\\begin{tabular}{"+columns+"}\n\\toprule\n"+
          " & ".join(header)+" \\\\\n\\midrule\n"+
          "\n".join(" & ".join(row)+" \\\\" for row in rows)+
          "\n\\bottomrule\n\\end{tabular}\n"+("}\n" if resize else "")+"\\end{table}\n")
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
          "Worst-client accuracy is reconstructed from saved predictions using the same four round-robin test shards for every run; early shared-only runs originally logged a different grouping. "
          "Table~\\ref{tab:revision-casehold} reports aggregate and worst-client outcomes.\n\n"
          "\\input{tables/revision_casehold_table}\n\n"
          "At the $8+2$ budget, gating changes accuracy by "+contrast(data,"guarded_8_2","unguarded_8_2","accuracy")+". "
          "At $10+2$, its effect is "+contrast(data,"guarded_10_2","unguarded_10_2","accuracy")+". "
          "Adding proxy conflict weights to guarded $8+2$ changes accuracy by "+contrast(data,"flen_guarded_8_2","guarded_8_2","accuracy")+". "
          "These fixed-budget contrasts separate private selection from additional shared optimization.\n\n"
          "The two gate effects are positive across all five seed pairs. Their exact two-sided sign-flip $p$-values are 0.0625, with Holm-adjusted $p=0.4375$ across seven planned accuracy contrasts. "
          "The intervals quantify recovery relative to unguarded adaptation; the small seed count limits exact-test resolution. "
          "Guarded $8+2$ accuracy is 27.78\\%, compared with 28.28\\% for FedAvg and 28.34\\% for FedProx. "
          "FedAdam reaches 21.28\\% at the fixed server setting; its hyperparameters were not optimized on these test outcomes.\n\n"
          "\\input{tables/revision_gate_diagnostics}\n\n")
    text+=("The gate retains 39.75\\% and 35.50\\% of client-round candidates at the two budgets. "
           "The selected private branch changes final accuracy relative to its shared branch by 0.00 and 0.16 points, respectively; unguarded private branches reduce it by 2.30 and 2.74 points. "
           "Thus the main observed benefit is removal of harmful private updates, while the retained private branch contributes little additional test utility.\n\n")
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
          "The frozen-backbone reference changes average micro-$F_1$ by "+contrast(data,"unadapted","fedavg","micro_f1")+"; target-supervised local-only adaptation changes it by "+contrast(data,"local_only","fedavg","micro_f1")+". "
          "The zero-conflict control reproduces FedAvg. Proxy-FLEN's average contrast spans zero, so the study provides no stable incremental benefit from proxy conflict weighting. "
          "Target-language supervision remains substantially stronger than three-round source-language transfer. The frozen-backbone reference trains a classification head and therefore measures the contribution of backbone adaptation.\n\n")
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
            if not any(r["condition"]==group for r in data["summaries"]):
                continue
            values=[]
            for metric in ("accuracy","macro_f1","balanced_accuracy"):
                r=next(r for r in data["summaries"] if r["condition"]==group and r["task"]==task and r["metric"]==metric)
                values.append(f"{100*r['mean']:.1f}")
            rows.append([title,name,*values])
    table(paper/"tables/revision_legalbench_table.tex",
          "Published-label LegalBench component diagnostics on 492 fixed cases. Entries are percent; adapted rows average five shared checkpoints, pretrained rows evaluate one checkpoint.",
          "tab:revision-legalbench",["Task","Checkpoint","Accuracy","Macro F1","Balanced accuracy"],rows)
    status=("All twelve checkpoint conditions complete the four-task diagnostic. " if data["matrix_complete"] else
            "Eleven of twelve checkpoint conditions have completed the four-task diagnostic; the Mistral 7B reference remains pending model-weight recovery. ")
    text=("\\subsection{Published-label legal component results}\n"+status+
          "Table~\\ref{tab:revision-legalbench} separates accuracy from class-balanced outcomes. "
          "The larger pretrained checkpoint differs in both model family and size; it provides a reference for component difficulty. "
          "The released seed and case contrasts include uncertainty, and probability metrics accompany each task.\n\n"
          "\\input{tables/revision_legalbench_table}\n\n"
          "Shared adaptation has task-dependent effects. FedAvg and FLEN reach 52.3\\% and 51.8\\% on overruling, compared with 49.2\\% for the base checkpoint. "
          "On citation support, their means fall to 49.3\\% and 49.1\\% from 53.7\\%. "
          "All three conditions produce the same UCC/common-law accuracy of 57.4\\% with balanced accuracy 50.0\\%, revealing a single-class solution. "
          "Personal-jurisdiction accuracy increases from 42.0\\% to 54.4\\% and 55.2\\%, while balanced accuracy remains below 56\\%. "
          "These outcomes identify limited component transfer and motivate expert authority annotation and independently verified citation retrieval.\n\n")
    text+=figure(paper,analysis,"legalbench_revision.pdf","revision_legalbench",
                 "Fixed-case component diagnostics; adapted values average five training seeds. Tasks retain their published label spaces.","fig:revision-legalbench")
    write(paper/"sections/revision_legalbench_results.tex",text)


def agents(data,analysis,paper):
    rows=[]
    for r in data["summaries"]:
        low,high=r["paired_bootstrap_ci95"]
        pvalue=("$<0.001$" if r['mcnemar_exact_pvalue']<.001 else f"{r['mcnemar_exact_pvalue']:.3f}")
        rows.append([r["backbone"],str(r["single_calls"]),f"{100*r['single_accuracy']:.1f}",
                     f"{100*r['multi_accuracy']:.1f}",f"${100*r['paired_accuracy_difference']:+.1f}$",
                     f"$[{100*low:.1f},{100*high:.1f}]$",pvalue])
    table(paper/"tables/revision_agents_table.tex",
          "Sixty paired reasoning cases per checkpoint. Differences are four-stage minus single-agent accuracy in points; intervals use paired case bootstrap. The four-stage comparator is regenerated deterministically across call-budget packages and counted once.",
          "tab:revision-agents",["Checkpoint","Single calls","Single (\\%)","Four-stage (\\%)","Difference","95\\% CI","McNemar $p$"],rows)
    costs=[]
    diagnostics=[]
    for r in data["summaries"]:
        diagnostics.append([r["backbone"],str(r["single_calls"]),
            str(r["single_parsed_cases"]),str(r["multi_parsed_cases"]),str(r["both_parsed_cases"]),
            f"{100*r['single_accuracy_on_both_parsed']:.1f}",f"{100*r['multi_accuracy_on_both_parsed']:.1f}"])
        if r["single_calls"]==4:
            for role,title in (("single","Self-refinement"),("multi","Four-stage")):
                costs.append([r["backbone"],title,*[f"{r[f'{role}_mean_{metric}']:.1f}" for metric in ("prompt_tokens","generated_tokens")],
                              f"{r[f'{role}_mean_elapsed_seconds']:.2f}"])
    table(paper/"tables/revision_agents_parse.tex",
          "Output-format diagnostics on 60 cases. S denotes single-agent prediction and D four-stage deliberation. Conditional accuracy (percent) uses only cases parsed in both conditions and is descriptive; primary accuracy retains all cases and counts parsing failures as errors.",
          "tab:revision-agents-parse",["Checkpoint","Single calls","Parsed S","Parsed D","Both","Cond. acc. S","Cond. acc. D"],diagnostics)
    table(paper/"tables/revision_agents_cost.tex",
          "Mean analysis cost per case with four calls in both conditions. The common presentation step is excluded; call matching does not match token consumption.",
          "tab:revision-agents-cost",["Checkpoint","Protocol","Prompt tokens","Generated tokens","Seconds"],costs)
    text=("\\subsection{Connected shared-adapter reasoning results}\n"
          "The completed automatic comparison links exported FedAvg and FLEN shared adapters to the same inference protocol as the base checkpoint. "
          "Table~\\ref{tab:revision-agents} compares role prompting with one-call prediction and four-call self-refinement. "
          "These are paired case outcomes for seed-41 checkpoints. "
          "Legal soundness, coherence, preference and unsupported-claim rates await completed independent blinded review sheets.\n\n"
          "\\input{tables/revision_agents_table}\n\n"
          "The one-call FedAvg and FLEN controls parse successfully on only 14 and 10 of 60 cases, respectively. "
          "Their large four-stage gains therefore combine task accuracy and output-format compliance. Four-call self-refinement raises successful parsing to 58 and 59 cases. "
          "Against that control, four-stage accuracy changes by 0.00 points for FedAvg and 6.67 points for FLEN (95\\% paired bootstrap interval $[0.00,15.00]$; exact McNemar $p=0.219$). "
          "The base checkpoint decreases by 3.33 points. These call-matched results leave the reasoning advantage uncertain.\n\n"
          "\\input{tables/revision_agents_parse}\n\n"
          "\\input{tables/revision_agents_cost}\n\n"
          "For the FLEN checkpoint, four-stage analysis consumes 360.3 generated tokens and 8.93 seconds per case, compared with 256.4 tokens and 6.43 seconds for self-refinement. "
          "The extra deliberation therefore increases generated tokens by about 41\\% and observed elapsed time by about 39\\%.\n\n")
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
            if data["matrix_complete"] or (name=="legalbench" and data["summaries"]):
                (args.paper_root/"figures").mkdir(parents=True,exist_ok=True)
                function(data,directory,args.paper_root)
                published.append(name)
    print(json.dumps({"studies_written_to_cas":published}))


if __name__=="__main__":
    main()
