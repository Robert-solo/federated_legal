"""Evaluate fixed LegalBench cases with base and federated shared adapters."""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fedlegal.evaluation.legalbench_diagnostic import TASKS, prompt_for, classification_metrics


def score_labels(model, tokenizer, prompt, labels, maximum):
    text=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    prefix=tokenizer(text,add_special_tokens=False).input_ids
    losses=[]
    truncated=False
    for label in labels:
        answer=tokenizer(" "+label,add_special_tokens=False).input_ids
        budget=maximum-len(answer)
        context=prefix
        if len(context)>budget:
            # Keep the instruction and the end of the supplied legal material.
            context=context[:128]+context[-(budget-128):]
            truncated=True
        inputs=torch.tensor([context+answer],device=model.device)
        target=inputs.clone()
        target[:,:len(context)]=-100
        with torch.no_grad():
            output=model(input_ids=inputs,labels=target)
        losses.append(float(output.loss))
    scores=-np.asarray(losses,dtype=float)
    probabilities=np.exp(scores-scores.max())
    probabilities/=probabilities.sum()
    return losses,probabilities.tolist(),truncated


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-root",type=Path,default=Path("data/raw/public_extended/LegalBench"))
    parser.add_argument("--model",required=True)
    parser.add_argument("--adapter",type=Path)
    parser.add_argument("--training-seed",type=int)
    parser.add_argument("--condition",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=240)
    parser.add_argument("--sample-seed",type=int,default=20261101)
    parser.add_argument("--max-length",type=int,default=1024)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the frozen LegalBench diagnostic.")
    torch.manual_seed(41)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    if args.adapter is not None:
        model=PeftModel.from_pretrained(model,str(args.adapter))
    model.eval()
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update({"label_source":"published_legalbench_test_labels","adapter_loaded":args.adapter is not None,
                   "scoring":"mean_answer_token_nll_then_label_softmax","truncation":"first_128_and_suffix",
                   "backbone_parameters":sum(p.numel() for p in model.parameters())})
    (args.output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    summaries={}
    started=time.time()
    with (args.output/"predictions.jsonl").open("w",encoding="utf-8") as handle:
        for name,task in TASKS.items():
            rows=[json.loads(line) for line in (args.raw_root/name/"test.jsonl").open(encoding="utf-8")]
            rows.sort(key=lambda row:row["id"])
            if len(rows)>args.limit:
                rows=random.Random(args.sample_seed).sample(rows,args.limit)
                rows.sort(key=lambda row:row["id"])
            predictions=[]
            for row in rows:
                losses,probabilities,truncated=score_labels(model,tokenizer,prompt_for(task,row),task.labels,args.max_length)
                result={"task":name,"id":row["id"],"gold":row["label"],"labels":task.labels,
                        "prediction":task.labels[int(np.argmin(losses))],"probabilities":probabilities,
                        "answer_nll":losses,"prompt_truncated":truncated}
                predictions.append(result)
                handle.write(json.dumps(result)+"\n")
            handle.flush()
            summaries[name]=classification_metrics(predictions,task.labels)
            print(json.dumps({"task":name,**summaries[name]}),flush=True)
    metrics={"condition":args.condition,"tasks":summaries,"elapsed_seconds":time.time()-started,
             "expert_conflict_construct_validated":False,"cross_jurisdiction_transfer_evaluated":False}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
