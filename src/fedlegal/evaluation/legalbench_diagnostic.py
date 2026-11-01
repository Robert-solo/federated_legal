"""Published-label diagnostics for authority regime, precedent and citation tasks."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClassificationTask:
    name: str
    labels: tuple[str, ...]
    question: str


TASKS = {
    "overruling": ClassificationTask("overruling",("No","Yes"),
        "Does the passage overrule a precedent?"),
    "ucc_v_common_law": ClassificationTask("ucc_v_common_law",("Common Law","UCC"),
        "Is the described contract governed by the UCC or the common law?"),
    "citation_prediction_classification": ClassificationTask("citation_prediction_classification",("No","Yes"),
        "Is the proposed case a correct supporting citation for the legal statement?"),
    "personal_jurisdiction": ClassificationTask("personal_jurisdiction",("No","Yes"),
        "Does the forum court have personal jurisdiction over the defendant under US law?"),
}


def prompt_for(task: ClassificationTask, row: dict) -> str:
    context = str(row["text"])
    if task.name == "citation_prediction_classification":
        context += "\n\nProposed citation: " + str(row["citation"])
    return (task.question+"\n\nMaterial:\n"+context+
            "\n\nReturn exactly one label: "+" | ".join(task.labels)+".")


def classification_metrics(rows: list[dict], labels: tuple[str,...]) -> dict:
    gold = np.asarray([labels.index(row["gold"]) for row in rows])
    predictions = np.asarray([labels.index(row["prediction"]) for row in rows])
    probabilities = np.asarray([row["probabilities"] for row in rows])
    f1s, recalls = [], []
    for label in range(len(labels)):
        tp = int(((gold==label)&(predictions==label)).sum())
        fp = int(((gold!=label)&(predictions==label)).sum())
        fn = int(((gold==label)&(predictions!=label)).sum())
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        if (gold==label).any():
            recalls.append(tp/(tp+fn))
    correct = (gold==predictions)
    confidence = probabilities.max(axis=1)
    ece = 0.
    for index in range(10):
        mask=(confidence>=index/10)&(confidence<(index+1)/10 if index<9 else confidence<=1)
        if mask.any():
            ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    targets=np.eye(len(labels))[gold]
    return {"examples":len(rows),"accuracy":float(correct.mean()),"macro_f1":float(np.mean(f1s)),
            "balanced_accuracy":float(np.mean(recalls)),
            "brier_multiclass":float(np.mean(np.sum((probabilities-targets)**2,axis=1))),
            "ece_10":float(ece),"majority_class_rate":float(np.bincount(gold).max()/len(gold)),
            "label_counts":{label:int((gold==i).sum()) for i,label in enumerate(labels)}}
