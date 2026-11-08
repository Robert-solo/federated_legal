"""Analyze frozen revision matrices; preserve all completed runs and paired seeds."""
from __future__ import annotations

import argparse
import csv
import json
import itertools
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_multieurlex_comparison import interval

CASEHOLD = ["fedavg_10", "fedprox_10", "ditto_8_2", "unguarded_8_2",
            "guarded_8_2", "unguarded_10_2", "guarded_10_2", "flen_guarded_8_2"]
PRIVACY = ["no_clip", "clipped_only", "gaussian_0_25", "gaussian_0_5", "gaussian_1_0"]
PRIVACY_LABELS = dict(zip(PRIVACY,["No clipping","Clipping only","Gaussian 0.25","Gaussian 0.5","Gaussian 1.0"]))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixed_test_shard_accuracy(predictions):
    """Use the frozen round-robin test assignment, including early shared-only runs."""
    shards = [[] for _ in range(4)]
    for row in predictions:
        shard = int(row["example_id"].rsplit("-", 1)[1]) % 4
        shards[shard].append(row["gold"] == row["prediction"])
    return min(statistics.fmean(values) for values in shards)


def load_casehold(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/test_metrics.json")):
        result = read_json(path)
        if result.get("protocol") != "revision_v2_dual_lora":
            continue
        config = read_json(path.parent / "resolved_config.json")
        if result["completed_rounds"] != 20 or result["test_examples"] != 1000:
            raise ValueError(f"Formal protocol differs: {path.parent}")
        if not config["test_evaluated_after_final_round_only"]:
            raise ValueError(f"Test protocol differs: {path.parent}")
        predictions = read_lines(path.parent / "test_predictions.jsonl")
        reconstructed = statistics.fmean(row["gold"] == row["prediction"] for row in predictions)
        if abs(reconstructed-result["accuracy"]) > 1e-8:
            raise ValueError(f"Prediction accuracy differs: {path.parent}")
        events = read_lines(path.parent / "communication_log.jsonl")
        gate_events = [event for event in events if "local_gate_accepted" in event["metrics"]]
        diagnostics = {}
        if gate_events:
            diagnostics = {
                "gate_acceptance_rate": statistics.fmean(e["metrics"]["local_gate_accepted"] for e in gate_events),
                "gate_candidate_accuracy_gain": statistics.fmean(e["metrics"]["local_gate_candidate_accuracy"]-e["metrics"]["local_gate_shared_accuracy"] for e in gate_events),
            }
        if all("personalization_accuracy_gain" in client for client in result["clients"]):
            diagnostics["test_personalization_accuracy_gain"] = statistics.fmean(client["personalization_accuracy_gain"] for client in result["clients"])
        rows.append({key: result[key] for key in (
            "spec", "seed", "accuracy", "worst_client_accuracy", "elapsed_seconds"
        )} | {
            "original_worst_client_accuracy": result["worst_client_accuracy"],
            "worst_client_accuracy": fixed_test_shard_accuracy(predictions),
            "condition": result["spec"], "path": str(path.parent),
            "communication_bytes": sum(event["num_bytes"] for event in events),
            "mean_client_drift_l2": statistics.fmean(event["metrics"]["client_drift_l2"] for event in events),
            "client_fit_seconds": sum(event["metrics"]["elapsed_sec"] for event in events),
            **diagnostics,
        })
    return rows


def load_privacy(root: Path) -> list[dict]:
    rows = []
    for condition in PRIVACY:
        for path in sorted(root.glob(f"revision_v2_privacy_{condition}_en-pl_seed*/final_metrics.json")):
            attack_path = path.parent / "privacy_attack" / "metrics.json"
            if not attack_path.exists():
                continue
            result, attack = read_json(path), read_json(attack_path)
            rows.append({
                "condition": condition, "seed": result["seed"], "path": str(path.parent),
                **{metric: result[f"en-pl_{metric}"] for metric in ("micro_f1", "macro_f1", "loss", "brier", "ece_15")},
                "attack_auc": attack["attack_auc"], "attack_balanced_accuracy": attack["balanced_accuracy"],
                "epsilon": result.get("dp_epsilon"),
                "communication_bytes": result["logical_communication_bytes"]*result["round"],
                "worst_language_micro_f1": min(result[f"{lang}_micro_f1"] for lang in ("en-de", "en-fr", "en-es", "en-pl")),
                "threshold_at_search_boundary": result["threshold_at_search_boundary"],
            })
    return rows


def summarize(rows, conditions, seeds, metrics, contrasts):
    index = {(row["condition"], row["seed"]): row for row in rows}
    missing = [{"condition": c, "seed": s} for c in conditions for s in seeds if (c, s) not in index]
    summaries, paired = [], []
    for condition in conditions:
        cohort = [index[(condition, s)] for s in seeds if (condition, s) in index]
        if cohort:
            for metric in metrics:
                if not all(metric in row for row in cohort):
                    continue
                summaries.append({"condition": condition, "metric": metric} | interval([row[metric] for row in cohort]))
    for treatment, control in contrasts:
        common = [s for s in seeds if (treatment, s) in index and (control, s) in index]
        if common:
            for metric in metrics:
                if not all(metric in index[(c,s)] for c in (treatment,control) for s in common):
                    continue
                differences = [index[(treatment,s)][metric]-index[(control,s)][metric] for s in common]
                paired.append({"treatment": treatment, "control": control, "metric": metric, "paired_seeds": common,
                               "paired_sign_flip_pvalue": sign_flip_pvalue(differences)} | interval(differences))
    # Treat all planned contrasts for one metric as one exploratory family.
    for metric in metrics:
        family = sorted([row for row in paired if row["metric"]==metric],key=lambda row:row["paired_sign_flip_pvalue"])
        previous = 0.
        for rank,row in enumerate(family):
            previous = max(previous,min(1.,(len(family)-rank)*row["paired_sign_flip_pvalue"]))
            row["holm_adjusted_pvalue"] = previous
            row["multiplicity_family"] = f"all_planned_contrasts:{metric}"
    return {"matrix_complete": not missing, "completed_runs": len(rows), "missing_runs": missing,
            "runs": rows, "summaries": summaries, "paired_contrasts": paired,
            "inference_unit": "matched_training_seed", "completed_runs_excluded_for_poor_utility": 0}


def sign_flip_pvalue(differences):
    observed = abs(statistics.fmean(differences))
    permuted = [abs(statistics.fmean(sign*value for sign,value in zip(signs,differences)))
                for signs in itertools.product((-1,1),repeat=len(differences))]
    return sum(value>=observed-1e-12 for value in permuted)/len(permuted)


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)


def reliability(rows, output):
    import numpy as np
    import matplotlib.pyplot as plt
    bins = []
    for run in rows:
        totals, p_sum, y_sum = np.zeros(15), np.zeros(15), np.zeros(15)
        for prediction in read_lines(Path(run["path"])/"predictions.jsonl"):
            if prediction["language"] != "en-pl":
                continue
            p = np.asarray(prediction["probabilities"])
            y = np.zeros(len(p))
            y[prediction["gold_labels"]] = 1
            indices = np.minimum((p*15).astype(int),14)
            totals += np.bincount(indices,minlength=15)
            p_sum += np.bincount(indices,weights=p,minlength=15)
            y_sum += np.bincount(indices,weights=y,minlength=15)
        for index,count in enumerate(totals):
            if count:
                bins.append({"condition":run["condition"], "seed":run["seed"], "bin":index,
                             "count":int(count), "mean_probability":p_sum[index]/count,
                             "observed_frequency":y_sum[index]/count})
    write_csv(output/"reliability_bins.csv",bins)
    figure,axis = plt.subplots(figsize=(5.4,4.6))
    for condition in PRIVACY:
        points = []
        for index in range(15):
            cohort = [r for r in bins if r["condition"]==condition and r["bin"]==index]
            count = sum(r["count"] for r in cohort)
            if count:
                points.append((sum(r["mean_probability"]*r["count"] for r in cohort)/count,
                               sum(r["observed_frequency"]*r["count"] for r in cohort)/count))
        if points:
            axis.plot(*zip(*points), marker="o", markersize=3, label=PRIVACY_LABELS[condition])
    axis.plot([0,1],[0,1],linestyle="--",color="gray",linewidth=1)
    axis.set(xlabel="Mean predicted label probability",ylabel="Observed positive-label frequency",
             xlim=(0,1),ylim=(0,1),title="Held-out pooled-label reliability")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output/"privacy_reliability.pdf",bbox_inches="tight")
    figure.savefig(output/"privacy_reliability.png",dpi=160,bbox_inches="tight")
    plt.close(figure)


def plot(result, output, kind, conditions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("accuracy", "Test accuracy"), ("worst_client_accuracy", "Worst-client test accuracy")] if kind == "casehold" else [
        ("micro_f1", "Held-out micro F1"), ("attack_auc", "Record-membership AUC"),
        ("loss", "Held-out BCE"), ("brier", "Brier score"), ("ece_15", "Pooled-label ECE (15 bins)")]
    if kind == "casehold":
        figure, axes = plt.subplots(1, 2, figsize=(10,4.7))
    else:
        figure, grid = plt.subplots(2, 3, figsize=(10.5,6.5))
        axes = grid.flat
        grid[1,2].axis("off")
        grid[1,2].text(.05,.8,"Three matched seeds\nAll 15 completed runs retained\nTrusted aggregator\nRecord-membership attack",va="top",fontsize=11)
    index = {(row["condition"],row["metric"]):row for row in result["summaries"]}
    for axis,(metric,label) in zip(axes,panels):
        available = [c for c in conditions if (c,metric) in index]
        data = [index[(c,metric)] for c in available]
        axis.bar(range(len(data)), [r["mean"] for r in data],
                 yerr=[r["standard_deviation"] for r in data], color="#356c92", capsize=3)
        axis.set_xticks(range(len(data)), [PRIVACY_LABELS.get(c,c.replace("_", " ")) for c in available], rotation=45, ha="right", fontsize=9)
        axis.set_ylabel(label)
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        if metric=="attack_auc":
            axis.axhline(.5, color="gray", linestyle="--", linewidth=1)
    figure.suptitle("Frozen revision protocol" + (" (partial matrix)" if not result["matrix_complete"] else "") + "; error bars: seed SD", fontsize=11)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{kind}_revision.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["casehold","privacy"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fedadam-input", type=Path)
    args = parser.parse_args()
    if args.kind=="casehold":
        rows, conditions, seeds = load_casehold(args.input), CASEHOLD, list(range(41,46))
        metrics = ["accuracy", "worst_client_accuracy", "communication_bytes", "mean_client_drift_l2", "elapsed_seconds",
                   "client_fit_seconds","gate_acceptance_rate","gate_candidate_accuracy_gain","test_personalization_accuracy_gain"]
        contrasts = [("guarded_8_2","unguarded_8_2"),("guarded_10_2","unguarded_10_2"),
                     ("flen_guarded_8_2","guarded_8_2"),("ditto_8_2","fedavg_10"),("fedprox_10","fedavg_10")]
        if args.fedadam_input:
            rows += load_casehold(args.fedadam_input)
            conditions = CASEHOLD + ["fedadam_10"]
            contrasts += [("fedadam_10","fedavg_10"),("flen_guarded_8_2","fedadam_10")]
        reference = None
        for row in rows:
            gold = {p["example_id"]: p["gold"] for p in read_lines(Path(row["path"])/"test_predictions.jsonl")}
            if reference is not None and gold != reference:
                raise ValueError(f"Fixed test records differ: {row['path']}")
            reference = gold
    else:
        rows, conditions, seeds = load_privacy(args.input), PRIVACY, [41,42,43]
        metrics = ["micro_f1", "macro_f1", "worst_language_micro_f1", "loss", "brier", "ece_15", "attack_auc", "attack_balanced_accuracy"]
        contrasts = [(c,"clipped_only") for c in PRIVACY if c not in {"no_clip","clipped_only"}] + [("clipped_only","no_clip")]
    result = summarize(rows, conditions, seeds, metrics, contrasts)
    result["scope"] = ("proxy_non_iid_casehold; worst_client_uses_fixed_round_robin_test_shards" if args.kind=="casehold"
                       else "trusted_aggregator_client_update_DP; black_box_record_membership_diagnostic")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for name,key in (("completed_runs","runs"),("summary","summaries"),("paired_contrasts","paired_contrasts")):
        write_csv(args.output/f"{name}.csv",result[key])
    if rows:
        plot(result,args.output,args.kind,conditions)
        if args.kind=="privacy":
            reliability(rows,args.output)
    print(json.dumps({key:result[key] for key in ("matrix_complete","completed_runs","missing_runs")},indent=2))


if __name__ == "__main__":
    main()
