"""Package numeric evidence and configs without checkpoints or benchmark text."""
from __future__ import annotations
import argparse
import io
import json
import tarfile
from pathlib import Path

CASEHOLD=["resolved_config.json","environment.json","acceptance.json","test_metrics.json",
          "test_predictions.jsonl","communication_log.jsonl","client_eval_log.jsonl",
          "server_rounds.jsonl","server_eval_rounds.jsonl","aggregation_rounds.jsonl"]
MULTIEURLEX=["run_config.json","acceptance.json","final_metrics.json","metrics.csv",
            "predictions.jsonl","threshold_calibration.json","aggregation_audit.json"]
LEGALBENCH=["run_config.json","metrics.json","predictions.jsonl"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with tarfile.open(args.output,"w:gz") as archive:
        groups=[(args.root/"outputs/revision_v2/casehold","*/test_metrics.json",CASEHOLD,"casehold/source_runs"),
                (args.root/"outputs/revision_v2/casehold_fedadam","*/test_metrics.json",CASEHOLD,"casehold_fedadam/source_runs"),
                (args.root/"outputs/experiments","revision_v2_multieurlex_*/acceptance.json",MULTIEURLEX,"multieurlex/source_runs"),
                (args.root/"outputs/revision_v2/legalbench","*/metrics.json",LEGALBENCH,"legalbench/source_runs")]
        for root,pattern,names,prefix in groups:
            for marker in sorted(root.glob(pattern)):
                for name in names:
                    path=marker.parent/name
                    if path.exists():
                        archive.add(path,arcname=f"{prefix}/{marker.parent.name}/{name}")
                        count+=1
        agents=args.root/"outputs/revision_v2/human_review"
        for marker in sorted(agents.glob("*/automatic_metrics.json")):
            prefix=f"agents/source_runs/{marker.parent.name}"
            for name in ("run_config.json","automatic_metrics.json","automatic_analysis.json"):
                path=marker.parent/name
                if path.exists():
                    archive.add(path,arcname=f"{prefix}/{name}")
                    count+=1
            numeric=[]
            for line in (marker.parent/"predictions.jsonl").read_text(encoding="utf-8").splitlines():
                row=json.loads(line)
                numeric.append({key:row[key] for key in ("case_id","gold_label","single_label","single_correct",
                    "multi_label","multi_correct","single_analysis_usage","multi_analysis_usage") if key in row})
            payload=("\n".join(json.dumps(row) for row in numeric)+"\n").encode("utf-8")
            entry=tarfile.TarInfo(f"{prefix}/predictions.jsonl")
            entry.size=len(payload)
            archive.addfile(entry,io.BytesIO(payload))
            count+=1
        analysis=args.root/"outputs/revision_v2/analysis"
        if analysis.exists():
            for path in sorted(analysis.rglob("*")):
                if path.is_file():
                    archive.add(path,arcname=f"analysis/{path.relative_to(analysis).as_posix()}")
                    count+=1
        paper=args.root/"papers/ipm/Federated Legal elsevier"
        for folder,suffix in [("sections","tex"),("tables","tex"),("figures","pdf")]:
            for stem in ("revision_casehold","revision_gate","revision_multieurlex","revision_legalbench","revision_agents"):
                for path in sorted((paper/folder).glob(f"{stem}*.{suffix}")):
                    archive.add(path,arcname=f"cas_assets/{folder}/{path.name}")
                    count+=1
    print(f"Packaged {count} evidence files in {args.output}")


if __name__=="__main__":
    main()
