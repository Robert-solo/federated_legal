# Frozen revision experiments (2026-11-01)

The maintained manuscript is `papers/ipm/Federated Legal elsevier/cas-sc-template.tex`.
Configuration: `configs/experiments/revision_v2.yaml`.

## Architecture and method correspondence

`scripts/run_casehold_revision.py` reuses the actual PEFT client and Flower
aggregation strategies from `scripts/run_real_flower_peft.py`. Four persistent
clients execute sequentially on one GPU. Flower FitRes/EvaluateRes objects and
aggregation execute in-process; communication numbers are logical payload bytes.
This run does not measure network transport latency or secure aggregation.

`src/fedlegal/training/dual_lora.py` implements independent shared and private
matrix updates, `B_shared A_shared + B_local A_local`. Only the shared branch is
uploaded. The historical `factor_residual` option remains for reproducing old
results; adding factors introduces cross terms and is a separate experiment.
Gate examples are held out from both shared and private training in all methods.

The gate contrasts are guarded versus unguarded at 8+2 and 10+2 shared/private
optimizer steps. FedAvg and FedProx use 10 shared steps. Ditto uses 8 shared and
2 private steps with proximal coefficient 0.01. FLEN uses an automatic prediction
proxy until genuine expert calibration is available.

## Execution

On the configured server, from `/remote_dir/home/junluo/IPM`:

```bash
sbatch --array=0-39%8 scripts/slurm_revision_casehold.sh
bash scripts/submit_revision_privacy.sh
sbatch --dependency=afterok:TRAIN_JOB scripts/slurm_revision_agents.sh
```

Use a new output root for a separate run. A failed run before training may be
moved to `outputs/revision_v2/failed_before_training/` and resubmitted with the
same frozen settings. Data are loaded into memory to avoid NFS dataset-cache
locks. Smoke runs use separate roots and do not enter formal analyses.

## Artifacts and analysis

- CaseHOLD: `outputs/revision_v2/casehold/SPEC_seedN/` contains resolved config,
  environment, per-round server/client logs, drift and communication, private
  branch decisions, shared adapter, and final independent-test metrics/predictions.
- Privacy: `outputs/experiments/revision_v2_privacy_*` contains config, training
  log, calibration audit, test probabilities, metrics, and membership-attack ROC.
- Agents: `outputs/revision_v2/human_review/` contains base/federated backbone
  crossed with one/four-call single-agent controls and four-stage deliberation.
  Review sheets are blinded with a separate condition key.

```bash
python scripts/analyze_revision_v2.py --kind casehold --input outputs/revision_v2/casehold --output outputs/revision_v2/analysis/casehold
python scripts/analyze_revision_v2.py --kind privacy --input outputs/experiments --output outputs/revision_v2/analysis/privacy
python -m pytest -q
```

Analysis writes JSON, run-level CSV, seed-level summaries, paired contrasts,
and PDF/PNG plots. Figures use seed SD; CSV/JSON also include paired Student-t
95% intervals. `matrix_complete` must be true before reporting the full matrix.
Completed privacy runs are retained regardless of threshold location or utility.

## Interpretation

The test sample is fixed before the revision runs and opened after round 20;
there is no test-driven checkpoint selection. Its round-robin allocation provides
a worst-client deployment diagnostic, not worst-jurisdiction generalization.
Training cohorts are heuristic. A separate authentic jurisdiction benchmark is
still required for the broader legal-transfer claim.

Privacy conditions are no clipping, clipping only, and Gaussian noise 0.25/0.5/1.0.
The central mechanism assumes a trusted aggregator and client adjacency.
The loss attack measures record membership and does not establish client-level
attack resistance. Report micro/macro F1, BCE, Brier, pooled binary-label ECE,
and attack AUC separately; compare noise with clipping only, then clipping with
the unclipped control. Source training probes and threshold-validation records
are disjoint. Nonmembers are source-language test records.

Human authority material is prepared by `scripts/prepare_expert_annotation.py`.
See `docs/experiments/HUMAN_ANNOTATION_GUIDE.md`. The 120-pair US federal pilot
requires actual annotation. One author can contribute one annotation record per
item; a repeat pass by that author remains single-person review. The supplied
blank files and model suggestions are not human gold labels.

## Remaining evidence

Expert conflict classification, authenticated jurisdiction transfer, citation
consistency and hallucination rates from human review require completed labels.
The agent protocol currently uses the shared FedAvg adapter; it tests the
federation/inference connection, not the effect of a target-private FLEN branch.
Additional strong methods such as SCAFFOLD/FedJudge require faithful implementations
and matched tuning budgets before any state-of-the-art claim.
