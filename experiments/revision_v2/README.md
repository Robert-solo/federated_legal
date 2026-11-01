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

## Frozen extensions

`configs/experiments/revision_v2_extensions.yaml` specifies the additional studies.
The original 40-run CaseHOLD matrix keeps its completion definition. FedAdam adds
five independent runs in `outputs/revision_v2/casehold_fedadam` using Flower's
`FedAdam` with fixed server settings, and identical FedAvg client training.
Server moments persist across rounds; the implementation test checks this.

MultiEURLEX starts a fresh 120-run matrix: six methods, four held-out languages,
and seeds 41--45. Each source client uses 500 training records and 64 separate
training-split probes; threshold selection uses validation records. The test
sample is fixed by evaluation seed 20261101, and predictions are written only
after round three. `local_only` has target supervision; `unadapted` freezes the
backbone while training its classification head. Search-boundary thresholds stay
in the analysis. No legacy or privacy run is reused as a cell of this matrix.

LegalBench adds four published-label component tasks with 492 fixed cases. The
conditions are base Qwen, five FedAvg shared adapters, five FLEN shared adapters,
and pretrained Mistral 7B. Mean label-token NLL followed by class softmax defines
the scores; prompts contain material and the citation candidate, but no gold.
Keep the pretrained larger-model comparison distinct from federated algorithms.
The LegalBench sample is never used to fit the adapters or choose checkpoints.
Truncation preserves the first 128 prompt tokens and the suffix within 1,024
tokens; record-level predictions report whether truncation occurred.

The agent extension adds the FLEN shared checkpoint to base and FedAvg. All use
the same 60 cases, one/four-call single-agent budgets, and four-stage deliberation.
The generator records prompt tokens, generated tokens, and elapsed analysis time.
The shared presentation step is excluded from analysis cost. The two deterministic
copies of multi-agent deliberation are not pooled as independent cases. Adapter
comparisons use seed-41 checkpoints; five-seed adapter inference is evaluated in
LegalBench. Human review sheets remain blank until people fill them.

```bash
sbatch --array=0-23%4 scripts/slurm_revision_multieurlex.sh
sbatch --array=0-4%2 --export=ALL,SPEC=fedadam_10,OUTPUT_ROOT=outputs/revision_v2/casehold_fedadam scripts/slurm_revision_casehold.sh
sbatch --array=0-1 --export=ALL,GROUP=base scripts/slurm_revision_legalbench.sh
sbatch --array=0-4%2 --export=ALL,GROUP=fedavg scripts/slurm_revision_legalbench.sh
sbatch --array=0-4%2 --export=ALL,GROUP=flen scripts/slurm_revision_legalbench.sh
sbatch scripts/slurm_revision_flen_agents.sh
python scripts/analyze_revision_v2.py --kind casehold --input outputs/revision_v2/casehold --fedadam-input outputs/revision_v2/casehold_fedadam --output outputs/revision_v2/analysis/casehold_extended
python scripts/analyze_revision_multieurlex.py --input outputs/experiments --output outputs/revision_v2/analysis/multieurlex
python scripts/analyze_revision_legalbench.py --input outputs/revision_v2/legalbench --output outputs/revision_v2/analysis/legalbench
python scripts/analyze_revision_agents.py --input outputs/revision_v2/human_review --output outputs/revision_v2/analysis/agents
python scripts/publish_revision_results.py --analysis-root outputs/revision_v2/analysis
python scripts/export_revision_evidence.py --output outputs/revision_v2/revision_evidence.tar.gz
```

Training dependencies must complete before adapter inference. `GROUP=base` needs
no adapter; all other groups require an exported final shared checkpoint. Check
that the output root is new before a separate run. Example commands assume these
dependencies have completed, whereas the submitted jobs use Slurm dependencies.

Analyses retain run metrics, seed summaries, paired Student-t intervals, exact
sign-flip p-values and Holm adjustment within each metric's contrast family.
Five matched pairs give a minimum two-sided exact p-value of 0.0625. Mean and
worst held-out-language scores are calculated within seed before inference.
Logical communication counts fit-phase model/probe payload; it excludes network
framing and separately invoked evaluation payload. Wall time includes the actual
recorded GPU and is descriptive rather than a hardware-normalized speed claim.

`publish_revision_results.py` writes tables, figures and CAS sections only for
completed matrices. The maintained CAS source selects these sections when they
exist and otherwise retains its explicitly labelled legacy results. The evidence
archive includes those generated CAS assets. Recompile the manuscript after
retrieval, and align the abstract/discussion/conclusion with the complete results
before submission; generated sections never count as completed human reviews.

## Remaining evidence

Expert conflict classification, authenticated jurisdiction transfer, citation
consistency and hallucination rates from human review require completed labels.
The agent protocol exports shared FedAvg and FLEN adapters; target-private
branches remain local and are not loaded into the shared inference checkpoint.
Additional strong methods such as SCAFFOLD/FedJudge require faithful implementations
and matched tuning budgets before any state-of-the-art claim.
