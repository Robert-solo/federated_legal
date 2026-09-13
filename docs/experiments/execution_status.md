# Experiment Execution Status

Last updated: 2026-09-11

## Completed Evidence

- A deterministic Qwen3.8-27B central API baseline completed on the first 100
  CaseHOLD validation examples with 77.0% accuracy, 100% parse coverage, zero API
  errors, and a passing acceptance artifact. Provider-side thinking was explicitly
  disabled and the model was constrained to label-only output.
- The public extended dataset manifest records 14 datasets and 28 splits, and all configured adapters load valid samples.
- The real Flower, HuggingFace, and PEFT pipeline completed a bounded three-round CaseHOLD run.
- Five FedAvg seeds completed three rounds with four clients and 240 evaluation examples per seed.
- The bounded five-seed result is 29.17% mean final accuracy with 0.98 percentage-point standard deviation.
- Five FedAvg seeds and three FedProx seeds completed the 20-round E1 CaseHOLD screening contract.
- The target-conditioned Flower runtime passed one-round lambda-zero and FLEN smoke tests.
- A three-round FLEN activation run showed non-uniform exposure and conflict-adjusted weights,
  with client-local residuals excluded from shared aggregation.
- Twenty-four matched 20-round runs now cover eight methods over seeds 41--43.
- The guarded FLEN result is 35.56% +/- 3.94 SD aggregate accuracy;
  its paired gain over the lambda-zero control is 1.67 points with a 95% interval
  from -4.80 to 8.13 points.

The CaseHOLD runs establish systems and optimization behavior only. The automatic proxy
calibration explicitly sets `formal_legal_claims_allowed = false`; it does not validate
legal-conflict semantics or authentic cross-jurisdiction transfer.

The Qwen API result is a central inference reference, not a federated participant.
Its current 100-example subset is not yet a paired comparison against the 240-example
Flower evaluation, so the raw accuracy gap must not be presented as a controlled
federated-versus-centralized effectiveness result.

## Completed Runs

The deterministic Qwen3.8-27B CaseHOLD API baseline is stored under
`outputs/api_baselines/qwen38_27b_casehold_100_nothinking_20260911/`. It completed
100 predictions in 28.22 seconds with 77 correct answers, parse rate 1.0, API error
rate 0.0, and `acceptance.json` reporting `passed = true`. The run stores every raw
response and a secret-free `run_config.json`. An earlier thinking-enabled run reached
only 31% parse coverage because its 256-token budget was consumed by truncated
reasoning; that run must not be reported as a clean 27% accuracy baseline.

The E12 round-10 checkpoint was subsequently evaluated at prediction level without
retraining. Its first 100 validation examples were paired by stable dataset order with
the deterministic API baseline. The API achieved 77/100 and E12 FedAvg achieved 27/100;
the API-minus-FedAvg accuracy difference was 0.50, with a paired bootstrap 95% interval
of [0.38, 0.61] and exact two-sided McNemar p = 1.04e-11. This is a diagnostic single
checkpoint comparison, not a multi-seed effectiveness claim. The export artifacts are
`checkpoint_predictions_round_0010.jsonl` and `paired_api_comparison_100.json` under
the E12 run directory.

The E1 CaseHOLD FedAvg 20-round development screening completed on the remote
Slurm cluster with seeds 41--45 as jobs `92973`--`92977`. All five runs contain
20 server rounds, 20 evaluation rounds, 80 client training events, 20 checkpoints,
and complete run summaries. Final accuracies are 0.3333, 0.3708, 0.3083,
0.3458, and 0.3042; each run reports `communication_bytes = 346030080`.

The matched E1 FedProx development screening completed with seeds 41--43 as
Slurm jobs `92985`--`92987`, using `fedprox_mu = 0.01`. All three passed the
20-round protocol with final accuracies 0.3208, 0.3750, and 0.3417.

The FLEN runtime smoke (`93045`), lambda-zero smoke (`93046`), and three-round
mechanism-activation preflight (`93048`) completed successfully. The activation
preflight used 24 disjoint probes per client; each round recorded all 72 directed
client--peer--probe comparisons per client and pair coverage 1.0. At least one
client had non-uniform conflict exposure in every round, producing nonzero changes
from base to final aggregation weights. Four local residual checkpoints were retained
client-side and excluded from the declared Flower upload payload.

The matched 20-round target-cohort and FLEN proxy-screening runs completed for
seeds 41--43 as jobs `93050`--`93055`. All six completed. The unguarded
FLEN variant averaged 25.28% accuracy versus 33.89% for lambda zero.

Component-isolation jobs `93072`--`93077` completed, with report-only recovery where
needed. Conflict-only aggregation averaged
35.42%, while unguarded personalization averaged 25.56%. Paired shared/personalized
evaluation showed that the unguarded residual reduced final accuracy by 8.19 points
on average.

Guarded personalization and guarded FLEN jobs `93081`--`93086` completed all
20 rounds. Two jobs initially failed only in report generation because the reporter
did not load `client_eval_log.jsonl`; their complete training artifacts were recovered
without retraining after the one-line reporting fix. All six final reports are complete.
Guarded personalization averaged 35.14%, and guarded FLEN averaged 35.56%. The gate
removed the unguarded collapse, but neither aggregate nor worst-client paired intervals
exclude zero with three seeds.

The unified analysis and paper-ready figures are stored under
`outputs/experiment_analysis/e1_flen_screening/`.

## Ready To Submit

| Task | Command | Acceptance artifact |
|---|---|---|
| E12 ten-round real run | `bash scripts/submit_e12_casehold_real.sh` | `outputs/real_flower_peft/E12_casehold_real_seed42_10r/acceptance.json` |
| E1 FedAvg 20-round screening | `bash scripts/submit_e1_casehold_fedavg_screening.sh` | `outputs/real_flower_peft/E1_casehold_fedavg_20r_seed*/acceptance.json` |
| E1 target-cohort/FLEN screening | `bash scripts/submit_e1_casehold_flen_screening.sh` | `outputs/real_flower_peft/E1_casehold_{target_cohort,flen}_20r_*/acceptance.json` |
| E1 component ablation | `bash scripts/submit_e1_casehold_flen_ablation.sh` | `outputs/real_flower_peft/E1_casehold_{conflict_only,personalization_only}_20r_*/acceptance.json` |
| E1 guarded residual | `bash scripts/submit_e1_casehold_flen_guarded.sh` | `outputs/real_flower_peft/E1_casehold_{personalization_guarded,flen_guarded}_20r_*/acceptance.json` |

All launch paths use `scripts/validate_real_flower_run.py` after training. The validator
checks completion, the final checkpoint, usable metrics, normalized final aggregation
weights, and the presence of the method-specific outputs needed for analysis. It tolerates
duplicate log entries from harmless restarts and does not treat bookkeeping details as
scientific evidence.

## Current Local Limitation

The local Windows environment has no visible NVIDIA GPU and does not contain the
real-training dependencies. Remote execution is available through the configured
SSH key, but Slurm accounting queries intermittently report user/partition
permission warnings; job state must therefore be checked with both `sacct` and
the run-level artifacts.

## Remaining Main Evidence

- E1 SCAFFOLD, FedNova, FedLoRA, local-only, centralized, and stronger personalized comparisons.
- Expand the screening from three to five matched seeds before any
  comparative effectiveness claim.
- Expert annotation, adjudication, inter-annotator agreement, and independent test data
  are still needed before formal legal-conflict claims.
- E2 legal reliability metrics and prediction-level evaluation.
- E12 paired prediction export is complete; multi-seed paired API comparisons remain
  optional because the existing five-seed Flower runs do not yet retain prediction files.
- E3 communication and training-dynamics integration from accepted long runs.
- E4 conflict-aware diagnostics and claim-release gate.
- E5 cross-jurisdiction transfer with authentic source/target labels.
- E6-E16 ablations, privacy, robustness, human review, sensitivity, and error analysis.
## 2026-09-11 Authentic-Data Priority Audit

- **ECtHR country holdout: blocked.** The remote LexGLUE `ecthr_a` and `ecthr_b` JSONL files
  contain `id`, `institution_id`, `institution_type`, `jurisdiction`, `label`,
  `legal_tradition`, and `paragraphs`, but no explicit respondent-state/country field. The
  current files must not be partitioned into countries using text, filenames, labels, or
  heuristics.
- **MultiEURLEX language holdout: data-ready, runtime-blocked.** The remote corpus contains
  four 55,000-record language pairs (`en-de`, `en-fr`, `en-es`, `en-pl`) with `celex_id`,
  explicit language fields, and a common EU regulatory authority. This supports a multilingual
  EU-law experiment, not independent national-jurisdiction transfer.
- **Code delivered:** explicit metadata provenance in the canonical adapter, required-field
  rejection, `scripts/validate_authentic_metadata.py`, a MultiEURLEX language-client config,
  and regression tests for the metadata gate.
- **Validation:** `tests/test_authentic_metadata.py` and
  `tests/test_target_conditioned_aggregation.py` pass locally (12 tests). The full local suite
  remains blocked by the pre-existing Python 3.10 `datetime.UTC` import in the agent module.
- **Next executable task:** generalize the Flower/PEFT runner beyond CaseHOLD with a multilabel
  task adapter, frozen CELEX-level split/partition manifest, and identical-budget FedAvg/FedProx/
  target-cohort/FLEN comparison. No result should enter the paper before that runner completes
  five matched seeds and per-language held-out evaluation.
- **Preparation added:** `scripts/prepare_multieurlex_experiment.py` now creates language-client
  files and rejects duplicate or cross-split CELEX identifiers before training. Its manifest
  explicitly records that the task is multilingual EU-law transfer rather than national
  jurisdiction transfer.
- **Frozen protocol added:** `configs/experiments/multieurlex_language_holdout.yaml` records the
  50-round, three-client training condition for each one-language holdout. It is a protocol
  contract only until a multilabel PEFT runner is implemented; the existing CaseHOLD runner and
  dry-run Flower orchestrator must not be used to populate its results.
- **Remote preparation result:** four symmetric holdouts (`en-de`, `en-fr`, `en-es`, `en-pl`)
  were generated on zhurong with 2,000 records per split and client. Every manifest reports
  `status=prepared` and an empty `cross_split_overlaps` set. These are frozen data artifacts,
  not model-performance results.
- **Training implementation added:** `scripts/run_multieurlex_federated.py` is a real
  HuggingFace/PEFT multilabel loop with language clients, LoRA adapter exchange, FedAvg,
  FedProx, FLEN-weighted aggregation, and per-language held-out evaluation. It is separate from
  the CaseHOLD-specific runner and must first pass smoke validation before formal runs.
- **Backbone adjustment:** the runner uses the cached Qwen backbone with a LoRA feature
  adapter and a 21-dimensional multilabel head. The original XLM-R smoke was blocked because
  Hugging Face is unreachable from the login node and XLM-R is not cached; GPU execution must
  use `scripts/slurm_multieurlex_federated.sh` with offline loading.
- **First GPU attempt:** job `104791` reached `gpu04` but failed before model loading because
  the cache was addressed by model name and Transformers attempted an offline HEAD lookup. The
  failure produced no metrics and is not counted as an experiment result. The Slurm launcher now
  passes the resolved cached snapshot path directly.
- **Second GPU attempt:** job `104796` failed before loading because the first resolved snapshot
  contained tokenizer files but not `config.json` or model weights. It produced no metrics and is
  not counted. The launcher now points to the cache snapshot containing both `config.json` and
  `model.safetensors`.
- **Third GPU attempt:** job `104799` was cancelled after accounting showed only 11 seconds of
  CPU activity and no output. Diagnostic job `104803` showed that `conda activate` hangs on the
  compute node during Slurm partition lookup. The launcher now bypasses activation and calls the
  environment's Python executable directly.
- **Fourth GPU attempt:** job `104805` loaded the cached model and reached the real runner, then
  failed on a missing `config` forwarding property in the custom classifier wrapper. It produced
  no metrics and is not counted. The wrapper contract and regression test are now fixed.
- **Successful real smoke:** job `104813` completed on `gpu25` with the Qwen backbone, LoRA
  feature adapter, multilabel head, three training-language clients, one round, and 50 records
  per client/split. For `en-pl` holdout, the resulting test micro-F1 was `0.1994`; other-language
  test micro-F1 values were `0.1804` (`en-de`), `0.1215` (`en-es`), and `0.1891` (`en-fr`). These
  are smoke diagnostics only: one seed, one round, and 50 examples, not paper evidence.
- **Formal first run submitted:** job `104815` is running on `gpu48` for the frozen `en-pl`
  holdout, FedAvg, three seeds, three rounds, and 500 records per client/split. Its outputs will
  be accepted only if all seed directories contain `final_metrics.json` and the run manifests
  pass the existing artifact checks.
