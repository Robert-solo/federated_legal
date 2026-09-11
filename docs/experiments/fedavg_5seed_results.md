# CaseHOLD FedAvg Five-Seed Results

## Completion Audit

The five bounded CaseHOLD runs for seeds 41--45 completed all three federated rounds. Each run contains four client updates and four client evaluations per round, zero recorded evaluation failures, a final report, and matching raw/report communication totals.

Slurm records parent job `90728` as `FAILED` with exit code `127:0`. The failure occurred after all five training reports were written, when the parent launcher invoked the summary script without an active Python environment. The launcher now uses the experiment environment's explicit interpreter. The validated training results remain usable; the original Slurm state must still be disclosed for provenance.

Job `90721` is not part of this result set. It is a currently running round-two experiment under `/remote_dir/home/junluo/Symbolic`, so neither its smoke outputs nor partial files are included in this analysis.

## Quantitative Findings

Across five seeds, round-three validation accuracy is **29.17% +/- 0.98 SD**, with a seed-level 95% t interval of **27.95% to 30.38%**. Mean accuracy rises from 23.92% in round one to 29.17% in round three, a gain of 5.25 percentage points. Mean validation loss decreases from 0.2851 to 0.1454, a 49.0% reduction.

Each seed communicates 49.50 MiB bidirectionally over three rounds. The mean client update norm relative to the broadcast adapter rises from 0.4515 to 0.5561 in round two and then falls to 0.5310 in round three. This is not the pairwise client-drift metric required by the full protocol, and the non-monotonic pattern does not demonstrate conflict mitigation.

Round-three client means range from 21.00% for `client_0` to 33.67% for `client_2`, a 12.67-point gap. The aggregate is stable across seeds, but the client-level disparity indicates meaningful non-IID difficulty that should be addressed by stronger federated baselines.

## Reviewer Interpretation

These results verify that the real Flower/PEFT pipeline trains, evaluates, logs communication, and converges above the five-choice chance reference under the bounded CaseHOLD protocol. They do not validate the paper's main conflict-aware, privacy-preserving, multi-agent, or cross-jurisdiction claims. The run uses one US legal benchmark, 240 validation examples, three rounds, and ten local optimization steps. The seed-level confidence interval measures training-seed variability on the same evaluation set and should not be interpreted as full dataset-sampling uncertainty.

## Artifacts

- Analysis command: `python scripts/analyze_fedavg_5seed.py`
- Machine-readable summary: `outputs/experiment_analysis/fedavg_5seed/results/summary.json`
- Seed metrics: `outputs/experiment_analysis/fedavg_5seed/results/seed_metrics.csv`
- Round metrics: `outputs/experiment_analysis/fedavg_5seed/results/round_metrics_summary.csv`
- Client metrics: `outputs/experiment_analysis/fedavg_5seed/results/client_final_summary.csv`
- Input file inventory: `outputs/experiment_analysis/fedavg_5seed/results/input_manifest.csv`
- Figures: `outputs/experiment_analysis/fedavg_5seed/results/fedavg_5seed_{convergence,robustness,efficiency}.{png,pdf,svg}`
