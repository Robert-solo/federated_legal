# Baseline FedLoRA Experiment

## Goal

Establish a Flower baseline for federated LoRA adapter training across simulated legal jurisdictions.

## Config

`configs/experiments/baseline_fedlora.yaml`

## Measured CaseHOLD Five-Seed Run

The controlled launcher uses the real Flower/PEFT runtime with five training seeds and a fixed heuristic jurisdiction partition. It is a FedAvg baseline only; it does not claim conflict-aware aggregation, formal privacy, multi-agent reasoning, or authentic cross-jurisdiction generalization.

Config: `experiments/baseline_fedlora/casehold_fedavg_5seed.yaml`

Submit on the configured Slurm cluster:

```bash
sbatch scripts/slurm_casehold_fedavg_multiseed.sh
```

Each run writes a resolved configuration, environment metadata, data counts, per-round logs, checkpoints, and a final report. The launcher then generates a CSV/Markdown/SVG summary under `outputs/evaluation/paper_casehold_fedavg_5seed`.

## Planned Outputs

- logs: `outputs/logs/baseline_fedlora`
- metrics: `outputs/metrics/baseline_fedlora`
- figures: `outputs/figures/baseline_fedlora`

## Reproducibility

```bash
python scripts/validate_config.py configs/experiments/baseline_fedlora.yaml
```

The broader multi-dataset FedLoRA comparison remains planned.

## E12 Ten-Round Acceptance Run

The E12 run extends the bounded three-round check to the minimum accepted ten-round real-training protocol. Its typed configuration is `configs/experiments/e12_casehold_real.yaml`.

Submit on the configured Slurm cluster:

```bash
bash scripts/submit_e12_casehold_real.sh
```

The submitted job writes `outputs/real_flower_peft/E12_casehold_real_seed42_10r/acceptance.json`. A run is accepted only when all ten server and evaluation rounds, all four client records per round, checkpoints, communication totals, and the final report agree.

## E1 FedAvg Screening

The first E1 baseline extends the same controlled CaseHOLD protocol to 20 rounds and five seeds. It remains a FedAvg-only screening component and must not be reported as the completed E1 comparison matrix.

```bash
bash scripts/submit_e1_casehold_fedavg_screening.sh
```

The launch manifest is `experiments/baseline_fedlora/casehold_e1_fedavg_screening.yaml`. FedProx, SCAFFOLD, FedNova, FedLoRA, centralized, local-only, and FLEN runs remain separate required E1 tasks.
