# MultiEURLEX comparison results

These files summarize zhurong Slurm jobs `107186`--`107193` and
`107196`--`107199`. The accepted matrix contains 36 runs: FedAvg, FedProx, and
the lambda-zero control across four held-out language pairs and seeds 41--43.

The source runs use Qwen2.5-0.5B-Instruct, three LoRA clients, three rounds, and
500 training, validation, and test records per language. Thresholds are selected
on source-language validation micro-F1. The held-out language is excluded from
selection. `scripts/analyze_multieurlex_comparison.py` validates the acceptance,
configuration, calibration, and prediction artifacts before producing these
tables.

This is a multilingual EU-law development screening. It does not include FLEN,
does not establish positive transfer relative to local-only or unadapted models,
and does not support independent national-jurisdiction rule migration.
