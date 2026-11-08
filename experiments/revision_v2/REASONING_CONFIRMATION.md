# New-case reasoning confirmation

This experiment tests whether the initial 60-case, one-checkpoint observations
replicate on 200 unused cases with five checkpoints per adapted group. It retains
the existing four-call self-refinement and four-stage prompts, parser, decoding,
and 160-token generation limit. The base model is evaluated once.

Configuration: `configs/experiments/reasoning_confirmation_20261108.yaml`.
`prepare_reasoning_confirmation.py` freezes case IDs and exclusions in
`outputs/revision_v2/reasoning_confirmation/protocol.json` before model evaluation.
The 1,000 final training-test cases and original 60 inference cases are excluded.

```bash
python scripts/prepare_reasoning_confirmation.py
sbatch --array=0-10%4 scripts/slurm_reasoning_confirmation.sh
# After the returned array job completes:
python scripts/analyze_reasoning_confirmation.py
```

Run configurations, predictions, timing/token usage and automatic metrics live in
`outputs/revision_v2/reasoning_confirmation/runs/<condition>/`. Scheduler logs live
in `outputs/slurm/ipm-reasoning-confirm-*`. The analyzer writes per-checkpoint
CSV, seed-level paired intervals, exact sign-flip tests with Holm adjustment for
the two group comparisons, and PDF/PNG visualizations under
`outputs/revision_v2/analysis/reasoning_confirmation/`.

The primary replication unit is the training seed, evaluated on the same frozen
200 cases. Case-bootstrap intervals describe an individual checkpoint; repeated
cases across seeds are not independent case observations. Conditional accuracy on
outputs parsed in both conditions is descriptive. All cases enter primary
accuracy, with parse failures counted as errors. Four calls match invocation
counts; actual token and elapsed costs are reported separately. Human legal
review remains a separate study.
