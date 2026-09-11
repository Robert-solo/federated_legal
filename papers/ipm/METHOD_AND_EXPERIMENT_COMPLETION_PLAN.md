# FLEN Method and Experiment Completion Plan

## 1. Evidence Principle

The remaining work must maximize evidential credibility rather than tune the protocol until every result is positive. Every hypothesis, primary metric, split, seed, baseline, and exclusion rule is frozen before the corresponding final run. A positive conclusion is permitted only when its pre-specified criterion is met; otherwise the paper reports a null or negative finding and narrows the claim.

The defensible final paper story is:

1. Legal disagreement is not equivalent to statistical drift.
2. FLEN separates transfer eligibility from unsupported or unresolved disagreement.
3. Target-conditioned aggregation protects compatible transfer while retaining jurisdiction-local capability.
4. Typed verification and privacy boundaries make the released output auditable.
5. The empirical contribution tests effectiveness, construct validity, reliability, privacy risk, and failure modes independently.

## 2. Implementation-Ready Method

### M1. Client Descriptor and Comparability Gate

Each client descriptor records six non-interchangeable fields: jurisdiction, authority scope, legal domain, language, institution type, and court level. Missing values remain explicit rather than being mapped to a generic jurisdiction.

For each probe-client pair, the gate checks:

- same task or an audited output mapping;
- same target jurisdiction or an audited authority mapping;
- compatible language or a frozen translation protocol;
- valid citation canonicalization;
- probe provenance and privacy-policy compliance.

Each comparable pair receives one routing state:

- `transferable`: may contribute to the target shared adapter;
- `authority_local`: legally valid but outside the target authority scope;
- `unsupported`: contains unsupported facts, authorities, or rule transfer;
- `unresolved`: insufficient evidence for transfer or rejection.

Authority eligibility is frozen from metadata. Model-dependent routing probabilities are calibrated only on the conflict-development set.

### M2. Probe and Conflict Benchmark

Create three immutable splits:

- `probe_train`: public or sanitized probes used during federated rounds;
- `conflict_calibration`: expert-labelled cases used to fit component weights and thresholds;
- `conflict_test`: untouched expert-labelled cases used only for final construct validation.

Each pair is double annotated as transferable, authority-local, unsupported, or unresolved. The annotation record includes target jurisdiction, task, admissible authorities, confidence, rationale category, and adjudication status. Report class prevalence, Cohen's kappa or Krippendorff's alpha, adjudication rate, and uncertainty.

The four conflict components are citation divergence, symmetric reasoning contradiction, verdict-distribution divergence, and typed rule-graph distance. Missing components are excluded and the remaining weights are renormalized per pair. Empty citations are unavailable, not zero conflict.

### M3. Target-Conditioned Aggregation

For target jurisdiction `j`, construct cohort `G_j` using the frozen comparability and authority gate. For client `k`, compute:

```text
base_mass_kj = authority_compatibility_kj * min(num_examples_k, sample_cap)
base_weight_kj = base_mass_kj / sum_l(base_mass_lj)
candidate_weight_kj ∝ base_weight_kj * exp(-lambda * conflict_exposure_kj)
floored_weight_kj = (1 - tau) * candidate_weight_kj + tau * base_weight_kj
weight_kj(t) = (1 - nu) * weight_kj(t-1) + nu * floored_weight_kj
```

Normalize the final weights within `G_j`. Log base, candidate, floored, smoothed, and final weights separately. The server must never use one global conflict scalar, citation volume, or agreement with the largest client as a reliability weight.

Required invariants:

- `lambda=0` reproduces the target-cohort base aggregation;
- equal exposure preserves base-weight ratios;
- increasing one client's unsupported exposure cannot increase its candidate weight;
- `tau>0` preserves at least the declared fraction of the base distribution;
- authority-incompatible clients cannot enter the shared target cohort;
- weights are finite, non-negative, deterministic, and sum to one;
- client ordering does not change the result.

### M4. Shared and Local LoRA Branches

Each client maintains two adapters over identical target modules:

- shared branch: initialized from the target adapter, trained in the federated local phase, clipped, and uploaded;
- local branch: initialized from the previous local checkpoint, trained in a fixed personalization phase with the shared branch frozen, and never uploaded.

Inference composes the base model, target shared adapter, and client-local adapter. Compare against shared-only FLEN and Ditto-style personalization to determine whether the residual adds utility or only complexity.

### M5. Privacy-Compatible Weighting

The baseline claim is data locality, not privacy. A formal privacy variant uses two stages:

1. compute weights from public probes and differentially private summaries, or through a declared secure computation;
2. publish weights, let clients scale clipped updates locally, and securely aggregate only the weighted sum.

The experiment must declare the protected unit, adjacency relation, clipping norm, sample rate, noise multiplier, accountant, final epsilon/delta, secure-aggregation dropout threshold, collusion assumption, and overhead. Update privacy and summary privacy are separate budgets.

### M6. Typed Multi-Agent Release Gate

Use the same backbone, retrieval corpus, maximum tokens, and tool budget for every agent comparison. The state graph emits schema-validated messages containing issue identifiers, target jurisdiction, canonical citations, verdict distribution, minimized reasoning summary, verifier result, conflict state, and privacy-policy status.

The judge releases an answer only if citation validity, authority compatibility, unresolved-conflict threshold, and privacy-policy checks pass. Otherwise it abstains with failed-check annotations. Evaluation therefore reports both quality and coverage; a system cannot improve hallucination rate merely by abstaining on every case.

## 3. Claim-to-Hypothesis Map

| Claim | Primary hypothesis | Primary endpoint | Minimum evidence |
|---|---|---|---|
| FLEN improves heterogeneous FL | H1: FLEN exceeds FedAvg/FedLoRA on worst-target utility | worst-target accuracy or macro-F1 | five paired seeds, same budget, corrected test |
| Conflict routing is legally meaningful | H2: scores identify unsupported/authority-local cases | macro AUPRC, ECE, coverage | held-out expert conflict test |
| FLEN reduces negative transfer | H3: fewer targets degrade versus local-only | negative-transfer rate | authentic metadata or audited common-task probes |
| Local residual protects minority clients | H4: residual improves worst-client utility without material global loss | worst-client utility and global non-inferiority | shared-only and personalization ablations |
| Privacy controls reduce leakage | H5: DP lowers attack success at bounded utility cost | attack AUC, epsilon, utility delta | actual accountant and attacks |
| Typed deliberation improves reliability | H6: full gate improves citation/hallucination risk--coverage | citation F1, hallucination, selective risk | frozen predictions and blinded review |

## 4. Experiment Program

### E0. Formula-Code Contract Tests

Implement the framework-independent aggregator before Flower integration. Unit tests cover all M3 invariants, missing conflict components, empty cohorts, client dropout, deterministic ordering, and local-residual isolation. Add a synthetic two-client and majority/minority counterexample proving that unconditional global downweighting is not used.

Deliverables:

- `src/fedlegal/aggregation/target_conditioned.py`
- `tests/test_target_conditioned_aggregation.py`
- `outputs/validation/formula_code_contract.json`

### E1. Same-Task Federated Baselines

Purpose: establish optimization effectiveness without claiming cross-jurisdiction law transfer.

Primary datasets (subject to the metadata gate below):

- CaseHOLD for five-way legal holding selection;
- ECtHR-A/B with authentic respondent-state metadata under a common ECHR authority space. The
  currently downloaded LexGLUE export does not contain this field, so it is blocked until a
  verified source version is acquired; no country holdout may be run on the current files;
- one additional same-task benchmark after adapter and metric validation.

Methods:

- local-only;
- centralized upper reference where licensing permits;
- FedAvg;
- FedProx;
- SCAFFOLD or FedNova;
- FedLoRA;
- clustered FL;
- Ditto-style personalization;
- target-cohort FLEN with `lambda=0`;
- full FLEN.

All methods use the same backbone, tokenizer, LoRA modules, split, client participation, local examples, local steps, optimizer, sequence length, and evaluation code.

Schedule:

- development: Qwen2.5-0.5B, 20 rounds, seeds 41--43;
- primary: Qwen2.5-1.5B, 50 rounds, seeds 41--45;
- confirmation: FedAvg, FedLoRA, and FLEN at 100 rounds;
- capacity check: one 7B backbone on three seeds after the protocol is frozen.

Primary endpoint: worst-client or worst-target standard task metric. Secondary endpoints: macro client utility, global utility, negative-transfer rate, drift, convergence stability, wall time, and payload.

### E2. Cross-Country, Cross-Language, and Cross-Jurisdiction Evaluation

These tracks are reported separately.

Track A uses ECtHR respondent-state holdout only when an explicit, verified respondent-state
field is present and passes the metadata audit. It tests generalization across country-specific
case distributions under a common Convention authority, not independent national law. Until
that condition is met, ECtHR is reported only as a common-authority multilabel benchmark.

Track B uses MultiEURLEX leave-one-language-out. It tests multilingual transfer and must not be described as independent-jurisdiction transfer.

Track C uses an expert-authored or expert-audited common-task probe benchmark. Identical fact patterns are instantiated with target-jurisdiction authority scopes and a shared outcome ontology. This is the only track that may support the strong cross-jurisdiction rule-transfer claim.

For each track, report in-target, held-out-target, worst-target, negative-transfer rate, and the train-target/test-target matrix. Do not average raw scores across different tasks.

### E3. Conflict Construct Validation

Evaluate each component and the combined router on `conflict_test` without model-utility tuning. Report macro AUROC/AUPRC, per-class recall, ECE, Brier score, selective risk, coverage, and failure cases. Compare against lexical similarity, embedding cosine, majority agreement, and a generic NLI baseline.

The aggregation experiment proceeds only after the router reaches a pre-specified reliability floor or is explicitly labelled experimental. A suitable initial gate is macro AUPRC above the class-prior baseline by at least 0.15 and ECE below 0.10; the final threshold should be reviewed with annotators.

### E4. Mechanism Ablation

Run the full model and the following paired variants on the same seeds:

- no comparability gate;
- no target cohort;
- `lambda=0`;
- no diversity floor;
- no weight smoothing;
- shared adapter only;
- no jurisdiction descriptor;
- remove each conflict component separately;
- replace calibrated router with majority disagreement.

Use CaseHOLD/ECtHR for optimization ablations and the expert probe benchmark for legal-routing ablations. The core causal evidence is the delta from full FLEN with paired seed-level confidence intervals, not whether every internal conflict score decreases.

### E5. Privacy and Attack Evaluation

Use sample-level DP-SGD for record protection and report RDP or PRV accounting. Recommended noise multipliers are selected on a validation grid and reported with their realized epsilon rather than interpreted directly. Secure aggregation is toggled independently.

Attacks:

- black-box membership inference on held-out members/non-members;
- adapter/update reconstruction similarity;
- canary or synthetic PII extraction;
- attribute or text-recovery probes against disclosed embeddings/summaries.

Report utility, attack AUC, true/false positive rates, extraction success, epsilon/delta, per-round bytes, protocol overhead, and runtime. Never place recovered sensitive text in the paper artifact.

### E6. Multi-Agent Legal Reliability

Compare:

- single-agent answer;
- ungated multi-agent deliberation;
- citation-verifier only;
- verifier plus conflict gate;
- full verifier/conflict/privacy release gate.

Use frozen predictions and blind method identifiers. Sample at least 200 outputs per primary comparison where feasible, with two legally trained annotators, adjudication, and inter-annotator agreement. Report task correctness, citation precision/recall/F1, legal soundness, unsupported fact rate, wrong-authority rate, hallucination rate, abstention coverage, and selective risk.

### E7. Robustness and Failure Analysis

Stress client counts 4/8/16, Dirichlet alpha 0.1/0.3/1.0, 20% client dropout, one stale client, one manipulated summary client, incomplete authority mappings, and class imbalance. Use 32 clients only if resources permit. Report worst-case degradation and failed-run rate rather than only the best setting.

Include four to six traceable case studies: successful compatible transfer, authority-local retention, unsupported citation rejection, unresolved abstention, minority-client protection, and at least two failures.

## 5. Statistical Protocol

- Use seeds 41--45 for all paired primary comparisons.
- Fix the split, partition, probe, calibration, model, tokenizer, and code version before primary runs.
- Report each seed, mean, standard deviation, and 95% Student-t interval.
- Use paired permutation tests on per-example predictions where the task permits.
- Report effect sizes and Holm-adjusted p-values for the declared comparison family.
- Separate run variability from example-sampling uncertainty.
- Report worst-target and negative-transfer metrics even if macro utility improves.
- Count crashed or invalid runs and apply the pre-specified rerun policy; do not silently replace weak seeds.

Suggested decision criteria for the main claim:

1. FLEN improves worst-target utility over FedAvg and FedLoRA by at least 2 percentage points or a task-appropriate standardized effect.
2. The adjusted paired test supports the direction of improvement.
3. Macro utility is non-inferior within a 1-point margin.
4. Negative-transfer rate and minimum client weight do not worsen.
5. The result repeats on at least two same-task settings and the expert cross-jurisdiction probe track.

These criteria are targets, not guaranteed outcomes. If any condition fails, the conclusion must identify which benefit was not established.

## 6. Implementation Sequence

### Phase 1: Core Method

1. Add typed routing, cohort, conflict-calibration, and local-residual schemas.
2. Implement framework-independent target-conditioned weights.
3. Add formula-code and minority-client regression tests.
4. Integrate a Flower strategy with complete per-round logs.
5. Implement separate shared/local LoRA checkpoints.

### Phase 2: Minimum Evidence

1. Run 20-round CaseHOLD screening for FedAvg, FedProx, FedLoRA, target-cohort `lambda=0`, and FLEN.
2. Fix only implementation errors using development seeds.
3. Freeze the primary protocol.
4. Run 50-round five-seed CaseHOLD and ECtHR experiments.
5. Generate tables and figures directly from manifests.

### Phase 3: Core Novelty Evidence

1. Complete expert conflict annotation and calibration.
2. Run construct validation and routing ablations.
3. Run authentic held-out-target evaluation.
4. Run local-residual, diversity-floor, and comparability-gate ablations.

### Phase 4: Trustworthiness Evidence

1. Implement DP accountant and secure aggregation.
2. Run privacy attacks and overhead measurement.
3. Run typed multi-agent reliability evaluation and blinded review.
4. Add robustness, adversarial client, and failure-case analyses.

## 7. Artifact Contract

Every run directory must contain:

- resolved YAML/JSON configuration;
- exact command and environment;
- source-tree or Git revision;
- dataset, split, partition, probe, model, and tokenizer versions;
- seed and hardware allocation;
- round-level train/evaluation/weight/conflict/privacy logs;
- per-example predictions with stable identifiers;
- communication bytes separated by tensor, serialization, transport, secure aggregation, and summaries;
- final metrics and completion status;
- provenance manifest linking every figure/table row to raw runs.

Synthetic, smoke, partial, failed, and final runs must remain distinguishable in both directory names and manifests.

## 8. Claim-Evidence Release Gates

| Manuscript phrase | Release gate |
|---|---|
| "FLEN improves heterogeneous federated legal modelling" | E1 primary criteria pass on at least two tasks |
| "conflict-aware aggregation" as an effective mechanism | E3 construct validity and E4 paired ablation pass |
| "cross-jurisdiction generalization" | E2 Track C passes; Tracks A/B alone are insufficient |
| "privacy-preserving" | formal accountant, secure runtime, and E5 attacks are reported |
| "multi-agent reasoning improves reliability" | E6 blinded evaluation and risk--coverage improve |
| "communication efficient" | payload and end-to-end protocol cost beat a declared baseline |

## 9. Five-Dimension Self-Review

- Contribution: the novel object is authority-scoped transfer eligibility, not generic disagreement weighting.
- Writing clarity: every method module now specifies input, execution, output, motivation, and measurable failure modes.
- Experimental strength: strong optimization, personalization, clustering, privacy, and agent baselines are required under equal budgets.
- Evaluation completeness: effectiveness, causality, construct validity, transfer, privacy, reliability, and failure cases are separated.
- Method soundness: majority suppression, missing conflict evidence, feedback loops, privacy leakage, and task/jurisdiction confounding have explicit controls.
