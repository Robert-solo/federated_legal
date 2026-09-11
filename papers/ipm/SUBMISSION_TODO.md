# IPM Submission Research Roadmap

This roadmap tracks the evidence that can materially change the paper's conclusions. It is organized by research question rather than by a long acceptance checklist.

## Current Evidence

The completed CaseHOLD study contains 24 matched twenty-round runs across eight methods and seeds 41--43. It establishes that the Flower/PEFT implementation executes, conflict weighting changes client contributions, unguarded local residuals can cause substantial degradation, and training-only selection avoids that collapse. It does not yet establish a statistically reliable accuracy gain or authentic cross-jurisdiction transfer.

## Priority 1: Effectiveness

Extend the matched comparison to seeds 44--45 and add FedLoRA, SCAFFOLD or FedNova, local-only adaptation, and a strong personalization baseline under the same data and compute setting. The main analysis should report aggregate and worst-client accuracy, paired effects, uncertainty intervals, negative transfer, runtime, and communication. This evidence determines whether FLEN offers a practical gain beyond its demonstrated failure-mode mitigation.

## Priority 2: Legal Conflict Validity

Create an expert-reviewed set of comparable legal problems with target-jurisdiction and authority metadata. Reviewers should use a concise annotation guide and provide a holistic transferability judgement with short rationales; factual support, authority compatibility, and unresolved ambiguity serve as explanatory dimensions rather than independent pass/fail boxes. Compare FLEN's conflict scores and routing decisions with these judgements, report agreement and calibration, and analyze representative disagreements.

## Priority 3: Cross-Jurisdiction Generalization

Evaluate held-out targets only where the task and label meaning remain comparable. Report source-to-target and personalized results per jurisdiction, with special attention to worst-target utility and negative transfer. Cross-language MultiEURLEX experiments should remain separate from legal-jurisdiction claims.

## Priority 4: Privacy and Reasoning

Treat privacy and multi-agent reasoning as separate empirical questions. For privacy, measure utility and communication under differential privacy and secure aggregation, then add one membership-inference and one update-reconstruction study. For multi-agent reasoning, conduct blinded legal review of single-agent and FLEN deliberation outputs using the concise annotation guide described above. These experiments are included only when they are implemented and measured.

## Paper Update Rule

Each central statement in the Abstract, Introduction, and Conclusion should point to a reported result. Unsupported benefits are either tested in the priorities above or presented as future work. Tables and figures are regenerated from the run-level metrics used in analysis.
