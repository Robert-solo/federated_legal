# Abstract revision evidence, 2026-11-02

The maintained CAS abstract uses four paragraphs: recent progress and task,
three limitations, three corresponding method components, and measured outcomes.
The second paragraph starts with “However”; the third starts with
“To address these limitations”. The three components are authority comparability
and target-conditioned weighting, independent shared/private branches with gating,
and typed analysis/challenge/verification/judgement.

The numerical claims use completed frozen experiments, not the earlier partial
snapshot or legacy factor-residual experiments. Remote analyses were rerun on
2026-11-02: CaseHOLD 40 original runs plus five FedAdam runs, MultiEURLEX 120 runs.
The analysis records used for this edit are saved in
`papers/ipm/results/revision_v2/abstract_revision_20261102/`.

| Abstract claim | Evidence | Interpretation |
| --- | --- | --- |
| Gate gain 2.50 points at 8+2 | guarded minus unguarded; five matched seeds; 95% paired t interval [1.63, 3.37] points | Positive observed mean gain; exact two-sided sign-flip p=.0625 and Holm p=.4375; no significance claim. |
| Gate gain 2.64 points at 10+2 | guarded minus unguarded; five matched seeds; interval [1.60, 3.68] points | Same-budget gate comparison, independent additive branches. |
| Proxy-FLEN micro F1 .336 | .336220, five seed means over four target holdouts | FedAvg .337315; paired difference -.001095, interval [-.006449, .004259]. |
| Frozen-backbone reference .310 | .310465, trained classification head | Backbone is frozen; the reference is supervised at the head. |

The abstract reports benefits of gating and descriptive closeness to FedAvg.
It makes no claim of overall FLEN superiority, statistical equivalence, validated
expert conflict labels, improved human reasoning quality, or inference privacy.
FedAdam and Ditto apply to CaseHOLD; unadapted/local-only references apply to
MultiEURLEX. The wording “FLEN components” preserves this mapping.

The abstract's method description states the proposed design. Expert annotation
remains necessary for validating its authority-scoped legal construct. The main
results text still distinguishes legacy studies from the revised protocol; a
separate full-results integration must update the body and conclusion using the
completed analyses before submission.

Self-review: the paragraph order is explicit, the three challenges map to the
three method steps, terms match the method section, and each numeric result has
a completed source analysis. The experimental paragraph avoids attributing a
FedAvg adaptation gain to FLEN and avoids pooling languages as independent seeds.
