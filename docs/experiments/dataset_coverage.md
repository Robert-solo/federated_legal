# Dataset Coverage for the Revised IPM Experiments

## Verified Local Holdings

The remote repository contains the original public collection under `data/raw/public` (about 11 GB) and the reviewer-requested extension under `data/raw/public_extended` (about 3.9 GB normalized JSONL). The extension download completed as Slurm job `90750` in 22 minutes 8 seconds with exit code `0:0`.

| Dataset group | License status | Primary experimental role | Readiness |
|---|---|---|---|
| LexGLUE CaseHOLD | CC-BY-4.0 | Flower/PEFT baselines and communication diagnostics | ready |
| LexGLUE SCOTUS, ECtHR-A/B, EURLEX, LEDGAR, UNFAIR-ToS | CC-BY-4.0 | multi-task legal accuracy and domain heterogeneity | ready; downloaded ECtHR lacks respondent-state metadata |
| CUAD / CUAD-QA | CC-BY-4.0 | contract extraction, clause reliability, hallucination cases | ready |
| CAIL2018 | unknown in Hub metadata | Chinese criminal judgment prediction | downloaded but quarantined from training claims pending license review |
| LeCaRDv2 queries/corpus/qrels | MIT | Chinese legal retrieval and citation/relevance consistency | ready |
| LegalBench citation prediction | CC-BY-4.0 | citation prediction and canonicalization stress tests | ready |
| LegalBench UCC v. common law and overruling | CC-BY-4.0 | authority-scope and conflict-gate stress tests | ready |
| LegalBench Canada tax outcomes | CC-BY-4.0 | Canadian judgment-outcome stress test | ready |
| LegalBench personal jurisdiction and rule QA | CC-BY-4.0 | reasoning, abstention, and rule-selection evaluation | ready |
| MultiEURLEX en-de/en-fr/en-es/en-pl | CC-BY-SA-4.0 upstream | same-label cross-language federated transfer | ready |

## Claim-to-Data Readiness

| Required experiment | Data status | Remaining non-data work |
|---|---|---|
| FedAvg/FedProx/SCAFFOLD/FedNova/FedLoRA comparison | ready on CaseHOLD and supported LexGLUE tasks | implement identical-budget runtime strategies and five-seed launcher |
| Multi-dataset legal accuracy | ready, except CAIL license limitation | implement task-specific heads/metrics and avoid averaging heterogeneous accuracies |
| Citation consistency and retrieval | ready for automatic LeCaRDv2/LegalBench diagnostics | add citation canonicalization and an expert-verified citation subset |
| Cross-language transfer | ready on four MultiEURLEX language pairs with CELEX identifiers | implement language-conditioned clients and held-out-language evaluation; report as multilingual EU-law transfer |
| ECtHR respondent-state holdout | blocked in the downloaded LexGLUE export | acquire and verify an explicit respondent-state field; never infer it from case text |
| Cross-jurisdiction transfer | partially ready across US, Canada, China, and EU sources | tasks and label spaces differ; create an audited common-task mapping or expert-authored comparable probe set |
| Conflict-component construct validation | public source cases are available | legally trained annotators must label transferable, authority-incompatible, unsupported, and unresolved pairs |
| Multi-agent coherence and hallucination | prompts and source cases are available | blinded double annotation, adjudication, and agreement reporting remain mandatory |
| Differential privacy and leakage attacks | existing train/test records are sufficient | implement accountant, membership inference, reconstruction, and embedding-leakage protocols |
| Secure aggregation overhead | no additional dataset required | implement protocol-compatible weighting and measure payload/latency overhead |

## Reproducibility Artifacts

- Download command: `sbatch scripts/slurm_download_review_datasets.sh`
- Typed preprocessing config: `configs/datasets/public_legal_extension.yaml`
- Remote provenance manifest: `data/raw/public_extended/manifest.json`
- Local copied manifest: `outputs/data_validation/public_extended_manifest.json`
- Adapter validation report: `outputs/data_validation/public_extended_validation.json`
- Bounded pipeline preflight: `outputs/data_preflight/public_extended_institution`

The extension manifest contains 14 dataset entries, 28 splits, 4,146,613,993 bytes, source revisions, schemas, license metadata, and declared evaluation roles. All 13 configured preprocessing sources passed adapter validation.

## Authentic-Metadata Gate

Run `scripts/validate_authentic_metadata.py` before any country, court, or jurisdiction holdout. The gate fails when the declared field is missing, under-populated, or associated with duplicate record identifiers. A configured dataset-level jurisdiction, filename, class label, or text-derived guess does not satisfy this gate. The current remote ECtHR audit is blocked because no respondent-state field is present. MultiEURLEX passes only a language-level interpretation: its language metadata is explicit, while all records remain within a common EU regulatory authority scope.
