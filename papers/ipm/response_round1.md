# Response to Reviewer Comments: Round 1

**Manuscript:** *Federated Legal Large Language Models for Privacy-Preserving Cross-Jurisdiction Judicial Reasoning*  
**Decision type:** Major Revision / Weak Reject  
**Package readiness:** `needs_author_input`

## Response Strategy Summary

The revised manuscript now treats the contribution as a formally specified framework with pilot-output alignment rather than a completed empirical comparison. Mathematical definitions, the aggregation algorithm, multi-agent protocol, legal-theoretical motivation, a novelty comparison table, and an explicit full-validation protocol have been added. Real benchmark results, statistical outputs, final hardware details, and verified privacy-attack results remain required before submission.

## Comment-Response Tracker

| ID | Reviewer concern | Revision action | Location in revised manuscript | Status |
|---|---|---|---|---|
| R1.1 | Conflict functions are conceptual rather than operational. | Defined citation, reasoning, verdict, and rule-alignment distances; specified weighted conflict score and client-exposure weights. | Methodology: *Operational legal conflict functions* and *Conflict-aware aggregation* | Addressed in text |
| R1.2 | Aggregation objective and pseudocode are insufficiently specified. | Replaced ambiguous correction rule by computable conflict-weighted LoRA update; revised algorithm and method figure. | Methodology: conflict-weight equations, Algorithm 1, Fig. 2 | Addressed in text |
| R1.3 | Real experiments and statistical tests are missing. | Added full experimental protocol, metrics, baseline controls, seed/testing requirements, and explicitly limited current artifacts to diagnostics. | Experiments: revised empirical protocol; *Alignment with current pilot outputs* | `AUTHOR_INPUT_NEEDED`: execute experiments and insert results |
| R1.4 | Novelty relative to adjacent systems is unclear. | Added dedicated positioning subsection and comparison table. | Related Work: *Distinction from adjacent systems*, novelty table | Addressed in text |
| R1.5 | Multi-agent communication, memory, debate, and state flow are underdefined. | Defined typed messages, three memory scopes, directed workflow, verification/privacy gates, and deliberation pseudocode. | Methodology: *Federated multi-agent legal reasoning*, Algorithm 2 | Addressed in text |
| R1.6 | Legal-theoretical grounding is insufficient. | Added motivation grounded in legal pluralism, comparative authority transfer, precedent hierarchy, and judicial discretion. | Introduction: *Legal-theoretical motivation* | Addressed in text; citations may be extended |
| R1.7 | Experimental implementation details are missing. | Added LoRA, privacy, partition, round, logging, and inference-test protocol fields; hardware is explicitly pending. | Experiments: revision protocol table | `AUTHOR_INPUT_NEEDED`: hardware/runtime and final configs |
| R1.8 | Metrics and privacy attacks require precise definition. | Defined legal accuracy, citation consistency, coherence, hallucination, communication, drift, stability, and attack categories/reporting. | Experiments: *Baselines, metrics, and ablations* | Addressed in protocol; measured values pending |
| R1.9 | Figures are placeholders and need substantive presentation. | Added publication-style template panels tied to replaceable CSV sources and explicit placeholder marking. | Experiments: *Planned result-reporting figures*; figure-template script | Layout addressed; real figures pending |
| R1.10 | Claims must match evidentiary status. | Revised Abstract, Introduction, Discussion, and Conclusion to avoid empirical-superiority claims prior to completed validation. | Abstract; Introduction; Discussion; Conclusion | Addressed in text |

## Draft Point-by-Point Response

### R1.1-R1.2 Mathematical formalization and algorithm specification

We agree that the original manuscript did not operationalize legal conflict sufficiently for implementation or evaluation. We have revised the Methodology section to define a sanitized cross-client reasoning tuple and explicit measures for citation conflict, reasoning conflict, verdict conflict, and rule-alignment distance. We further define the combined score with normalized component weights and use it to compute each client's conflict exposure. The aggregation rule is now a directly computable conflict-weighted LoRA update, and Algorithm 1 and Figure 2 have been revised accordingly.

### R1.3 Real experimental validation and significance testing

We agree that dry-run or placeholder outputs cannot support comparative performance claims. The revised manuscript now labels all current artifacts as pipeline diagnostics only and removes any implication of empirical superiority. It specifies the required full validation protocol over CAIL, CaseHOLD, ECtHR, and CUAD, controlled centralized/FedAvg/FedProx/conflict-aware comparisons, at least 100 federated rounds, multiple seeds, and paired inference procedures. Completion of these runs and replacement of placeholder figures and tables remain necessary before submission.

### R1.4 Novelty relative to existing work

We have added a dedicated comparison subsection and table that distinguish the proposed framework from federated optimizers, legal language models, and role-based legal-agent workflows. The revision identifies the specific joint contribution: private cross-jurisdiction training, operational legal-conflict-aware aggregation, and verified multi-agent judicial reasoning in one framework.

### R1.5 Multi-agent design

We have formalized the multi-agent design as a directed state graph with typed messages, case/jurisdiction/citation memory scopes, prosecutor-defence deliberation, citation verification, conflict detection, privacy auditing, and judge-side output gating. Algorithm 2 specifies the deliberation protocol and its admissible disclosures.

### R1.6 Legal-theoretical motivation

We have added a legal-theoretical motivation subsection explaining why cross-jurisdiction heterogeneity can reflect legitimate plural legal orders rather than optimization noise. The revision explicitly connects rule transfer to authority hierarchy, comparative-law alignment, and judicial discretion.

### R1.7-R1.8 Experimental and evaluation details

We have added a detailed protocol table and metric definitions covering the backbone and LoRA plan, non-IID design, communication schedule, privacy controls, communication logging, legal reliability measures, client drift, aggregation stability, and privacy attack categories. Final hardware/runtime information and measured privacy-attack outcomes will be added when the full experiments have been executed.

### R1.9 Figures and result presentation

We have prepared result-reporting figure templates for main legal performance, convergence and client drift, operational conflict diagnostics, held-out jurisdiction transfer, and privacy-utility trade-offs. All currently synthetic values are visibly marked as placeholders and backed by replaceable CSV files; they will not be used as submitted empirical evidence.

### R1.10 Claim positioning

We have revised the central positioning throughout the manuscript. The paper is now presented as a formally operationalized framework with verified pipeline-output alignment, while full empirical validation is explicitly listed as the outstanding requirement for a publishable comparative contribution.

## Outstanding Author Inputs Before Submission

- Run the planned multi-dataset experiments using the finalized configuration and populate all result CSVs/tables with real values.
- Supply GPU model/count/memory, runtime, optimizer/batch-size decisions, software versions, and final privacy configuration.
- Supply multi-seed means/standard deviations and the selected paired significance procedure and outputs.
- Supply privacy-attack results and details of any expert/human evaluation used for reasoning coherence.
- Check and extend comparative-law/legal-theory citations where required by the final journal version.

## 中文核对

- 当前修订已经解决方法形式化、创新定位、多智能体协议、法律理论动机和实验报告结构问题。
- 当前图表中的比较数值仍是明确标注的占位数据，不能作为投稿结果。
- 投稿前必须补入四类基准数据集上的真实运行结果、统计检验、硬件与隐私攻击评价信息。
