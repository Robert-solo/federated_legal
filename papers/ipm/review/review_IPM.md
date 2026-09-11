# IPM 投稿级论文全面修改任务

## 0. 总体目标

请你对当前论文进行一次**面向 Information Processing & Management (IPM) 投稿的系统性重构与深度修改**。

目标不是简单润色英文，而是将论文从“一个有效的方法 + 实验结果”提升为：

> **一个具有清晰 Information Processing & Management 问题属性、明确理论/方法学贡献、严谨实验验证、充分可解释性与可复现性的研究工作。**

请严格以当前论文已有内容、实验结果和代码实现为依据，不允许虚构实验结果、数据、baseline、引用或研究结论。

修改时优先级：

1. **Research Problem / Research Gap**
2. **Scientific Contribution**
3. **Methodological Novelty**
4. **Experimental Validity**
5. **Ablation / Robustness / Generalization**
6. **Information Processing & Management 领域契合度**
7. **论文逻辑结构**
8. **英文表达**

不要把主要精力放在语言润色上而忽略上述核心问题。

---

# 1. 首要问题：重新定义论文的 Research Problem

目前论文最大的潜在问题不是英文，而是：

> 审稿人可能认为这是一篇“LLM + Multi-Agent + Legal Reasoning”的应用型方法论文，而没有充分体现 IPM 所关心的信息处理、信息检索、知识组织、信息融合、信息推理和决策支持问题。

因此请重新检查 Introduction。

必须明确回答：

### Q1. 当前法律智能系统究竟存在什么信息处理问题？

不要只写：

* LLM hallucination
* legal reasoning is difficult
* multi-hop reasoning is challenging
* existing LLMs lack reliability

这些太泛。

需要进一步定义具体的信息处理困难，例如：

* 多跳法律信息如何组织？
* 不同法律条文之间如何建立可执行关系？
* 法律事实、法律规则、判例依据之间如何进行信息对齐？
* 多个 agent 产生的异构 reasoning evidence 如何融合？
* 如何保证最终法律判断能够追溯到具体法律依据？
* 如何避免 LLM 在多步推理过程中传播错误信息？
* 如何进行 rule-level verification？
* 如何将非结构化法律文本转化为可验证的 reasoning structure？

最终形成一个明确的 research problem。

---

# 2. 强化 Research Gap，而不是简单罗列 Related Work

请重新组织 Related Work。

不要采用：

> LLM → Legal LLM → Multi-Agent → Our method

这种非常常见的结构。

建议重新围绕以下三个 gap 建立论文逻辑：

### Gap 1：Reasoning Gap

现有 Legal LLM 能够生成法律推理，但：

* 推理链可能不可验证；
* 多跳推理中错误容易累积；
* 最终 judgment 与中间法律依据之间缺乏结构化对应关系。

### Gap 2：Evidence / Information Integration Gap

现有方法通常把法律文本作为 context 输入模型，但是：

> 没有解决 heterogeneous legal evidence 如何被组织、关联、验证和融合的问题。

强调：

**retrieval ≠ information integration**

**generation ≠ evidence-grounded reasoning**

### Gap 3：Traceability / Verification Gap

现有方法可能能够提高 prediction accuracy，但是：

* 为什么得到这个结果？
* 哪条法律规定支持这个结论？
* 哪一步推理产生了错误？
* 不同 agent 是否存在冲突？
* 最终 decision 是否违反 statutory constraints？

这些问题没有得到系统解决。

最终将论文定位为：

> **evidence-grounded, authority-bounded and traceable legal information processing / reasoning**

而不是单纯的：

> multi-agent legal LLM.

---

# 3. 重新检查论文标题

标题必须体现：

1. Legal reasoning / decision support
2. Multi-agent / neuro-symbolic mechanism
3. Traceability / verification
4. Information processing aspect

避免标题堆砌过多 buzzwords。

尤其检查当前标题是否存在：

* Neuro-Symbolic
* Multi-Agent
* Traceable
* Executable
* Statutory
* Legal Judgment Prediction

等概念堆叠问题。

如果一个概念没有在方法中被严格定义和实验验证，就不要放在标题里。

请给出：

* 推荐主标题
* 2–3 个备选标题
* 每个标题的优势和潜在审稿风险

并最终选择一个最适合 IPM 的标题。

---

# 4. 重新定义 Contributions

当前贡献部分不要写成普通的：

1. propose a framework
2. conduct experiments
3. achieve better performance

这种贡献对于 IPM 来说不够强。

请重写成三个层次。

## Contribution 1 — Problem / Framework

提出一种针对法律多跳推理的：

> traceable and evidence-grounded information processing framework

明确说明解决什么 information processing problem。

## Contribution 2 — Methodological Contribution

重点解释：

* multi-agent reasoning 如何工作；
* symbolic constraints 如何工作；
* statutory rules 如何被执行；
* evidence 如何在 agent 之间流动；
* conflict 如何检测；
* reasoning 如何验证；
* final judgment 如何产生。

必须说明：

> **真正的新机制是什么。**

不要把“使用多个 LLM agent”本身作为创新点。

## Contribution 3 — Empirical / Analytical Contribution

不要只说：

> extensive experiments demonstrate superiority.

而应该强调：

* performance improvement；
* reasoning traceability；
* evidence consistency；
* conflict detection；
* robustness；
* ablation；
* generalization；
* efficiency / cost；
* failure analysis。

如果当前实验还不支持这些结论，请标记为：

> REQUIRED EXPERIMENT

不要虚构结果。

---

# 5. Method 部分必须进行结构化重写

建议将 Method 重构成：

## 3.1 Problem Formulation

明确：

* 输入是什么；
* 输出是什么；
* legal evidence 是什么；
* reasoning path 是什么；
* statutory constraint 是什么；
* final judgment 是什么。

给出数学形式化。

例如定义：

[
x = \text{case facts}
]

[
E = {e_1,e_2,\ldots,e_n}
]

[
R = {r_1,r_2,\ldots,r_m}
]

[
P = \text{reasoning path}
]

[
y = \text{final judgment}
]

然后定义：

[
P = f(x,E,R)
]

以及最终：

[
y = g(P,R,E)
]

具体形式必须根据论文真实方法调整，不要机械套公式。

---

# 6. 明确定义 Neuro-Symbolic 到底是什么

这是当前论文最容易被审稿人质疑的地方之一。

如果论文使用了“Neuro-Symbolic”这个术语，必须回答：

### Neural component 是什么？

例如：

* LLM reasoning
* evidence extraction
* semantic matching
* legal argument generation

### Symbolic component 是什么？

例如：

* statutory rules
* logical constraints
* rule execution
* consistency checking
* structured legal graph
* deterministic verification

### 两者如何交互？

必须给出明确的数据流：

LLM → structured representation → symbolic verification → feedback → LLM reasoning → final decision

如果目前系统实际上只是：

> LLM + prompt + rule checking

则不要过度宣称 Neuro-Symbolic。

可以考虑降低术语强度，避免审稿人认为存在概念包装。

---

# 7. Multi-Agent 部分不能只描述“多个 Agent”

必须回答：

> 为什么一定需要 multi-agent？

请加入明确的 agent specialization。

例如：

* Evidence Agent
* Statutory Agent
* Reasoning Agent
* Verification Agent
* Judge Agent

但必须以论文真实系统为准。

对于每个 agent 给出：

| Agent | Input | Output | Role | Constraint |
| ----- | ----- | ------ | ---- | ---------- |

重点说明：

> Multi-agent architecture 是否真正产生了功能分解？

而不是：

> 让多个 LLM 分别回答，然后 majority vote。

如果只是后者，创新性会明显下降。

---

# 8. 强化“Executable Statutory Control”的理论含义

如果论文继续使用：

> Executable Statutory Control

必须明确说明：

### Statute → Rule Representation → Rule Execution → Verification

法律条文如何转换成：

* predicates
* conditions
* constraints
* logical relations
* executable rules

然后说明：

> 哪一步由 LLM 完成？

> 哪一步是 deterministic / symbolic？

> 哪一步可以保证不依赖 LLM 的随机生成？

尤其要强调：

> symbolic control 不是简单 prompt instruction。

如果当前实现还没有真正达到“executable”，必须谨慎修改术语。

---

# 9. 增加一个完整的 Pipeline Figure

Figure 应该能够让审稿人不看正文也理解方法。

建议：

Case Facts
↓
Evidence Retrieval / Extraction
↓
Legal Knowledge Construction
↓
Multi-Agent Reasoning
↓
Statutory Rule Execution
↓
Conflict / Consistency Verification
↓
Reasoning Revision
↓
Traceable Judgment

图中必须明确：

* neural components
* symbolic components
* information flow
* feedback loop
* final output

不要画成普通的“几个 Agent + LLM API”。

---

# 10. 必须强化 Ablation Study

这是投稿 IPM 前非常重要的一项。

至少考虑：

### A1. Full Model

完整系统。

### A2. – Multi-Agent

单 Agent。

### A3. – Symbolic Control

去掉 statutory constraints。

### A4. – Verification

去掉 verification module。

### A5. – Evidence Traceability

去掉 evidence grounding。

### A6. – Feedback / Revision

去掉 iterative refinement。

如果当前代码允许，建议全部补齐。

最终形成：

| Model | Accuracy | Macro-F1 | Evidence Consistency | Reasoning Validity |
| ----- | -------: | -------: | -------------------: | -----------------: |

不要只比较 Accuracy。

---

# 11. 增加 Reasoning Quality Evaluation

如果论文强调：

> traceable reasoning

那么仅仅报告 LJP Accuracy 是不够的。

必须增加 reasoning-level evaluation。

至少考虑：

### Evidence Grounding

最终结论引用的 evidence 是否真正支持结论。

### Citation / Evidence Accuracy

引用法律条文是否正确。

### Logical Consistency

推理过程中是否存在前后矛盾。

### Rule Compliance

最终结论是否符合 statutory rules。

### Traceability

是否能够从最终 prediction 回溯：

Prediction
→ Reasoning step
→ Rule
→ Evidence
→ Case fact

建议建立一个新的指标：

> Traceability / Evidence-Grounded Reasoning Score

如果无法严格设计 quantitative metric，可以进行 expert evaluation，但必须说明：

* evaluator；
* criteria；
* sample size；
* inter-rater agreement；
* evaluation protocol。

---

# 12. 增加 Conflict Resolution 实验

如果系统声称能够处理多 agent reasoning conflict，必须证明。

构造或识别：

* agent disagreement；
* conflicting evidence；
* conflicting statutes；
* incorrect reasoning chain。

然后比较：

* majority voting；
* single-agent；
* proposed verification mechanism。

观察：

> proposed method 是否能够识别并纠正错误 reasoning。

这会比单纯 benchmark accuracy 更有说服力。

---

# 13. 增加 Error Analysis

至少分析：

### Failure Type 1

Retrieval error

### Failure Type 2

Evidence interpretation error

### Failure Type 3

Statutory mapping error

### Failure Type 4

Reasoning error

### Failure Type 5

Conflict resolution error

### Failure Type 6

Final judgment error

最好给出真实案例。

形成：

| Error Type | Frequency | Example | Cause | Solution |
| ---------- | --------: | ------- | ----- | -------- |

这会明显增强论文的研究深度。

---

# 14. 增加 Robustness Evaluation

至少考虑：

### Prompt robustness

不同 prompt 是否稳定。

### Model robustness

不同 backbone LLM 是否仍然有效。

例如：

* GPT 系列
* Qwen
* Llama
* DeepSeek

具体根据实际可获得模型进行。

### Evidence noise

加入 irrelevant / noisy evidence。

### Missing evidence

删除部分 evidence。

### Contradictory evidence

加入 conflicting evidence。

如果 proposed framework 的核心价值是 verification，那么这些实验尤其重要。

---

# 15. 增加 Efficiency / Cost Analysis

Multi-agent 方法最大的审稿人质疑之一：

> 性能提高是不是只是因为用了更多 LLM calls？

因此必须分析：

* number of LLM calls；
* tokens；
* latency；
* computational cost；
* accuracy gain per cost。

至少增加：

| Method | Accuracy | # Calls | Tokens | Latency | Cost |

最终回答：

> Does the proposed reasoning architecture justify its additional computational cost?

这是非常重要的。

---

# 16. Baseline 必须重新检查

不要只比较：

* traditional ML；
* GPT；
* one or two Legal LLMs。

至少形成几个层次：

### Traditional / Classical

### Single LLM

### Legal LLM

### Retrieval-Augmented LLM

### Multi-Agent LLM

### Neuro-Symbolic / Rule-Grounded method

### Proposed

如果某类 baseline 无法获得，必须解释原因。

所有 baseline 必须保证：

* 相同 test set；
* 相同 evaluation metric；
* 尽可能相同 information access；
* 相同 / 公平的 prompt setting。

---

# 17. 避免数据集泄漏问题

Legal benchmark + LLM 论文特别容易被质疑：

> benchmark data may have appeared in LLM pretraining.

因此需要在论文中讨论：

* benchmark contamination；
* model knowledge leakage；
* retrieval setting；
* whether external legal knowledge is allowed；
* whether test cases can be retrieved online；
* closed-book vs open-book setting。

如果当前实验使用了外部法律知识，需要明确：

> 哪些 information 可以访问？

> 哪些 information 不可以访问？

---

# 18. 增加 Reproducibility Section

IPM 审稿人可能关注：

* model version；
* temperature；
* max tokens；
* prompt；
* number of agents；
* number of iterations；
* retrieval corpus；
* hardware；
* random seed；
* evaluation scripts。

建议增加：

## Implementation Details

并在 appendix 提供：

* system prompts；
* agent prompts；
* rule templates；
* pseudocode；
* examples。

---

# 19. 增加完整 Algorithm

建议加入：

> Algorithm 1: Traceable Multi-Agent Legal Reasoning

伪代码至少包含：

1. input case；
2. extract evidence；
3. retrieve statutes；
4. initialize agents；
5. generate reasoning；
6. construct structured reasoning;
7. execute statutory constraints；
8. detect conflict；
9. revise reasoning；
10. produce final judgment；
11. output evidence trace。

Algorithm 必须与代码实际实现一致。

---

# 20. Related Work 需要减少“论文堆砌”

不要：

> Author A did...
> Author B did...
> Author C did...

应该形成 taxonomy：

### Legal Judgment Prediction

### Legal LLMs

### Retrieval-Augmented Legal Reasoning

### Multi-Agent Reasoning

### Neuro-Symbolic Reasoning

### Explainable / Traceable AI

然后每一类最后都回答：

> Existing methods solve X, but remain limited in Y.

最终自然引出本文。

---

# 21. Introduction 最后一段必须形成完整逻辑闭环

建议 Introduction 最后形成：

### Problem

法律多跳推理需要整合 heterogeneous evidence。

↓

### Limitation

Existing LLMs generate plausible reasoning but cannot reliably enforce legal constraints.

↓

### Gap

Existing multi-agent methods improve reasoning diversity but do not provide executable statutory verification and end-to-end traceability.

↓

### Solution

We propose ...

↓

### Mechanism

multi-agent + structured evidence + executable statutory control + verification.

↓

### Evidence

experiments demonstrate improvements in prediction, reasoning validity, traceability and robustness.

这才是完整的论文故事。

---

# 22. Abstract 必须重新写

Abstract 不要从：

> Large Language Models have shown great potential...

开始长篇铺垫。

建议四段逻辑：

### Background / Problem

一句话说明问题。

### Gap

现有方法缺什么。

### Method

提出什么。

### Results

具体提升多少。

最后一句：

> implications for reliable legal information processing / decision support.

结果部分必须使用真实数字。

---

# 23. Conclusion 不要重复 Abstract

Conclusion 必须包含：

1. What problem was solved?
2. What methodological insight was obtained?
3. What empirical evidence supports it?
4. What limitations remain?
5. Future research direction.

特别加入：

> Limitations

不要只写：

> In the future, we will explore more datasets.

应该讨论：

* LLM dependence；
* computational cost；
* jurisdiction specificity；
* statutory formalization difficulty；
* expert evaluation limitations；
* benchmark limitations。

这反而会提高可信度。

---

# 24. 必须增加 Limitations

建议单独设置：

## Limitations

讨论：

### L1. Jurisdiction limitation

不同司法辖区的法律规则不同。

### L2. Formalization limitation

自然语言法律条文并不总能被无损转换为 executable rules。

### L3. LLM dependence

neural reasoning 仍然可能产生错误。

### L4. Computational overhead

multi-agent + verification 增加推理成本。

### L5. Benchmark limitation

现有 legal benchmarks 无法完全衡量真实司法决策。

不要刻意隐藏这些问题。

---

# 25. 全文最重要的概念统一

全文统一以下 terminology。

不要在不同地方混用：

* legal reasoning
* statutory reasoning
* legal inference
* legal judgment prediction
* legal decision making

需要定义它们的关系。

同样统一：

* evidence
* legal evidence
* statutory evidence
* legal rules
* statutes
* authorities

以及：

* traceability
* explainability
* interpretability
* verification

特别注意：

> Traceability ≠ Explainability

如果论文真正实现的是 evidence-level provenance，应主要使用：

> traceability / provenance / evidence grounding

而不要泛化成 explainability。

---

# 26. IPM 领域定位必须贯穿全文

请检查全文是否出现足够明确的：

> information processing

但不要机械增加关键词。

应该把整个 framework 理解为：

**Legal Information Processing Pipeline**

Case Information
→ Evidence Extraction
→ Information Organization
→ Knowledge Alignment
→ Rule-based Processing
→ Reasoning
→ Verification
→ Decision Support

这样才能自然体现 IPM scope。

---

# 27. 最终需要建立一个“IPM审稿人视角”检查表

修改完成后，请模拟至少 3 类 reviewer：

## Reviewer A — Information Processing

问题：

> Is this really an information processing contribution?

## Reviewer B — AI / ML

问题：

> What is technically novel compared with existing LLM multi-agent systems?

## Reviewer C — Legal AI

问题：

> Does the system actually perform valid legal reasoning and provide trustworthy evidence?

分别列出：

* Major Concern
* Evidence in Paper
* Current Status
* Required Revision

---

# 28. 建立“投稿前硬性验收标准”

修改完成后，不允许直接认为论文完成。

请逐项检查：

### Research

* [ ] Research problem clearly defined
* [ ] Research gap convincing
* [ ] Contributions non-trivial
* [ ] Novelty clearly distinguished from existing multi-agent LLMs

### Method

* [ ] Problem formulation
* [ ] Formal definition
* [ ] Architecture
* [ ] Algorithm
* [ ] Neural component
* [ ] Symbolic component
* [ ] Interaction mechanism
* [ ] Verification mechanism

### Experiment

* [ ] Strong baselines
* [ ] Ablation
* [ ] Robustness
* [ ] Generalization
* [ ] Reasoning evaluation
* [ ] Evidence evaluation
* [ ] Conflict evaluation
* [ ] Error analysis
* [ ] Efficiency / cost analysis

### Reproducibility

* [ ] Model versions
* [ ] Prompts
* [ ] Hyperparameters
* [ ] Dataset details
* [ ] Experimental protocol
* [ ] Code / appendix details

### Writing

* [ ] Abstract
* [ ] Introduction
* [ ] Related Work
* [ ] Method
* [ ] Experiments
* [Discussion
* [ ] Limitations
* [ ] Conclusion

### IPM Positioning

* [ ] Information processing problem is explicit
* [ ] Information integration is explicit
* [ ] Evidence provenance is explicit
* [ ] Decision-support implication is explicit
* [ ] Contribution goes beyond generic LLM application

---

# 29. 修改执行顺序

请不要从第一段开始逐句润色。

严格按照以下顺序执行：

### Phase 1 — Research Story

重新确定：

> Problem → Gap → Research Question → Method → Contribution

### Phase 2 — Method

重新检查：

> Neuro-Symbolic + Multi-Agent + Statutory Control + Verification

### Phase 3 — Experiment

检查现有实验，并建立：

> Baseline + Ablation + Robustness + Reasoning Evaluation + Error Analysis + Cost

### Phase 4 — IPM Positioning

重新修改：

> Title + Abstract + Introduction + Related Work + Discussion + Conclusion

### Phase 5 — Language

最后才进行：

> Academic English polishing

---

# 30. 最重要的原则

整个修改过程中必须遵守：

## 原则一

**不要为了让论文看起来更强而虚构实验结果。**

缺实验就明确标记：

> [REQUIRED EXPERIMENT]

## 原则二

**不要为了 IPM 而机械加入“information processing”。**

必须让 information processing 成为论文真实的问题和贡献。

## 原则三

**不要把 Multi-Agent 本身当成核心创新。**

核心创新必须是：

> 如何组织、约束、验证和追溯法律信息与多跳推理。

## 原则四

**不要过度使用 Neuro-Symbolic。**

只有当论文真正存在 neural + symbolic 的结构化交互时才使用。

## 原则五

**所有新增 claim 都必须能够被实验、理论分析或引用支持。**

## 原则六

最终目标不是：

> “把论文写得更漂亮”

而是：

> **把论文从一个“LLM-based legal system”提升为一个“具有明确信息处理科学问题和方法学贡献的法律智能研究”。**

---

# 最终输出要求

完成修改后，请给我以下内容：

1. **Revised Title**
2. **Revised Abstract**
3. **Revised Introduction**
4. **Revised Contributions**
5. **Revised Method Structure**
6. **Required New Experiments**
7. **Revised Experiment Section**
8. **Limitations**
9. **Conclusion**
10. **完整修改清单**

同时建立一个表格：

| Issue | Severity | Current Problem | Proposed Modification | Need New Experiment? | Status |
| ----- | -------- | --------------- | --------------------- | -------------------- | ------ |

Severity 分为：

* CRITICAL
* MAJOR
* MINOR

重点优先解决 CRITICAL 和 MAJOR 问题。

最后，请以 **IPM Reviewer #1 / Reviewer #2 / Reviewer #3** 的身份重新审阅修改后的论文，并分别给出：

* Overall Assessment
* Major Concerns
* Minor Concerns
* Strengths
* Weaknesses
* Recommendation
* Estimated Acceptance Risk

不要为了“鼓励作者”而降低评价标准，要按照真正的 IPM 审稿标准进行压力测试。
