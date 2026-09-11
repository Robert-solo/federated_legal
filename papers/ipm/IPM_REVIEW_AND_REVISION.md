# IPM 审稿报告与修订记录

## 总体结论

**审稿建议：Reject，完成核心实现与真实实验后重新投稿。**

论文选题重要，契合 Information Processing & Management 对法律信息处理、可信人工智能、隐私计算和跨域信息系统的关注。核心洞见——法律异质性可能代表合法的规范差异，而不只是统计噪声——具有研究价值。然而，审查前版本仍是“框架设计 + 单次工程试跑”，并将固定随机种子生成的合成图表按已完成主实验呈现。该 claim–evidence 冲突属于投稿前必须消除的拒稿级风险。

## 评分

| 维度 | 审查前 | 本轮文字修订后 | 说明 |
|---|---:|---:|---|
| 问题重要性 | 8/10 | 8/10 | 法律数据孤岛与跨法域冲突具有现实意义 |
| IPM 契合度 | 8/10 | 8/10 | 属于法律信息处理与可信协同学习交叉方向 |
| 创新潜力 | 5/10 | 6/10 | target-conditioned conflict routing 更清楚，但尚未验证 |
| 方法严谨性 | 2/10 | 5/10 | 已修正多数中心性降权、可比性与隐私边界 |
| 实验充分性 | 1/10 | 1/10 | 仍只有单次 CaseHOLD smoke test |
| Claim–evidence 对齐 | 1/10 | 7/10 | 合成主结果已从论文移除，证据边界已写明 |
| 可复现性 | 2/10 | 4/10 | 协议已明确，完整 provenance 仍待真实运行 |
| 写作与组织 | 6/10 | 7/10 | 摘要、贡献、实验、讨论和结论已重写 |
| 隐私与治理 | 2/10 | 5/10 | 已加入 threat model 边界和伦理治理章节 |
| **综合评分** | **3.0/10** | **4.9/10** | **仍不具备投稿条件** |

审稿信心：5/5。

## 主要审稿意见

### 1. 合成结果被当作真实主实验

`scripts/generate_topconf_figure_templates.py` 明确用固定随机种子生成 benchmark、conflict、transfer、ablation、privacy 和 robustness CSV；`TOPCONF_FIGURES.md` 也将其定义为 synthetic layout data。审查前 PDF 却隐藏了 synthetic footer，并在摘要、讨论、结论和数据声明中将图 4–11 描述为五随机种子受控实验。

**本轮处理：** 已从 `main.tex` 删除全部合成主结果图和结果性陈述，并在 Evaluation Status 中明确其仅为布局模板。

### 2. 原聚合规则会压制合法少数法域

原始权重 `p_k exp(-lambda delta_k)` 将“与多数不一致”等同于“不可靠”。两个客户端时，对称冲突项归一化后退化为 FedAvg；两个多数客户端加一个少数客户端时，少数客户端会因合法差异被系统性降权。这与保留法律多样性的动机矛盾。

**本轮处理：** 方法改为 target-conditioned transfer cohort，新增 comparability gate、jurisdiction-local residual 和 diversity floor。不可比或权威范围不同的差异不再进入惩罚项。

### 3. 冲突指标缺乏构造效度

原 citation Jaccard 会把不同法域的正常法源差异判为最大冲突；空引用集合被记为零会奖励不引用；NLI 未对称化；JSD 未说明对数底；rule distance 未使用图边且不能在完全匹配时严格归零；共享 probe 的来源与数据隔离未定义。

**本轮处理：** 已加入独立 probe set、任务/标签/权威可比性条件、空值处理、双向 NLI、base-2 JSD、节点与边联合 rule distance，以及专家标注 conflict benchmark 的验证协议。

### 4. 隐私主张超过实现证据

联邦训练、adapter 交换、sanitization 和 `privacy_tag` 都不自动构成隐私保证。当前仓库没有 clipping、accountant、最终 epsilon/delta、secure aggregation runtime、summary protection 或真实攻击结果。标准 secure aggregation 还与服务器读取逐客户端摘要并计算权重存在协议冲突。

**本轮处理：** 标题和正文改用 data-local/privacy-aware 定位；新增 threat model 与 privacy boundary；说明 DP 和 secure aggregation 的必要参数、两阶段兼容协议及当前未实现状态。

### 5. 论文方法与代码实现不一致

`src/fedlegal/federated/strategy.py` 将 `conflict_aware` 映射为普通 FedAvg。`src/fedlegal/aggregation/aggregators.py` 对所有客户端乘同一个 shrinkage 后再归一化，使冲突惩罚完全抵消。原冲突表来自 placeholder adapter 和 `total_examples=0` 的 dry run。

**本轮处理：** 正文明确当前 pilot 只实现 FedAvg，并删除 dry-run 冲突表。下一步必须实现论文中的 target-conditioned 权重并增加公式—代码一致性测试。

### 6. 跨法域与跨任务变量混淆

CAIL、CaseHOLD、ECtHR、CUAD同时改变语言、任务、标签空间、法律体系和文体。直接平均 accuracy 无法解释为跨法域性能；contract law 是领域而不是法域；CaseHOLD 的 pseudo-jurisdiction 还与 gold label 相关。

**本轮处理：** 数据表分离 jurisdiction、language、domain 和 task；各数据集保留标准指标；禁止跨异质任务直接平均 raw accuracy；跨法域主实验要求同任务或经审计的输出映射和真实 jurisdiction metadata。

### 7. 多智能体流程尚未形成可复现方法

原稿未完整定义边、提示、阈值、错误传播和 Judge 聚合；Conflict Detector 不影响放行条件；prosecutor/defence 也不适用于合同、人权和多数民事任务。

**本轮处理：** 角色泛化为 argument/counterargument，并将 citation、authority、conflict threshold 和 privacy-policy 同时纳入 abstention gate。真实提示、模型版本、阈值和专家评估仍待实现与记录。

### 8. 真实 pilot 只能证明流程可运行

唯一真实实验为 Qwen2.5-0.5B、4 clients、1,200 train、240 validation、3 rounds、单 seed 的 FedAvg run。28.75% 没有 round-0、强基线或置信区间；49.50 MiB 只是 tensor bytes；0.5130 是相对广播 adapter 的 update norm，不是正文定义的 pairwise client drift。

**本轮处理：** 全部指标重新命名并限定解释，不再将该试跑作为 FLEN 有效性、隐私或跨法域泛化证据。

## 修订后论文主线

- **Opening：** 法律机构需要 data-local 协作，但任务差异、统计偏移与规范冲突不能混为一谈。
- **Challenge：** 单一全局平均和无条件冲突降权可能造成多数法域支配与错误规则迁移。
- **Method：** 用独立 probe 和 comparability gate 判断可转移关系，再做 target-conditioned aggregation 并保留 local residual。
- **Reasoning：** typed message、citation/authority/conflict/policy gates 和 abstention 约束推理输出。
- **Evidence：** 当前只有 CaseHOLD orchestration smoke test；完整受控实验是预注册式验证协议。
- **Boundary：** data locality 不等于 formal privacy；系统不用于自动裁判，必须保留人类责任与申诉路径。

## 段落角色检查

| 章节 | 段落角色 | 核心消息 |
|---|---|---|
| Abstract | challenge | 标准 FL 不能区分统计噪声与合法法律分歧 |
| Abstract | method | FLEN 结合可比性、冲突诊断、条件聚合和 typed deliberation |
| Abstract | evidence | 真实证据仅支持四客户端三轮 Flower/PEFT 可运行性 |
| Introduction | motivation | 跨法域协作同时受数据本地性与权威边界约束 |
| Introduction | contribution | 三项方法贡献与一项严格限定的工程贡献 |
| Method | safeguard | comparability gate 与 diversity floor 防止少数法域被自动压低 |
| Experiments | provenance | synthetic templates 与 measured run 明确分离 |
| Discussion | limitation | 当前尚未回答 RQ1–RQ3，也未实现正式隐私协议 |
| Ethics | governance | 人类决策责任、申诉、数据治理和少数法域公平不可省略 |
| Conclusion | takeaway | 论文目前是技术框架与可复现验证议程，不是完成的实证优越性主张 |

## Claim–Evidence 映射

| Claim | Evidence | 状态 |
|---|---|---|
| Flower + PEFT 联邦流程可以端到端运行 | CaseHOLD 真实日志、server rounds、communication log | supported，限 smoke test |
| 三轮内验证 accuracy 从 23.75% 到 28.75% | `server_eval_rounds.jsonl` | supported，描述性、不可归因 |
| 逻辑 tensor payload 为 49.50 MiB | `communication_log.jsonl` / run report | supported，不含协议开销 |
| target-conditioned conflict aggregation 优于 FedAvg | 无真实实现或对照 | needs experiment |
| 冲突指标对应真实法律冲突 | 无专家标注 construct benchmark | needs experiment |
| FLEN 改善跨法域泛化 | 无真实 held-out-jurisdiction run | needs experiment |
| 多智能体提高 citation/coherence 并降低 hallucination | 无真实 agent 输出与人评 | needs experiment |
| DP / secure aggregation 提供正式隐私 | 无 accountant、runtime 或攻击结果 | unsupported；已从结果主张移除 |
| 合成 topconf 图代表真实五种子结果 | 生成脚本证明为 synthetic | contradicted；已从稿件移除 |

## 五维自审

### Contribution

- Pass：研究问题重要，法律冲突与普通 non-IID 的区分具有潜在新意。
- Needs evidence：必须证明 conflict labels 与法律专家判断一致，并优于 generic personalization/clustering。

### Writing clarity

- Pass：FLEN 已展开；data locality、privacy、jurisdiction、domain、task 的术语边界更清楚。
- Needs revision：Figure 1–3 仍需最终按 target-conditioned 方法重新绘制并统一术语。

### Experimental strength

- Needs new experiment：当前没有同 backbone、同 split、同 budget 的强基线与五 seed 结果。
- Needs new experiment：缺少 minority-jurisdiction、恶意客户端、负迁移与失败案例。

### Evaluation completeness

- Needs new experiment：缺 privacy accountant、MIA/reconstruction/embedding leakage。
- Needs new experiment：缺双人法律标注、agreement、adjudication 和错误分类。

### Method soundness

- Improved：已修复无条件多数中心性降权的理论矛盾。
- Needs implementation：代码仍未实现 target-conditioned aggregation、local residual 或 diversity floor。
- Needs validation：authority mapping、probe provenance、恶意摘要和反馈循环仍需验证。

## 投稿前最低完成条件

1. 实现 target-conditioned Flower strategy，并用单元测试验证权重、local residual 和 diversity floor。
2. 使用同一 backbone、split、LoRA、local steps、rounds 和 seeds 比较强基线。
3. 完成专家标注 conflict benchmark，报告 agreement、calibration、AUROC/AUPRC 和失败案例。
4. 使用真实 jurisdiction metadata 与同任务输出空间完成 held-out-jurisdiction 评估。
5. 报告 per-jurisdiction、worst-jurisdiction、negative transfer 和权重公平性。
6. 实现并报告 DP accountant、secure aggregation runtime、隐私攻击和完整通信开销。
7. 实现真实多智能体推理，保存 prompts、模型版本、阈值、逐样本输出和盲法人评。
8. 每个结果必须绑定 provenance manifest；绘图脚本不得仅通过隐藏 footer 将模板标记为 final data。

完成以上条件后，再重新撰写真正的 Results 段落，按“发现—数值/CI—统计检验—机制解释—失败边界”报告，而不是用图注代替分析。
