# IPM 论文实验 Codex 可执行任务卡

本文档把 IPM 投稿前需要补齐的实验整理为可交给 Codex 执行的任务卡。目标是用真实实验结果替换 `papers/ipm/tables/experiment_source_data_topconf/` 下的模板数据，并让每个主张都有可复现证据支撑。

## 0. 总交付目标

最终交付物：

- 真实实验原始日志：`outputs/`
- 每个实验的配置文件：`configs/experiments/` 或 `configs/experiments/generated/`
- 论文图表 CSV：`papers/ipm/tables/experiment_source_data_topconf/*.csv`
- 论文最终图：`papers/ipm/figures/experiments_topconf/`
- 实验复现实录：`outputs/reports/ipm_experiment_manifest.md`

统一验收标准：

- 所有核心结果至少覆盖 `seed={1,2,3,4,5}`，除非任务卡明确写明是 smoke test 或人工评审。
- 每条 CSV 结果必须能追溯到 `run_id`、配置文件、数据版本、模型版本、代码提交或文件快照。
- 不允许把 `generate_topconf_figure_templates.py` 生成的合成模板数值当成真实结果。
- 所有主表指标必须报告均值、95% CI，并在报告中给出与 `FedAvg`、`FedLoRA` 的配对检验或 bootstrap 检验。
- 若实验结果没有达到论文预期，保留结果并在报告中标明风险，不得手工改数。

最终渲染命令：

```bash
cd papers/ipm
python scripts/generate_topconf_figure_templates.py --use-existing-data --final-data
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

## 1. 数据下载与来源清单

### D0.1 已有脚本可直接接入的数据

输入数据：

- LexGLUE：Hugging Face `coastalcph/lex_glue`
- CUAD 合同语料：Hugging Face `theatticusproject/cuad`
- CUAD-QA：Hugging Face `theatticusproject/cuad-qa`
- CAIL2018：优先使用 Hugging Face `china-ai-law-challenge/cail2018`；若沿用当前脚本失败，需要把 `scripts/download_public_legal_datasets.py` 中的 `CAIL_SPEC.repo_id` 从 `cail2018` 修正为 `china-ai-law-challenge/cail2018`

下载命令：

```powershell
python scripts/download_public_legal_datasets.py --output-dir data/raw/public --tasks lexglue_case_hold lexglue_scotus lexglue_ecthr_a lexglue_ecthr_b lexglue_eurlex lexglue_ledgar lexglue_unfair_tos cuad_contracts cuad_qa --trust-remote-code

python scripts/download_public_legal_datasets.py --output-dir data/raw/public --include-cail --trust-remote-code

python scripts/process_datasets.py configs/datasets/public_legal_corpus.yaml --partition
python scripts/partition_datasets.py configs/datasets/public_legal_corpus_dirichlet.yaml
```

验收标准：

- `data/raw/public/manifest.json` 存在，并记录每个数据集的 `repo_id`、split 数量、schema、license。
- `data/processed/public_legal_corpus/` 和 `data/partitions/public_legal_corpus*/` 存在。
- 每个数据集至少包含 train/validation/test 或任务卡允许的替代 split。

### D0.2 需要人工确认或新增接入的数据

FEDLEGAL：

- 下载来源：`https://github.com/SMILELab-FL/FedLegal`
- 用途：中文法律 FL 基准、真实非 IID 对照、隐私任务参考。
- 下载方式：按仓库 README 的 Google Drive 数据链接下载，放入 `data/raw/external/fedlegal/`。
- Codex 任务：新增 `scripts/import_fedlegal.py`，转换为本仓库 JSONL schema。

FedPII：

- 下载来源：`https://github.com/SMILELab-FL/FedPII`
- 数据来源：Hugging Face `FedPII/FedPII-dataset` 中的 `fed_pii_v1.parquet`。
- 用途：E9 跨客户端 PII 抽取、membership inference、隐私-效用实验。
- 建议命令：

```powershell
huggingface-cli download FedPII/FedPII-dataset fed_pii_v1.parquet --repo-type dataset --local-dir data/raw/external/fedpii
```

LeCaRD / LeCaRDv2：

- LeCaRDv2 下载来源：`https://github.com/THUIR/LeCaRDv2`
- LeCaRD v1 下载来源：`https://github.com/myx666/LeCaRD`
- 用途：法律检索、跨法域/跨任务 case study、citation/retrieval error analysis。
- Codex 任务：新增 importer；不要把 retrieval 指标混入主分类指标，除非论文新增对应实验小节。

FedJudge：

- 代码来源：`https://github.com/yuelinan/FedJudge`
- 用途：方法和 baseline 参考；其 README 使用公开法律语料模拟法院、咨询、教培三个客户端。
- 注意：不作为本仓库主数据源，除非明确复现实验协议并记录数据许可。

暂不使用或需先核验：

- `LJPIV`、`ELAM` 当前仓库未接入，也未在本轮核验到稳定官方下载入口。若论文保留这些名字，先建独立任务卡核验来源、许可、schema，再决定是否纳入实验。

## 2. 论文图表 CSV Schema

Codex 汇总实验结果时必须保持以下列名。

`benchmark_results.csv`：

```text
seed,dataset,method,legal_accuracy,citation_f1,reasoning_coherence,hallucination_rate
```

`training_dynamics.csv`：

```text
seed,round,method,legal_accuracy,client_drift,cumulative_communication_mb
```

`conflict_diagnostics.csv`：

```text
seed,round,method,component,score
```

`cross_jurisdiction.csv`：

```text
seed,train_jurisdiction,test_jurisdiction,method,legal_accuracy
```

`ablation.csv`：

```text
seed,variant,legal_accuracy,citation_f1,hallucination_rate
```

`privacy_utility.csv`：

```text
seed,noise_multiplier,secure_aggregation,legal_accuracy,attack_auc,communication_mb_per_round
```

`robustness.csv`：

```text
seed,num_clients,dirichlet_alpha,method,legal_accuracy,total_communication_gb
```

## 3. 实验任务卡

### E1. 多数据集主性能实验

实验目标：

- 证明 `Full framework` 在 CAIL、CaseHOLD、ECtHR、CUAD 上相对 `FedAvg`、`FedLoRA`、`Conflict-aware` 有稳定提升。

输入：

- 数据：D0.1 中的 CAIL2018、LexGLUE CaseHOLD、LexGLUE ECtHR-A/B、CUAD/CUAD-QA。
- 配置：`configs/datasets/public_legal_corpus.yaml`、`configs/datasets/public_legal_corpus_dirichlet.yaml`。
- 方法：`Centralized`、`Local-only`、`FedAvg`、`FedProx`、`SCAFFOLD`、`FedNova`、`FedLoRA`、`Conflict-aware`、`Full framework`。

执行：

- 先运行数据下载和分区。
- 为每个 dataset、method、seed 生成独立配置文件。
- 若当前 `scripts/run_federated.py` 仍为 dry run，Codex 需要优先补齐真实训练或接入 `scripts/run_real_flower_peft.py` 的多数据集版本。

输出：

- 原始日志：`outputs/runs/E1_main_benchmark/{dataset}/{method}/seed_{seed}/`
- 汇总报告：`outputs/reports/E1_main_benchmark.md`
- 论文 CSV：`papers/ipm/tables/experiment_source_data_topconf/benchmark_results.csv`

指标：

- `legal_accuracy`
- `citation_f1`
- `reasoning_coherence`
- `hallucination_rate`

验收标准：

- 每个 dataset-method 至少 5 个 seed。
- 所有方法使用相同 split、相同最大样本数、相同评价脚本。
- `benchmark_results.csv` 行数不少于 `5 seeds * 4 datasets * 9 methods = 180`。
- 报告必须包含 `Full framework - FedAvg`、`Full framework - FedLoRA` 的均值差、95% CI 和显著性检验。

### E2. 法律可靠性实验

实验目标：

- 证明性能提升不是单纯 accuracy 提升，而是同时提高引用一致性、推理连贯性，并降低幻觉。

输入：

- E1 的 predictions。
- 引用抽取与规范化模块：`src/fedlegal/data/citations.py`。
- 人工或自动 verifier 协议：若无 gold citation，必须建立 verifier 标准。

执行：

```bash
python scripts/run_evaluation.py configs/experiments/evaluation_default.yaml --predictions outputs/runs/E1_main_benchmark/.../predictions.jsonl
```

输出：

- 逐样本可靠性结果：`outputs/evaluation/E2_legal_reliability/reliability_items.jsonl`
- 错误分类：`outputs/evaluation/E2_legal_reliability/error_taxonomy.csv`
- 更新 E1 的 `benchmark_results.csv` 中 `citation_f1`、`reasoning_coherence`、`hallucination_rate` 列。

指标：

- citation precision/recall/F1
- reasoning coherence score
- unsupported citation rate
- unsupported fact rate
- hallucination rate

验收标准：

- 每个主数据集至少抽样 200 条或全量验证。
- 引用指标必须说明 gold 来源；没有 gold 时，报告 verifier 规则和人工抽检一致性。
- 幻觉定义必须区分 `wrong citation`、`unsupported legal rule`、`unsupported fact`、`wrong jurisdiction transfer`。

### E3. 联邦优化动态实验

实验目标：

- 证明方法在 100 轮通信中收敛稳定，并降低 client drift / communication cost。

输入：

- E1 中所有联邦方法的 round-level logs。
- 方法：`FedAvg`、`FedProx`、`SCAFFOLD`、`FedNova`、`FedLoRA`、`Conflict-aware`、`Full framework`。

执行：

- 训练时记录每轮 validation accuracy、client drift、uplink/downlink bytes。
- 若使用 `run_real_flower_peft.py`，保留 `communication_log.jsonl`、`server_rounds.jsonl`、`server_eval_rounds.jsonl`。

输出：

- `outputs/runs/E3_dynamics/{method}/seed_{seed}/communication_log.jsonl`
- `outputs/runs/E3_dynamics/{method}/seed_{seed}/server_eval_rounds.jsonl`
- `papers/ipm/tables/experiment_source_data_topconf/training_dynamics.csv`

指标：

- `legal_accuracy` by round
- `client_drift`
- `cumulative_communication_mb`
- aggregation stability over last 10 rounds

验收标准：

- 所有主比较方法先完成 20-round screening，再完成 50-round 五种子主实验；FedAvg、FedLoRA 和 FLEN 追加 100-round 收敛确认。
- `training_dynamics.csv` 必须覆盖每个实际运行的 method-seed-round，禁止用固定行数掩盖失败或缺失运行。
- communication 统计必须来自真实参数/adapter payload byte size，不允许使用手写常数。

### E4. Conflict-aware 聚合有效性实验

实验目标：

- 证明 conflict-aware aggregation 能降低法律冲突，而不是只改变优化轨迹。

输入：

- 配置：`configs/experiments/conflict_aware.yaml`、`configs/experiments/conflict_aware_fedavg.yaml`、`configs/experiments/ablations/*.yaml`。
- 方法：`FedAvg`、`FedLoRA`、`Conflict-aware`、`Full framework`。
- 数据：优先使用具有同一任务和可比较输出空间的 CaseHOLD、ECtHR 分区，以及专家标注的冲突 probe；CAIL 在许可证确认前不得进入训练主张。

执行：

- 每轮计算四类冲突：citation、reasoning、verdict、rule alignment。
- 冲突组件权重、阈值和缺失组件处理必须在独立专家标注 calibration split 上确定；测试集不得用于选择权重。
- 每个 probe-pair 先分类为 `transferable`、`authority_local`、`unsupported` 或 `unresolved`，只有可比较 cohort 内的 unsupported/unresolved exposure 进入聚合权重。

输出：

- `outputs/conflict/E4_conflict_diagnostics/{method}/seed_{seed}.jsonl`
- `papers/ipm/tables/experiment_source_data_topconf/conflict_diagnostics.csv`

指标：

- citation conflict score
- reasoning conflict score
- verdict conflict score
- rule-alignment distance
- weighted conflict score

验收标准：

- `component` 必须包含 `citation`、`reasoning`、`verdict`、`rule alignment`、`total`。
- 每个 method-seed-round 必须有 5 条 component 记录。
- 主判据是专家标签上的 AUROC/AUPRC、校准误差与 coverage，以及 held-out jurisdiction 的 worst-jurisdiction utility；内部 total conflict 下降只能作为次要诊断。
- 若 FLEN 不优于 FedAvg，必须报告无效或负迁移案例，不得通过修改权重或删除种子获得有利结论。

### E5. 可解释的跨法域与跨语言迁移实验

实验目标：

- 检验模型是否在相同任务、共同标签空间和明确权威边界下改善 held-out jurisdiction 泛化。
- 将跨语言、跨国家案件分布和真正跨法律体系规则迁移作为三个不同实验报告。

输入：

- Track A：ECtHR 按 respondent state 或 court metadata 做 country holdout，在共同 ECHR 权威下测试跨国家案件分布泛化。
- Track B：MultiEURLEX en/de/fr/es/pl 做 held-out-language transfer，仅支持跨语言结论。
- Track C：使用法律专家审计的共同事实模式、共同任务标签和目标法域权威映射构建 cross-jurisdiction probe benchmark。
- CAIL、CaseHOLD、ECtHR、CUAD/LEDGAR 任务和标签空间不同，禁止直接组成 raw-accuracy train/test jurisdiction matrix。
- 方法：`Local-only`、`Centralized`、`FedAvg`、`FedProx`、`FedLoRA`、`Conflict-aware`、`Full framework`。

执行：

- Track A/C 采用 leave-one-jurisdiction-out；Track B 采用 leave-one-language-out。
- 所有映射在训练前冻结，probe/calibration/test 三者互斥，主结果报告 target-specific 和 worst-target utility。

输出：

- `outputs/transfer/E5_cross_jurisdiction/{train}_{test}/{method}/seed_{seed}/metrics.json`
- `papers/ipm/tables/experiment_source_data_topconf/cross_jurisdiction.csv`

指标：

- held-out jurisdiction legal accuracy
- in-domain vs out-of-domain delta
- worst-jurisdiction accuracy
- macro average across jurisdictions

验收标准：

- 每个可比较 target-method-seed 必须有独立记录、预测文件和映射版本；不再规定会鼓励伪映射的固定 560 行。
- 必须分别报告同域、held-out country、held-out language 和专家 probe，不得混合求一个“跨法域平均准确率”。
- 缺少共同任务或审计映射的数据集只能用于多任务或领域异质性实验，不能支持跨法域结论。

### E6. 组件消融实验

实验目标：

- 证明论文关键模块各自有贡献。

输入：

- Full framework checkpoint/config。
- Ablation variants：
  - `No jurisdiction embedding`
  - `No citation conflict`
  - `No reasoning conflict`
  - `No verdict conflict`
  - `No rule alignment`
  - `No citation verifier`
  - `No debate agents`
  - `No privacy auditor`

执行：

- 每个 variant 使用与 full model 相同 seed、split、训练预算。
- 关闭单个模块时不得改变其他超参。

输出：

- `outputs/ablations/E6_components/{variant}/seed_{seed}/metrics.json`
- `papers/ipm/tables/experiment_source_data_topconf/ablation.csv`

指标：

- legal accuracy
- citation F1
- hallucination rate
- delta vs full framework

验收标准：

- 每个 variant 至少 5 个 seed。
- `ablation.csv` 包含 full variant 和所有 ablation variants。
- 报告必须说明每个 ablation 对应论文哪一个技术主张。

### E7. 隐私-效用权衡实验

实验目标：

- 证明差分隐私和安全聚合的成本、效用下降与攻击风险下降之间的权衡。

输入：

- 数据：E1 主数据，优先 CaseHOLD + CAIL；隐私攻击部分可接 FedPII。
- DP noise multiplier：`0.0,0.2,0.5,0.8,1.0,1.5`。
- secure aggregation：`False,True`。

执行：

- 对每个 noise-secure-seed 组合运行训练。
- 记录 DP clipping norm、noise multiplier、accountant、epsilon/delta。

输出：

- `outputs/privacy/E7_privacy_utility/noise_{noise}/secure_{flag}/seed_{seed}/metrics.json`
- `papers/ipm/tables/experiment_source_data_topconf/privacy_utility.csv`

指标：

- legal accuracy
- attack AUC
- communication MB per round
- privacy budget epsilon

验收标准：

- `privacy_utility.csv` 至少包含 `5 seeds * 6 noise * 2 secure = 60` 行。
- attack AUC 必须来自实际 attack/evaluator，不允许用 DP 噪声公式直接推导。
- 报告必须给出推荐运行点，例如 `noise_multiplier=0.8 + secure aggregation`，并说明选择理由。

### E8. 鲁棒性与扩展性实验

实验目标：

- 验证方法对客户端数量和非 IID 强度的鲁棒性。

输入：

- client counts：`4,8,16,32`
- Dirichlet alpha：`0.1,0.3,1.0`
- 方法：`FedAvg`、`FedProx`、`FedLoRA`、`Conflict-aware`、`Full framework`

执行：

- 为每个 client-alpha-method-seed 生成配置。
- 使用 `configs/datasets/public_legal_corpus_dirichlet.yaml` 作为模板，动态修改 client 数和 alpha。

输出：

- `outputs/robustness/E8_scale/{clients}c_alpha_{alpha}/{method}/seed_{seed}/metrics.json`
- `papers/ipm/tables/experiment_source_data_topconf/robustness.csv`

指标：

- legal accuracy
- total communication GB / 100 rounds
- client drift
- failed-client rate, if simulated

验收标准：

- `robustness.csv` 至少包含 `5 seeds * 4 client settings * 3 alpha settings * 5 methods = 300` 行。
- 每个 alpha 下都必须报告方法排名和 worst-case drop。
- 若 32-client run 资源不足，可先跑 16 clients 并在报告中标为未完成，不得补假值。

### E9. FedPII / Membership Inference 攻击实验

实验目标：

- 验证联邦法律 LLM 在跨客户端隐私泄露上的风险，并给 E7 的 `attack_auc` 提供真实来源。

输入：

- FedPII：`data/raw/external/fedpii/fed_pii_v1.parquet`
- 或本仓库带 PII 标注/可脱敏标签的数据分区。
- 攻击设置：membership inference、PII extraction、adapter leakage probe。

执行：

- 先新增或接入 attack runner，例如 `scripts/run_privacy_attack.py`。
- 对 DP off/on、secure aggregation off/on、FedLoRA/Full framework 运行攻击。

输出：

- `outputs/privacy/E9_fedpii_attack/{setting}/attack_predictions.jsonl`
- `outputs/privacy/E9_fedpii_attack/{setting}/metrics.json`
- 回填 `privacy_utility.csv` 的 `attack_auc`。

指标：

- attack AUC
- PII extraction coverage
- extraction efficiency
- false positive rate
- victim-exclusive PII recovery rate

验收标准：

- 攻击样本必须区分 train member / non-member 或 victim-exclusive / attacker-known。
- 报告必须列出 PII 类型：name、address、birthday、case identifier 等。
- 不得输出未脱敏真实敏感信息到论文目录；原始攻击日志只能保留在受控 `outputs/privacy/`。

### E10. 人工法律可靠性评审

实验目标：

- 补强 IPM 级别的法律可靠性证据，避免完全依赖自动指标。

输入：

- E1/E2 中每个方法的 predictions。
- 抽样策略：每个数据集每个核心方法至少 30 条，优先比较 `FedAvg`、`FedLoRA`、`Conflict-aware`、`Full framework`。

执行：

- 生成人工评审包：事实、模型输出、gold label、引用、盲化 method id。
- 由至少两名法律背景标注者或一名法律背景标注者加一名复核者评分。

输出：

- `outputs/human_review/E10/review_sheet.csv`
- `outputs/human_review/E10/adjudicated_labels.csv`
- `outputs/human_review/E10/iaa_report.json`

指标：

- reasoning coherence
- legal soundness
- citation correctness
- hallucination label
- inter-annotator agreement, Cohen's kappa or Krippendorff's alpha

验收标准：

- 输出盲评协议和评分 rubric。
- IAA 必须报告；如果 IAA 偏低，必须给出仲裁流程。
- 论文中只使用脱敏、可公开展示的案例。

### E11. Conflict weight sensitivity 实验

实验目标：

- 验证 conflict penalty `lambda` 和四类冲突权重不是过度调参。

输入：

- `lambda`: `0.0,0.05,0.1,0.2,0.4,0.8`
- 权重设置：
  - uniform
  - paper default
  - citation-heavy
  - reasoning-heavy
  - rule-alignment-heavy

执行：

- 固定 dataset、seed、训练预算，扫描 lambda 和权重组合。

输出：

- `outputs/conflict/E11_weight_sensitivity/sensitivity.csv`
- `outputs/conflict/E11_weight_sensitivity/report.md`

指标：

- legal accuracy
- weighted conflict score
- citation F1
- hallucination rate
- stability over last 10 rounds

验收标准：

- 结果必须展示推荐超参附近存在稳定区间，而不是单点偶然最优。
- 若最佳 lambda 不是论文默认值，报告必须建议是否修改论文默认设置。

### E12. CaseHOLD 真实 Flower/PEFT 扩展实验

实验目标：

- 用已存在的 `scripts/run_real_flower_peft.py` 跑真实 Flower + HuggingFace + PEFT 训练，作为最低限度真实系统证据。

输入：

- 数据：LexGLUE CaseHOLD。
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`，可加 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`。
- 脚本：`scripts/run_real_flower_peft.py`、`scripts/slurm_real_flower_peft.sh`。

执行：

```bash
python scripts/run_real_flower_peft.py --mode prepare-data --run-name E12_casehold_real --num-clients 4 --num-rounds 10 --max-train-samples 1200 --max-eval-samples 240
python scripts/run_real_flower_peft.py --mode prepare-model --run-name E12_casehold_real --model-name Qwen/Qwen2.5-0.5B-Instruct
```

GPU/Slurm 环境优先运行：

```bash
bash scripts/submit_e12_casehold_real.sh
```

输出：

- `outputs/real_flower_peft/E12_casehold_real_seed42_10r/real_experiment_report.json`
- `outputs/real_flower_peft/E12_casehold_real_seed42_10r/communication_log.jsonl`
- `outputs/real_flower_peft/E12_casehold_real_seed42_10r/server_eval_rounds.jsonl`
- `outputs/real_flower_peft/E12_casehold_real_seed42_10r/acceptance.json`

指标：

- final CaseHOLD accuracy
- best accuracy
- communication bytes
- mean client drift
- wall-clock time

验收标准：

- 至少完成 4 clients、10 rounds 的真实训练。
- 至少有一组可扩展到 100 rounds 或给出资源受限说明。
- 报告明确区分真实训练结果和 dry-run 结果。

### E13. Backbone 对比实验

实验目标：

- 证明方法不是依赖单一基座模型。

输入：

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`、`TinyLlama/TinyLlama-1.1B-Chat-v1.0`、`mistralai/Mistral-7B-Instruct-v0.2`；资源允许时加入 Llama/Qwen 7B。
- 数据：CaseHOLD + CUAD；资源允许时加入 CAIL/ECtHR。

执行：

- 使用 `scripts/submit_real_model_comparison.sh` 或生成等价配置。

输出：

- `outputs/model_comparison/E13_backbone/backbone_results.csv`
- `outputs/model_comparison/E13_backbone/report.md`

指标：

- legal accuracy
- citation F1
- communication MB per round
- training time

验收标准：

- 至少 2 个 backbone 完成 3 seeds；目标版本为 3 个 backbone 完成 5 seeds。
- 报告中必须说明每个模型参数量、LoRA rank、max length、训练预算。

### E14. LoRA rank / adapter 设置对比实验

实验目标：

- 证明通信效率和效果之间的选择合理。

输入：

- LoRA rank：`4,8,16,32`
- LoRA alpha：默认 `2 * rank`
- target modules：`q_proj,v_proj`；扩展组可加入 `k_proj,o_proj`。

执行：

- 固定 dataset、model、method，只改变 LoRA 设置。

输出：

- `outputs/peft/E14_lora_rank/lora_rank_results.csv`
- `outputs/peft/E14_lora_rank/report.md`

指标：

- legal accuracy
- adapter parameter count
- communication MB per round
- client drift
- hallucination rate

验收标准：

- 至少 rank 4/8/16/32 各 5 seeds。
- 报告必须解释论文默认 rank 的选择，而不是只报告最高 accuracy。

### E15. Citation verifier 错误分析

实验目标：

- 找出 citation verifier 对论文可靠性结论的影响和失败模式。

输入：

- E2 的 `reliability_items.jsonl`。
- 人工评审子集 E10。
- 引用规范化规则和 citation extractor。

执行：

- 对 false positive / false negative citation decisions 进行分桶。
- 输出可复核的错误样例，但必须脱敏。

输出：

- `outputs/evaluation/E15_citation_error/citation_error_taxonomy.csv`
- `outputs/evaluation/E15_citation_error/examples.md`

指标：

- verifier precision/recall on human subset
- wrong authority-level rate
- malformed citation rate
- unsupported citation rate

验收标准：

- 至少 100 条人工核验样本。
- 每类错误至少给出 2 个脱敏案例或说明该错误未出现。
- 若 verifier precision 低于 0.85，论文中 citation F1 需标注为 verifier-assisted estimate。

### E16. 定性案例研究

实验目标：

- 用少量案例展示跨辖区冲突、引用纠错、多 agent debate 的机制价值。

输入：

- E1/E4/E5 中代表性成功和失败案例。
- 方法：`FedAvg`、`FedLoRA`、`Conflict-aware`、`Full framework`。

执行：

- 选择 4-6 个案例：至少 2 个成功、2 个失败、1 个跨辖区冲突案例。
- 对每个案例保存输入事实、模型输出、引用、冲突分数、最终裁决。

输出：

- `outputs/case_studies/E16/case_studies.jsonl`
- `outputs/case_studies/E16/case_studies_for_paper.md`

指标：

- qualitative correctness
- citation repair
- conflict reduction
- human reviewer note

验收标准：

- 所有案例必须脱敏。
- 不得只选成功案例；失败案例要说明边界条件。
- 每个案例必须能追溯到原始 experiment run。

## 4. 最终集成任务

### F1. 汇总结果到论文 CSV

输入：

- E1-E8 的原始日志和中间 CSV。

输出：

- `benchmark_results.csv`
- `training_dynamics.csv`
- `conflict_diagnostics.csv`
- `cross_jurisdiction.csv`
- `ablation.csv`
- `privacy_utility.csv`
- `robustness.csv`

验收标准：

- 所有 CSV schema 与第 2 节一致。
- 每个 CSV 旁边生成 `*.manifest.json`，记录来源 run。
- 用以下命令重新生成图，图中不得出现 synthetic/template 标记：

```bash
cd papers/ipm
python scripts/generate_topconf_figure_templates.py --use-existing-data --final-data
```

### F2. 论文证据一致性审查

输入：

- `papers/ipm/main.tex`
- E1-E16 报告
- 最终 figures/tables

输出：

- `outputs/reports/ipm_claim_evidence_map.md`
- `papers/ipm/SUBMISSION_TODO.md` 更新版

验收标准：

- Abstract 和 Introduction 中每个强主张都有对应实验编号。
- 若某个主张缺少实验，不删正文主张，而是在 TODO 中标为 `needs evidence`，并指出缺口实验。
- `main.tex` 编译通过。

## 5. Codex 执行优先级

第一批必须完成：

- 实现 target-conditioned cohort、client-specific conflict exposure、diversity floor、weight smoothing 和 local residual，并通过公式--代码一致性测试。
- 完成 conflict probe 的专家标注协议、calibration split 与可比性 gate。
- E1 同任务主性能实验的 20-round screening 和 50-round 五种子闭环。
- E3 真实动态、通信与失败运行日志。
- F1 带 provenance manifest 的 CSV 汇总脚本。

第二批补强 IPM 说服力：

- E2 法律可靠性与 blinded review
- E4 conflict construct validation 与聚合效果
- E5 分离的 cross-country、cross-language、cross-jurisdiction 实验
- E6 核心机制消融
- E7/E9 DP、secure aggregation、攻击与隐私--效用

第三批增强审稿抗性：

- E8 鲁棒性
- E10 人工评审
- E11 超参敏感性
- E13/E14 模型与 LoRA 设置
- E15/E16 错误分析和案例研究
