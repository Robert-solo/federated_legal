# 工作交接：论文、代码与修订实验

更新时间：2026-11-01，Asia/Shanghai。本文件记录实际完成状态；运行中的矩阵以服务器输出为准。

## 目标与当前判断

本轮要统一方法和代码，完成相同训练预算的门控消融、强基线与独立测试、
权威元数据/人工冲突验证、联邦 adapter 与推理连接、隐私对照，然后更新论文。
目前已修复关键方法实现差异并启动修正版实验。完整投稿证据仍取决于正式矩阵完成、
真实法律标注与独立盲评、真实法域泛化数据。不能把实验启动当作结论已经成立。

## 唯一维护的论文版本

- 主文件：`papers/ipm/Federated Legal elsevier/cas-sc-template.tex`。
- PDF：同目录 `cas-sc-template.pdf`，使用系统 TeX Live / latexmk 成功编译。
- 只在该模板及其 tables/figures 下更新论文内容。
- 旧 `papers/ipm/main.tex` 及旧表格有此前未提交的改动，本轮保留；不继续维护该稿。
- Data and code availability 已使用 `https://github.com/Robert-solo/federated_legal.git`。
- 改写了旧实验中把“门控+训练预算同时变化”归因为门控效果的句子。
- 最新 PDF 已更新为 20 页，隐私新表、概率可靠性图、方法公式和 Figure 2 已渲染检查。
- 方法公式明确为共享 LoRA 因子平均与独立双分支矩阵输出；算法伪码同步更新。
- Figure 2 已重绘，源脚本 `scripts/create_revision_method_figure.py`。旧 Visio 文件属于
  旧图版本，不要用它覆盖新 PDF。其他框架图仍描述完整设计协议。

## 已完成的代码工作

1. `src/fedlegal/training/dual_lora.py`：共享 default 和私有 local adapter 独立，
   输出为 `B_shared A_shared + gamma B_local A_local`，只上传共享因子。
   重置私有 B、保留 A 的可训练初始化，避免双因子都置零导致梯度消失。
2. `scripts/run_real_flower_peft.py`：默认 dual 分支；保留 factor_residual 复现旧实验。
   所有方法预留相同 gate 数据，均排除在共享/私有训练之外。增加 Ditto 个性化近端训练。
3. `scripts/run_casehold_revision.py`：四个真实 PEFT 客户端、真实 Flower 策略在单进程执行。
   固定 20 轮、5 个种子、1200 训练、80 验证、24 probes、1000 最终测试。
   测试集只在训练结束后进入模型评价，导出 shared_adapter 和逐条测试预测。
4. MultiEURLEX：从训练池单独预留 probes，不再与阈值验证集共用；只在最后一轮测测试集。
   隐私增加仅裁剪对照、0.25 噪声、Brier、15-bin pooled-label ECE；非成员来自源语言测试集。
5. `scripts/run_multi_agent_blind_eval.py`：可加载真实导出的联邦 adapter；增加四次调用的
   单智能体自修正对照，匹配四阶段推理的分析调用数。
6. 分析脚本：`scripts/analyze_revision_v2.py` 导出运行 CSV、种子汇总、配对差异、
   PDF/PNG 和隐私可靠性图；不按性能高低或阈值边界选择性删运行。
7. 修复 training/federated 包循环导入，以及 Python 3.10 的 timezone 导入兼容问题。
8. 服务器测试：76 passed，1 条 PEFT 词表配置提示。独立分支测试检查实际矩阵输出、
   私有梯度、共享冻结及重置；分析测试检查配对种子与缺失运行处理。
   本地普通 Python 缺 PEFT，不要据此宣称本地完整训练测试通过。

## 服务器与同步

- 登录：`junluo@172.29.15.117`。
- 密钥文件：`C:/Users/a/.ssh/zhurong_codex_20260830_ed25519`，不复制密钥内容到仓库。
- 项目：`/remote_dir/home/junluo/IPM`；服务器目录没有 `.git`。
- Python：`/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python`，3.11.15。
- 模型和数据已缓存，Slurm 使用离线加载。
- 初始服务器代码备份：`/remote_dir/home/junluo/IPM-code-before-20261101.tar.gz`。
- 本地备份：`tmp/alignment_20261101/remote_before.tar.gz`，已按文本比较代码差异。
- 本轮先对齐本地较新的配置/元数据检查，再将 src/scripts/configs/tests 同步服务器。
  后续用提交后的 Git 源码归档同步维护文件；输出、缓存、模型不覆盖。
- 所有后续代码变更先检查正在运行的任务是否已加载；冻结科学设置不随测试结果改动。

## 已提交的作业与路径

| 作业 | 内容 | 状态说明 |
| --- | --- | --- |
| 168484 | guarded 8+2 一轮 GPU smoke | 完成，仅预检 |
| 168489 | Ditto smoke | 完成，仅预检 |
| 168490 | FLEN guarded smoke | 完成，仅预检 |
| 168496，array 0–39，最多并行 8 | CaseHOLD 正式 8 方法 × 5 种子 | 运行中 |
| 168511，array 0 | FedAvg seed41 补交 | 依相同配置运行，替代训练前退出的 168496_0 |
| 168491 | 隐私 no_clip，41–43 | 已完成 |
| 168492 | 隐私 clipped_only，41–43 | 已完成 |
| 168493 | 隐私 Gaussian 0.25，41–43 | 已完成 |
| 168494 | 隐私 Gaussian 0.5，41–43 | 已完成 |
| 168495 | 隐私 Gaussian 1.0，41–43 | 已完成 |
| 168512 | base/federated adapter 的代理对照与盲评包 | afterok:168511，等待训练 |
| 168518 | 自动生成正式矩阵分析和图 | afterany:168496:168511:168491:168492:168493:168494:168495 |

CaseHOLD 输出：`outputs/revision_v2/casehold/SPEC_seedN/`。
8 方法为 fedavg_10、fedprox_10、ditto_8_2、unguarded_8_2、guarded_8_2、
unguarded_10_2、guarded_10_2、flen_guarded_8_2。
必须分别比较固定 8+2 和固定 10+2 内的门控差异。

隐私输出：`outputs/experiments/revision_v2_privacy_CONDITION_en-pl_seedN/`。
分析输出：`outputs/revision_v2/analysis/{casehold,privacy}/`。
代理输出：`outputs/revision_v2/human_review/{base,federated}_{1,4}pass/`。
后续实际完成数量见本地 `papers/ipm/results/revision_v2/` 的快照；部分矩阵只作进度展示。

隐私 15/15 已完成并同步最新模板：Gaussian 0.5 相对仅裁剪的攻击 AUC
0.615→0.525，配对差异区间 [-0.2145, 0.0343]；Brier 0.227→0.481，
配对增加区间 [0.1448, 0.3639]。攻击效果仍不确定，概率质量恶化有直接测量支持。
新表/图由 `scripts/update_revision_privacy_table.py` 从完整分析生成。
`privacy/source_runs/` 保留 15 运行的配置、逐轮日志、预测概率和成员攻击损失特征。
本地重建完整汇总，另对 seed41 的 no_clip 和 Gaussian 0.5 重跑攻击，AUC 与原结果一致。

## 人工标注：用户现在需要做什么

用户询问自己能否替代两位法律评审。已说明：作者可以贡献一位标注者；有相应法律训练
可作为一位法律评审；同一人重复两轮仍为单人复核。只有单人时如实报告 author-annotated
pilot；要报告两位独立评审还需要另一位。用户尚未提供法律资历或第二位评审。

- 可填写材料：`annotation/authority_v3/`。
- 7 份真实官方判决 PDF，120 对候选材料，开发/校准/测试按案件家族各 40 对。
- `pilot_15.csv` 供先试标，均来自开发分区；两份正式 reviewer CSV 留给不同标注者。
- 指南：`docs/experiments/HUMAN_ANNOTATION_GUIDE.md`。
- 下载/生成：`scripts/prepare_expert_annotation.py`。
- 导入汇总：`scripts/analyze_expert_annotations.py`。
- 当前 human_confirmed=0；空表和模型建议均不构成人工金标准。
- 本地分享包：`outputs/annotation/authority_v3_review_pack.zip`，包含 PDF 与填写说明。
- authority_v1/v2 是早期下载不完整版本；使用 v3。

这是美国联邦权威及时间适用性试点。只有 7 个案件、按来源家族划分，正式研究仍需扩充
独立案件和法域覆盖；不能将该试点当作跨国法域冲突验证。不同问题之间的差异先标不可比，
不能把结论不同直接记作法律冲突。模型可提供 suggested 列，最终标签需逐条人工确认。

代理盲评包生成后：两人各收自己的 reviewer 表，不收 condition_key；独立填写偏好、
法律正确性、连贯性、无依据主张和简短理由，冻结后再讨论。作者参与评价须披露。

## 尚未解决与下一步

1. 检查正式作业完成/失败原因，确认 CaseHOLD 40 个均有完整结果；隐私 15 个已完成。
   只补交真正失败项，保持相同配置；不因结果不利更换种子或测试集。
2. 下载结果与日志，运行分析，核对预测重构与配对种子，再更新最新 CAS 的数值和图表。
   168518 会在训练结束后生成服务器分析；它不自动修改论文或推送后来生成的结果。
3. 168512 成功后完成 base/federated × 单次/四次控制的自动分析，并分发真实人工盲评。
   当前用 FedAvg shared_adapter 测试联邦训练与推理的连接；尚未验证目标私有 FLEN adapter。
4. 用户完成试标后冻结定义，扩充真实权威样本，取得第二位独立标注，统计一致性与分歧，
   才能训练/校准法律冲突分类器并运行专家校准 FLEN。
5. 真实法域分区仍未解决。CaseHOLD 是启发式非 IID；固定 round-robin 测试最差客户端
   是部署诊断，不是最差法域指标。MultiEURLEX 是共享 EU 权威下语言迁移。
6. 旧 MultiEURLEX 72 运行结果保留为 legacy；修正 probe/threshold 分离后尚未重新完成
   六方法 × 四语言 × 三种子的完整矩阵，不要混为新版结果。
7. 更强的近期方法（如 SCAFFOLD、FedJudge）还需忠实实现和公平调参，不能声称已达到 SOTA。
8. 隐私当前为可信聚合器下的客户端更新保护，攻击测记录成员身份。按攻击对象、
   micro/macro F1 与 BCE、Brier/ECE 分开解释；尚无推理消息隐私/重构攻击证据。
9. 引用一致性、推理连贯性与幻觉率仍需相应标注/评审数据。不能用分类准确率代替。
10. 投稿前由用户确认作者真实姓名和单位；当前模板沿用 Robert J. / Independent Research Project。

## 已踩的坑

- 旧代码将本地残差加到 LoRA 因子，产生交叉项；与独立矩阵分支不等价。
- 旧门控对照 8+2 对 10+2 混入共享训练预算，无法单独归因门控。
- 168496_0 在训练前因 HuggingFace datasets 的 NFS FileLock stale handle 退出。
  已改为 Dataset.from_list(read_jsonl(...))，科学设置不变。原空目录移动到
  `outputs/revision_v2/failed_before_training/fedavg_10_seed41_168496_0`，然后补交 168511。
- 原始 slip opinion 链接部分失效；国会图书馆正式报告 PDF 可用。
  本机 urllib 曾 403，而 curl 可正常下载，生成器现在使用 curl。
- 不能以阈值触边排除噪声较大的运行，否则造成结果选择；新版仅作诊断。
- BCE 增大不能自动推断所有任务效用下降，macro F1 可能同时上升；直接看校准指标。
- 旧代理实验只加载 base model，新版本才接入 adapter；旧 60 案结果不能归因联邦学习。
- 系统 Python 有 matplotlib，Codex bundled Python 有 pypdf；后者没有 matplotlib。
  服务器训练环境包含全部实验依赖。latexmk 必须在最新模板目录运行。
- CAS 首页面有一个类内部 overfull 提示；已渲染检查题名/摘要未见裁切。
  部分预印本参考文献 pages 字段为空会有 BibTeX 提示，不是编译失败。

## 常用检查与复现

```powershell
ssh -i C:/Users/a/.ssh/zhurong_codex_20260830_ed25519 junluo@172.29.15.117
```

```bash
cd /remote_dir/home/junluo/IPM
squeue -u junluo
sacct -j 168496,168511,168491,168492,168493,168494,168495,168512,168518 --format=JobID,State,ExitCode,Elapsed
/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python -m pytest -q
/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python scripts/analyze_revision_v2.py --kind casehold --input outputs/revision_v2/casehold --output outputs/revision_v2/analysis/casehold
/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python scripts/analyze_revision_v2.py --kind privacy --input outputs/experiments --output outputs/revision_v2/analysis/privacy
```

在最新模板目录编译：`latexmk -pdf -interaction=nonstopmode -halt-on-error cas-sc-template.tex`。
完整协议见 `experiments/revision_v2/README.md`。

## 投稿前复查

贡献定义与代码双分支已对齐；写作中的门控归因已纠正；新版实验正在补齐公平对照；
人工与真实法域评价尚缺结果；方法效果需要正式配对区间支持。以上五点均应在最终
投稿前重新核对，尤其不能沿用旧图表宣称修正版已获证实。
