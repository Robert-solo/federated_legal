# 人工标注与法律盲评操作指南

## 谁来标

作者可以担任一位标注者，并如实记录法律训练、研究经历和作者身份。
一个人做两轮是单人复核，不是两位独立评审者；不能据此计算标注者间一致性。
没有法律训练的作者可先整理材料和做试标，最终专家标签由具有相应训练者确认。
如只有作者标注，论文报告 author annotation / single-annotator pilot。

## 批量生成的是待标材料

```bash
python scripts/prepare_expert_annotation.py --output annotation/authority_new
```

脚本下载美国最高法院官方意见 PDF，保留法院、日期、网址、页码和原文，
生成 120 对候选材料和两份独立 CSV。脚本不生成金标准。
已生成的可填写版本在 `annotation/authority_v3/`：7 份官方判决 PDF，
development、calibration、test 各 40 对。先填写 `pilot_15.csv`；
这 15 对均来自 development，试标完成后将标签按 item_id 合并回 reviewer_1.csv。
reviewer_2.csv 留给第二位独立标注者。
来源覆盖联邦刑法、宪法与行政法中的权威范围和时间适用性；
这些材料不是跨国法律冲突基准，也还不能替代真实法域训练集。

## 你需要填写哪些内容

每个 item 先读原文页和必要的上下文，再填写：

1. `issue`：双方讨论的具体法律问题。
2. `authority_status_A/B`：majority holding / concurrence / dissent / syllabus / quotation。
   判决摘要、异议意见和多数意见不能混用为同等权威。
3. `operative_rule_A/B`：各段适用的规则，用自己的简短句子概括。
4. `comparable_yes_no`：同一法律问题、事实条件与目标权威范围是否可比。
5. `routing_state`：`transferable`（可以共同使用）、`authority_local`（仅在原范围有效）、
   `unsupported`（主张缺乏依据）、`unresolved`（目前材料无法判定）。
6. `reason_and_pinpoint`：一两句理由，包含页码及适用条件。
7. `reviewer_id`、`confirmed_by_human=yes`：标注来源。

`page_A/B` 是 PDF 文件页序号，从 1 开始；正式引证同时写判决正文页码。
PDF 抽取偶有字符错码，以同目录 `sources/` 中的原 PDF 为准。
`unsupported` 用于可比条件下缺乏原文支持的规则主张；材料不足以判断时填
`unresolved`。两段谈不同法律问题时先记 `comparable_yes_no=no`，并说明范围，
不要仅凭结论不同就记为冲突。

先试标 10–15 对，解决标签含义后再冻结正式规则。
development 用于制定规则，calibration 用于选择阈值，test 用于最终检验；
同一案件及其高度相关案件不应横跨这些分区。脚本按来源家族分区，
应由专家确认问题范围和样本覆盖并冻结后才训练冲突分类器。

模型可以建议规则概括与标签，建议保存在额外的 `suggested_*` 列。
人工确认应在独立的最终标签列完成，记录修改；全自动输出只能作为银标。
不用标签平衡或预期性能来决定最终标签。

批量处理分三步：脚本抽取官方来源与候选对 → 人工填写或逐条修正模型建议 →
汇总已确认记录。已有公开数据的标签只有在定义、权威范围、版本一致时才能映射。
不能用模型一键填满的 CSV 代替人工证据。

```bash
python scripts/analyze_expert_annotations.py --sheets annotation/authority_v3/reviewer_1.csv annotation/authority_v3/reviewer_2.csv --output annotation/authority_v3/annotation_summary.json
```

程序统计实际确认数量、待标数量和标签分布；两位不同标注者有共同完成项时，
再计算原始一致率、Cohen's kappa 和待讨论的分歧列表。
若只有一位完成，报告单人 pilot 的数量和分布。120 对是流程试点，
正式法律冲突有效性研究还要扩充独立案件和目标法域，按实际标签分布确定样本规模。

## A/B 法律推理盲评

修正版输出位于 `outputs/revision_v2/human_review/`。
每位评审者仅接收自己的 `reviewer_N_sheet.csv`，不能接收 `condition_key.json`。
按给定案情选择 A/B/tie，分别评价法律正确性和推理连贯性（1–5），
标记是否有无依据事实或法律主张，并用一句话解释关键判断。
同一人不得伪装成两位评审者。两位先独立完成、冻结文件后再讨论分歧。
作者知道方法或参与生成时，应记录其评价的作者参与性质，不能宣称完全独立盲评。

## 交付

交回填写后的 CSV 和评审者资历说明；不在公开仓库写私人联系方式。
两位记录必须保留原始文件，裁决另存。论文报告完成数量、弃标数、
一致性、裁决规则和最终指标。没有收回人工文件前，结果状态为 pending。
