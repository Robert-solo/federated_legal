# 待标材料：法律权威与时间适用性试点

先打开 `pilot_15.csv` 试标，再填写 `reviewer_1.csv`。第二位独立标注者填写
`reviewer_2.csv`。每张正式表有 120 对：开发、校准、测试各 40 对。
同一案件家族只属于一个分区，试标项来自开发分区。

官方判决：Johnson、Welch、Borden、Elonis、Counterman、Chevron、Loper Bright。
来源清单和日期在 `source_manifest.json`；已检查下载文件首页的案件名称与日期。
`sources/` 保存原 PDF；CSV 的 page_A/B 是 PDF 页序号，从 1 开始。
原文抽取中的少量字符错码请对照 PDF 阅读。

请填写：法律问题、可比性、双方权威地位、双方规则、路由标签、理由与精确页码、
标注者编号及 `confirmed_by_human=yes`。具体标签解释见
`docs/experiments/HUMAN_ANNOTATION_GUIDE.md`。

一人重复填写仍然是单人复核。120 对用于流程试点，真实跨法域研究还需要扩大
权威来源和独立案件；当前没有已完成人工标签。

```bash
python scripts/analyze_expert_annotations.py --sheets annotation/authority_v3/reviewer_1.csv annotation/authority_v3/reviewer_2.csv --output annotation/authority_v3/annotation_summary.json
```

批量生成或更新候选材料时使用新的输出目录，保留已填写文件：

```bash
python scripts/prepare_expert_annotation.py --output annotation/authority_new
```
