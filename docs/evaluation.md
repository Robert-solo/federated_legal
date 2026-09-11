# Evaluation Framework

The evaluation framework computes legal and federated metrics and writes automatic artifacts under `outputs/evaluation`.

## Metrics

- `citation_accuracy`: citation precision against gold citations or citation syntax when no gold set is supplied
- `legal_consistency`: explicit consistency score average or predicted/gold label agreement
- `hallucination_rate`: unsupported generation rate using citation and fact overlap
- `communication_cost`: total bytes from federated communication JSONL logs
- `client_drift`: explicit client drift average or update-norm pairwise distance

## Input Format

Prediction files can be JSONL or JSON:

```json
{
  "example_id": "case-001",
  "predicted_label": "liability_supported",
  "gold_label": "liability_supported",
  "predicted_citations": ["42 U.S.C. § 1983"],
  "gold_citations": ["42 U.S.C. § 1983"],
  "generated_text": "The claim is supported by 42 U.S.C. § 1983.",
  "supported_facts": ["claim is supported"]
}
```

## Run

```bash
python scripts/run_evaluation.py configs/experiments/evaluation_default.yaml --predictions predictions.jsonl
```

Optional communication log:

```bash
python scripts/run_evaluation.py configs/experiments/evaluation_default.yaml \
  --predictions predictions.jsonl \
  --communication-log outputs/logs/communication/fedavg_qwen_lora.jsonl
```

## Outputs

```text
outputs/evaluation/<run_name>/metrics.json
outputs/evaluation/<run_name>/metrics_table.csv
outputs/evaluation/<run_name>/metrics_table.md
outputs/evaluation/<run_name>/metrics_bar.svg
outputs/evaluation/<run_name>/report.md
```
