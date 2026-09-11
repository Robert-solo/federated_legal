# Dataset Pipeline

The legal dataset pipeline supports CAIL, LexGLUE, CaseHOLD, and CUAD as local raw files. It does not download datasets automatically, because licensing and access differ across legal corpora.

## Supported Input Formats

Each configured source may point to `.jsonl`, `.json`, or `.csv`.

Preferred layout:

```text
data/raw/
  CAIL/train.jsonl
  LexGLUE/train.jsonl
  CaseHOLD/train.jsonl
  CUAD/train.jsonl
```

JSONL rows are ordinary JSON objects. Dataset-specific text fields are configured in YAML, with defaults for common field names.

## Processing Steps

The preprocessing pipeline performs:

1. Dataset-specific field normalization
2. Canonical legal record creation
3. Citation extraction from text and configured citation fields
4. Jurisdiction, legal-tradition, institution, and institution-type tagging
5. Tokenization through HuggingFace `AutoTokenizer` when configured, or a deterministic simple tokenizer fallback
6. JSONL persistence under `data/processed`
7. Manifest writing to `data/processed/manifest.json`

Run:

```bash
python scripts/process_datasets.py configs/datasets/legal_supported.yaml
```

Processed outputs:

```text
data/processed/CAIL/train.jsonl
data/processed/LexGLUE/train.jsonl
data/processed/CaseHOLD/train.jsonl
data/processed/CUAD/train.jsonl
data/processed/manifest.json
```

## Federated Non-IID Partitioning

Supported partition strategies:

- `dirichlet`: label-skewed non-IID partitioning using a seeded Dirichlet sampler
- `jurisdiction_based`: routes records by jurisdiction tag to the matching client
- `jurisdiction_holdout`: alias of jurisdiction-based routing for cross-jurisdiction holdout experiments
- `institution_based`: routes records by institution ID

Run preprocessing and partitioning together:

```bash
python scripts/process_datasets.py configs/datasets/legal_supported.yaml --partition
```

Run partitioning from existing processed files:

```bash
python scripts/partition_datasets.py configs/datasets/legal_jurisdiction_partition.yaml
python scripts/partition_datasets.py configs/datasets/legal_institution_partition.yaml
```

Partition outputs:

```text
data/partitions/<strategy>/<client_id>.jsonl
data/partitions/<strategy>/manifest.json
```

## Canonical Record Schema

Processed records include:

- `record_id`
- `dataset`
- `split`
- `text`
- `label`
- `citations`
- `jurisdiction`
- `legal_tradition`
- `institution_id`
- `institution_type`
- `metadata`
- `input_ids`
- `attention_mask`
- `tokens`

## Reproducibility

Use the dataset config seed for deterministic simple tokenization and partitioning. Dirichlet partitioning is deterministic for a fixed seed, alpha, client list, and processed input order.
