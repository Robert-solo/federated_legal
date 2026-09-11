# Data Module

Federated judicial dataset metadata and non-IID jurisdiction partitioning.

## Architecture

This module corresponds to Layer 1 in `paper.md`. It will manage local corpora, legal knowledge bases, citation graphs, and reasoning trajectories without centralizing raw judicial data.

Current scope:

- CAIL, LexGLUE, CaseHOLD, CUAD, LeCaRDv2, LegalBench, and MultiEURLEX local file adapters
- LexGLUE and CUAD subtask metadata without collapsing multiple tasks into one output
- canonical preprocessing into JSONL
- citation extraction
- jurisdiction and institution tagging
- tokenizer pipeline with HuggingFace or deterministic fallback
- Dirichlet, jurisdiction-based, and institution-based partitioning

Future scope:

- legal entity sanitization before preprocessing
- local citation graph construction

## Usage

```python
from fedlegal.config import load_experiment_config
from fedlegal.data import partition_processed_datasets, preprocess_datasets

config = load_experiment_config("configs/experiments/baseline_fedlora.yaml")
preprocess_datasets(config.data)
partition_processed_datasets(config.data)
```

Download public legal corpora into raw JSONL files:

```bash
python scripts/download_public_legal_datasets.py \
  --output-dir data/raw/public \
  --include-cail
```

Download the reviewer-requested extension set for retrieval, citation, rule-reasoning, and cross-language stress tests:

```bash
python scripts/download_public_legal_datasets.py \
  --output-dir data/raw/public_extended \
  --extended-only
```

The extension set records repository revisions, evaluation roles, license metadata, byte counts, and schemas for each exported split. CAIL2018 remains quarantined from training claims until its unknown license is independently resolved.

Validate preprocessing and jurisdiction partitioning after download:

```bash
python scripts/process_datasets.py configs/datasets/public_legal_extension.yaml --partition
```

Sample each configured adapter before training:

```bash
python scripts/validate_downloaded_datasets.py \
  configs/datasets/public_legal_extension.yaml \
  data/raw/public_extended/manifest.json
```

Run a bounded end-to-end preprocessing and partition preflight without overwriting full outputs:

```bash
python scripts/process_datasets.py \
  configs/datasets/public_legal_extension.yaml \
  --limit-per-source 5 \
  --output-root outputs/data_preflight/public_extended \
  --partition
```

The downloader writes `manifest.json` with source repository, split counts,
field schema, and license metadata. CAIL is marked as `license: unknown` by the
Hub metadata, so downstream experiments should report that limitation.
