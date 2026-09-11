# Model Comparison Plan

The comparison matrix separates models that can participate in real Flower +
LoRA training from closed API models that can only be evaluated as inference
baselines.

## Federated Trainable Models

These models expose HuggingFace weights and can be cached before Slurm GPU jobs:

- `Qwen/Qwen2.5-0.5B-Instruct`: smoke and reproducibility baseline.
- `Qwen/Qwen2.5-1.5B-Instruct`: efficient bilingual baseline.
- `Qwen/Qwen2.5-7B-Instruct`: stronger bilingual baseline.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`: DeepSeek reasoning baseline.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`: larger DeepSeek reasoning baseline.
- `mistralai/Mistral-7B-Instruct-v0.3`: strong English legal baseline.
- `openai/gpt-oss-20b`: larger open-weight baseline when GPU memory permits.

Each trainable model should be run with FedAvg, FedProx, Scaffold, and later the
conflict-aware aggregation strategy over IID, Dirichlet, and jurisdiction-based
partitions.

## API Inference Baselines

OpenAI GPT-5.3/GPT-5.4 and DeepSeek API models should be evaluated centrally on
held-out prompts with identical scoring code. They should not be reported as
federated PEFT participants unless compatible model weights are available.

Required environment variables:

- `OPENAI_API_KEY`
- `OPENAI_GPT53_MODEL`
- `OPENAI_GPT54_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

Exact OpenAI model IDs must be verified from the user's API account before
running, because API-visible model names can differ by account and date.

## Reporting

Every comparison run should record:

- model ID and resolved revision or API model name
- dataset task and split
- partition strategy
- FL algorithm
- per-round accuracy or F1
- communication bytes
- client drift
- cross-jurisdiction generalization
- prompt template and API sampling parameters for closed baselines
