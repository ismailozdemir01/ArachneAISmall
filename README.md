# ArachneAISmall

**ArachneAISmall is a from-scratch, Turkish–English language-model research and training project in the ArachneAI ecosystem.**

The repository contains the implementation for preparing licensed training data, building a tokenizer, training a decoder-only Transformer, evaluating model artifacts, and serving a verified model through a local API and web interface.

- **Primary languages:** Turkish and English
- **Target model size:** 1.5B-class
- **Configured parameter count:** 1,523,714,048
- **Training approach:** random initialization; no third-party pretrained weights
- **Project status:** training and release pipeline implemented; trained model weights are not yet published

[🇹🇷 Türkçe README](README_TR.md)

---

## Project goals

ArachneAISmall is being developed to provide a transparent, reproducible path from authorized text corpora to a versioned language-model artifact. The goal is to build and train the model itself rather than adapt an existing pretrained model.

The 1.5B configuration is an architecture target, **not evidence that a model of this size has already been trained**. Turkish and English are the intended languages; language quality and capability claims require training and published evaluations.

## Architecture

The configured decoder-only Transformer uses:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped-Query Causal Attention (GQA)
- SwiGLU feed-forward layers
- Residual connections
- Optional gradient checkpointing
- Configurable model dimensions and training settings

Inspect the configured parameter count directly:

```bash
arachne-small count-parameters --config configs/1_5b.yaml
```

## What is implemented

| Area | Implementation |
|---|---|
| Corpus preparation | Unicode normalization, deterministic document-level splitting, disk-backed deduplication and provenance manifest |
| Tokenizer | SentencePiece BPE training from the training split |
| Token datasets | Binary token streams with checksum and tokenizer metadata sidecars |
| Training | PyTorch training loop, AdamW, gradient accumulation, clipping, warmup/cosine schedule and evaluation |
| Checkpointing | Atomic checkpoint writes and validated resume |
| Model artifacts | SafeTensors weights, tokenizer, model configuration and SHA-256 manifest |
| Artifact verification | File integrity, tokenizer identity, tensor names/shapes and architecture checks |
| Inference | FastAPI endpoints and a Turkish/English browser interface |
| Feedback | Local SQLite storage with explicit user consent and mandatory operator review/approval before training export |
| Continual learning | Candidate fine-tuning from a verified base artifact, baseline comparison, gated promotion and rollback |

Implementation and operational details are documented in the `docs/` directory.

## Quick start

### Requirements

- Python 3.10 or newer
- A PyTorch installation appropriate for the target CPU or CUDA environment
- Authorized training text in UTF-8 JSONL format

Install the project:

```bash
python -m venv .venv
# Activate the virtual environment
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Prepare a corpus

Each JSONL line must contain a `text` field:

```json
{"text":"A licensed Turkish or English training document."}
```

Prepare the corpus and record its origin and license:

```bash
arachne-small prepare-corpus \
  --input data/source.jsonl \
  --output-dir data/split \
  --validation-fraction 0.01 \
  --seed 1337 \
  --source "Dataset origin and version" \
  --license "Exact license or written permission"
```

Train the tokenizer only on the training split, then encode both splits:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000

arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

The data pipeline writes integrity metadata alongside token files. Training checks the token files and tokenizer identity before optimization begins.

### Train

```bash
arachne-small train \
  --config configs/1_5b.yaml \
  --train-data data/processed/train.bin \
  --validation-data data/processed/validation.bin \
  --tokenizer artifacts/tokenizer/arachne.model
```

The 1.5B target requires substantial accelerator memory, storage, and training time. Use `configs/dev.yaml` for a small local software smoke test with matching tokenizer and data dimensions; that test does not train or validate the 1.5B model.

Only use datasets for which model-training rights are documented. Do not commit raw corpora, personal data, or confidential material to this repository.

## Local inference

After training, set the path to a verified model artifact and start the service.

**Windows PowerShell**

```powershell
$env:ARACHNE_SMALL_ARTIFACT = "artifacts/arachne-small-1.5b"
arachne-small serve
```

**Linux / macOS**

```bash
export ARACHNE_SMALL_ARTIFACT="artifacts/arachne-small-1.5b"
arachne-small serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web interface calls the actual inference API. It does not provide placeholder or simulated model answers. If a verified model artifact is unavailable, generation is disabled and the API returns HTTP 503.

The server binds to loopback by default. Remote binding requires explicit opt-in; any remote deployment must use an authenticated TLS reverse proxy and network access controls.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service health |
| `GET /api/status` | Loaded artifact and runtime status |
| `POST /api/generate` | Text generation |
| `POST /api/feedback` | Explicit-consent feedback submission |
| `GET /docs` | Interactive OpenAPI documentation |

See [Inference API](docs/INFERENCE_API.md).

## Controlled continual learning

The feedback workflow is opt-in and local. Feedback must be explicitly reviewed and approved by an operator before eligible corrections can be used to train a candidate initialized from a verified base artifact. The candidate is evaluated against the baseline on the same validation token stream. Promotion is blocked when the configured validation-loss gate fails; promoted releases are content-addressed and support verified rollback.

This workflow requires an already trained base artifact and a matching validation dataset. It does not mean continual learning has already been run on a released model.

See [Controlled Continual Learning](docs/CONTINUAL_LEARNING.md).

## Documentation

- [Model Card](docs/MODEL_CARD.md)
- [Dataset Policy](docs/DATASET_POLICY.md)
- [Artifact Format](docs/ARTIFACT_FORMAT.md)
- [Inference API](docs/INFERENCE_API.md)
- [Controlled Continual Learning](docs/CONTINUAL_LEARNING.md)

## Validation and release status

The current automated test suite was run locally on Python 3.14.7 with CPU-only PyTorch: **33 tests passed**. Python bytecode compilation and the command-line help check also completed successfully in that environment.

These checks validate software behavior covered by the test suite. They are **not** a trained-model evaluation, benchmark, throughput measurement, or evidence of Turkish/English language quality. No trained weights or training corpus are currently published in this repository.

A trained release requires an actual training run, recorded evaluation metrics, artifact verification, and hardware performance measurements.

## Repository boundaries

- **ArachneAISmall:** model architecture, data preparation, tokenizer, training, evaluation and artifact creation.
- **ArachneAIOpen:** separate inference/runtime project consuming the documented artifact contract.
- **ArachneAI:** separate commercial SaaS product.

## License

The ArachneAISmall source code in this repository is licensed under the **Apache License 2.0**. You may use, modify, and distribute the code under its terms. The license includes an express patent grant and requires preservation of applicable notices and provision of the license with redistributed copies.

See [LICENSE](LICENSE) for the complete terms.

The license for this repository's code does not automatically apply to external datasets, third-party assets, or future trained model weights. Those materials must carry their own applicable license and usage terms.
