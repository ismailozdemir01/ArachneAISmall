# ArachneAISmall

ArachneAISmall is the from-scratch model research and training repository for the ArachneAI ecosystem. It is intentionally separate from the commercial SaaS repository and from the open-source inference runtime.

Türkçe dokümantasyon: [README_TR.md](README_TR.md).

## Model design

The initial target configuration is a 1.5B-class decoder-only Transformer. The checked-in dimensions resolve to approximately 1.524B trainable parameters, calculated from the implemented layers and embeddings with RMSNorm, rotary position embeddings (RoPE), grouped-query causal attention, and SwiGLU feed-forward layers. The dimensions are explicit and configurable; run the parameter counter rather than assuming a label equals an exact parameter count.

The implementation starts from random initialization. It does not download, wrap, or fine-tune third-party pretrained weights.

## Install

Use Python 3.10+ and install a PyTorch build appropriate for the target CPU/CUDA environment:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data contract and provenance

Training input is JSONL with one document per line, for example:

```json
{"text":"Açık ve lisanslı eğitim metni."}
```

Only use corpora for which you have documented rights to train. Keep raw data out of Git. Record source, license, collection date, filtering, deduplication, and consent/provenance in your own dataset manifest. Do not include personal or confidential data without a lawful basis and appropriate authorization.

Prepare the source corpus before tokenization. This step normalizes Unicode and whitespace, removes exact normalized duplicates with a disk-backed index, performs a deterministic document-level train/validation split, and writes a provenance manifest. Source and license are mandatory metadata:

```bash
arachne-small prepare-corpus --input data/source.jsonl --output-dir data/split --validation-fraction 0.01 --seed 1337 --source "Dataset origin and version" --license "Exact license or written grant"
```

Train a SentencePiece BPE tokenizer from the authorized training documents only, so validation text does not leak into tokenizer training:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000
```

Encode the two document splits independently. Each binary token stream is written atomically with a `.bin.json` sidecar containing its SHA-256 checksum, tokenizer SHA-256, vocabulary size, EOS ID, source checksum, and token count. Training refuses mismatched tokenizers or altered token files:
```bash
arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

## Training

The tokenizer file is required at training time. The CLI verifies its SHA-256 against both token dataset sidecars before optimization starts.

```bash
arachne-small count-parameters --config configs/1_5b.yaml
arachne-small train --config configs/1_5b.yaml --train-data data/processed/train.bin --validation-data data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model
```

For a genuine end-to-end code smoke run on a small machine, use `configs/dev.yaml` with a tokenizer/data set whose vocabulary and sequence length match that configuration. The 1.5B target requires substantial accelerator memory, storage, and training time; the configuration is not a claim that weights have already been trained.

Resume from a checkpoint with `--resume path/to/checkpoints/step-XXXXXXXX.pt`. The completed run exports `model.safetensors`, the exact `tokenizer.model`, `config.json`, and `artifact-manifest.json`; the package contract is defined in [docs/ARTIFACT_FORMAT.md](docs/ARTIFACT_FORMAT.md). Checkpoints contain model and optimizer state. Resume files are trusted local training artifacts; do not load files from untrusted sources. Resume restores model and optimizer state, but does not guarantee bit-for-bit replay of the data-loader shuffle position.

## Local inference API and user interface

After training and verifying an artifact, start the local inference server:

```bash
# Linux/macOS
export ARACHNE_SMALL_ARTIFACT="artifacts/arachne-small-1.5b"
arachne-small serve

# Windows PowerShell
$env:ARACHNE_SMALL_ARTIFACT = "artifacts/arachne-small-1.5b"
arachne-small serve
```

Open `http://127.0.0.1:8000`. The browser UI calls the real inference API; it does not contain a simulated model response. If no verified artifact is configured, the interface reports that the model is unavailable and generation returns HTTP 503. The server binds to loopback by default. Remote binding is blocked unless explicitly enabled with `ARACHNE_SMALL_ALLOW_REMOTE=1`; remote deployments must be placed behind an authenticated TLS reverse proxy and network access controls.

API endpoints:
- `GET /api/health`: process health.
- `GET /api/status`: loaded artifact, device and model dimensions.
- `POST /api/generate`: text generation with bounded prompt and generation parameters.
- `GET /docs`: interactive OpenAPI documentation.

The model is loaded once at startup and inference requests are serialized per process to prevent concurrent generation from exhausting accelerator memory. See [docs/INFERENCE_API.md](docs/INFERENCE_API.md) for the API contract and deployment notes.

## Model card and data governance

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model's release status and evaluation requirements, and [docs/DATASET_POLICY.md](docs/DATASET_POLICY.md) for dataset intake and provenance rules.

## Controlled continual learning

The repository includes an explicit-consent, local SQLite feedback store and an end-to-end continual-learning command. Eligible corrections are encoded with the verified base artifact's tokenizer; candidate training initializes from that artifact rather than random weights. A candidate is evaluated against the baseline on the same fixed validation stream and is rejected on validation-loss regression by default. Passing candidates are stored as immutable content-addressed releases behind an atomic current-version pointer, with verified rollback support. There is no hidden telemetry or automatic collection of raw prompts. See [Controlled Continual Learning](docs/CONTINUAL_LEARNING.md).

This pipeline requires a trained base artifact and a matching validation token stream. No trained weights are included in the repository; the pipeline is source-complete but has not been executed on the user's training hardware.

## Code license

No code license has been selected in this repository. Making the repository public does not by itself grant permission to reuse, modify, or redistribute its code. Add the owner's chosen license before describing the project as open source.

## Repository boundary

This repository owns tokenizer training, data encoding, model architecture, optimization, checkpointing, evaluation, and model artifact production. ArachneAIOpen owns inference and must consume the versioned artifact format documented there. ArachneAI remains the commercial SaaS product.

## Current implementation status

The repository contains the complete source pipeline for producing a trained artifact, but no trained weights or training corpus are committed. A completed run exports a versioned SafeTensors package with the exact tokenizer, architecture config, and SHA-256 manifest. Verify a completed package with:

```bash
arachne-small verify-artifact --path artifacts/arachne-small-1.5b
```

Training, numerical validation, artifact verification, and hardware throughput measurements must be run on the target machine before claiming a trained release.
