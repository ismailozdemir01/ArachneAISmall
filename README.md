# ArachneAISmall

ArachneAISmall is the from-scratch model research and training repository for the ArachneAI ecosystem. It is intentionally separate from the commercial SaaS repository and from the open-source inference runtime.

## Model design

The initial target configuration is a 1.5B-class decoder-only Transformer. The checked-in dimensions resolve to approximately 1.524B trainable parameters, calculated from the implemented layers and embeddings with RMSNorm, rotary position embeddings (RoPE), grouped-query causal attention, and SwiGLU feed-forward layers. The dimensions are explicit and configurable; run the parameter counter rather than assuming a label equals an exact parameter count.

The implementation starts from random initialization. It does not download, wrap, or fine-tune third-party pretrained weights.

## Install

Use Python 3.10+ and install a PyTorch build appropriate for the target CPU/CUDA environment:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -e .
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

```bash
arachne-small count-parameters --config configs/1_5b.yaml
arachne-small train --config configs/1_5b.yaml --train-data data/processed/train.bin --validation-data data/processed/validation.bin
```

For a genuine end-to-end code smoke run on a small machine, use `configs/dev.yaml` with a tokenizer/data set whose vocabulary and sequence length match that configuration. The 1.5B target requires substantial accelerator memory, storage, and training time; the configuration is not a claim that weights have already been trained.

Resume from a checkpoint with `--resume path/to/checkpoints/step-XXXXXXXX.pt`. Checkpoints contain model and optimizer state. Resume files are trusted local training artifacts; do not load files from untrusted sources. Resume restores model and optimizer state, but does not guarantee bit-for-bit replay of the data-loader shuffle position.

## Repository boundary

This repository owns tokenizer training, data encoding, model architecture, optimization, checkpointing, evaluation, and model artifact production. ArachneAIOpen owns inference and must consume the versioned artifact format documented there. ArachneAI remains the commercial SaaS product.

## Current implementation status

The training implementation is source code only. No model weights or training corpus are committed. Final model artifacts are versioned (`artifact_version: 1`) and record the tokenizer SHA-256 so the inference runtime can verify the exact tokenizer. Training, numerical validation, and hardware throughput measurements must be run on the target machine before claiming a trained release.
