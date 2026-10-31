# ArachneAISmall

ArachneAISmall is the from-scratch model research and training repository for the ArachneAI ecosystem. It is intentionally separate from the commercial SaaS repository and from the open-source inference runtime.

## Model design

The initial target configuration is a 1.5B-class decoder-only Transformer with RMSNorm, rotary position embeddings (RoPE), grouped-query causal attention, and SwiGLU feed-forward layers. The dimensions are explicit and configurable; run the parameter counter rather than assuming a label equals an exact parameter count.

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

Train a SentencePiece BPE tokenizer from your authorized UTF-8 text corpus:

```bash
arachne-small tokenizer --input data/corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000
```

Encode JSONL documents into a compact little-endian uint32 binary token stream. The encoder streams documents and the training dataset memory-maps the resulting file, avoiding loading the full token corpus into RAM:

```bash
arachne-small prepare-data --input data/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

Create train and validation source files before encoding; keep documents isolated between splits to prevent leakage.

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

The training implementation is source code only. No model weights or training corpus are committed. Training, numerical validation, and hardware throughput measurements must be run on the target machine before claiming a trained release.
