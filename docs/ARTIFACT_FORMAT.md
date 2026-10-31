# Arachne Small Artifact Format

**Format identifier:** `arachne-small`  
**Format version:** `1`

A completed training run writes an inference package into the configured `output_dir`:

```text
output_dir/
├── model.safetensors
├── tokenizer.model
├── config.json
├── artifact-manifest.json
├── training.jsonl
└── checkpoints/
    └── step-XXXXXXXX.pt
```

## Inference files

- `model.safetensors`: model state dictionary. It is a SafeTensors file; inference loaders must not use Python pickle to load model weights.
- `tokenizer.model`: exact SentencePiece model used to encode the training and validation token streams.
- `config.json`: canonical architecture and tokenizer configuration.
- `artifact-manifest.json`: SHA-256 checksums and format metadata.

Example configuration:

```json
{
  "format": "arachne-small",
  "format_version": 1,
  "architecture": "decoder-only-transformer",
  "model": {
    "vocab_size": 32000,
    "d_model": 2048,
    "n_layers": 32,
    "n_heads": 16,
    "n_kv_heads": 4,
    "max_seq_len": 4096,
    "intermediate_size": 5376,
    "rope_theta": 10000.0,
    "norm_eps": 0.00001,
    "tie_embeddings": false
  },
  "parameter_count": 1523714048,
  "tokenizer": {
    "file": "tokenizer.model",
    "sha256": "<sha256>",
    "vocab_size": 32000
  },
  "weights": {
    "file": "model.safetensors",
    "format": "safetensors"
  }
}
```

## Loader requirements

A compatible runtime must:

1. Parse `config.json` and reject unknown format identifiers or unsupported major format versions.
2. Resolve artifact filenames relative to the artifact directory; reject absolute paths and path traversal.
3. Compute SHA-256 for `model.safetensors` and `tokenizer.model`, then compare with `artifact-manifest.json` and `config.json` before loading.
4. Verify tokenizer vocabulary size equals `model.vocab_size`.
5. Instantiate the decoder architecture with the exact configuration and load SafeTensors weights strictly; missing or unexpected parameter names are fatal.
6. Keep model weights and tokenizer immutable after validation.

The architecture state-dictionary names are the names emitted by `arachne_small.model.ArachneSmall`. The runtime must implement the same RMSNorm, RoPE, grouped-query causal attention, SwiGLU, residual ordering, embedding policy, and parameter names.

## Training checkpoints

Files under `checkpoints/` are resumable training checkpoints containing optimizer/scaler state. They are trusted local training artifacts and use PyTorch's checkpoint format. They are not inference artifacts and must not be accepted by an untrusted model-serving endpoint.

## Compatibility policy

The format version changes only when the artifact schema or model-loading contract changes incompatibly. Additive manifest metadata may be introduced without changing the version if existing loaders can safely ignore it. A runtime must fail closed on unsupported versions; it must never guess tensor shapes or silently remap model weights.
