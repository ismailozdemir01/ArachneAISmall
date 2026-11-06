# ArachneAI Small — Model Card

## Identity

- **Name:** ArachneAI Small
- **Purpose:** From-scratch Turkish and English decoder-only language model research and training.
- **Architecture:** Decoder-only Transformer with RMSNorm, RoPE, grouped-query causal attention and SwiGLU.
- **Target scale:** 1,523,714,048 parameters in `configs/1_5b.yaml`; the autonomous first-run bootstrap is a separate approximately 1.8M-parameter model.
- **Initialization:** Random initialization; no third-party pretrained model is used as a base. The self-taught worker acquires licensed text and trains its bootstrap model from scratch.
- **Tokenizer:** SentencePiece BPE, trained on the authorized training split.
- **Artifact format:** SafeTensors inference artifact plus a portable `arachne_small.pt` weights-only PyTorch checkpoint; checkpoint loading uses `weights_only=True`.

## Release status

The repository contains model and training implementation code; trained weights are generated locally and are not committed. Running `arachne-small serve` starts the global research worker, which discovers public HTTPS sources across domains, records source provenance and trains only on pages with explicitly approved reuse licenses. A local run publishes `arachne_small.pt` only after a candidate has completed training, artifact verification, held-out evaluation and successful promotion. Failed acquisition or evaluation leaves the active model unchanged. The bootstrap is not the 1.5B target model. A configuration describing a 1.5B-parameter model is not evidence that the model has been trained.

A trained release must include actual training provenance, dataset/license records, completed-step count, validation metrics, tokenizer checksum, model-weight checksum, and hardware/training details. No quality, safety, reasoning, multilinguality, or benchmark claim should be made until measured on a documented evaluation suite.

## Intended use

The model is intended for research and development of Turkish-English language modeling and for use through a compatible ArachneAIOpen runtime after a trained artifact is produced and validated.

## Limitations

- Performance is unknown until training and evaluation are completed.
- The model may produce incorrect, fabricated, biased, or unsafe text.
- Training data quality, licensing, language balance, contamination, and representativeness directly affect results.
- The target configuration requires substantial compute and memory; hardware requirements must be measured on the actual training setup.
- This project does not provide a guarantee of suitability for high-impact decisions.

## Data and privacy

Only use training data with documented permission and a valid training-use basis. Keep raw corpora, personal data, credentials, and private datasets out of Git. Record dataset source, version, license, collection date, filtering, deduplication, and any consent or retention constraints. Remove sensitive data before tokenization.

## Evaluation required before a trained release

- Held-out Turkish and English cross-entropy/perplexity.
- Language-specific and mixed-language instruction/task evaluations, where applicable.
- Memorization and training-data contamination checks.
- Safety and refusal evaluations appropriate to the intended deployment.
- Robustness, long-context, and generation-quality evaluations.
- Hardware, throughput, memory, and inference latency measurements.


## Autonomous bootstrap status

The automatic bootstrap model uses a 128-dimensional hidden size, 4 Transformer layers, 4 attention heads and a 256-token context. Its exact parameter count depends on the tokenizer vocabulary (configured target 4,000). It is intended to establish a real end-to-end training loop, not to claim production assistant capability or 1.5B-scale quality. The worker repeats acquisition and candidate training every six hours by default and promotes only candidates that do not regress on the cycle's held-out validation set.

The training corpus is acquired from public web pages whose HTML declares an exact CC0, CC BY or CC BY-SA license URL on the crawler's allowlist. Canonical URL, domain, license, retrieval time and content hash are retained locally. Discovered pages without an approved license are excluded from gradient training. The automated filter is not a legal determination; see [Dataset Policy](DATASET_POLICY.md) and [Self-Taught Learning](SELF_TAUGHT.md).
