# ArachneAI Small — Model Card

## Identity

- **Name:** ArachneAI Small
- **Purpose:** From-scratch Turkish and English decoder-only language model research and training.
- **Architecture:** Decoder-only Transformer with RMSNorm, RoPE, grouped-query causal attention and SwiGLU.
- **Target scale:** 1,523,714,048 parameters.
- **Initialization:** Random initialization; no third-party pretrained model is used as a base.
- **Tokenizer:** SentencePiece BPE, trained on the authorized training split.
- **Artifact format:** Arachne Small format v1; SafeTensors weights.

## Release status

The repository contains model and training implementation code. It does not currently contain trained weights or a released trained checkpoint. A configuration describing a 1.5B-parameter model is not evidence that the model has been trained.

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
