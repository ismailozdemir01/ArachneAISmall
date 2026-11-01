# Security Policy

## Scope

This repository contains model architecture, dataset processing, training, checkpoint, and artifact-verification code.

## Reporting a vulnerability

Please do not publish exploitable vulnerability details in a public issue. Use GitHub's private vulnerability reporting feature for this repository when it is enabled. Include affected commit/version, reproduction steps, impact, and any relevant logs with secrets and personal data removed.

## Dataset and artifact handling

- Never commit credentials, private corpora, personal data, or customer content.
- Treat training checkpoints as trusted local files; do not load checkpoints supplied by untrusted parties.
- Prefer SafeTensors for inference weights.
- Verify artifact checksums and schema before loading a model.
- Keep model-serving endpoints and training hosts isolated according to their deployment threat model.
- Remote serving must use `ARACHNE_SMALL_ALLOW_REMOTE=1` together with a unique `ARACHNE_SMALL_API_TOKEN` of at least 32 characters. Inference and feedback write endpoints reject requests without a valid Bearer token.
- Never commit API tokens, feedback databases, feedback exports, checkpoints, or private corpora. Restrict feedback storage and backups to authorized operators.
- Review and explicitly approve each feedback record before it can enter a training export. Treat user corrections as untrusted data even after consent; inspect them for quality, licensing, privacy, and prompt-injection content before approval.
