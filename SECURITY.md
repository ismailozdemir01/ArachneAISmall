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
