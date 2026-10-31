# Contributing

ArachneAISmall is the from-scratch model research and training component of the ArachneAI ecosystem.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Contribution requirements

- Keep model architecture and artifact-format changes synchronized with `docs/ARTIFACT_FORMAT.md`.
- Add focused tests for behavior changes; run them locally before submitting a pull request.
- Do not add mock model weights, placeholder training logic, fabricated benchmark results, or unlicensed corpora.
- Never commit generated datasets, token streams, checkpoints, or model weights.
- Document any new configuration fields and their validation constraints.
- Preserve strict tokenizer and artifact checksum verification.
- Do not add CI workflows or trigger GitHub Actions without explicit maintainer approval.

## Pull requests

Describe the technical change, compatibility impact, local test commands and actual results. Do not claim tests passed unless they were executed.
