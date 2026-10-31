# Controlled Continual Learning

ArachneAISmall supports an explicit-consent feedback path and a gated incremental-training cycle. It does not train on arbitrary inference traffic and does not update the active model while users are chatting.

## 1. Collect feedback with consent

Feedback is stored locally in SQLite. Consent is mandatory; prompt, model response, user correction, rating, and language are validated. Exact duplicate submissions are idempotent. Nothing is uploaded.

```bash
arachne-small feedback-add --database data/feedback.sqlite3 --consent \\
  --prompt "..." --response "..." --correction "..." --rating 5 --language tr
```

Only consented entries rated at or above the configured threshold are eligible. Corrections, not the model's original answer, become the target text. The response is retained for audit context but is not treated as ground truth.

## 2. Run a continual-learning cycle

The base artifact must already be trained, complete, and verifiable. The fixed validation token stream must have been prepared using the exact tokenizer bundled in that artifact.

```bash
arachne-small learn-cycle \\
  --database data/feedback.sqlite3 \\
  --base-artifact artifacts/arachne-small-1.5b \\
  --config configs/1_5b.yaml \\
  --validation-data data/processed/validation.bin \\
  --work-dir artifacts/learning-cycle \\
  --release-dir artifacts/releases
```

The command exports newly eligible consented feedback with hashes, encodes it with the base artifact tokenizer, initializes from verified base weights (not random weights), trains a candidate with a fresh optimizer, evaluates baseline and candidate on the same validation stream, and rejects any candidate whose validation loss exceeds the baseline plus `--max-loss-regression` (default `0.0`). Only a verified candidate that passes the gate is copied to an immutable content-addressed release directory and made current through an atomic pointer update. Feedback is marked consumed only after successful promotion; failed cycles can be retried.

For manual inspection, use `feedback-export`, `evaluate-artifact`, and `promote-candidate` separately. `rollback-release` restores a verified previous version or a specific content-hash version.

## 3. Rollback

```bash
arachne-small rollback-release --release-dir artifacts/releases
```

The release pointer is an atomic JSON file. Rollback validates the target artifact before changing the pointer. The inference server currently loads the artifact configured by `ARACHNE_SMALL_ARTIFACT` at process startup; point that variable at the selected immutable release directory and restart the server to activate it.

## Operational constraints

- No hidden telemetry, automatic raw-prompt harvesting, or network upload is implemented.
- Consent is explicit and local; operators remain responsible for lawful basis, retention, deletion, and access controls on the SQLite database.
- High user ratings are a selection signal, not proof of factual correctness. Review feedback quality and dataset contamination before running a cycle.
- A single validation loss is a narrow regression gate, not a complete safety, factuality, bias, or capability evaluation.
- Training is compute-intensive, synchronous, and requires enough data for at least one complete model sequence block. It does not schedule background jobs or train while serving.
- A passed candidate is promoted in the release registry; the running API process is not hot-swapped.
- No trained base weights are committed. The pipeline becomes operational only after a verified base artifact and fixed validation set exist.
