# Verified module discovery and activation

ArachneAISmall supports self-generated and externally discovered modules through a deliberately constrained declarative format.

## Trust boundary

Modules are JSON operation pipelines, not Python plugins. The runtime does not import downloaded code, evaluate expressions, execute shell commands, install packages, or grant modules filesystem/network access. The initial operation set is intentionally small: `trim`, `lowercase`, `uppercase`, `prepend`, `append`, and literal string replacement.

This boundary is intentional. Running arbitrary third-party Python inside the model server would turn a discovered module into arbitrary code execution. A Python subprocess is not a security sandbox, so this implementation does not pretend that AST checks or a subprocess alone can make arbitrary code safe.

## Module manifest

A module JSON document contains exactly these fields:

- `id`: lowercase identifier, 3-64 characters
- `version`: semantic version `MAJOR.MINOR.PATCH`
- `description`: 1-500 characters
- `operations`: 1-32 allowlisted operations
- `tests`: 2-100 deterministic input/expected-output vectors

The validator rejects unknown fields, unsupported operations, malformed arguments, excessive inputs/outputs and failed test vectors. A candidate is content-addressed with SHA-256. Activation rechecks the digest, schema and all behavior vectors, writes an immutable content-addressed module file, then atomically switches its active pointer.

The test vectors demonstrate the module's declared behavior; they are not a proof that a module is universally correct. External modules must be treated as untrusted even when they come from a known repository.

## Internet discovery

Only operator-configured registry index URLs are polled. The only accepted host is `raw.githubusercontent.com`, over HTTPS, without redirects, credentials, query strings or fragments. Indexes are bounded to 1 MiB, module documents to 256 KiB, and each registry to 100 entries. Each entry must include a SHA-256 matching the downloaded module document.

An index has this format:

```json
{
  "format": "arachne-module-index-v1",
  "modules": [
    {
      "url": "https://raw.githubusercontent.com/OWNER/REPO/0123456789abcdef0123456789abcdef01234567/modules/example.json",
      "sha256": "64-character-lowercase-hex-digest"
    }
  ]
}
```

Use immutable commit URLs in registry entries. The digest protects the downloaded document against a mismatch with the index; it does not independently prove that the publisher is trustworthy. Registry URLs are trust roots selected by the operator.

When `ARACHNE_SMALL_MODULE_REGISTRIES` contains semicolon-separated trusted index URLs, the API starts a periodic discovery worker. The scan interval is configured by `ARACHNE_SMALL_MODULE_SCAN_INTERVAL` in seconds and is clamped to 300-86400 seconds. Discovery runs in a worker thread, has network timeouts and size limits, and does not block inference. A failed scan leaves existing active modules unchanged.

## Self-generation and API

A trained Arachne model can propose a declarative module from a natural-language goal through `POST /api/modules/plan`. The generated response must be strict JSON and pass the same schema and deterministic behavior gates before activation. Operators or an external planner can submit a prebuilt manifest to `POST /api/modules/create`. Both endpoints are protected by the same Bearer-token middleware as inference and feedback when an API token is configured. Since the 1.5B model has not yet been trained, autonomous model-driven planning remains unavailable until a verified trained artifact is loaded.

- `GET /api/modules`: list active modules
- `POST /api/modules/plan`: ask the loaded model to propose, test and activate a declarative module
- `POST /api/modules/create`: validate and activate a supplied declarative module
- `POST /api/modules/discover`: discover from explicitly supplied trusted indexes
- `POST /api/modules/{module_id}/run`: execute one active module

CLI equivalents:

- `arachne-small module-create --spec module.json`
- `arachne-small module-discover --registry-url INDEX_URL`
- `arachne-small module-list`
- `arachne-small module-run --id MODULE_ID --input "text"`

Set `ARACHNE_SMALL_MODULE_DIR` to choose the module registry directory. The default is `data/modules`.

## Automatic reviewed-feedback training

The server can run the existing feedback fine-tuning/evaluation/release pipeline periodically. It is opt-in and requires a trained base artifact plus an exact training config and validation token stream:

- `ARACHNE_SMALL_AUTO_LEARN=1`
- `ARACHNE_SMALL_ARTIFACT=/path/to/verified/base-artifact`
- `ARACHNE_SMALL_TRAINING_CONFIG=/path/to/training.yaml`
- `ARACHNE_SMALL_VALIDATION_TOKENS=/path/to/validation.bin`
- Optional: `ARACHNE_SMALL_FEEDBACK_DB`, `ARACHNE_SMALL_RELEASE_DIR`, `ARACHNE_SMALL_LEARN_WORK_DIR`
- Optional controls: `ARACHNE_SMALL_LEARN_INTERVAL` (default 21600 seconds, clamped to 1800-604800), `ARACHNE_SMALL_LEARN_MIN_RATING` (default 4), `ARACHNE_SMALL_MAX_LOSS_REGRESSION` (default 0), `ARACHNE_SMALL_EVAL_BATCH_SIZE` (default 1), `ARACHNE_SMALL_EVAL_MAX_BATCHES` (default 200)

Only explicitly consented, reviewed and approved feedback is eligible. Each cycle uses a fresh work directory, starts from the latest verified release when one exists, evaluates against the same fixed validation stream, and promotes only when the configured loss gate passes. Failed cycles leave the active release pointer unchanged. The currently loaded inference process is not hot-swapped; restart the service to load a newly promoted artifact.

Automatic training is not started unless `ARACHNE_SMALL_AUTO_LEARN=1`. It cannot train the 1.5B model until a trained base artifact, suitable compute, training configuration and validation corpus are supplied.

## Current model limitation

The 1.5B parameter configuration is an architecture target, not a trained model. The module lifecycle can validate and activate constrained modules independently, but autonomous semantic module planning by the Arachne language model requires a trained model artifact and an explicitly enabled planner. Passing deterministic module tests does not establish broad semantic correctness.
