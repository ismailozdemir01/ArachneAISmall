# Inference API

## Runtime lifecycle

The ASGI application is exposed as `arachne_small.server:app`. The console command `arachne-small serve` starts Uvicorn. At startup, the server reads `ARACHNE_SMALL_ARTIFACT`, verifies the complete artifact manifest, checks SafeTensors tensor names and shapes, validates tokenizer metadata, and loads the model onto CUDA when available or CPU otherwise.

If the environment variable is absent, the HTTP process remains available for status and UI inspection, but generation is unavailable. It never fabricates a response or silently initializes random weights for inference.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARACHNE_SMALL_ARTIFACT` | unset | Path to a completed, verified artifact directory |
| `ARACHNE_SMALL_HOST` | `127.0.0.1` | Bind address |
| `ARACHNE_SMALL_PORT` | `8000` | TCP port |
| `ARACHNE_SMALL_ALLOW_REMOTE` | `0` | Must equal `1` to permit non-loopback binding |
| `ARACHNE_SMALL_FEEDBACK_DB` | `data/feedback.sqlite3` | Local SQLite file for explicitly consented correction feedback |

Loopback binding is the safe default. Enabling remote binding does not add application authentication. Put the service behind an authenticated TLS reverse proxy, restrict network access, and apply request-rate and connection limits before exposing it to other machines or the public internet. Do not expose the development server directly to the internet.

## Endpoints

### `GET /api/health`

Returns process health. This endpoint does not claim that model weights are loaded.

### `GET /api/status`

Returns:
- `ready`: whether a verified model and tokenizer are loaded.
- `state`: `ready`, `artifact_missing`, or `error`.
- `device` and `cuda_available`.
- model version and artifact directory name (the absolute local path is not disclosed).
- parameter count, vocabulary size and context length when loaded.

### `POST /api/generate`

Request JSON:

```json
{
  "prompt": "Merhaba, kendini tanıt.",
  "history": [
    {"role": "user", "content": "Merhaba."},
    {"role": "assistant", "content": "Merhaba, nasıl yardımcı olabilirim?"}
  ],
  "max_new_tokens": 256,
  "temperature": 0.8,
  "top_k": 50,
  "language": "auto"
}
```

Constraints:
- prompt: non-blank, at most 20,000 characters.
- max_new_tokens: 1–1024.
- temperature: 0.05–2.0.
- top_k: 0–500; zero disables top-k filtering.
- language: `auto`, `tr`, or `en`.
- history: up to 20 previous `user`/`assistant` messages, each at most 4,000 characters and 20,000 characters total. System/developer roles are not accepted.

Successful response:

```json
{
  "model": "arachne-small",
  "text": "Generated model output",
  "prompt_tokens": 12,
  "completion_tokens": 42,
  "finish_reason": "length"
}
```

HTTP 503 means no trained artifact is loaded or inference is temporarily unavailable. HTTP 422 indicates invalid request data. Unexpected internal errors are logged server-side; raw exception details are not returned to the client.

### `POST /api/feedback`

Stores a correction only when the request explicitly sets `consent: true`. Feedback remains in the local SQLite database configured by `ARACHNE_SMALL_FEEDBACK_DB`; the endpoint does not transmit it elsewhere.

Request JSON:

```json
{
  "consent": true,
  "prompt": "Original user prompt",
  "response": "Model response",
  "correction": "User-provided corrected answer",
  "rating": 5,
  "language": "tr"
}
```

The API validates text length, language, rating, obvious credential/contact-data patterns, and whether the correction provides a distinct learning signal. Duplicate records are idempotent. A successful response returns a local feedback record ID. No feedback is stored without explicit consent.

## Operational behavior

- Artifact integrity is verified before model loading.
- The model and tokenizer are loaded once during application startup.
- A process-wide re-entrant lock serializes generation and protects the single model instance.
- Prompt input is bounded at the API boundary and tokenized input is truncated to the model context window.
- The browser UI inserts user and model text using DOM text nodes, not HTML interpretation.
- The UI sends a bounded previous-turn history with each generation request; history is held in browser memory and cleared by starting a new chat. No conversation content is persisted on the server or across page reloads.
- The API currently provides synchronous generation, not token streaming or multi-user scheduling.

A trained artifact is a prerequisite for inference. A configured architecture or an untrained checkpoint must not be represented as a released trained model.
