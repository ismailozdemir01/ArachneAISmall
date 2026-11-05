# Autonomous Self-Taught Learning

## Runtime behavior

The `arachne-small serve` command enables the self-taught worker by default. It runs in the background and does not block API startup. Set `ARACHNE_SMALL_SELF_TAUGHT=0` before starting the server to disable it. Direct ASGI launches can enable it with `ARACHNE_SMALL_SELF_TAUGHT=1`.

The first cycle starts from random weights; it does not download an arbitrary pretrained checkpoint. It:

1. Queries Turkish and English Wikipedia's public REST APIs for random article summaries.
2. Retrieves each page's own REST metadata and accepts only pages whose reported license URL is CC BY-SA 3.0 or CC BY-SA 4.0.
3. Records the page URL, history URL, page/revision IDs, language, license and retrieval timestamp with each document.
4. Excludes pages categorized as living people and documents containing obvious credential/contact patterns.
5. Deduplicates and splits documents before training the tokenizer.
6. Trains a SentencePiece tokenizer and a conservative, from-scratch bootstrap Transformer.
7. Writes `arachne_small.pt` immediately after model initialization, updates it at checkpoint intervals, and marks it `trained` only after final validation succeeds.
8. Creates a SafeTensors inference artifact and promotes the first completed model to the local release registry.

The initial bootstrap profile is intentionally small (128 hidden dimensions, 4 layers, 256-token context; approximately 1.8 million parameters depending on tokenizer vocabulary). It is a functioning training bootstrap, not the configured 1.5B target model and not a claim of assistant-level capability.

## Continual loop

After bootstrap, the worker repeats acquisition and training every six hours by default. It reuses the current model's tokenizer and architecture, initializes candidate weights from the current verified release, appends newly acquired documents to a persistent deduplicated replay corpus, and evaluates baseline and candidate on the fixed held-out validation set created during bootstrap. The validation set is never added to training. A candidate is promoted only when validation loss is no worse than the baseline. Promotion is content-addressed and atomic. The server loads the promoted model only after successful validation; failed acquisition, training, evaluation or activation leaves the current model active.

The worker never trains on raw chat traffic or its own generated answers. User corrections remain on the separate explicit-consent and review workflow. This avoids silently harvesting conversations and reduces self-reinforcing synthetic-data errors.

## Files

- `data/self_taught/cycle-*/wikipedia.jsonl`: acquired text with per-page provenance.
- `data/self_taught/cycle-*/split/`: per-cycle deduplicated train/validation documents and manifest.
- `data/self_taught/replay/train.jsonl`: cumulative training replay corpus, including new licensed pages from successful acquisition cycles.
- `data/self_taught/replay/validation.jsonl` and `validation.bin`: fixed bootstrap holdout, never used for gradient updates.
- `data/self_taught/seen-pages.sqlite3`: cross-cycle page-ID and content-hash deduplication index.
- `artifacts/self_taught/cycle-*/arachne_small.pt`: PyTorch model checkpoint.
- `artifacts/self_taught/cycle-*/checkpoints/`: optimizer-aware resume checkpoints.
- `artifacts/releases/current.json`: active content-addressed release pointer.
- `artifacts/releases/<sha256>/arachne_small.pt`: promoted portable PyTorch checkpoint.
- `artifacts/releases/<sha256>/`: verified SafeTensors inference artifact.

These directories are local runtime data and must not be committed to Git.

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `ARACHNE_SMALL_SELF_TAUGHT` | enabled by `arachne-small serve` | Set to `0` to disable the worker |
| `ARACHNE_SMALL_SELF_TAUGHT_INTERVAL` | `21600` | Seconds between completed cycles; bounded to 30 minutes–7 days |
| `ARACHNE_SMALL_SELF_TAUGHT_ARTICLES_PER_LANGUAGE` | `40` | Per-cycle target per language; 20–500 |
| `ARACHNE_SMALL_SELF_TAUGHT_DATA` | `data/self_taught` | Local acquisition and corpus directory |
| `ARACHNE_SMALL_SELF_TAUGHT_ARTIFACTS` | `artifacts/self_taught` | Local candidate model and checkpoint directory |
| `ARACHNE_SMALL_RELEASE_DIR` | `artifacts/releases` | Content-addressed active model releases |

The status endpoint `GET /api/status` includes `self_taught.enabled`, `self_taught.state`, `self_taught.last_cycle` and `self_taught.last_error`.

## License and source limitations

Wikimedia page metadata is checked individually; public visibility alone is not treated as permission. Wikimedia documents commonly use CC BY-SA 3.0 or 4.0, but page-level licensing and attribution obligations must be preserved. Every record keeps a page link and history link. The crawler rejects pages whose returned license is missing or outside the explicit allowlist.

Wikimedia's own documentation warns that license and attribution metadata can be incomplete. This automated check is a technical filter, not a legal opinion or guarantee. Before distributing trained weights or a service trained on this corpus, review applicable attribution, ShareAlike, privacy, and model-output obligations for the intended jurisdiction and distribution model. See [Wikimedia content reuse](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse), [REST page metadata](https://www.mediawiki.org/wiki/API:REST_API/Reference), and [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Operational boundaries

- Network collection is limited to the two Wikimedia Wikipedia REST/API origins and uses a descriptive User-Agent and a one-request-per-second throttle.
- Only text extracts are collected; images and media are not downloaded.
- The worker does not execute downloaded code, install external modules, or fetch third-party model checkpoints.
- The bootstrap data is split at document level. The validation set remains fixed and is never used for gradient updates. Later acquisitions are appended only to the cumulative training replay corpus.
- Page IDs and normalized document content are deduplicated across cycles to reduce validation contamination.
- A rejected candidate never replaces the active model.
- The bootstrap model is deliberately small. Moving to the 1.5B configuration requires suitable compute and a separate measured scaling/training plan; the worker does not silently claim or fabricate a 1.5B model.
