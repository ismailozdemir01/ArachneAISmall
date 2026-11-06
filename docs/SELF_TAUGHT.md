# Autonomous Self-Taught Learning

## Runtime behavior

The `arachne-small serve` command enables the self-taught worker by default. It runs in the background and does not block API startup. Set `ARACHNE_SMALL_SELF_TAUGHT=0` before starting the server to disable it. Direct ASGI launches can enable it with `ARACHNE_SMALL_SELF_TAUGHT=1`.

The first cycle starts from random weights; it does not download an arbitrary pretrained checkpoint. It now uses the independent global research engine rather than treating Wikimedia as its training provider:

1. Searches the public web using broad multilingual research queries (customizable through `ARACHNE_RESEARCH_QUERIES`).
2. Follows discovered HTTPS pages across different domains, observes each site's `robots.txt`, uses a descriptive User-Agent, limits response sizes and applies per-host request pacing.
3. Extracts readable HTML text and source metadata without executing scripts, downloading binaries, logging in, or running page-provided code.
4. Records canonical URL, domain, title, retrieval time, content hash, and the page's explicit reuse license.
5. Adds content to the training corpus only when the page declares a license on the explicit CC0/CC BY/CC BY-SA allowlist. Public visibility alone is not treated as permission to train.
6. Rejects pages without an approved license, pages with obvious credential/contact patterns, malformed/oversized responses, and pages disallowed by robots policy.
7. Deduplicates and splits eligible documents before training the tokenizer.
8. Trains a SentencePiece tokenizer and a conservative, from-scratch bootstrap Transformer.
9. Promotes a candidate only after artifact verification and held-out evaluation succeed.

The global research crawler is deliberately provider-independent at the page-fetching layer. When a web-search cycle finds no explicitly licensed training documents, the worker makes a bounded fallback attempt against Wikimedia's REST APIs, verifies each page's CC BY-SA license and excludes living-person category pages. Wikimedia is therefore a fallback source, not the sole research provider.

The initial bootstrap profile is intentionally small (128 hidden dimensions, 4 layers, 256-token context; approximately 1.8 million parameters depending on tokenizer vocabulary). It is a functioning training bootstrap, not the configured 1.5B target model and not a claim of assistant-level capability.
## Continual loop

After bootstrap, the worker repeats acquisition and training every six hours by default. A failed cycle is retried after 15 minutes (or the configured interval if shorter). It reuses the current model's tokenizer and architecture, initializes candidate weights from the current verified release, appends only the current cycle's training split to a persistent deduplicated replay corpus, and evaluates baseline and candidate on that cycle's fresh held-out validation split. No cycle's validation documents are ever added to gradient training. A candidate is promoted only when validation loss is no worse than the baseline. Promotion is content-addressed and atomic. The server loads the promoted model only after successful validation; failed acquisition, training, evaluation or activation leaves the current model active.

The worker never trains on raw chat traffic or its own generated answers. User corrections remain on the separate explicit-consent and review workflow. This avoids silently harvesting conversations and reduces self-reinforcing synthetic-data errors.

## Files

- `data/self_taught/cycle-*/global-research.jsonl`: eligible public-web text with per-page provenance, used for model training when found.
- `data/self_taught/cycle-*/wikipedia-licensed.jsonl`: verified Wikimedia CC BY-SA fallback corpus, used only when the global web cycle finds no eligible training documents.
- `data/self_taught/seen-wikipedia-pages.sqlite3`: cross-cycle Wikimedia page-ID and content-hash deduplication index.
- `data/self_taught/cycle-*/research-index.jsonl`: source metadata and short excerpts for discovered pages, including pages excluded from training; this is a retrieval/source index, not training data.
- `data/self_taught/seen-global-sources.sqlite3`: cross-cycle URL and content-hash deduplication for training-eligible pages.
- `data/self_taught/cycle-*/split/`: per-cycle deduplicated train/validation documents and manifest.
- `data/self_taught/replay/train.jsonl`: cumulative training replay corpus, including new licensed pages from successful acquisition cycles.
- `data/self_taught/cycle-*/split/validation.jsonl` and `data/self_taught/cycle-*/validation.bin`: fresh per-cycle holdout, never used for gradient updates.
- `data/self_taught/seen-wikipedia-pages.sqlite3`: cross-cycle Wikimedia page-ID and content-hash deduplication index.
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
| `ARACHNE_SMALL_SELF_TAUGHT_ARTICLES_PER_LANGUAGE` | `40` | Legacy-compatible setting used to derive a bounded global page budget; 20–500 |
| `ARACHNE_SEARCH_ENDPOINTS` | DuckDuckGo HTML + Mojeek | Optional HTTPS search endpoint URLs separated by `||` |
| `ARACHNE_RESEARCH_QUERIES` | Built-in multilingual topics | Optional research queries separated by `||`; at most 20 are used per cycle |
| `ARACHNE_SMALL_SELF_TAUGHT_DATA` | `data/self_taught` | Local acquisition and corpus directory |
| `ARACHNE_SMALL_SELF_TAUGHT_ARTIFACTS` | `artifacts/self_taught` | Local candidate model and checkpoint directory |
| `ARACHNE_SMALL_RELEASE_DIR` | `artifacts/releases` | Content-addressed active model releases |

The status endpoint `GET /api/status` includes `self_taught.enabled`, `self_taught.state`, `self_taught.last_cycle` and `self_taught.last_error`.

## License and source limitations

The research engine searches across public websites and records source metadata in a local research index, but it does not assume that public accessibility grants training rights. Pages without an approved license can remain as source references/snippets in the research index; they are never copied into the gradient-training corpus. A page enters the model-training corpus only when its own HTML declares an exact license URL on the explicit CC0, CC BY or CC BY-SA allowlist. Pages without machine-verifiable reuse metadata remain excluded from training. The crawler records the source URL, canonical URL, domain, license and retrieval timestamp.

This automated metadata check is a conservative technical filter, not a legal opinion or guarantee that every page element has the same rights status. Before distributing trained weights or a service trained on this corpus, review applicable attribution, ShareAlike, privacy, and model-output obligations for the intended jurisdiction and distribution model.
## Operational boundaries

- Discovery uses public web search results; page collection is restricted to HTTPS, robots-permitted HTML pages.
- The crawler applies per-host request pacing, response-size limits, bounded page counts and a descriptive User-Agent.
- Only text is extracted; images, media, executables and arbitrary binary downloads are not collected.
- Scripts and page-provided code are never executed. The worker does not log in, bypass access controls, install external modules, or fetch third-party model checkpoints.
- Training data requires an explicit allowlisted reuse license. Unlicensed or unverifiable pages cannot enter gradient training.
- Every acquisition batch is split at document level. Only its training split is appended to the cumulative replay corpus; its fresh validation split is used for that cycle's paired baseline/candidate evaluation and is never used for gradient updates.
- A rejected candidate never replaces the active model.
- Failed acquisition/training cycles are retried; successful cycles use the configured repeat interval.
- The bootstrap model is deliberately small. Moving to the 1.5B configuration requires suitable compute and a separate measured scaling/training plan; the worker does not silently claim or fabricate a 1.5B model.
