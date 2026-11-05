# ArachneAISmall

<p align="center">
  <strong>Türkçe / English</strong><br>
  Sıfırdan Türkçe–İngilizce dil modeli geliştirme, eğitim ve güvenli yerel çalışma altyapısı.<br>
  From-scratch Turkish–English language-model engineering, training, and secure local runtime.
</p>

[🇹🇷 Türkçe](#türkçe) · [🇬🇧 English](#english) · [Kurulum / Installation](KURULUM.md) · [Model Kartı](docs/MODEL_CARD.md)

> **Güncel sürüm durumu / Current release status**
>
> **TR:** Git deposunda eğitilmiş 1.5B model ağırlıkları yayımlanmaz. `arachne-small serve` otomatik self-taught işçisini başlatır; Türkçe/İngilizce Wikipedia sayfalarını sayfa bazında lisans kontrolüyle toplar ve yaklaşık 1.8M parametreli küçük bootstrap modeli sıfırdan eğiterek `arachne_small.pt` üretir. Bu model 1.5B hedefi değildir. Model ancak eğitim, doğrulama ve yayınlama tamamlanınca kullanıma açılır.
>
> **EN:** The source code, training pipeline, API, and web interface are available. The Git repository does not ship trained weights. Running `arachne-small serve` starts an autonomous self-taught worker that acquires page-level license-checked Turkish/English Wikipedia text, trains a small bootstrap model from random initialization, and writes `arachne_small.pt`. The initial bootstrap is approximately 1.8M parameters, not the 1.5B target. Generation becomes ready only after training, validation and verified promotion complete. See [Autonomous Self-Taught Learning](docs/SELF_TAUGHT.md).

---

## Türkçe

**ArachneAISmall**, ArachneAI ekosistemi için sıfırdan geliştirilen, Türkçe ve İngilizce odaklı bir dil modeli mühendisliği projesidir. Veri hazırlama, tokenizer eğitimi, Transformer eğitimi, model artifact doğrulama, yerel API/web arayüzü, rızaya dayalı geri bildirim, kontrollü sürekli öğrenme ve güvenlik sınırları belirlenmiş bildirimsel modül yönetimini kapsar.

### Özellikler ve kabiliyetler

- **Model mimarisi:** PyTorch tabanlı decoder-only Transformer; RMSNorm, RoPE, Grouped-Query Attention (GQA), SwiGLU ve residual bağlantılar.
- **Hedef ölçek:** `configs/1_5b.yaml` içinde yapılandırılmış 1.523.714.048 parametreli 1.5B sınıfı hedef mimari.
- **Veri hattı:** UTF-8 JSONL içe aktarma, kaynak/lisans kaydı, Unicode ve boşluk normalizasyonu, disk tabanlı mükerrer temizliği, deterministik eğitim/doğrulama ayrımı.
- **Tokenizer:** SentencePiece BPE eğitimi; tokenizer kimliği ve özel token doğrulaması.
- **Token hazırlama:** Little-endian uint32 token akışları, SHA-256 checksum ve kaynak metaverisi.
- **Eğitim:** AdamW, gradient accumulation, gradient clipping, warmup/cosine öğrenme oranı, doğrulama değerlendirmesi, isteğe bağlı gradient checkpointing ve precision ayarları.
- **Checkpoint yönetimi:** Atomik checkpoint yazımı, doğrulanmış devam eğitimi ve doğrulanmış artifact üzerinden yeni optimizer ile devam.
- **Artifact güvenliği:** SafeTensors, sürümlü manifest, SHA-256, tokenizer/tensor şekli/parametre sayısı doğrulaması; yol geçişi ve sembolik bağlantı kontrolleri.
- **Yerel çalışma zamanı:** FastAPI + Uvicorn, Türkçe/İngilizce web arayüzü, OpenAPI dokümantasyonu ve model durum uç noktaları.
- **Metin üretimi:** Yalnızca doğrulanmış model artifact yüklendiğinde; token sayısı, temperature, top-k, dil ve konuşma geçmişi sınırlarıyla.
- **Bildirimsel modüller:** Şema ve semantik sürüm kontrolü, deterministik test vektörleri, aday doğrulama, etkinleştirme, listeleme, çalıştırma ve güvenilir registry keşfi.
- **Modül güvenliği:** Sınırlı metin dönüşüm işlemleri; keyfî Python, shell, import veya `eval` çalıştırma yoktur. Registry kaynakları HTTPS, değişmez Git commit URL'si ve SHA-256 ile doğrulanır.
- **Geri bildirim yönetişimi:** Açık kullanıcı rızası, yerel SQLite saklama, operatör onayı/reddi ve bütünlük metaverili dışa aktarma.
- **Kontrollü sürekli öğrenme:** Onaylanmış geri bildirimlerle aday ince ayarı, sabit doğrulama akışında baseline karşılaştırması, yayın kapısı ve doğrulanmış rollback.
- **Güvenlik ve gizlilik:** Varsayılan loopback erişimi, uzaktan erişimde açık opt-in ve Bearer token, istek boyutu sınırları, güvenli artifact yükleme ve yerel geri bildirim saklama.

### Önemli model durumu

<strong>1.5B, yapılandırılmış hedef mimari ölçeğidir; eğitilmiş ve yayımlanmış bir model olduğu anlamına gelmez.</strong> Depoda eğitim ve çalışma zamanı yazılımı bulunur; eğitilmiş ağırlıklar ve eğitim korpusu yayımlanmış değildir. Bu nedenle doğrulanmış model artifact'i sağlanmadan servis ve web arayüzü açılır, ancak metin üretimi kullanılamaz ve API HTTP 503 döndürür. Sistem rastgele ağırlıklarla sahte yanıt üretmez.

### Hızlı başlatma

```powershell
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
arachne-small serve
```

Tarayıcı: http://127.0.0.1:8000  
Sağlık: http://127.0.0.1:8000/api/health  
Durum: http://127.0.0.1:8000/api/status  
API: http://127.0.0.1:8000/docs

Windows ve Linux/WSL için PyTorch kurulumu, model artifact ayarı ve ayrıntılı komutlar: [KURULUM.md](KURULUM.md).

---

## English

**ArachneAISmall** is a from-scratch Turkish–English language-model engineering project in the ArachneAI ecosystem. It covers data preparation, tokenizer training, Transformer training, model-artifact verification, a local API/web interface, consent-based feedback, controlled continual learning, and a security-constrained declarative module system.

### Features and capabilities

- **Model architecture:** PyTorch decoder-only Transformer with RMSNorm, RoPE, Grouped-Query Attention (GQA), SwiGLU, and residual connections.
- **Target scale:** A 1.5B-class target architecture configured for 1,523,714,048 parameters in `configs/1_5b.yaml`.
- **Data pipeline:** UTF-8 JSONL ingestion, source/license provenance, Unicode and whitespace normalization, disk-backed deduplication, and deterministic train/validation splits.
- **Tokenizer:** SentencePiece BPE training with tokenizer identity and special-token validation.
- **Token preparation:** Little-endian uint32 token streams with SHA-256 checksums and provenance metadata.
- **Training:** AdamW, gradient accumulation and clipping, warmup/cosine learning-rate scheduling, validation evaluation, optional gradient checkpointing, and precision settings.
- **Checkpoint management:** Atomic checkpoint writes, validated resume, and continued training from a verified artifact with a fresh optimizer.
- **Artifact security:** SafeTensors, versioned manifests, SHA-256, tokenizer/tensor-shape/parameter-count validation, path-traversal and symlink checks.
- **Local runtime:** FastAPI + Uvicorn, Turkish/English browser interface, OpenAPI documentation, and model-status endpoints.
- **Text generation:** Available only when a verified model artifact is loaded; request controls bound token count, temperature, top-k, language, and conversation history.
- **Declarative modules:** Schema and semantic-version validation, deterministic test vectors, candidate verification, activation, listing, execution, and trusted-registry discovery.
- **Module security:** Bounded text transformations only; no arbitrary Python, shell, import, or `eval` execution. Registry sources require HTTPS, immutable Git commit URLs, and SHA-256 verification.
- **Feedback governance:** Explicit user consent, local SQLite storage, operator approval/rejection, and export with integrity metadata.
- **Controlled continual learning:** Candidate fine-tuning from approved feedback, baseline comparison on a fixed validation stream, promotion gates, and verified rollback.
- **Security and privacy:** Loopback access by default, explicit opt-in and Bearer-token authentication for remote access, bounded requests, safe artifact loading, and local feedback storage.

### Important model status

<strong>1.5B describes the configured target architecture; it does not mean a trained model has been released.</strong> The repository contains the training and runtime software, but trained weights and a training corpus have not been published. Without a verified model artifact, the service and web interface can start, but text generation remains unavailable and the API returns HTTP 503. The application does not generate fabricated responses from random weights.

### Quick start

```powershell
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
arachne-small serve
```

Web interface: http://127.0.0.1:8000  
Health: http://127.0.0.1:8000/api/health  
Runtime status: http://127.0.0.1:8000/api/status  
API documentation: http://127.0.0.1:8000/docs

See [KURULUM.md](KURULUM.md) for the complete Windows and Linux/WSL setup, PyTorch installation, artifact configuration, and operational commands. The detailed English technical reference follows.

---

## Contents

- [Project scope](#project-scope)
- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Corpus and tokenizer pipeline](#corpus-and-tokenizer-pipeline)
- [Training and checkpoints](#training-and-checkpoints)
- [Artifact verification and inference](#artifact-verification-and-inference)
- [HTTP API](#http-api)
- [Declarative module system](#declarative-module-system)
- [Consent-based feedback and continual learning](#consent-based-feedback-and-continual-learning)
- [CLI reference](#cli-reference)
- [Security and privacy](#security-and-privacy)
- [Configuration](#configuration)
- [Tests and release status](#tests-and-release-status)
- [Repository boundaries](#repository-boundaries)
- [Documentation](#documentation)
- [License](#license)

## Project scope

ArachneAISmall is designed to provide a reproducible path from training data with documented usage rights to a versioned, verifiable language-model artifact. The model is trained from random initialization rather than adapted from a third-party pretrained checkpoint.

The project separates model development from runtime operation. Training produces a defined artifact contract; inference verifies that contract before loading weights. Feedback is not silently harvested from conversations, and it is not automatically treated as ground truth.

## Architecture

The configured decoder-only Transformer includes:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped-Query Causal Attention (GQA)
- SwiGLU feed-forward layers
- Residual connections
- Configurable model dimensions and sequence length
- Optional gradient checkpointing
- SentencePiece BPE tokenizer

Inspect the configured parameter count:

```bash
arachne-small count-parameters --config configs/1_5b.yaml
```

The 1.5B configuration is a compute-intensive target. Actual hardware requirements, throughput, quality, and language capabilities must be measured after a real training run.

## Capabilities

### Data governance and preprocessing

- Reads UTF-8 JSONL documents using a configurable text field.
- Records dataset source and declared license/permission information.
- Normalizes Unicode and whitespace.
- Deduplicates documents using disk-backed processing.
- Creates deterministic document-level training and validation splits.
- Exports text for tokenizer training.
- Trains SentencePiece BPE on the training split only.
- Encodes train and validation data into little-endian uint32 token streams.
- Writes SHA-256 checksums and tokenizer/provenance metadata beside token files.
- Validates token stream and tokenizer identity before training.

### Model training

- PyTorch decoder-only Transformer training.
- AdamW optimizer.
- Gradient accumulation and gradient clipping.
- Warmup and cosine learning-rate schedule.
- Validation evaluation at configured intervals.
- Optional gradient checkpointing and precision configuration.
- Atomic checkpoint writes and validated resume.
- Initialization from a verified artifact for continued training with a fresh optimizer.

### Artifact integrity and compatibility

- SafeTensors for inference weights; no pickle-based inference weight loading.
- Versioned artifact and manifest formats.
- SHA-256 verification for model configuration, tokenizer, and weights.
- Tokenizer vocabulary and special-token ID checks.
- Exact architecture parameter-count validation.
- Tensor-name and tensor-shape validation before loading.
- Strict state-dictionary loading.
- Path traversal and symbolic-link rejection for artifact files.
- Content-addressed promoted releases and verified rollback.

### Local inference and web interface

- FastAPI application and Uvicorn server.
- Turkish/English browser interface.
- Prompt and bounded conversation-history validation.
- Configurable generation length, temperature, top-k, and language hint.
- Runtime status including model readiness, device, CUDA availability, and model metadata.
- OpenAPI documentation at `/docs`.
- Explicit unavailable state when a verified artifact is absent; no placeholder model output.

### Declarative module system

ArachneAISmall can create, validate, activate, discover, list, and run small declarative text-processing modules.

- Strict JSON manifest schema and semantic version validation.
- Bounded module IDs, descriptions, operation counts, test counts, and input sizes.
- Supported operations: `trim`, `lowercase`, `uppercase`, `prepend`, `append`, and `replace_literal`.
- Deterministic input/expected test vectors are required before activation.
- Candidate modules are validated and tested before being activated.
- Content is identified with SHA-256.
- Registry discovery is restricted to HTTPS `raw.githubusercontent.com` sources using immutable 40-character Git commit paths.
- Redirects, untrusted hosts, oversized documents, malformed manifests, and missing/invalid source digests are rejected.
- A failed registry entry is audited and does not prevent later entries from being processed.
- Rejection records are appended to a local JSONL audit file; failed candidate artifacts are cleaned up when activation fails.
- Arbitrary Python, shell, imports, `eval`, or generated executable code are not part of the module format.

The module system is intentionally a bounded declarative transformation engine, not an unrestricted plugin runtime.

### Consent-based feedback and controlled continual learning

- Feedback is stored in a local SQLite database only after explicit user consent.
- Prompt, response, correction, rating, and language are validated.
- Obvious credential/contact-data patterns and low-signal duplicates are rejected.
- Operators can inspect pending records and explicitly approve or reject them.
- Review decisions are immutable; rejected records cannot be silently re-approved.
- Only reviewed, approved, consented corrections meeting the rating threshold can be exported.
- Exported records include integrity/provenance metadata.
- A candidate is fine-tuned from a verified base artifact, then evaluated against the baseline on the same fixed validation token stream.
- Promotion is blocked when the configured validation-loss regression gate fails.
- Successful releases are content-addressed and can be rolled back after verification.
- Failed cycles do not consume feedback records; feedback is marked consumed only after successful promotion.

This workflow does not train on arbitrary live chat traffic and does not hot-swap the model inside a running inference process.

## Installation

### Requirements

- Python 3.10 or newer
- Git
- PyTorch compatible with the target CPU or CUDA environment
- Authorized UTF-8 JSONL training data for training workflows

### Clone and create a virtual environment

**Windows PowerShell**

```powershell
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / WSL / macOS**

```bash
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for the intended hardware. For a CPU-only environment:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, select the command matching the installed driver and target platform at [the official PyTorch installation page](https://pytorch.org/get-started/locally/).

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "import torch, fastapi, sentencepiece, safetensors; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
arachne-small --help
```

For the complete Windows and Linux walkthrough, see [KURULUM.md](KURULUM.md).

## Corpus and tokenizer pipeline

Each source JSONL line must be an object containing a non-empty `text` field (or the field selected with `--text-field`):

```json
{"text":"A training document whose usage rights have been verified."}
```

Prepare a reproducible corpus split and record provenance:

```bash
arachne-small prepare-corpus \
  --input data/source.jsonl \
  --output-dir data/split \
  --validation-fraction 0.01 \
  --seed 1337 \
  --source "Dataset source and version" \
  --license "Exact license or written permission"
```

Export training text and train the tokenizer only on the training split:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000
```

Encode both splits with the same tokenizer:

```bash
arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

The pipeline writes sidecar metadata containing token counts, source and token checksums, tokenizer identity, vocabulary size, and special-token IDs. Training validates these files before optimization.

## Training and checkpoints

Train with the 1.5B target configuration:

```bash
arachne-small train \
  --config configs/1_5b.yaml \
  --train-data data/processed/train.bin \
  --validation-data data/processed/validation.bin \
  --tokenizer artifacts/tokenizer/arachne.model
```

Use `configs/dev.yaml` only for small local software checks with matching tokenizer and token data. It is not a substitute for the 1.5B configuration and does not validate 1.5B quality or performance.

Training checkpoints support validated resume. To continue from a verified model artifact while starting a fresh optimizer state, use `--initial-artifact`; to resume a compatible training checkpoint, use `--resume`.

Do not commit raw datasets, personal data, private corpora, local checkpoints, or model weights to Git.

## Artifact verification and inference

Verify a completed artifact before serving:

```bash
arachne-small verify-artifact --path artifacts/arachne-small-1.5b
```

The artifact must include the format-defined configuration, manifest, tokenizer, and SafeTensors weights. See [Artifact Format](docs/ARTIFACT_FORMAT.md).

Set the artifact path and start the local service:

**Windows PowerShell**

```powershell
$env:ARACHNE_SMALL_ARTIFACT = "artifacts/arachne-small-1.5b"
arachne-small serve
```

**Linux / macOS**

```bash
export ARACHNE_SMALL_ARTIFACT="artifacts/arachne-small-1.5b"
arachne-small serve
```

Open:

- Web interface: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/api/health
- Runtime status: http://127.0.0.1:8000/api/status
- Interactive API docs: http://127.0.0.1:8000/docs

Without `ARACHNE_SMALL_ARTIFACT`, the service can run for health/status/UI inspection, but generation is unavailable. A successful health response alone does not prove that model weights are loaded; check `/api/status` and require `ready: true`.

## HTTP API

| Method | Endpoint | Capability |
|---|---|---|
| GET | `/api/health` | Process health |
| GET | `/api/status` | Model readiness, device and artifact status |
| POST | `/api/generate` | Bounded text generation using the loaded model |
| POST | `/api/feedback` | Store explicitly consented correction feedback |
| GET | `/api/modules` | List active verified modules |
| POST | `/api/modules/create` | Validate and activate a declarative module |
| POST | `/api/modules/plan` | Ask a loaded model to draft a declarative module, then validate it |
| POST | `/api/modules/discover` | Discover and verify modules from trusted registry URLs |
| POST | `/api/modules/{module_id}/run` | Execute an active declarative module |
| GET | `/docs` | OpenAPI documentation |

Generation requests support bounded history (up to 20 messages), `max_new_tokens` (1–1024), temperature (0.05–2.0), top-k (0–500), and `auto`/`tr`/`en` language selection. The server validates request shape and total history size.

See [Inference API](docs/INFERENCE_API.md) for request/response contracts and operational details.

## Declarative module system

A module manifest contains exactly five fields: `id`, `version`, `description`, `operations`, and `tests`. Each module must provide 1–32 supported operations and 2–100 deterministic test vectors. Module input is bounded to 100,000 characters.

Example manifest:

```json
{
  "id": "text-cleaner",
  "version": "1.0.0",
  "description": "Normalize whitespace and casing in text.",
  "operations": [
    {"op": "trim"},
    {"op": "lowercase"}
  ],
  "tests": [
    {"input": "  HELLO  ", "expected": "hello"},
    {"input": " Arachne ", "expected": "arachne"}
  ]
}
```

Create and activate a validated module:

```bash
arachne-small module-create --spec module.json --registry-dir data/modules
arachne-small module-list --registry-dir data/modules
arachne-small module-run --registry-dir data/modules --id text-cleaner --input "  HELLO  "
```

Discover modules from one or more trusted immutable registry index URLs:

```bash
arachne-small module-discover \
  --registry-url "https://raw.githubusercontent.com/OWNER/REPO/<40-character-commit>/registry.json" \
  --registry-dir data/modules
```

Registry entries must point to immutable raw GitHub commit URLs and carry a verifiable SHA-256 source digest. Discovery rejects redirects and untrusted hosts. Rejection history is retained locally in the registry directory. The module format does not load or execute arbitrary source code.

Model-assisted planning is available through `POST /api/modules/plan` only when a verified trained model is loaded. The generated JSON is still subject to the same local schema and deterministic tests; model output is not trusted or activated without validation.

## Consent-based feedback and continual learning

The feedback endpoint and CLI require explicit consent. Feedback remains in the local SQLite database and is not uploaded by this project. Operator review is mandatory before export.

Example review workflow:

```bash
arachne-small feedback-list --database data/feedback.sqlite3 --limit 100
arachne-small feedback-review --database data/feedback.sqlite3 --id 12 --approve
arachne-small feedback-review --database data/feedback.sqlite3 --id 13 --reject
arachne-small feedback-export --database data/feedback.sqlite3 --output data/approved-feedback.jsonl --min-rating 4
```

A controlled learning cycle requires a verified base artifact, matching tokenizer/validation stream, training configuration, and a fresh work directory:

```bash
arachne-small learn-cycle \
  --database data/feedback.sqlite3 \
  --base-artifact artifacts/arachne-small-1.5b \
  --config configs/1_5b.yaml \
  --validation-data data/processed/validation.bin \
  --work-dir artifacts/learning-cycle-001 \
  --release-dir artifacts/releases
```

The candidate is evaluated against the baseline on the same validation stream. The default maximum allowed validation-loss regression is zero. Only a verified candidate that passes the gate is promoted. Rollback validates the selected release before updating the release pointer:

```bash
arachne-small rollback-release --release-dir artifacts/releases
```

The inference process loads its configured artifact at startup; restart it with the selected release path to use a promoted or rolled-back version. This is not hot model swapping.

## CLI reference

| Command | Purpose |
|---|---|
| `prepare-corpus` | Normalize, deduplicate, split and record corpus provenance |
| `export-text` | Export JSONL text for tokenizer training |
| `tokenizer` | Train SentencePiece BPE |
| `prepare-data` | Encode JSONL into token streams with integrity metadata |
| `train` | Train or continue model training |
| `count-parameters` | Calculate parameter count from a config |
| `verify-artifact` | Verify a model artifact |
| `serve` | Start API and web UI |
| `feedback-add/list/review/export` | Collect and govern consented corrections |
| `evaluate-artifact` | Evaluate an artifact on a fixed validation stream |
| `promote-candidate` | Gate and promote a candidate against a baseline |
| `rollback-release` | Restore a verified release |
| `learn-cycle` | Run the controlled feedback fine-tuning workflow |
| `module-create/discover/list/run` | Manage declarative modules |

Run `arachne-small --help` or `arachne-small <command> --help` for command-specific options.

## Security and privacy

- Local loopback binding is the default (`127.0.0.1:8000`).
- Remote binding is disabled unless explicitly enabled with `ARACHNE_SMALL_ALLOW_REMOTE=1`.
- Remote mode requires a unique `ARACHNE_SMALL_API_TOKEN` of at least 32 characters; protected API routes require Bearer authentication.
- Use TLS, an authenticated reverse proxy, network restrictions, and rate/connection limits before remote deployment.
- API responses receive anti-sniffing, anti-framing, no-referrer, and no-store headers.
- Request bodies, prompts, histories, module manifests, and feedback fields are bounded and validated.
- Feedback collection requires explicit consent and local storage.
- Training-data provenance and usage rights must be documented by the operator.
- Artifact paths, checksums, tokenizer metadata, tensor names, and tensor shapes are verified before inference loading.
- Registry discovery uses a host allowlist, immutable commit URLs, digest checks, no redirects, and bounded downloads.
- Declarative modules cannot execute arbitrary Python or shell code.
- Keep tokens, private datasets, user data, checkpoints, and model weights out of public Git history.

Remote deployment is an operator responsibility. Do not expose the development server directly to the public internet.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ARACHNE_SMALL_ARTIFACT` | unset | Verified inference artifact directory |
| `ARACHNE_SMALL_HOST` | `127.0.0.1` | Bind address |
| `ARACHNE_SMALL_PORT` | `8000` | TCP port |
| `ARACHNE_SMALL_ALLOW_REMOTE` | `0` | Explicit remote-bind opt-in |
| `ARACHNE_SMALL_API_TOKEN` | unset | Bearer token; required for remote binding |
| `ARACHNE_SMALL_FEEDBACK_DB` | `data/feedback.sqlite3` | Local feedback database |
| `ARACHNE_SMALL_MODULE_DIR` | `data/modules` | Local module registry and audit data |
| `ARACHNE_SMALL_MODULE_REGISTRIES` | unset | Semicolon-separated registry URLs for startup discovery |
| `ARACHNE_SMALL_MODULE_SCAN_INTERVAL` | `3600` | Module discovery interval, bounded by the server |
| `ARACHNE_SMALL_AUTO_LEARN` | `0` | Enable the optional background learning worker |
| `ARACHNE_SMALL_LEARN_INTERVAL` | `21600` | Learning worker interval |
| `ARACHNE_SMALL_TRAINING_CONFIG` | unset | Training config required by automatic learning |
| `ARACHNE_SMALL_VALIDATION_TOKENS` | unset | Fixed validation token stream for automatic learning |
| `ARACHNE_SMALL_RELEASE_DIR` | `artifacts/releases` | Content-addressed release directory |

Automatic learning remains disabled unless explicitly enabled and correctly configured. It requires a verified base artifact, a training configuration, and validation tokens. See [Inference API](docs/INFERENCE_API.md) and [Controlled Continual Learning](docs/CONTINUAL_LEARNING.md).

## Tests and release status

Run the local suite with:

```bash
python -m pytest -q --tb=short
```

This command runs tests locally; it does not require GitHub Actions. Report test results only for the exact code revision and environment on which the command was executed.

The repository currently contains the implementation and validation pipeline, not a published trained model. Software tests do not establish language quality, factuality, safety, benchmark performance, training completion, or inference throughput. A trained release requires a real training run, dataset provenance, recorded evaluation metrics, artifact verification, and hardware measurements.

## Repository boundaries

- **ArachneAISmall:** model architecture, data preparation, tokenizer, training, evaluation, artifacts, local inference, feedback governance, controlled learning, and declarative modules.
- **ArachneAIOpen:** separate runtime project consuming the documented artifact contract.
- **ArachneAI:** separate commercial SaaS product.

## Documentation

- [Turkish installation and startup guide](KURULUM.md)
- [Model Card](docs/MODEL_CARD.md)
- [Dataset Policy](docs/DATASET_POLICY.md)
- [Artifact Format](docs/ARTIFACT_FORMAT.md)
- [Inference API](docs/INFERENCE_API.md)
- [Controlled Continual Learning](docs/CONTINUAL_LEARNING.md)
- [Autonomous Self-Taught Learning](docs/SELF_TAUGHT.md)

## License

The source code is licensed under the [Apache License 2.0](LICENSE). The repository code license does not automatically apply to external datasets, third-party assets, or future trained model weights; those materials require their own applicable terms.



## Autonomous Self-Taught Learning

The `arachne-small serve` command enables the background self-taught worker by default. It collects Turkish and English Wikipedia article extracts through the public REST APIs, checks each page's reported license against the CC BY-SA 3.0/4.0 allowlist, and retains page/revision attribution. It filters obvious credentials/contact patterns and living-person categories, deduplicates documents, creates document-level train/validation splits, trains a tokenizer, and trains the initial Transformer from random weights.

The first bootstrap is intentionally small (128 hidden dimensions, 4 layers, 256-token context; approximately 1.8M parameters). It is not the 1.5B target model. Training writes `arachne_small.pt`, updates it at checkpoint intervals, and marks it trained only after final validation. Later cycles reuse the active tokenizer and architecture, append new material to a cumulative deduplicated replay corpus, keep the bootstrap holdout out of gradient updates, and promote only when fixed-set validation loss does not regress. The active model is atomically replaced only after the candidate release has been verified.

- Default repeat interval: 6 hours.
- No raw chat harvesting and no training on the model's own generated answers.
- `GET /api/status` reports the self-taught worker state.
- Set `ARACHNE_SMALL_SELF_TAUGHT=0` to disable the worker.
- Local corpora, checkpoints and model weights remain outside Git.

See [docs/SELF_TAUGHT.md](docs/SELF_TAUGHT.md) for data sources, licensing, files, configuration and operational limits.
