from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import sentencepiece as spm

from .config import Config, ModelConfig, TrainingConfig, load_config
from .corpus import prepare_corpus, export_text
from .data import encode_jsonl, sha256_file
from .trainer import train
from .artifacts import verify_artifact
from .learning import promote_candidate, _read_release_version

class WikimediaRateLimited(RuntimeError):
    """A Wikimedia 429 response with a bounded server-provided cooldown."""

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(60.0, min(float(retry_after_seconds), 21600.0))
        super().__init__(f"Wikimedia rate limit reached; retry after {self.retry_after_seconds:.0f} seconds.")


def _retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 900.0
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return 900.0
    if not (seconds >= 0 and seconds < float("inf")):
        return 900.0
    return max(60.0, min(seconds, 21600.0))


LOGGER = logging.getLogger("arachne_small.self_taught")
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
USER_AGENT = "ArachneAISmall/0.1 (autonomous research crawler; contact: github.com/ismailozdemir01/ArachneAISmall)"
WIKIPEDIA_LANGUAGES = ("tr", "en")
_ALLOWED_LICENSES = {
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/licenses/by-sa/3.0/",
}
_MIN_DOCUMENT_CHARS = 350
_MAX_DOCUMENT_CHARS = 24000
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_CONTACT_OR_SECRET = re.compile(
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b)",
    re.IGNORECASE,
)


def _get_json(url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    global _LAST_REQUEST_AT
    with _REQUEST_LOCK:
        wait = 1.0 - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        response_context = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise WikimediaRateLimited(_retry_after_seconds(exc.headers.get("Retry-After"))) from exc
        raise
    with response_context as response:
        final = urllib.parse.urlparse(response.geturl())
        original = urllib.parse.urlparse(url)
        if final.scheme != "https" or final.hostname != original.hostname:
            raise ValueError("Wikimedia request redirected outside its HTTPS origin.")
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("Wikimedia response exceeded the configured size limit.")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Wikimedia API returned an invalid JSON object.")
    return value


def _is_approved_license(page: dict[str, Any]) -> tuple[str, str] | None:
    license_data = page.get("license")
    if not isinstance(license_data, dict):
        return None
    title = license_data.get("title")
    url = license_data.get("url")
    if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
        return None
    normalized_url = url if url.startswith("https:") else "https:" + url if url.startswith("//") else url
    if normalized_url.rstrip("/") + "/" not in _ALLOWED_LICENSES:
        return None
    return title, normalized_url


def _is_living_person_category(language: str, categories: list[dict[str, Any]]) -> bool:
    prefixes = ("living people", "yaşayan insanlar") if language == "tr" else ("living people",)
    for category in categories:
        if not isinstance(category, dict):
            continue
        name = str(category.get("title", "")).casefold()
        if any(prefix in name for prefix in prefixes):
            return True
    return False


def _fetch_page(language: str) -> dict[str, Any] | None:
    host = f"{language}.wikipedia.org"
    summary = _get_json(f"https://{host}/api/rest_v1/page/random/summary")
    title = summary.get("title")
    extract = summary.get("extract")
    if not isinstance(title, str) or not isinstance(extract, str):
        return None
    text = " ".join(extract.split())
    if len(text) < _MIN_DOCUMENT_CHARS or len(text) > _MAX_DOCUMENT_CHARS:
        return None
    if _CONTACT_OR_SECRET.search(text):
        return None

    encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
    page = _get_json(f"https://{host}/w/rest.php/v1/page/{encoded_title}/bare")
    approved = _is_approved_license(page)
    if approved is None:
        LOGGER.debug("Skipped %s:%s because its page license is not on the allowlist.", language, title)
        return None

    page_id = page.get("id")
    revision = page.get("latest")
    if not isinstance(page_id, int) or not isinstance(revision, dict):
        return None
    revision_id = revision.get("id")
    if not isinstance(revision_id, int):
        return None

    category_params: dict[str, str] = {
        "action": "query", "format": "json", "prop": "categories",
        "cllimit": "max", "titles": title,
    }
    categories: list[dict[str, Any]] = []
    for category_page in range(10):
        query = urllib.parse.urlencode(category_params)
        categories_response = _get_json(f"https://{host}/w/api.php?{query}")
        pages = categories_response.get("query", {}).get("pages", {})
        if isinstance(pages, dict):
            for page_data in pages.values():
                if isinstance(page_data, dict) and isinstance(page_data.get("categories"), list):
                    categories.extend(page_data["categories"])
        if _is_living_person_category(language, categories):
            return None
        continuation = categories_response.get("continue")
        if not isinstance(continuation, dict):
            break
        if category_page == 9:
            LOGGER.warning("Skipped %s:%s because category metadata exceeded the safety page limit.", language, title)
            return None
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in continuation.items()):
            return None
        category_params.update(continuation)

    license_title, license_url = approved
    page_url = f"https://{host}/wiki/{encoded_title}"
    return {
        "text": f"<|language:{language}|>\n{title}\n{text}",
        "provenance": {
            "source": "Wikimedia Wikipedia",
            "language": language,
            "title": title,
            "page_url": page_url,
            "history_url": page_url + "?action=history",
            "revision_id": revision_id,
            "page_id": page_id,
            "license": license_title,
            "license_url": license_url,
            "retrieved_at": int(time.time()),
            "modification": "Extract normalized and prefixed with language marker for language-model training.",
        },
    }


def acquire_wikipedia_corpus(
    output_jsonl: str | Path,
    *,
    articles_per_language: int = 40,
    request_delay_seconds: float = 1.1,
    max_attempts_per_language: int | None = None,
    seen_database: str | Path | None = None,
) -> dict[str, Any]:
    """Acquire only per-page CC BY-SA 3.0/4.0 Wikipedia extracts with provenance."""
    if isinstance(articles_per_language, bool) or not isinstance(articles_per_language, int) or not 20 <= articles_per_language <= 500:
        raise ValueError("articles_per_language must be between 20 and 500.")
    if not isinstance(request_delay_seconds, (int, float)) or request_delay_seconds < 1.0:
        raise ValueError("Wikimedia request delay must be at least one second.")
    target = Path(output_jsonl).expanduser()
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"Corpus output already exists or is a symbolic link: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    attempts_limit = max_attempts_per_language or articles_per_language * 8
    counts = {language: 0 for language in WIKIPEDIA_LANGUAGES}
    seen: set[tuple[str, int]] = set()
    seen_content: set[str] = set()
    database = None
    if seen_database is not None:
        seen_path = Path(seen_database).expanduser()
        if seen_path.is_symlink():
            raise ValueError("Self-taught seen-page database cannot be a symbolic link.")
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(seen_path, timeout=30)
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("CREATE TABLE IF NOT EXISTS seen_pages (language TEXT NOT NULL, page_id INTEGER NOT NULL, PRIMARY KEY(language, page_id))")
        database.execute("CREATE TABLE IF NOT EXISTS seen_content (content_sha256 TEXT PRIMARY KEY)")
        database.execute("BEGIN IMMEDIATE")
    records = 0
    try:
        with target.open("x", encoding="utf-8", newline="\n") as output:
            for language in WIKIPEDIA_LANGUAGES:
                attempts = 0
                while counts[language] < articles_per_language and attempts < attempts_limit:
                    attempts += 1
                    try:
                        record = _fetch_page(language)
                        if record is not None:
                            provenance = record["provenance"]
                            identity = (language, provenance["page_id"])
                            content_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                            already_seen = identity in seen or content_hash in seen_content
                            if database is not None and not already_seen:
                                already_seen = database.execute(
                                    "SELECT 1 FROM seen_pages WHERE language=? AND page_id=?",
                                    identity,
                                ).fetchone() is not None
                                if not already_seen:
                                    already_seen = database.execute(
                                        "SELECT 1 FROM seen_content WHERE content_sha256=?",
                                        (content_hash,),
                                    ).fetchone() is not None
                            if not already_seen:
                                seen.add(identity)
                                seen_content.add(content_hash)
                                if database is not None:
                                    database.execute(
                                        "INSERT INTO seen_pages(language,page_id) VALUES(?,?)",
                                        identity,
                                    )
                                    database.execute(
                                        "INSERT INTO seen_content(content_sha256) VALUES(?)",
                                        (content_hash,),
                                    )
                                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                                output.flush()
                                counts[language] += 1
                                records += 1
                    except WikimediaRateLimited:
                        # A 429 is a source-wide throttle signal, not a bad random page.
                        # Abort this cycle immediately and let the worker honor Retry-After.
                        raise
                    except (OSError, ValueError, KeyError, TypeError, AttributeError, urllib.error.URLError) as exc:
                        LOGGER.warning("Wikimedia %s acquisition attempt %d failed: %s", language, attempts, exc)
                    time.sleep(float(request_delay_seconds))
        if min(counts.values()) < 20:
            raise RuntimeError(
                f"Insufficient licensed documents were acquired: {counts}. "
                "No training cycle was started."
            )
        if database is not None:
            database.commit()
        return {
            "records": records,
            "by_language": counts,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source": "Wikimedia Wikipedia REST and Action APIs",
            "license_policy": sorted(_ALLOWED_LICENSES),
            "output": str(target),
        }
    except Exception:
        if database is not None:
            database.rollback()
        target.unlink(missing_ok=True)
        raise
    finally:
        if database is not None:
            database.close()


def _write_tokenizer(corpus_text: Path, output_prefix: Path, vocab_size: int = 4000) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_text),
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        hard_vocab_limit=False,
        input_sentence_size=0,
        shuffle_input_sentence=False,
    )
    model_path = output_prefix.with_suffix(".model")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.vocab_size() < 256:
        raise RuntimeError("Acquired corpus is too small to train a usable tokenizer.")
    return model_path


def _bootstrap_config(vocab_size: int, output_dir: Path) -> Config:
    # Conservative from-scratch bootstrap model; it is not the 1.5B target model.
    model = ModelConfig(
        vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4,
        n_kv_heads=2, max_seq_len=256, intermediate_size=352,
        tie_embeddings=False, gradient_checkpointing=False,
    )
    training = TrainingConfig(
        seed=1337, batch_size=2, gradient_accumulation_steps=1,
        max_steps=80, learning_rate=3e-4, min_learning_rate=3e-5,
        warmup_steps=8, weight_decay=0.1, grad_clip=1.0,
        eval_interval=20, save_interval=20, precision="fp32",
        num_workers=0, output_dir=str(output_dir),
    )
    config = Config(model, training)
    config.model.validate()
    config.training.validate()
    return config


def _continued_config(base_artifact: Path, output_dir: Path) -> Config:
    raw = json.loads((base_artifact / "config.json").read_text(encoding="utf-8"))
    model = ModelConfig(**raw["model"])
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / "self_taught.yaml") if (
        Path(__file__).resolve().parents[2] / "configs" / "self_taught.yaml"
    ).is_file() else _bootstrap_config(model.vocab_size, output_dir)
    training = TrainingConfig(
        seed=config.training.seed, batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps, learning_rate=config.training.learning_rate,
        min_learning_rate=config.training.min_learning_rate, warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay, grad_clip=config.training.grad_clip,
        eval_interval=config.training.eval_interval, save_interval=config.training.save_interval,
        precision="fp32", num_workers=0, output_dir=str(output_dir),
    )
    return Config(model, training)



def _atomic_copy_or_append(source: Path, destination: Path, *, append: bool) -> None:
    """Atomically copy a new corpus file or append new documents without partial writes."""
    import shutil
    import tempfile

    if source.is_symlink() or not source.is_file() or destination.is_symlink():
        raise ValueError("Replay corpus files must be regular, non-symlink files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not append:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if append and destination.is_file():
                with destination.open("rb") as existing:
                    shutil.copyfileobj(existing, output, length=1024 * 1024)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def run_self_taught_cycle(
    *,
    work_root: str | Path = "data/self_taught",
    artifact_root: str | Path = "artifacts/self_taught",
    release_root: str | Path = "artifacts/releases",
    articles_per_language: int = 40,
) -> dict[str, Any]:
    """Acquire licensed data, bootstrap or improve the model, evaluate, and promote safely."""
    work_root = Path(work_root).expanduser()
    artifact_root = Path(artifact_root).expanduser()
    release_root = Path(release_root).expanduser()
    for path in (work_root, artifact_root, release_root):
        if path.is_symlink():
            raise ValueError(f"Self-taught path cannot be a symbolic link: {path}")
    cycle_id = f"cycle-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    cycle = work_root / cycle_id
    cycle.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=True)
    release_root.mkdir(parents=True, exist_ok=True)

    raw_path = cycle / "wikipedia.jsonl"
    acquisition = acquire_wikipedia_corpus(
        raw_path, articles_per_language=articles_per_language,
        seen_database=work_root / "seen-pages.sqlite3",
    )
    split_dir = cycle / "split"
    dataset_manifest = prepare_corpus(
        str(raw_path), str(split_dir), text_field="text",
        validation_fraction=0.2, seed=1337,
        source="Wikimedia Wikipedia per-page REST metadata; see raw JSONL provenance",
        license_name="Mixed per-page CC BY-SA 3.0/4.0; exact page license and attribution are recorded per document.",
    )
    split_train_jsonl = split_dir / "train.jsonl"
    validation_jsonl = split_dir / "validation.jsonl"
    replay_dir = work_root / "replay"
    if replay_dir.is_symlink():
        raise ValueError("Persistent replay directory cannot be a symbolic link.")
    replay_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = replay_dir / "train.jsonl"
    replay_tokenizer = replay_dir / "tokenizer.model"
    validation_tokens = cycle / "validation.bin"
    train_tokens = cycle / "train.bin"
    if any(path.is_symlink() for path in (train_jsonl, replay_tokenizer, validation_tokens, Path(str(validation_tokens) + ".json"))):
        raise ValueError("Persistent replay files cannot be symbolic links.")
    if train_jsonl.exists():
        _atomic_copy_or_append(split_train_jsonl, train_jsonl, append=True)
    else:
        _atomic_copy_or_append(split_train_jsonl, train_jsonl, append=False)

    current_version = _read_release_version(release_root)
    base_artifact = release_root / current_version if current_version else None
    if base_artifact is None:
        configured_base = os.environ.get("ARACHNE_SMALL_ARTIFACT", "").strip()
        if configured_base:
            configured_path = Path(configured_base).expanduser()
            if not configured_path.is_symlink() and configured_path.is_dir() and (configured_path / "artifact-manifest.json").is_file():
                base_artifact = configured_path
        if base_artifact is None:
            from .artifacts import discover_model_artifact
            discovered = discover_model_artifact("artifacts")
            if discovered is not None:
                discovered_path = Path(discovered[2])
                if discovered_path.is_dir() and (discovered_path / "artifact-manifest.json").is_file():
                    base_artifact = discovered_path
    if base_artifact is not None:
        base_summary = verify_artifact(base_artifact)
        parameter_count = int(base_summary["parameter_count"])
        if parameter_count > 20_000_000:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Self-taught fine-tuning of {parameter_count:,} parameters is disabled on CPU. "
                    "The current verified model remains active; CUDA with sufficient free memory is required."
                )
            free_memory, _ = torch.cuda.mem_get_info()
            required_memory = parameter_count * 20 + 2 * 1024**3
            if free_memory < required_memory:
                raise RuntimeError(
                    f"Insufficient free CUDA memory for safe self-taught fine-tuning: "
                    f"{free_memory:,} bytes available, approximately {required_memory:,} bytes required."
                )
        if replay_tokenizer.exists():
            if sha256_file(replay_tokenizer) != sha256_file(base_artifact / "tokenizer.model"):
                raise ValueError("Persistent replay tokenizer differs from the active model tokenizer.")
        else:
            _atomic_copy_or_append(base_artifact / "tokenizer.model", replay_tokenizer, append=False)
        tokenizer_path = replay_tokenizer
    else:
        if replay_tokenizer.is_file():
            tokenizer_path = replay_tokenizer
        else:
            tokenizer_prefix = cycle / "tokenizer" / "arachne"
            tokenizer_text = cycle / "tokenizer-corpus.txt"
            export_text(str(train_jsonl), str(tokenizer_text))
            generated_tokenizer = _write_tokenizer(tokenizer_text, tokenizer_prefix)
            _atomic_copy_or_append(generated_tokenizer, replay_tokenizer, append=False)
            tokenizer_path = replay_tokenizer
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    encode_jsonl(str(validation_jsonl), str(validation_tokens), str(tokenizer_path), "text", processor.eos_id())
    encode_jsonl(str(train_jsonl), str(train_tokens), str(tokenizer_path), "text", processor.eos_id())

    candidate_dir = artifact_root / cycle_id
    config = (
        _continued_config(base_artifact, candidate_dir)
        if base_artifact is not None
        else _bootstrap_config(processor.vocab_size(), candidate_dir)
    )
    model_path = train(
        config, str(train_tokens), str(validation_tokens), str(tokenizer_path),
        initial_artifact=str(base_artifact) if base_artifact is not None else None,
    )
    candidate_dir = model_path.parent
    candidate_summary = verify_artifact(candidate_dir)
    result: dict[str, Any] = {
        "cycle_id": cycle_id,
        "acquisition": acquisition,
        "dataset": dataset_manifest,
        "candidate_artifact": str(candidate_dir),
        "candidate_weights_sha256": candidate_summary["weights_sha256"],
        "parameter_count": candidate_summary["parameter_count"],
        "training_steps": json.loads((candidate_dir / "artifact-manifest.json").read_text(encoding="utf-8"))["training_steps_completed"],
        "validation_loss": json.loads((candidate_dir / "artifact-manifest.json").read_text(encoding="utf-8"))["final_validation_loss"],
        "bootstrap": base_artifact is None,
        "promoted": False,
    }
    if base_artifact is not None:
        gate = promote_candidate(
            candidate_dir, base_artifact, validation_tokens, release_root,
            max_loss_regression=0.0, batch_size=1, max_batches=200,
        )
        result["promotion"] = gate
        result["promoted"] = True
    else:
        # The first completed training run becomes the initial immutable release.
        from .learning import _release_content_hash
        version = _release_content_hash(candidate_summary)
        destination = release_root / version
        if destination.exists():
            raise FileExistsError(f"Initial model release already exists: {destination}")
        import shutil
        shutil.copytree(candidate_dir, destination)
        pointer = release_root / "current.json"
        temporary = release_root / f".current-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps({
            "format": "arachne-small-release", "format_version": 1,
            "version": version, "artifact": version,
        }, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, pointer)
        result["promotion"] = {"version": version, "artifact": str(destination), "bootstrap": True}
        result["promoted"] = True
    return result
