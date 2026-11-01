from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .artifacts import load_model, verify_artifact
from .data import TokenBlockDataset, sha256_file
from .trainer import evaluate


class FeedbackStore:
    """Explicit-consent, local-only store for reviewed training examples."""

    def __init__(self, path: str | Path) -> None:
        requested_path = Path(path).expanduser()
        if requested_path.is_symlink():
            raise ValueError("Feedback database path cannot be a symbolic link.")
        self.path = requested_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    consent INTEGER NOT NULL CHECK(consent = 1),
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    correction TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    language TEXT NOT NULL CHECK(language IN ('tr','en')),
                    reviewed INTEGER NOT NULL DEFAULT 0 CHECK(reviewed IN (0,1)),
                    approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0,1)),
                    exported INTEGER NOT NULL DEFAULT 0 CHECK(exported IN (0,1)),
                    content_sha256 TEXT NOT NULL UNIQUE
                )
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(feedback)")}
            if "exported" not in columns:
                db.execute("ALTER TABLE feedback ADD COLUMN exported INTEGER NOT NULL DEFAULT 0 CHECK(exported IN (0,1))")
            if "approved" not in columns:
                db.execute("ALTER TABLE feedback ADD COLUMN approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0,1))")
            db.execute("CREATE INDEX IF NOT EXISTS feedback_reviewed_idx ON feedback(reviewed, approved, exported, id)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            self._secure_private_file(self.path)
            self._secure_private_file(Path(str(self.path) + "-wal"))
            self._secure_private_file(Path(str(self.path) + "-shm"))
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _secure_private_file(path: Path) -> None:
        if os.name == "posix" and path.exists():
            os.chmod(path, 0o600)

    @staticmethod
    def _open_private_temp(path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")

    @staticmethod
    def _validate_text(name: str, value: str, limit: int) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        value = value.strip()
        if not value or len(value) > limit or "\x00" in value:
            raise ValueError(f"{name} must contain 1..{limit} valid characters.")
        return value

    def add(self, *, consent: bool, prompt: str, response: str,
            correction: str, rating: int, language: str) -> int:
        if consent is not True:
            raise PermissionError("Explicit consent is required before storing feedback.")
        prompt = self._validate_text("prompt", prompt, 8000)
        response = self._validate_text("response", response, 8000)
        correction = self._validate_text("correction", correction, 8000)
        combined = "\n".join((prompt, response, correction))
        if any(ord(char) < 32 and char not in "\n\r\t" for char in combined):
            raise ValueError("Feedback contains unsupported control characters.")
        sensitive_patterns = (
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s]+",
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        )
        if any(re.search(pattern, combined) for pattern in sensitive_patterns):
            raise ValueError("Feedback appears to contain credentials or personal contact data; it was not stored.")
        if response.casefold() == correction.casefold():
            raise ValueError("Correction must differ from the original response to provide a learning signal.")
        if prompt.casefold() == correction.casefold():
            raise ValueError("Correction cannot duplicate the prompt.")
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise ValueError("rating must be an integer from 1 to 5.")
        if language not in {"tr", "en"}:
            raise ValueError("language must be 'tr' or 'en'.")
        canonical = json.dumps([prompt, response, correction, rating, language],
                               ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO feedback(created_at,consent,prompt,response,correction,rating,language,content_sha256) VALUES(?,?,?,?,?,?,?,?)",
                (int(time.time()), 1, prompt, response, correction, rating, language, digest),
            )
            row = db.execute("SELECT id FROM feedback WHERE content_sha256=?", (digest,)).fetchone()
            return int(row["id"])

    def list_pending(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000.")
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,created_at,prompt,response,correction,rating,language "
                "FROM feedback WHERE reviewed=0 AND consent=1 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def review(self, record_id: int, *, approved: bool) -> None:
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 1:
            raise ValueError("record_id must be a positive integer.")
        if not isinstance(approved, bool):
            raise TypeError("approved must be a boolean.")
        with self._connect() as db:
            row = db.execute(
                "SELECT reviewed,exported FROM feedback WHERE id=? AND consent=1",
                (record_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Feedback record does not exist: {record_id}")
            if row["reviewed"]:
                raise ValueError("Feedback review decisions are immutable; submit a new correction instead.")
            if row["exported"]:
                raise ValueError("Exported feedback cannot be re-reviewed; create a new correction instead.")
            db.execute(
                "UPDATE feedback SET reviewed=1, approved=? WHERE id=? AND consent=1",
                (1 if approved else 0, record_id),
            )

    def mark_exported(self, record_hashes: list[str]) -> None:
        if not record_hashes:
            return
        with self._connect() as db:
            db.executemany(
                "UPDATE feedback SET exported=1 WHERE content_sha256=? AND consent=1 AND reviewed=1 AND approved=1",
                [(value,) for value in record_hashes],
            )

    def export_approved(self, destination: str | Path, *, min_rating: int = 4) -> dict[str, Any]:
        if isinstance(min_rating, bool) or not isinstance(min_rating, int) or not 1 <= min_rating <= 5:
            raise ValueError("min_rating must be between 1 and 5.")
        requested_target = Path(destination).expanduser()
        if requested_target.is_symlink():
            raise ValueError("Feedback export destination cannot be a symbolic link.")
        target = requested_target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        if target.exists() or manifest_path.exists():
            raise FileExistsError(
                f"Feedback export destination already exists: {target}. "
                "Choose a new output path; existing exports are never overwritten."
            )
        temporary = target.with_name(target.name + ".tmp")
        count = 0
        hashes: list[str] = []
        temporary_created = False
        try:
            with self._connect() as db:
                output_handle = self._open_private_temp(temporary)
                temporary_created = True
                with output_handle as output:
                    rows = db.execute(
                        "SELECT id,prompt,correction,language,content_sha256 "
                        "FROM feedback WHERE reviewed=1 AND approved=1 AND exported=0 "
                        "AND consent=1 AND rating>=? ORDER BY id",
                        (min_rating,),
                    )
                    for row in rows:
                        # Corrections are preferred targets; the original answer is not treated as ground truth.
                        record = {
                            "text": f"<|language:{row['language']}|>\nUser: {row['prompt']}\nAssistant: {row['correction']}",
                            "provenance": {
                                "source": "explicit-user-feedback",
                                "feedback_id": row["id"],
                                "content_sha256": row["content_sha256"],
                            },
                        }
                        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                        hashes.append(row["content_sha256"])
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
        finally:
            if temporary_created:
                temporary.unlink(missing_ok=True)
        manifest = {
            "format": "arachne-small-feedback-export",
            "format_version": 1,
            "records": count,
            "minimum_rating": min_rating,
            "record_hashes": hashes,
            "export_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "created_at": int(time.time()),
        }
        tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_temp_created = False
        try:
            output_handle = self._open_private_temp(tmp_manifest)
            manifest_temp_created = True
            with output_handle as output:
                output.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp_manifest, manifest_path)
        finally:
            if manifest_temp_created:
                tmp_manifest.unlink(missing_ok=True)
        return manifest


def evaluate_artifact(artifact: str | Path, validation_tokens: str | Path,
                      *, batch_size: int = 1, max_batches: int = 200,
                      precision: str = "fp32") -> dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer.")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16.")
    summary = verify_artifact(artifact)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 evaluation requires CUDA.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 evaluation requested but this CUDA device does not support it.")
    model, tokenizer = load_model(artifact, device)
    dataset = TokenBlockDataset(str(validation_tokens), model.config.max_seq_len)
    if not len(dataset):
        raise ValueError("Validation token dataset has no complete blocks.")
    sidecar = Path(str(validation_tokens) + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing token provenance sidecar: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Validation token provenance sidecar must contain a JSON object.")
    if metadata.get("format_version") != 1 or metadata.get("token_format") != "little-endian-uint32":
        raise ValueError("Unsupported validation token dataset format.")
    if metadata.get("token_file") != Path(validation_tokens).name:
        raise ValueError("Validation token filename does not match its provenance sidecar.")
    if metadata.get("token_count") != dataset.tokens.numel():
        raise ValueError("Validation token count does not match its provenance sidecar.")
    if metadata.get("tokenizer_vocab_size") != model.config.vocab_size:
        raise ValueError("Validation token vocabulary does not match the evaluated artifact.")
    if metadata.get("tokenizer_sha256") != summary["tokenizer_sha256"]:
        raise ValueError("Validation data tokenizer does not match the evaluated artifact.")
    if metadata.get("token_file_sha256") != sha256_file(validation_tokens):
        raise ValueError("Validation token checksum does not match its sidecar.")
    actual_special_tokens = {
        "pad_id": tokenizer.pad_id(),
        "unk_id": tokenizer.unk_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
    }
    if metadata.get("special_tokens") != actual_special_tokens or metadata.get("eos_id") != tokenizer.eos_id():
        raise ValueError("Validation token special-token metadata does not match the evaluated tokenizer.")
    minimum_token = int(dataset.tokens.min().item())
    maximum_token = int(dataset.tokens.max().item())
    if minimum_token < 0 or maximum_token >= model.config.vocab_size:
        raise ValueError("Validation token stream contains IDs outside the model vocabulary.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss = evaluate(model, loader, device, precision=precision, max_batches=max_batches)
    return {
        "artifact": str(Path(artifact).resolve()),
        "weights_sha256": summary["weights_sha256"],
        "validation_loss": loss,
        "validation_perplexity": float(torch.exp(torch.tensor(min(80.0, loss))).item()),
        "validation_tokens_sha256": metadata["token_file_sha256"],
        "batches_evaluated": min(len(loader), max_batches),
    }


def promote_candidate(candidate: str | Path, baseline: str | Path,
                      validation_tokens: str | Path, release_dir: str | Path,
                      *, max_loss_regression: float = 0.0,
                      batch_size: int = 1, max_batches: int = 200) -> dict[str, Any]:
    if (isinstance(max_loss_regression, bool)
            or not isinstance(max_loss_regression, (int, float))
            or not math.isfinite(max_loss_regression)
            or max_loss_regression < 0):
        raise ValueError("max_loss_regression must be a finite non-negative number.")
    baseline_summary = verify_artifact(baseline)
    candidate_summary = verify_artifact(candidate)
    baseline_config = json.loads((Path(baseline) / "config.json").read_text(encoding="utf-8"))
    candidate_config = json.loads((Path(candidate) / "config.json").read_text(encoding="utf-8"))
    if baseline_config.get("model") != candidate_config.get("model"):
        raise ValueError("Baseline and candidate architectures must match for a valid paired comparison.")
    if baseline_summary["tokenizer_sha256"] != candidate_summary["tokenizer_sha256"]:
        raise ValueError("Baseline and candidate must use the same tokenizer.")
    baseline_metrics = evaluate_artifact(baseline, validation_tokens, batch_size=batch_size, max_batches=max_batches)
    candidate_metrics = evaluate_artifact(candidate, validation_tokens, batch_size=batch_size, max_batches=max_batches)
    allowed = baseline_metrics["validation_loss"] + max_loss_regression
    if candidate_metrics["validation_loss"] > allowed:
        raise RuntimeError(
            f"Candidate rejected: validation loss {candidate_metrics['validation_loss']:.8f} "
            f"exceeds baseline threshold {allowed:.8f}."
        )
    candidate_path = Path(candidate).expanduser().resolve()
    baseline_path = Path(baseline).expanduser().resolve()
    requested_release = Path(release_dir).expanduser()
    if requested_release.is_symlink():
        raise ValueError("Release directory cannot be a symbolic link.")
    release = requested_release.resolve()

    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    if overlaps(release, candidate_path) or overlaps(release, baseline_path):
        raise ValueError("Release directory must not overlap candidate or baseline artifacts.")
    release.mkdir(parents=True, exist_ok=True)
    import shutil
    version = candidate_metrics["weights_sha256"]
    baseline_version = baseline_metrics["weights_sha256"]
    if version == baseline_version:
        raise RuntimeError("Candidate weights are identical to the baseline; no release was created.")

    def install_immutable(source: Path, digest: str) -> None:
        destination = release / digest
        if destination.is_symlink():
            raise ValueError("Content-addressed release directories cannot be symbolic links.")
        if destination.exists():
            verify_artifact(destination)
            if verify_artifact(destination)["weights_sha256"] != digest:
                raise ValueError("Content-addressed release directory contains the wrong artifact.")
            return
        staging = release / (digest + ".staging")
        if staging.is_symlink():
            raise ValueError("Release staging path cannot be a symbolic link.")
        if staging.exists():
            raise FileExistsError(
                f"Release staging path already exists and will not be removed: {staging}"
            )
        staging.mkdir()
        try:
            source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            inference_files = {
                "config.json",
                "artifact-manifest.json",
                source_config["tokenizer"]["file"],
                source_config["weights"]["file"],
            }
            for filename in inference_files:
                shutil.copyfile(source / filename, staging / filename)
                if os.name == "posix":
                    os.chmod(staging / filename, 0o600)
            if verify_artifact(staging)["weights_sha256"] != digest:
                raise ValueError("Staged release artifact hash does not match its content address.")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    install_immutable(baseline_path, baseline_version)
    install_immutable(candidate_path, version)
    previous_version = _read_release_version(release) or baseline_version
    pointer = {
        "format": "arachne-small-release",
        "format_version": 1,
        "version": version,
        "artifact": version,
        "previous_version": previous_version,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "promoted_at": int(time.time()),
    }
    _atomic_json(release / "current.json", pointer)
    return pointer


def _read_release_version(release: Path) -> str | None:
    pointer = release / "current.json"
    if pointer.is_symlink():
        raise ValueError("Release pointer cannot be a symbolic link.")
    if not pointer.is_file():
        return None
    data = json.loads(pointer.read_text(encoding="utf-8"))
    if (not isinstance(data, dict)
            or data.get("format") != "arachne-small-release"
            or data.get("format_version") != 1):
        raise ValueError("Existing release pointer has an unsupported format.")
    version = data.get("version")
    version_path = release / version if isinstance(version, str) else release
    if (not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version)
            or data.get("artifact") != version
            or version_path.is_symlink() or not version_path.is_dir()):
        raise ValueError("Existing release pointer is invalid; refusing to overwrite it.")
    summary = verify_artifact(version_path)
    if summary["weights_sha256"] != version:
        raise ValueError("Release directory name does not match its artifact weight hash.")
    return version


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("Atomic JSON destination cannot be a symbolic link.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if path.is_symlink():
            raise ValueError("Atomic JSON destination became a symbolic link.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rollback_release(release_dir: str | Path, version: str | None = None) -> dict[str, Any]:
    requested_release = Path(release_dir).expanduser()
    if requested_release.is_symlink():
        raise ValueError("Release directory cannot be a symbolic link.")
    release = requested_release.resolve()
    current = _read_release_version(release)
    if current is None:
        raise FileNotFoundError("No current release exists.")
    pointer_path = release / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    target = version or pointer.get("previous_version")
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-f]{64}", target) or target == current:
        raise ValueError("A distinct, valid content-hash release version is required for rollback.")
    target_path = release / target
    if target_path.is_symlink():
        raise ValueError("Rollback target cannot be a symbolic link.")
    if not target_path.is_dir():
        raise FileNotFoundError(f"Release version does not exist: {target}")
    target_summary = verify_artifact(target_path)
    if target_summary["weights_sha256"] != target:
        raise ValueError("Rollback target directory does not match its artifact weight hash.")
    pointer.update({
        "previous_version": current,
        "version": target,
        "artifact": target,
        "rolled_back_at": int(time.time()),
    })
    _atomic_json(pointer_path, pointer)
    return pointer


def run_learning_cycle(*, database: str | Path, base_artifact: str | Path,
                       config_path: str | Path, validation_tokens: str | Path,
                       work_dir: str | Path, release_dir: str | Path,
                       min_rating: int = 4, max_loss_regression: float = 0.0,
                       batch_size: int = 1, max_batches: int = 200) -> dict[str, Any]:
    """Run one complete opt-in feedback -> fine-tune -> evaluate -> gated release cycle."""
    from dataclasses import replace
    from .config import load_config
    from .data import encode_jsonl

    base = Path(base_artifact).expanduser().resolve()
    base_summary = verify_artifact(base)
    artifact_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    tokenizer_path = base / artifact_config["tokenizer"]["file"]
    tokenizer = __import__("sentencepiece").SentencePieceProcessor(model_file=str(tokenizer_path))
    requested_work = Path(work_dir).expanduser()
    requested_release = Path(release_dir).expanduser()
    if requested_work.is_symlink():
        raise ValueError("Learning work directory cannot be a symbolic link.")
    if requested_release.is_symlink():
        raise ValueError("Release directory cannot be a symbolic link.")
    work = requested_work.resolve()
    release = requested_release.resolve()
    validation = Path(validation_tokens).expanduser().resolve()

    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    for label, protected in (("base artifact", base), ("release directory", release),
                             ("validation token file", validation)):
        if overlaps(work, protected):
            raise ValueError(f"Work directory must not overlap the {label} path.")
    if overlaps(base, release):
        raise ValueError("Base artifact and release directory must not overlap.")
    if work.exists():
        if not work.is_dir():
            raise NotADirectoryError(work)
        if any(work.iterdir()):
            raise FileExistsError(
                f"Learning work directory is not empty: {work}. "
                "Use a new directory for each learning cycle; existing files are preserved."
            )
    else:
        work.mkdir(parents=True, exist_ok=False)
    export_path = work / "approved-feedback.jsonl"
    manifest = FeedbackStore(database).export_approved(export_path, min_rating=min_rating)
    if manifest["records"] == 0:
        raise RuntimeError("No new consented feedback meets the minimum rating; no training was started.")
    train_tokens = work / "feedback-train.bin"
    encode_jsonl(str(export_path), str(train_tokens), str(tokenizer_path), "text", tokenizer.eos_id())
    config = load_config(config_path)
    training_output = work / "candidate"
    candidate_config = replace(config, training=replace(config.training, output_dir=str(training_output)))
    from .trainer import train
    train(candidate_config, str(train_tokens), str(validation_tokens),
          str(tokenizer_path), initial_artifact=str(base))
    gate = promote_candidate(training_output, base, validation_tokens, release_dir,
                             max_loss_regression=max_loss_regression,
                             batch_size=batch_size, max_batches=max_batches)
    FeedbackStore(database).mark_exported(manifest["record_hashes"])
    return {
        "feedback_records": manifest["records"],
        "feedback_export_sha256": manifest["export_sha256"],
        "base_weights_sha256": base_summary["weights_sha256"],
        "candidate_artifact": str(training_output),
        "release": gate,
    }
