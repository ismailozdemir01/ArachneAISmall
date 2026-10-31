from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
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
        self.path = Path(path).expanduser().resolve()
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
                    exported INTEGER NOT NULL DEFAULT 0 CHECK(exported IN (0,1)),
                    content_sha256 TEXT NOT NULL UNIQUE
                )
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(feedback)")}
            if "exported" not in columns:
                db.execute("ALTER TABLE feedback ADD COLUMN exported INTEGER NOT NULL DEFAULT 0 CHECK(exported IN (0,1))")
            db.execute("CREATE INDEX IF NOT EXISTS feedback_reviewed_idx ON feedback(reviewed, exported, id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

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

    def mark_exported(self, record_hashes: list[str]) -> None:
        if not record_hashes:
            return
        with self._connect() as db:
            db.executemany("UPDATE feedback SET exported=1 WHERE content_sha256=? AND consent=1",
                           [(value,) for value in record_hashes])

    def export_approved(self, destination: str | Path, *, min_rating: int = 4) -> dict[str, Any]:
        if isinstance(min_rating, bool) or not isinstance(min_rating, int) or not 1 <= min_rating <= 5:
            raise ValueError("min_rating must be between 1 and 5.")
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        count = 0
        hashes: list[str] = []
        try:
            with self._connect() as db, temporary.open("w", encoding="utf-8") as output:
                rows = db.execute(
                    "SELECT id,prompt,correction,language,content_sha256 FROM feedback WHERE reviewed=0 AND exported=0 AND consent=1 AND rating>=? ORDER BY id",
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
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        try:
            tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_manifest, manifest_path)
        finally:
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
    model, _ = load_model(artifact, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    dataset = TokenBlockDataset(str(validation_tokens), model.config.max_seq_len)
    if not len(dataset):
        raise ValueError("Validation token dataset has no complete blocks.")
    sidecar = Path(str(validation_tokens) + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing token provenance sidecar: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("tokenizer_sha256") != summary["tokenizer_sha256"]:
        raise ValueError("Validation data tokenizer does not match the evaluated artifact.")
    if metadata.get("token_file_sha256") != sha256_file(validation_tokens):
        raise ValueError("Validation token checksum does not match its sidecar.")
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
    release = Path(release_dir).expanduser().resolve()
    release.mkdir(parents=True, exist_ok=True)
    import shutil
    version = candidate_metrics["weights_sha256"]
    baseline_version = baseline_metrics["weights_sha256"]
    if version == baseline_version:
        raise RuntimeError("Candidate weights are identical to the baseline; no release was created.")

    def install_immutable(source: Path, digest: str) -> None:
        destination = release / digest
        if destination.exists():
            verify_artifact(destination)
            if verify_artifact(destination)["weights_sha256"] != digest:
                raise ValueError("Content-addressed release directory contains the wrong artifact.")
            return
        staging = release / (digest + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if verify_artifact(staging)["weights_sha256"] != digest:
            shutil.rmtree(staging)
            raise ValueError("Staged release artifact hash does not match its content address.")
        os.replace(staging, destination)

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
    if not pointer.is_file():
        return None
    data = json.loads(pointer.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version) or not (release / version).is_dir():
        raise ValueError("Existing release pointer is invalid; refusing to overwrite it.")
    verify_artifact(release / version)
    return version


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rollback_release(release_dir: str | Path, version: str | None = None) -> dict[str, Any]:
    release = Path(release_dir).expanduser().resolve()
    current = _read_release_version(release)
    if current is None:
        raise FileNotFoundError("No current release exists.")
    pointer_path = release / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    target = version or pointer.get("previous_version")
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-f]{64}", target) or target == current:
        raise ValueError("A distinct, valid content-hash release version is required for rollback.")
    target_path = release / target
    if not target_path.is_dir():
        raise FileNotFoundError(f"Release version does not exist: {target}")
    verify_artifact(target_path)
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
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    export_path = work / "approved-feedback.jsonl"
    manifest = FeedbackStore(database).export_approved(export_path, min_rating=min_rating)
    if manifest["records"] == 0:
        raise RuntimeError("No new consented feedback meets the minimum rating; no training was started.")
    train_tokens = work / "feedback-train.bin"
    train_tokens.with_name(train_tokens.name + ".json").unlink(missing_ok=True)
    train_tokens.unlink(missing_ok=True)
    encode_jsonl(str(export_path), str(train_tokens), str(tokenizer_path), "text", tokenizer.eos_id())
    config = load_config(config_path)
    training_output = work / "candidate"
    if training_output.exists():
        import shutil
        shutil.rmtree(training_output)
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
