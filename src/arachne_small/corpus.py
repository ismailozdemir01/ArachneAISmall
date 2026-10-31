import hashlib
import json
import os
import sqlite3
import unicodedata
from pathlib import Path


def _normalize_document(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def prepare_corpus(
    input_path: str,
    output_dir: str,
    text_field: str = "text",
    validation_fraction: float = 0.01,
    seed: int = 1337,
    source: str = "",
    license_name: str = "",
) -> dict:
    """Deduplicate and deterministically split JSONL documents using disk-backed SQLite."""
    source_path = Path(input_path)
    destination = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not text_field:
        raise ValueError("text_field must not be empty.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if not source.strip() or not license_name.strip():
        raise ValueError("Dataset source and license must be documented.")
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    manifest_path = destination / "dataset-manifest.json"
    database_path = destination / ".dedup-index.sqlite3"
    temporary_paths = [
        destination / ".train.jsonl.tmp",
        destination / ".validation.jsonl.tmp",
        destination / ".dataset-manifest.json.tmp",
    ]
    if any(path.exists() for path in (train_path, validation_path, manifest_path)):
        raise FileExistsError(f"Prepared dataset outputs already exist in {destination}")
    input_digest = hashlib.sha256()
    train_count = validation_count = duplicate_count = skipped_count = 0
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY) WITHOUT ROWID")
        with source_path.open("rb") as raw_input, temporary_paths[0].open("w", encoding="utf-8") as train_out, temporary_paths[1].open("w", encoding="utf-8") as validation_out:
            for line_no, raw_line in enumerate(raw_input, 1):
                input_digest.update(raw_line)
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid UTF-8 JSONL at {source_path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Each JSONL row must be an object at {source_path}:{line_no}")
                value = row.get(text_field)
                if not isinstance(value, str):
                    skipped_count += 1
                    continue
                normalized = _normalize_document(value)
                if not normalized:
                    skipped_count += 1
                    continue
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                try:
                    connection.execute("INSERT INTO seen(digest) VALUES (?)", (digest,))
                except sqlite3.IntegrityError:
                    duplicate_count += 1
                    continue
                row[text_field] = normalized
                split_digest = hashlib.sha256(f"{seed}:{digest}".encode("ascii")).digest()
                split_value = int.from_bytes(split_digest[:8], "big") / 2**64
                output = validation_out if split_value < validation_fraction else train_out
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if output is validation_out:
                    validation_count += 1
                else:
                    train_count += 1
        connection.commit()
        if train_count == 0 or validation_count == 0:
            raise ValueError(
                "The deterministic split produced an empty train or validation set. "
                "Provide more unique documents or adjust validation_fraction."
            )
        manifest = {
            "format_version": 1,
            "input": str(source_path),
            "input_sha256": input_digest.hexdigest(),
            "source": source.strip(),
            "license": license_name.strip(),
            "text_field": text_field,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train_documents": train_count,
            "validation_documents": validation_count,
            "duplicates_removed": duplicate_count,
            "invalid_or_empty_documents_skipped": skipped_count,
            "train_file": train_path.name,
            "validation_file": validation_path.name,
        }
        temporary_paths[2].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_paths[0], train_path)
        os.replace(temporary_paths[1], validation_path)
        os.replace(temporary_paths[2], manifest_path)
        return manifest
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        for path in temporary_paths:
            path.unlink(missing_ok=True)
