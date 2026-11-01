import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unicodedata
from pathlib import Path


def _normalize_document(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    source_path = Path(input_path).expanduser().resolve()
    requested_destination = Path(output_dir).expanduser()
    if requested_destination.is_symlink():
        raise ValueError("Corpus output directory cannot be a symbolic link.")
    destination = requested_destination.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not text_field:
        raise ValueError("text_field must not be empty.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not source.strip() or not license_name.strip():
        raise ValueError("Dataset source and license must be documented.")
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    manifest_path = destination / "dataset-manifest.json"
    if any(path.exists() or path.is_symlink() for path in (train_path, validation_path, manifest_path)):
        raise FileExistsError(f"Prepared dataset outputs already exist in {destination}")
    temporary_root = Path(tempfile.mkdtemp(prefix=".arachne-corpus-", dir=destination))
    database_path = temporary_root / "dedup-index.sqlite3"
    temporary_paths = [
        temporary_root / "train.jsonl",
        temporary_root / "validation.jsonl",
        temporary_root / "dataset-manifest.json",
    ]
    input_digest = hashlib.sha256()
    train_count = validation_count = duplicate_count = skipped_count = 0
    connection = sqlite3.connect(database_path)
    published_paths: list[Path] = []
    success = False
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
            "train_sha256": _sha256_path(temporary_paths[0]),
            "validation_file": validation_path.name,
            "validation_sha256": _sha256_path(temporary_paths[1]),
        }
        temporary_paths[2].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for staged, final in zip(temporary_paths, (train_path, validation_path, manifest_path)):
            if final.exists() or final.is_symlink():
                raise FileExistsError(f"Prepared dataset output appeared during processing: {final}")
            os.replace(staged, final)
            published_paths.append(final)
        success = True
        return manifest
    finally:
        connection.close()
        if not success:
            for path in published_paths:
                path.unlink(missing_ok=True)
        shutil.rmtree(temporary_root, ignore_errors=True)


def export_text(input_path: str, output_path: str, text_field: str = "text") -> int:
    """Export normalized JSONL document text as one UTF-8 document per line."""
    source = Path(input_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not text_field:
        raise ValueError("text_field must not be empty.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    count = 0
    try:
        with source.open("r", encoding="utf-8") as src, temporary.open("w", encoding="utf-8", newline="\n") as dst:
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {source}:{line_no}") from exc
                if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                    continue
                text = _normalize_document(row[text_field])
                if text:
                    dst.write(text + "\n")
                    count += 1
        if count == 0:
            raise ValueError(f"No usable text documents were found in {source}")
        os.replace(temporary, target)
        return count
    finally:
        temporary.unlink(missing_ok=True)
