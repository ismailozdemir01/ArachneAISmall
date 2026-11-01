import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
import torch
from torch.utils.data import Dataset

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

class TokenBlockDataset(Dataset):
    """Memory-mapped uint32 token stream; each sample is one fixed-length next-token window."""
    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        size = self.path.stat().st_size
        if size % 4:
            raise ValueError(f"Token file size must be divisible by 4 bytes: {self.path}")
        token_count = size // 4
        if token_count < seq_len + 1:
            raise ValueError(f"Token file contains {token_count} tokens; at least {seq_len + 1} are required.")
        self.tokens = torch.from_file(str(self.path), shared=False, size=token_count, dtype=torch.int32)
        self.seq_len = seq_len
        self.length = (token_count - 1) // seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        start = index * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1].to(dtype=torch.long)
        return chunk[:-1], chunk[1:]

def encode_jsonl(input_path: str, output_path: str, tokenizer_path: str,
                 text_field: str = "text", eos_id: int = 3) -> int:
    """Encode JSONL documents to an atomic little-endian uint32 stream plus provenance metadata."""
    import sentencepiece as spm
    source = Path(input_path)
    target = Path(output_path).expanduser()
    tokenizer_file = Path(tokenizer_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(tokenizer_file)
    if not text_field:
        raise ValueError("text_field must not be empty.")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    if not isinstance(eos_id, int) or isinstance(eos_id, bool) or eos_id != tokenizer.eos_id():
        raise ValueError(f"eos_id must equal the tokenizer's configured EOS ID ({tokenizer.eos_id()}).")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_name(target.name + ".json")
    if any(path.exists() or path.is_symlink() for path in (target, metadata_path)):
        raise FileExistsError(f"Token output or provenance metadata already exists: {target}")
    token_fd, token_temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    metadata_fd, metadata_temp_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(token_temp_name)
    metadata_temporary = Path(metadata_temp_name)
    metadata_fd_open = True
    token_published = False
    metadata_published = False
    input_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(token_fd, "wb") as dst:
            with source.open("rb") as src:
                for line_no, raw_line in enumerate(src, 1):
                    input_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid UTF-8 JSONL at {source}:{line_no}: {exc}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Each JSONL row must be an object at {source}:{line_no}")
                    text = row.get(text_field)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    ids = tokenizer.encode(text, out_type=int)
                    ids.append(eos_id)
                    if any(token < 0 or token >= tokenizer.vocab_size() for token in ids):
                        raise ValueError(f"Tokenizer produced an out-of-range token at {source}:{line_no}")
                    if ids:
                        encoded = struct.pack(f"<{len(ids)}I", *ids)
                        dst.write(encoded)
                        token_digest.update(encoded)
                        total += len(ids)
        if total == 0:
            raise ValueError(f"No usable text documents were found in {source}")
        metadata = {
            "format_version": 1,
            "token_format": "little-endian-uint32",
            "token_count": total,
            "token_file": target.name,
            "token_file_sha256": token_digest.hexdigest(),
            "input_file": source.name,
            "input_sha256": input_digest.hexdigest(),
            "tokenizer_file": tokenizer_file.name,
            "tokenizer_sha256": sha256_file(tokenizer_file),
            "tokenizer_vocab_size": tokenizer.vocab_size(),
            "eos_id": eos_id,
            "special_tokens": {
                "pad_id": tokenizer.pad_id(),
                "unk_id": tokenizer.unk_id(),
                "bos_id": tokenizer.bos_id(),
                "eos_id": tokenizer.eos_id(),
            },
            "text_field": text_field,
        }
        with os.fdopen(metadata_fd, "w", encoding="utf-8") as metadata_output:
            metadata_fd_open = False
            metadata_output.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            metadata_output.flush()
            os.fsync(metadata_output.fileno())
        if target.exists() or target.is_symlink() or metadata_path.exists() or metadata_path.is_symlink():
            raise FileExistsError(f"Token output appeared during encoding: {target}")
        os.replace(temporary, target)
        token_published = True
        os.replace(metadata_temporary, metadata_path)
        metadata_published = True
        if os.name == "posix":
            os.chmod(target, 0o600)
            os.chmod(metadata_path, 0o600)
    finally:
        if metadata_fd_open:
            os.close(metadata_fd)
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        if token_published and not metadata_published:
            target.unlink(missing_ok=True)
    return total
