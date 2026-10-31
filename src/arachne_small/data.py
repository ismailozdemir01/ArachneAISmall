import json
import struct
from pathlib import Path
import torch
from torch.utils.data import Dataset

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
        return chunk[:-1], chunk

def encode_jsonl(input_path: str, output_path: str, tokenizer_path: str,
                 text_field: str = "text", eos_id: int | None = 3) -> int:
    """Encode JSONL documents to a little-endian uint32 token stream without holding the corpus in RAM."""
    import sentencepiece as spm
    source = Path(input_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not text_field:
        raise ValueError("text_field must not be empty.")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    if eos_id is not None and not 0 <= eos_id < tokenizer.vocab_size():
        raise ValueError(f"eos_id {eos_id} is outside tokenizer vocabulary.")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with source.open("r", encoding="utf-8") as src, target.open("wb") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Each JSONL row must be an object at {source}:{line_no}")
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, out_type=int)
            if eos_id is not None:
                ids.append(eos_id)
            if any(token < 0 or token >= tokenizer.vocab_size() for token in ids):
                raise ValueError(f"Tokenizer produced an out-of-range token at {source}:{line_no}")
            if ids:
                dst.write(struct.pack(f"<{len(ids)}I", *ids))
                total += len(ids)
    if total == 0:
        target.unlink(missing_ok=True)
        raise ValueError(f"No usable text documents were found in {source}")
    return total
