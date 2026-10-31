import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TokenBlockDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        tokens: list[int] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                values = row.get("input_ids")
                if not isinstance(values, list) or not all(isinstance(x, int) for x in values):
                    raise ValueError(f"Invalid input_ids at {self.path}:{line_no}")
                tokens.extend(values)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.length = max(0, (len(self.tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        start = index * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        if chunk.numel() != self.seq_len + 1:
            raise IndexError(index)
        return chunk[:-1], chunk

def encode_jsonl(input_path: str, output_path: str, tokenizer_path: str,
                 text_field: str = "text", eos_id: int | None = None) -> int:
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    count = 0
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, out_type=int)
            if eos_id is not None:
                ids.append(eos_id)
            dst.write(json.dumps({"input_ids": ids}, ensure_ascii=False) + "\n")
            count += len(ids)
    return count
