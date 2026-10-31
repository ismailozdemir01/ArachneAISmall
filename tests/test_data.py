import json
import struct

import sentencepiece as spm
import torch

from arachne_small.data import TokenBlockDataset, encode_jsonl


def test_binary_token_dataset_returns_shifted_windows(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(struct.pack("<10I", *range(10)))
    dataset = TokenBlockDataset(path, seq_len=4)

    assert len(dataset) == 2
    inputs, targets = dataset[0]
    assert torch.equal(inputs, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets, torch.tensor([1, 2, 3, 4]))
    inputs, targets = dataset[1]
    assert torch.equal(inputs, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(targets, torch.tensor([5, 6, 7, 8]))


def test_binary_token_dataset_rejects_corrupt_file_size(tmp_path):
    path = tmp_path / "broken.bin"
    path.write_bytes(b"abc")
    try:
        TokenBlockDataset(path, seq_len=4)
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:
        raise AssertionError("Corrupt binary token stream was accepted.")


def test_encode_jsonl_records_tokenizer_and_file_provenance(tmp_path):
    corpus = tmp_path / "tokenizer-corpus.txt"
    corpus.write_text(
        "\n".join(
            f"Türkçe English data pipeline document {index} "
            "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    prefix = tmp_path / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=64,
        model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps({"text": "Türkçe ve English eğitim belgesi."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "encoded" / "tokens.bin"

    count = encode_jsonl(str(source), str(output), str(tokenizer_path))
    metadata = json.loads(output.with_name(output.name + ".json").read_text(encoding="utf-8"))

    assert count == metadata["token_count"]
    assert metadata["tokenizer_vocab_size"] == processor.vocab_size()
    assert metadata["eos_id"] == processor.eos_id()
    assert metadata["special_tokens"]["eos_id"] == processor.eos_id()
    assert metadata["token_file_sha256"]
    assert TokenBlockDataset(output, seq_len=2)[0][0].numel() == 2
