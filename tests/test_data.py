import struct

import torch

from arachne_small.data import TokenBlockDataset


def test_binary_token_dataset_returns_shifted_windows(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(struct.pack("<10I", *range(10)))
    dataset = TokenBlockDataset(path, seq_len=4)

    assert len(dataset) == 2
    inputs, targets = dataset[0]
    assert torch.equal(inputs, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets, torch.tensor([0, 1, 2, 3, 4]))
    inputs, targets = dataset[1]
    assert torch.equal(inputs, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(targets, torch.tensor([4, 5, 6, 7, 8]))


def test_binary_token_dataset_rejects_corrupt_file_size(tmp_path):
    path = tmp_path / "broken.bin"
    path.write_bytes(b"abc")
    try:
        TokenBlockDataset(path, seq_len=4)
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:
        raise AssertionError("Corrupt binary token stream was accepted.")
