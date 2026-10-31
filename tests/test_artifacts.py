from pathlib import Path

import pytest

from arachne_small.artifacts import _artifact_file, _expected_state_shapes
from arachne_small.config import ModelConfig


def test_artifact_file_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        _artifact_file(tmp_path, "../outside.safetensors")


def test_expected_artifact_shapes_cover_all_transformer_parameters():
    config = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4,
        n_kv_heads=2, max_seq_len=16, intermediate_size=64,
    )
    shapes = _expected_state_shapes(config)
    assert shapes["token_embedding.weight"] == (64, 32)
    assert shapes["layers.0.attn.k_proj.weight"] == (16, 32)
    assert shapes["layers.1.mlp.down_proj.weight"] == (32, 64)
    assert shapes["lm_head.weight"] == (64, 32)
