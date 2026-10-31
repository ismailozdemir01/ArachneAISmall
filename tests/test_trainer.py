import hashlib
import json
import struct

from arachne_small.config import Config, ModelConfig, TrainingConfig
from arachne_small.trainer import train


def _write_tokens(path, token_ids, tokenizer_hash):
    encoded = struct.pack(f"<{len(token_ids)}I", *token_ids)
    path.write_bytes(encoded)
    metadata = {
        "format_version": 1,
        "token_format": "little-endian-uint32",
        "token_count": len(token_ids),
        "token_file": path.name,
        "token_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "tokenizer_vocab_size": 16,
        "tokenizer_sha256": tokenizer_hash,
    }
    path.with_name(path.name + ".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_tiny_training_writes_versioned_model_and_checkpoint(tmp_path):
    from safetensors.torch import load_file

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"tiny-tokenizer-artifact")
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    train_path = tmp_path / "train.bin"
    validation_path = tmp_path / "validation.bin"
    _write_tokens(train_path, [index % 16 for index in range(80)], tokenizer_hash)
    _write_tokens(validation_path, [index % 16 for index in range(80)], tokenizer_hash)

    config = Config(
        model=ModelConfig(
            vocab_size=16, d_model=16, n_layers=1, n_heads=2,
            n_kv_heads=1, max_seq_len=8, intermediate_size=32,
        ),
        training=TrainingConfig(
            batch_size=2, gradient_accumulation_steps=1, max_steps=1,
            learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=0,
            eval_interval=1, save_interval=1, precision="fp32",
            num_workers=0, output_dir=str(tmp_path / "run"),
        ),
    )

    artifact_path = train(config, str(train_path), str(validation_path), str(tokenizer_path))
    artifact = load_file(str(artifact_path))
    artifact_config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "run" / "artifact-manifest.json").read_text(encoding="utf-8"))

    assert "token_embedding.weight" in artifact
    assert artifact_config["format_version"] == 1
    assert artifact_config["tokenizer"]["sha256"] == tokenizer_hash
    assert manifest["weights_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert (tmp_path / "run" / "tokenizer.model").read_bytes() == tokenizer_path.read_bytes()
    assert (tmp_path / "run" / "checkpoints" / "step-00000001.pt").is_file()
    assert artifact and all(tensor.is_floating_point() for tensor in artifact.values())
