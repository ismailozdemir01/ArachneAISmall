import hashlib
import json
import struct

import sentencepiece as spm

from arachne_small.config import Config, ModelConfig, TrainingConfig
from arachne_small.trainer import train


def _write_tokens(path, token_ids, tokenizer_hash, vocab_size, special_tokens):
    encoded = struct.pack(f"<{len(token_ids)}I", *token_ids)
    path.write_bytes(encoded)
    metadata = {
        "format_version": 1,
        "token_format": "little-endian-uint32",
        "token_count": len(token_ids),
        "token_file": path.name,
        "token_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_hash,
        "eos_id": special_tokens["eos_id"],
        "special_tokens": special_tokens,
    }
    path.with_name(path.name + ".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_tiny_training_writes_versioned_model_and_checkpoint(tmp_path):
    from safetensors.torch import load_file
    from arachne_small.artifacts import verify_artifact

    tokenizer_prefix = tmp_path / "tokenizer"
    tokenizer_corpus = tmp_path / "tokenizer-corpus.txt"
    tokenizer_corpus.write_text(
        "\n".join(
            f"Türkçe English ArachneAI language model training sample {index} "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789"
            for index in range(500)
        ),
        encoding="utf-8",
    )
    spm.SentencePieceTrainer.train(
        input=str(tokenizer_corpus), model_prefix=str(tokenizer_prefix),
        vocab_size=128, model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = processor.vocab_size()
    special_tokens = {
        "pad_id": processor.pad_id(), "unk_id": processor.unk_id(),
        "bos_id": processor.bos_id(), "eos_id": processor.eos_id(),
    }
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    train_path = tmp_path / "train.bin"
    validation_path = tmp_path / "validation.bin"
    token_ids = [index % vocab_size for index in range(80)]
    _write_tokens(train_path, token_ids, tokenizer_hash, vocab_size, special_tokens)
    _write_tokens(validation_path, token_ids, tokenizer_hash, vocab_size, special_tokens)

    config = Config(
        model=ModelConfig(
            vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2,
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
    verified = verify_artifact(tmp_path / "run")

    assert verified["parameter_count"] == config.model.parameter_count()
    assert "token_embedding.weight" in artifact
    assert artifact_config["format_version"] == 1
    assert artifact_config["tokenizer"]["sha256"] == tokenizer_hash
    assert manifest["weights_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["training_steps_completed"] == 1
    assert manifest["parameter_count"] == config.model.parameter_count()
    assert manifest["training_tokens_processed"] > 0
    assert manifest["final_validation_loss"] >= 0
    assert (tmp_path / "run" / "tokenizer.model").read_bytes() == tokenizer_path.read_bytes()
    assert (tmp_path / "run" / "checkpoints" / "step-00000001.pt").is_file()
    assert artifact and all(tensor.is_floating_point() for tensor in artifact.values())
