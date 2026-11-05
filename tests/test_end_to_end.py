import hashlib
import json
import struct

from fastapi.testclient import TestClient

from arachne_small.artifacts import verify_artifact
from arachne_small.config import Config, ModelConfig, TrainingConfig
from arachne_small.learning import FeedbackStore, rollback_release, run_learning_cycle
from arachne_small.server import app
from arachne_small.trainer import train


def _write_token_stream(path, ids, tokenizer_hash, vocab_size, special_tokens):
    raw = struct.pack(f"<{len(ids)}I", *ids)
    path.write_bytes(raw)
    metadata = {
        "format_version": 1,
        "token_format": "little-endian-uint32",
        "token_count": len(ids),
        "token_file": path.name,
        "token_file_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_hash,
        "eos_id": special_tokens["eos_id"],
        "special_tokens": special_tokens,
    }
    path.with_name(path.name + ".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_full_verified_learning_release_and_module_activation_flow(tmp_path, monkeypatch):
    import sentencepiece as spm

    # Build a real, tiny baseline artifact; no pretrained weights or fake model are used.
    corpus = tmp_path / "tokenizer-corpus.txt"
    corpus.write_text(
        "\n".join(
            f"Turkish English Arachne model training example number {i} "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789"
            for i in range(400)
        ),
        encoding="utf-8",
    )
    prefix = tmp_path / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=128,
        model_type="bpe", character_coverage=1.0, hard_vocab_limit=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = tokenizer.vocab_size()
    special = {
        "pad_id": tokenizer.pad_id(), "unk_id": tokenizer.unk_id(),
        "bos_id": tokenizer.bos_id(), "eos_id": tokenizer.eos_id(),
    }
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    token_ids = [(i % (vocab_size - 4)) + 4 for i in range(96)]
    train_tokens = tmp_path / "train.bin"
    validation_tokens = tmp_path / "validation.bin"
    _write_token_stream(train_tokens, token_ids, tokenizer_hash, vocab_size, special)
    _write_token_stream(validation_tokens, token_ids, tokenizer_hash, vocab_size, special)

    base_dir = tmp_path / "base-artifact"
    config = Config(
        model=ModelConfig(
            vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2,
            n_kv_heads=1, max_seq_len=8, intermediate_size=32,
        ),
        training=TrainingConfig(
            batch_size=2, gradient_accumulation_steps=1, max_steps=1,
            learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=0,
            eval_interval=1, save_interval=1, precision="fp32",
            num_workers=0, output_dir=str(base_dir),
        ),
    )
    train(config, str(train_tokens), str(validation_tokens), str(tokenizer_path))

    # Artifact verification is the first gate.
    baseline = verify_artifact(base_dir)
    assert baseline["parameter_count"] == config.model.parameter_count()

    # API auth is checked before creating/activating a generated module.
    token = "integration-test-token-" + ("x" * 40)
    monkeypatch.setenv("ARACHNE_SMALL_API_TOKEN", token)
    monkeypatch.delenv("ARACHNE_SMALL_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("ARACHNE_SMALL_MODULE_REGISTRIES", raising=False)
    monkeypatch.delenv("ARACHNE_SMALL_AUTO_LEARN", raising=False)
    monkeypatch.delenv("ARACHNE_SMALL_ARTIFACT", raising=False)
    module_dir = tmp_path / "modules"
    monkeypatch.setenv("ARACHNE_SMALL_MODULE_DIR", str(module_dir))
    module_spec = {
        "id": "text-cleaner",
        "version": "1.0.0",
        "description": "Trim and normalize text.",
        "operations": [{"op": "trim"}, {"op": "lowercase"}],
        "tests": [
            {"input": " HELLO ", "expected": "hello"},
            {"input": " World ", "expected": "world"},
        ],
    }
    with TestClient(app) as client:
        denied = client.post("/api/modules/create", json={"module": module_spec})
        created = client.post(
            "/api/modules/create",
            json={"module": module_spec},
            headers={"Authorization": f"Bearer {token}"},
        )
        executed = client.post(
            "/api/modules/text-cleaner/run",
            json={"text": "  Arachne  "},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert denied.status_code == 401
    assert created.status_code == 200
    assert created.json()["state"] == "active"
    assert executed.status_code == 200
    assert executed.json()["output"] == "arachne"

    # Consented feedback must be explicitly reviewed before it can train.
    database = tmp_path / "feedback.sqlite3"
    feedback = FeedbackStore(database)
    record_id = feedback.add(
        consent=True,
        prompt="What does Arachne do?",
        response="An incorrect short answer.",
        correction="Arachne is a Turkish-English language model project.",
        rating=5,
        language="en",
    )
    feedback.review(record_id, approved=True)
    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        "model:\n"
        f"  vocab_size: {vocab_size}\n"
        "  d_model: 16\n  n_layers: 1\n  n_heads: 2\n"
        "  n_kv_heads: 1\n  max_seq_len: 8\n  intermediate_size: 32\n"
        "training:\n"
        "  batch_size: 1\n  gradient_accumulation_steps: 1\n"
        "  max_steps: 1\n  learning_rate: 0.0001\n"
        "  min_learning_rate: 0.00001\n  warmup_steps: 0\n"
        "  eval_interval: 1\n  save_interval: 1\n"
        "  precision: fp32\n  num_workers: 0\n"
        f"  output_dir: '{(tmp_path / 'unused-output').as_posix()}'\n",
        encoding="utf-8",
    )

    # One real fine-tuning cycle, fixed validation evaluation and gated release.
    result = run_learning_cycle(
        database=database,
        base_artifact=base_dir,
        config_path=training_config,
        validation_tokens=validation_tokens,
        work_dir=tmp_path / "learning-work",
        release_dir=tmp_path / "releases",
        min_rating=4,
        max_loss_regression=1_000_000.0,
        batch_size=1,
        max_batches=1,
    )
    release = result["release"]
    candidate_version = release["version"]
    baseline_version = release["previous_version"]
    assert result["feedback_records"] == 1
    assert release["candidate"]["batches_evaluated"] == 1
    assert (tmp_path / "releases" / candidate_version).is_dir()
    assert (tmp_path / "releases" / baseline_version).is_dir()

    # Roll back to the preserved baseline and verify the restored artifact again.
    rolled_back = rollback_release(tmp_path / "releases", baseline_version)
    assert rolled_back["version"] == baseline_version
    restored = verify_artifact(tmp_path / "releases" / baseline_version)
    assert restored["weights_sha256"] == baseline["weights_sha256"]
