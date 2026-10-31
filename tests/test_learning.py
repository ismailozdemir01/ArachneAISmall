import json

import pytest

from arachne_small.learning import FeedbackStore


def test_feedback_requires_explicit_consent(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    with pytest.raises(PermissionError):
        store.add(
            consent=False,
            prompt="Soru",
            response="Yanıt",
            correction="Düzeltme",
            rating=5,
            language="tr",
        )


def test_feedback_is_validated_and_duplicate_submission_is_idempotent(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    values = {
        "consent": True,
        "prompt": "Soru",
        "response": "İlk yanıt",
        "correction": "Doğru yanıt",
        "rating": 5,
        "language": "tr",
    }
    first = store.add(**values)
    second = store.add(**values)
    assert first == second
    with pytest.raises(ValueError):
        store.add(**{**values, "rating": 6})
    with pytest.raises(ValueError):
        store.add(**{**values, "language": "de"})


def test_export_uses_corrections_and_only_eligible_unexported_records(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    low = store.add(
        consent=True, prompt="Düşük puanlı soru", response="Yanıt",
        correction="Düzeltme", rating=2, language="tr",
    )
    accepted = store.add(
        consent=True, prompt="Yüksek puanlı soru", response="Yanlış yanıt",
        correction="Onaylanan düzeltme", rating=5, language="tr",
    )
    output = tmp_path / "approved.jsonl"
    manifest = store.export_approved(output, min_rating=4)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["records"] == 1
    assert accepted in [row["provenance"]["feedback_id"] for row in rows]
    assert low not in [row["provenance"]["feedback_id"] for row in rows]
    assert "Onaylanan düzeltme" in rows[0]["text"]
    assert "Yanlış yanıt" not in rows[0]["text"]
    store.mark_exported(manifest["record_hashes"])
    second = store.export_approved(output, min_rating=4)
    assert second["records"] == 0


def test_feedback_export_manifest_hash_matches_file(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.add(
        consent=True, prompt="Question", response="Response",
        correction="Correction", rating=4, language="en",
    )
    output = tmp_path / "approved.jsonl"
    manifest = store.export_approved(output)
    import hashlib
    assert manifest["export_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(".jsonl.manifest.json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["records"] == 1
