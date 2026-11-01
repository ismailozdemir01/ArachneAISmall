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
    pending = store.export_approved(output, min_rating=4)
    assert pending["records"] == 0
    store.review(accepted, approved=True)
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


def test_feedback_rejects_credentials_and_non_learning_corrections(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    with pytest.raises(ValueError, match="credentials or personal contact"):
        store.add(
            consent=True,
            prompt="What is this?",
            response="Fine",
            correction="Contact admin@example.com for details",
            rating=5,
            language="en",
        )
    with pytest.raises(ValueError, match="must differ"):
        store.add(
            consent=True,
            prompt="Explain this",
            response="Same answer",
            correction="Same answer",
            rating=5,
            language="en",
        )



def test_feedback_must_be_explicitly_reviewed_and_approved_before_export(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    record_id = store.add(
        consent=True, prompt="Question", response="Wrong answer",
        correction="Reviewed correction", rating=5, language="en",
    )
    output = tmp_path / "approved.jsonl"
    assert store.export_approved(output)["records"] == 0

    store.review(record_id, approved=False)
    assert store.export_approved(output)["records"] == 0

    store.review(record_id, approved=True)
    assert store.export_approved(output)["records"] == 1

    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    store.mark_exported(manifest["record_hashes"])
    with pytest.raises(ValueError, match="cannot be re-reviewed"):
        store.review(record_id, approved=False)


def test_feedback_review_rejects_unknown_ids_and_invalid_actions(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    with pytest.raises(KeyError):
        store.review(999, approved=True)
    with pytest.raises(ValueError):
        store.review(0, approved=True)
    with pytest.raises(TypeError):
        store.review(1, approved=1)



def test_pending_feedback_queue_exposes_review_context_and_excludes_decided_records(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    pending_id = store.add(
        consent=True, prompt="Review this prompt", response="Model output",
        correction="Suggested correction", rating=5, language="en",
    )
    rejected_id = store.add(
        consent=True, prompt="Reject this prompt", response="Model output",
        correction="Rejected correction", rating=4, language="tr",
    )
    store.review(rejected_id, approved=False)

    pending = store.list_pending(limit=10)
    assert [record["id"] for record in pending] == [pending_id]
    assert pending[0]["prompt"] == "Review this prompt"
    assert pending[0]["response"] == "Model output"
    assert pending[0]["correction"] == "Suggested correction"

    with pytest.raises(ValueError):
        store.list_pending(limit=0)
