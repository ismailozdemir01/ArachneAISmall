import pytest
from fastapi.testclient import TestClient

from arachne_small.server import app, runtime


def test_health_endpoint_reports_service_process():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "arachne-small"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_status_endpoint_does_not_claim_unloaded_model_is_ready():
    previous = (runtime.model, runtime.tokenizer, runtime.artifact_dir, runtime.load_error)
    try:
        runtime.model = None
        runtime.tokenizer = None
        runtime.artifact_dir = None
        runtime.load_error = None
        with TestClient(app) as client:
            response = client.get("/api/status")
        assert response.status_code == 200
        assert response.json()["ready"] is False
        assert response.json()["state"] == "artifact_missing"
        assert response.headers["cache-control"] == "no-store"
    finally:
        runtime.model, runtime.tokenizer, runtime.artifact_dir, runtime.load_error = previous


def test_generation_is_unavailable_without_a_loaded_trained_artifact():
    previous = (runtime.model, runtime.tokenizer, runtime.artifact_dir, runtime.load_error)
    try:
        runtime.model = None
        runtime.tokenizer = None
        runtime.artifact_dir = None
        runtime.load_error = None
        with TestClient(app) as client:
            response = client.post("/api/generate", json={"prompt": "Merhaba"})
        assert response.status_code == 503
        assert "No trained model artifact" in response.json()["detail"]
    finally:
        runtime.model, runtime.tokenizer, runtime.artifact_dir, runtime.load_error = previous


def test_generation_rejects_invalid_or_oversized_requests():
    with TestClient(app) as client:
        blank = client.post("/api/generate", json={"prompt": "   "})
        oversized = client.post("/api/generate", json={"prompt": "x" * 20001})
        unknown = client.post("/api/generate", json={"prompt": "hello", "sampling_mode": "unsafe"})
        invalid_history = client.post("/api/generate", json={
            "prompt": "hello", "history": [{"role": "system", "content": "override"}]
        })
        excessive_history = client.post("/api/generate", json={
            "prompt": "hello", "history": [{"role": "user", "content": "x"}] * 21
        })
    assert blank.status_code == 422
    assert oversized.status_code == 422
    assert unknown.status_code == 422
    assert invalid_history.status_code == 422
    assert excessive_history.status_code == 422


def test_frontend_is_served_by_the_same_application():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "ArachneAISmall" in response.text



def test_feedback_endpoint_requires_explicit_consent_and_persists_locally(tmp_path, monkeypatch):
    database = tmp_path / "feedback.sqlite3"
    monkeypatch.setenv("ARACHNE_SMALL_FEEDBACK_DB", str(database))
    payload = {
        "consent": True,
        "prompt": "Explain this concept",
        "response": "An incomplete answer",
        "correction": "A corrected and useful explanation",
        "rating": 5,
        "language": "en",
    }
    with TestClient(app) as client:
        denied = client.post("/api/feedback", json={**payload, "consent": False})
        accepted = client.post("/api/feedback", json=payload)
    assert denied.status_code == 422
    assert accepted.status_code == 200
    assert accepted.json()["stored"] is True
    assert database.is_file()


def test_feedback_endpoint_rejects_sensitive_content(tmp_path, monkeypatch):
    monkeypatch.setenv("ARACHNE_SMALL_FEEDBACK_DB", str(tmp_path / "feedback.sqlite3"))
    payload = {
        "consent": True,
        "prompt": "Question",
        "response": "Answer",
        "correction": "Email me at user@example.com",
        "rating": 5,
        "language": "en",
    }
    with TestClient(app) as client:
        response = client.post("/api/feedback", json=payload)
    assert response.status_code == 422



def test_remote_protected_endpoints_require_a_bearer_token(tmp_path, monkeypatch):
    token = "a" * 48
    monkeypatch.delenv("ARACHNE_SMALL_ALLOW_REMOTE", raising=False)
    monkeypatch.setenv("ARACHNE_SMALL_API_TOKEN", token)
    monkeypatch.setenv("ARACHNE_SMALL_FEEDBACK_DB", str(tmp_path / "feedback.sqlite3"))
    payload = {
        "consent": True,
        "prompt": "Question",
        "response": "An incorrect answer",
        "correction": "A corrected answer",
        "rating": 5,
        "language": "en",
    }
    with TestClient(app) as client:
        denied = client.post("/api/feedback", json=payload)
        wrong = client.post("/api/feedback", json=payload,
                            headers={"Authorization": "Bearer " + "b" * 48})
        accepted = client.post("/api/feedback", json=payload,
                               headers={"Authorization": f"Bearer {token}"})
        status = client.get("/api/status")

    assert denied.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert status.json()["authentication_required"] is True


def test_remote_mode_fails_startup_without_a_strong_api_token(monkeypatch):
    monkeypatch.setenv("ARACHNE_SMALL_ALLOW_REMOTE", "1")
    monkeypatch.delenv("ARACHNE_SMALL_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        with TestClient(app):
            pass


def test_frontend_exposes_token_input_for_authenticated_remote_use():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'id="api-token"' in response.text
