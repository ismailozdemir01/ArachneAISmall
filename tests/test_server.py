from fastapi.testclient import TestClient

from arachne_small.server import app, runtime


def test_health_endpoint_reports_service_process():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "arachne-small"}


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
    assert blank.status_code == 422
    assert oversized.status_code == 422
    assert unknown.status_code == 422


def test_frontend_is_served_by_the_same_application():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "ArachneAISmall" in response.text
