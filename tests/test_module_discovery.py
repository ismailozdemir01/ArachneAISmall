import hashlib
import json

from arachne_small.modules import ModuleRegistry
import arachne_small.modules as backend


def test_trusted_registry_discovery_auto_activates_digest_verified_module(tmp_path, monkeypatch):
    spec = {
        "id": "text-cleaner",
        "version": "1.0.0",
        "description": "Normalize simple text input.",
        "operations": [{"op": "trim"}, {"op": "lowercase"}],
        "tests": [
            {"input": " HELLO ", "expected": "hello"},
            {"input": " World ", "expected": "world"},
        ],
    }
    raw_module = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw_module).hexdigest()
    index = {
        "format": "arachne-module-index-v1",
        "modules": [{
            "url": "https://raw.githubusercontent.com/example/repo/0123456789abcdef0123456789abcdef01234567/modules/text-cleaner.json",
            "sha256": digest,
        }],
    }
    raw_index = json.dumps(index).encode()

    def retrieve(url, limit):
        return raw_index if url.endswith("/index.json") else raw_module

    monkeypatch.setattr(backend, "_fetch_json", retrieve)
    registry = ModuleRegistry(tmp_path / "modules")
    result = registry.discover([
        "https://raw.githubusercontent.com/example/repo/0123456789abcdef0123456789abcdef01234567/index.json"
    ])
    assert len(result) == 1
    assert result[0]["state"] == "active"
    assert registry.run("text-cleaner", "  HELLO  ") == "hello"
