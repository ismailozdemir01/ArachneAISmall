import hashlib
import json

import pytest

from arachne_small.modules import ModuleRegistry, ModuleRejected, execute_module


def _spec():
    return {
        "id": "text-cleaner",
        "version": "1.0.0",
        "description": "Normalize simple text input.",
        "operations": [{"op": "trim"}, {"op": "replace_literal", "old": "  ", "new": " "}],
        "tests": [
            {"input": "  hello  ", "expected": "hello"},
            {"input": "  a  b  ", "expected": "a b"},
            {"input": "word", "expected": "word"},
        ],
    }


def test_self_generated_module_is_verified_then_atomically_activated(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    candidate = registry.create_candidate(_spec())
    assert candidate["state"] == "verified_candidate"
    assert candidate["tests_passed"] == 3
    assert registry.list_active() == []

    activated = registry.activate(candidate["sha256"])
    assert activated["state"] == "active"
    assert registry.run("text-cleaner", "  A  B  ") == "A B"
    assert registry.list_active() == [{
        "id": "text-cleaner",
        "version": "1.0.0",
        "sha256": candidate["sha256"],
    }]


def test_module_creation_rejects_untrusted_executable_operations(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    spec = _spec()
    spec["operations"] = [{"op": "python", "code": "import os; os.system('whoami')"}]
    with pytest.raises(ModuleRejected, match="Unsupported"):
        registry.create_candidate(spec)


def test_module_creation_rejects_failed_behavior_vectors(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    spec = _spec()
    spec["tests"][0]["expected"] = "incorrect"
    with pytest.raises(ModuleRejected, match="tests failed"):
        registry.create_candidate(spec)


def test_module_candidate_hash_tampering_blocks_activation(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    candidate = registry.create_candidate(_spec())
    path = registry.candidates / f"{candidate['sha256']}.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ModuleRejected, match="hash"):
        registry.activate(candidate["sha256"])


def test_module_execution_enforces_input_and_output_limits(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    candidate = registry.create_candidate(_spec())
    registry.activate(candidate["sha256"])
    with pytest.raises(ValueError, match="100000"):
        registry.run("text-cleaner", "x" * 100001)


def test_external_module_source_requires_verified_digest(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    with pytest.raises(ModuleRejected, match="source digest"):
        registry.create_candidate(_spec(), source="https://raw.githubusercontent.com/example/repo/main/module.json")


def test_module_source_rejects_untrusted_hosts(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    spec = _spec()
    with pytest.raises(ModuleRejected, match="raw.githubusercontent.com"):
        registry.create_candidate(
            spec,
            source="https://example.com/module.json",
            source_sha256=hashlib.sha256(json.dumps(spec).encode()).hexdigest(),
        )


def test_active_module_pointer_symlink_is_rejected(tmp_path):
    registry = ModuleRegistry(tmp_path / "modules")
    candidate = registry.create_candidate(_spec())
    registry.activate(candidate["sha256"])
    pointer = registry.active / "text-cleaner.current.json"
    target = tmp_path / "pointer.json"
    target.write_text(pointer.read_text(encoding="utf-8"), encoding="utf-8")
    pointer.unlink()
    try:
        pointer.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform.")
    with pytest.raises(ModuleRejected, match="symbolic link"):
        registry.run("text-cleaner", "hello")
