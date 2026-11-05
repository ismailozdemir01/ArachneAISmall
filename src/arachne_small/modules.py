from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_MODULE_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_VERSION = re.compile(r"^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$")
_ALLOWED_HOSTS = {"raw.githubusercontent.com"}
_MAX_DOCUMENT_BYTES = 256 * 1024
_MAX_REGISTRY_BYTES = 1024 * 1024
_MAX_OPERATIONS = 32
_MAX_TESTS = 100


class ModuleRejected(ValueError):
    """A discovered or generated module failed the local safety policy."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _safe_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ModuleRejected("Module URL is invalid.")
    parsed = urllib.parse.urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.query or parsed.fragment):
        raise ModuleRejected("Only HTTPS raw.githubusercontent.com module sources are allowed.")
    return value


def _fetch_json(url: str, limit: int) -> bytes:
    request = urllib.request.Request(
        _safe_url(url),
        headers={"Accept": "application/json", "User-Agent": "ArachneAISmall-ModuleDiscovery/1.0"},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            if response.status != 200:
                raise ModuleRejected(f"Registry returned HTTP {response.status}.")
            if response.geturl() != url:
                raise ModuleRejected("Redirects are not allowed for module registry resources.")
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > limit:
                raise ModuleRejected("Registry document exceeds the configured size limit.")
            data = response.read(limit + 1)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if isinstance(exc, ModuleRejected):
            raise
        raise ModuleRejected(f"Could not retrieve module registry resource: {exc}") from exc
    if len(data) > limit:
        raise ModuleRejected("Registry document exceeds the configured size limit.")
    return data


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModuleRejected("HTTP redirects are not allowed during module discovery.")


def validate_module(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != {
        "id", "version", "description", "operations", "tests"
    }:
        raise ModuleRejected("Module manifest must contain exactly id, version, description, operations and tests.")
    module_id = spec["id"]
    version = spec["version"]
    description = spec["description"]
    operations = spec["operations"]
    tests = spec["tests"]
    if not isinstance(module_id, str) or not _MODULE_ID.fullmatch(module_id):
        raise ModuleRejected("Module id must be a lowercase identifier of 3-64 safe characters.")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ModuleRejected("Module version must use MAJOR.MINOR.PATCH.")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 500:
        raise ModuleRejected("Module description must contain 1-500 characters.")
    if not isinstance(operations, list) or not 1 <= len(operations) <= _MAX_OPERATIONS:
        raise ModuleRejected("A module must define 1-32 operations.")
    if not isinstance(tests, list) or not 2 <= len(tests) <= _MAX_TESTS:
        raise ModuleRejected("A module must provide 2-100 deterministic test vectors.")

    normalized_ops: list[dict[str, str]] = []
    for operation in operations:
        if not isinstance(operation, dict) or not isinstance(operation.get("op"), str):
            raise ModuleRejected("Every operation must be an object with an op field.")
        name = operation["op"]
        if name in {"trim", "lowercase", "uppercase"} and set(operation) == {"op"}:
            normalized_ops.append({"op": name})
        elif name in {"prepend", "append"} and set(operation) == {"op", "value"}:
            value = operation["value"]
            if not isinstance(value, str) or len(value) > 4096:
                raise ModuleRejected(f"{name} value must be a string of at most 4096 characters.")
            normalized_ops.append({"op": name, "value": value})
        elif name == "replace_literal" and set(operation) == {"op", "old", "new"}:
            old, new = operation["old"], operation["new"]
            if not isinstance(old, str) or not old or len(old) > 1024:
                raise ModuleRejected("replace_literal.old must be a non-empty string of at most 1024 characters.")
            if not isinstance(new, str) or len(new) > 1024:
                raise ModuleRejected("replace_literal.new must be a string of at most 1024 characters.")
            normalized_ops.append({"op": name, "old": old, "new": new})
        else:
            raise ModuleRejected(f"Unsupported or malformed operation: {name!r}")

    normalized_tests: list[dict[str, str]] = []
    for case in tests:
        if (not isinstance(case, dict) or set(case) != {"input", "expected"}
                or not isinstance(case["input"], str) or not isinstance(case["expected"], str)
                or len(case["input"]) > 16_000 or len(case["expected"]) > 16_000):
            raise ModuleRejected("Each test vector must contain bounded string input and expected fields.")
        normalized_tests.append({"input": case["input"], "expected": case["expected"]})

    return {
        "id": module_id,
        "version": version,
        "description": description.strip(),
        "operations": normalized_ops,
        "tests": normalized_tests,
    }


def execute_module(spec: dict[str, Any], text: str) -> str:
    module = validate_module(spec)
    if not isinstance(text, str) or len(text) > 100_000:
        raise ValueError("Module input must be a string of at most 100000 characters.")
    result = text
    for operation in module["operations"]:
        name = operation["op"]
        if name == "trim":
            result = result.strip()
        elif name == "lowercase":
            result = result.lower()
        elif name == "uppercase":
            result = result.upper()
        elif name == "prepend":
            result = operation["value"] + result
        elif name == "append":
            result += operation["value"]
        elif name == "replace_literal":
            result = result.replace(operation["old"], operation["new"])
        if len(result) > 100_000:
            raise ValueError("Module output exceeds the 100000-character limit.")
    return result


class ModuleRegistry:
    """Content-addressed, non-executable module registry.

    Modules are declarative pipelines. Arbitrary Python, shell commands, imports,
    network calls and filesystem operations are intentionally not supported.
    """

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise ModuleRejected("Module registry root cannot be a symbolic link.")
        self.root = requested.resolve()
        self.candidates = self.root / "candidates"
        self.active = self.root / "active"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates.mkdir(exist_ok=True)
        self.active.mkdir(exist_ok=True)
        if self.candidates.is_symlink() or self.active.is_symlink():
            raise ModuleRejected("Module registry directories cannot be symbolic links.")

    def create_candidate(self, spec: dict[str, Any], *, source: str = "self-generated",
                         source_sha256: str | None = None) -> dict[str, Any]:
        module = validate_module(spec)
        if source != "self-generated":
            source = _safe_url(source)
            if (not isinstance(source_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
                raise ModuleRejected("Discovered modules require a verified source SHA-256.")
        elif source_sha256 is not None:
            raise ModuleRejected("Self-generated modules cannot claim an external source digest.")

        failures = []
        for index, case in enumerate(module["tests"]):
            actual = execute_module(module, case["input"])
            if actual != case["expected"]:
                failures.append(index)
        if failures:
            raise ModuleRejected(f"Module behavior tests failed at cases: {failures}.")

        document = {
            "format": "arachne-small-module",
            "format_version": 1,
            "module": module,
            "source": source,
            "source_sha256": source_sha256,
        }
        digest = hashlib.sha256(_json_bytes(document)).hexdigest()
        candidate_path = self.candidates / f"{digest}.json"
        self._write_new(candidate_path, _json_bytes(document))
        return {"id": module["id"], "version": module["version"],
                "sha256": digest, "state": "verified_candidate",
                "tests_passed": len(module["tests"]), "source": source}

    def discover(self, registry_urls: list[str]) -> list[dict[str, Any]]:
        if not isinstance(registry_urls, list) or len(registry_urls) > 20:
            raise ValueError("Provide at most 20 trusted registry URLs.")
        discovered = []
        for registry_url in registry_urls:
            raw_index = _fetch_json(registry_url, _MAX_REGISTRY_BYTES)
            try:
                index = json.loads(raw_index)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModuleRejected(f"Invalid registry JSON at {registry_url}.") from exc
            if not isinstance(index, dict) or set(index) != {"format", "modules"} or index["format"] != "arachne-module-index-v1":
                raise ModuleRejected(f"Unsupported module index format: {registry_url}")
            entries = index["modules"]
            if not isinstance(entries, list) or len(entries) > 100:
                raise ModuleRejected("A registry index must contain at most 100 module entries.")
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"url", "sha256"}:
                    raise ModuleRejected("Registry entries must contain only url and sha256.")
                source_url = _safe_url(entry["url"])
                expected_hash = entry["sha256"]
                if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                    raise ModuleRejected("Registry entry has an invalid SHA-256.")
                raw_module = _fetch_json(source_url, _MAX_DOCUMENT_BYTES)
                actual_hash = hashlib.sha256(raw_module).hexdigest()
                if actual_hash != expected_hash:
                    raise ModuleRejected(f"Module source digest mismatch: {source_url}")
                try:
                    spec = json.loads(raw_module)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ModuleRejected(f"Invalid module JSON: {source_url}") from exc
                discovered.append(self.create_candidate(
                    spec, source=source_url, source_sha256=actual_hash
                ))
        return discovered

    def activate(self, digest: str) -> dict[str, Any]:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ModuleRejected("Invalid module content digest.")
        candidate = self.candidates / f"{digest}.json"
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(candidate)
        raw = candidate.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ModuleRejected("Candidate content hash verification failed.")
        document = json.loads(raw)
        if (not isinstance(document, dict) or document.get("format") != "arachne-small-module"
                or document.get("format_version") != 1):
            raise ModuleRejected("Unsupported module candidate format.")
        module = validate_module(document.get("module"))
        for case in module["tests"]:
            if execute_module(module, case["input"]) != case["expected"]:
                raise ModuleRejected("Module verification changed after candidate creation.")

        active_path = self.active / f"{digest}.json"
        if not active_path.exists():
            self._write_new(active_path, raw)
        pointer = {
            "format": "arachne-small-active-module",
            "format_version": 1,
            "id": module["id"],
            "version": module["version"],
            "sha256": digest,
            "file": active_path.name,
        }
        pointer_path = self.active / f"{module['id']}.current.json"
        self._atomic_replace(pointer_path, _json_bytes(pointer))
        return {**pointer, "state": "active", "tests_passed": len(module["tests"])}

    def run(self, module_id: str, text: str) -> str:
        if not isinstance(module_id, str) or not _MODULE_ID.fullmatch(module_id):
            raise ValueError("Invalid module id.")
        pointer_path = self.active / f"{module_id}.current.json"
        if pointer_path.is_symlink() or not pointer_path.is_file():
            raise FileNotFoundError(f"No active module: {module_id}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        digest = pointer.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ModuleRejected("Active module pointer is invalid.")
        module_path = self.active / f"{digest}.json"
        if module_path.is_symlink() or not module_path.is_file():
            raise ModuleRejected("Active module file is missing or unsafe.")
        raw = module_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ModuleRejected("Active module integrity check failed.")
        document = json.loads(raw)
        module = validate_module(document.get("module"))
        if pointer.get("id") != module["id"] or pointer.get("version") != module["version"]:
            raise ModuleRejected("Active module pointer does not match module metadata.")
        return execute_module(module, text)

    def list_active(self) -> list[dict[str, Any]]:
        result = []
        for pointer_path in sorted(self.active.glob("*.current.json")):
            if pointer_path.is_symlink():
                raise ModuleRejected("Active module pointer cannot be a symbolic link.")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            result.append({"id": pointer["id"], "version": pointer["version"],
                           "sha256": pointer["sha256"]})
        return result

    @staticmethod
    def _write_new(path: Path, content: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_replace(path: Path, content: bytes) -> None:
        if path.is_symlink():
            raise ModuleRejected("Module pointer cannot be a symbolic link.")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            if path.is_symlink():
                raise ModuleRejected("Module pointer became a symbolic link.")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
