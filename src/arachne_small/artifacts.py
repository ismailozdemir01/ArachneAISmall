import json
import os
import re
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .config import ModelConfig
from .model import ArachneSmall
from .data import sha256_file


def _artifact_file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Artifact file name must be a plain relative filename: {name!r}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Artifact files cannot be symbolic links: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _expected_state_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    d, f, h, kv, hd = (
        config.d_model, config.intermediate_size, config.n_heads,
        config.n_kv_heads, config.head_dim,
    )
    expected = {
        "token_embedding.weight": (config.vocab_size, d),
        "final_norm.weight": (d,),
        "lm_head.weight": (config.vocab_size, d),
    }
    for index in range(config.n_layers):
        prefix = f"layers.{index}."
        expected.update({
            prefix + "attn_norm.weight": (d,),
            prefix + "ffn_norm.weight": (d,),
            prefix + "attn.q_proj.weight": (h * hd, d),
            prefix + "attn.k_proj.weight": (kv * hd, d),
            prefix + "attn.v_proj.weight": (kv * hd, d),
            prefix + "attn.o_proj.weight": (d, d),
            prefix + "mlp.gate_proj.weight": (f, d),
            prefix + "mlp.up_proj.weight": (f, d),
            prefix + "mlp.down_proj.weight": (d, f),
        })
    return expected


def verify_artifact(artifact_dir: str | Path) -> dict:
    """Verify artifact schema, file integrity, tokenizer IDs and exact tensor names/shapes."""
    root = Path(artifact_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    config_path = _artifact_file(root, "config.json")
    manifest_path = _artifact_file(root, "artifact-manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("format") != "arachne-small"
            or config.get("format_version") != 1
            or config.get("architecture") != "decoder-only-transformer"):
        raise ValueError("Unsupported Arachne Small artifact format.")
    if not isinstance(manifest, dict) or manifest.get("format") != "arachne-small-artifact" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported Arachne Small artifact manifest.")
    model_data = config.get("model")
    tokenizer_data = config.get("tokenizer")
    weights_data = config.get("weights")
    if not isinstance(model_data, dict) or not isinstance(tokenizer_data, dict) or not isinstance(weights_data, dict):
        raise ValueError("Artifact config is missing model, tokenizer or weights metadata.")
    model_config = ModelConfig(**model_data)
    model_config.validate()
    parameter_count = model_config.parameter_count()
    if config.get("parameter_count") != parameter_count or manifest.get("parameter_count") != parameter_count:
        raise ValueError("Artifact parameter count does not match the architecture configuration.")
    tokenizer_path = _artifact_file(root, tokenizer_data.get("file"))
    weights_path = _artifact_file(root, weights_data.get("file"))
    if weights_data.get("format") != "safetensors":
        raise ValueError("Only SafeTensors inference weights are supported.")
    if manifest.get("model_config_sha256") != sha256_file(config_path):
        raise ValueError("Artifact config checksum mismatch.")
    if manifest.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("Artifact tokenizer checksum mismatch.")
    if tokenizer_data.get("sha256") != manifest.get("tokenizer_sha256"):
        raise ValueError("Tokenizer checksums in config and manifest do not match.")
    if manifest.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("Artifact weights checksum mismatch.")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    actual_special_tokens = {
        "pad_id": tokenizer.pad_id(),
        "unk_id": tokenizer.unk_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
    }
    expected_special_tokens = {key: tokenizer_data.get(key) for key in actual_special_tokens}
    if tokenizer.vocab_size() != model_config.vocab_size or tokenizer_data.get("vocab_size") != model_config.vocab_size:
        raise ValueError("Artifact tokenizer vocabulary does not match model vocabulary.")
    if actual_special_tokens != expected_special_tokens:
        raise ValueError("Artifact tokenizer special-token IDs do not match the tokenizer file.")
    if manifest.get("tokenizer_special_tokens") != actual_special_tokens:
        raise ValueError("Artifact manifest special-token IDs do not match the tokenizer file.")
    expected_shapes = _expected_state_shapes(model_config)
    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        actual_keys = set(weights.keys())
        if actual_keys != set(expected_shapes):
            missing = sorted(set(expected_shapes) - actual_keys)
            unexpected = sorted(actual_keys - set(expected_shapes))
            raise ValueError(f"Model tensor names do not match architecture. Missing={missing}; unexpected={unexpected}")
        for name, expected_shape in expected_shapes.items():
            if tuple(weights.get_slice(name).get_shape()) != expected_shape:
                raise ValueError(f"Tensor shape mismatch for {name}.")
    return {
        "artifact_dir": str(root),
        "format_version": 1,
        "parameter_count": parameter_count,
        "vocab_size": tokenizer.vocab_size(),
        "weights_sha256": manifest["weights_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "model_config_sha256": manifest["model_config_sha256"],
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }


def load_model(artifact_dir: str | Path, device: str | torch.device = "cpu") -> tuple[ArachneSmall, spm.SentencePieceProcessor]:
    """Load a verified SafeTensors artifact or a trained, weights-only arachne_small.pt checkpoint."""
    requested = Path(artifact_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Model artifact path cannot be a symbolic link.")
    if requested.is_dir() and (requested / "arachne_small.pt").is_file():
        requested = requested / "arachne_small.pt"
    if requested.is_file() and requested.suffix.lower() == ".pt":
        if requested.is_symlink():
            raise ValueError("Model checkpoint cannot be a symbolic link.")
        root = requested.parent
        tokenizer_path = _artifact_file(root, "tokenizer.model")
        checkpoint = torch.load(requested, map_location="cpu", weights_only=True)
        if (not isinstance(checkpoint, dict)
                or checkpoint.get("format") != "arachne-small-pytorch-checkpoint"
                or checkpoint.get("format_version") != 1):
            raise ValueError("Unsupported Arachne Small PyTorch checkpoint.")
        training = checkpoint.get("training")
        if not isinstance(training, dict) or training.get("status") != "trained":
            raise ValueError("Arachne Small checkpoint has not completed training.")
        model_data = checkpoint.get("model_config")
        state = checkpoint.get("model_state_dict")
        if not isinstance(model_data, dict) or not isinstance(state, dict):
            raise ValueError("PyTorch checkpoint is missing model configuration or weights.")
        model_config = ModelConfig(**model_data)
        model_config.validate()
        expected_shapes = _expected_state_shapes(model_config)
        if set(state) != set(expected_shapes):
            raise ValueError("PyTorch checkpoint tensor names do not match the model architecture.")
        for name, shape in expected_shapes.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"PyTorch checkpoint tensor shape mismatch: {name}")
            if not tensor.is_floating_point():
                raise ValueError(f"PyTorch checkpoint contains non-floating weights: {name}")
            flat = tensor.reshape(-1)
            for chunk in flat.split(1_048_576):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError(f"PyTorch checkpoint contains non-finite weights: {name}")
        tokenizer_hash = checkpoint.get("tokenizer_sha256")
        if not isinstance(tokenizer_hash, str) or sha256_file(tokenizer_path) != tokenizer_hash:
            raise ValueError("PyTorch checkpoint tokenizer checksum does not match tokenizer.model.")
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if tokenizer.vocab_size() != model_config.vocab_size:
            raise ValueError("PyTorch checkpoint tokenizer vocabulary does not match model configuration.")
        model = ArachneSmall(model_config)
        model.load_state_dict(state, strict=True, assign=True)
        del state, checkpoint
        model.to(device)
        model.eval()
        return model, tokenizer

    summary = verify_artifact(requested)
    root = Path(summary["artifact_dir"])
    config_data = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model_config = ModelConfig(**config_data["model"])
    model = ArachneSmall(model_config)
    state = load_file(str(root / config_data["weights"]["file"]), device=str(device))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(root / config_data["tokenizer"]["file"]))
    return model, tokenizer


def discover_model_artifact(search_root: str | Path = "artifacts",
                            device: str | torch.device = "cpu"):
    """Load the newest valid trained model found below the local artifacts directory."""
    root = Path(search_root).expanduser()
    if not root.exists() or not root.is_dir() or root.is_symlink():
        return None
    release_root = root / "releases"
    current_pointer = release_root / "current.json"
    if current_pointer.is_file() and not current_pointer.is_symlink():
        try:
            pointer = json.loads(current_pointer.read_text(encoding="utf-8"))
            version = pointer.get("version") if isinstance(pointer, dict) else None
            if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
                return None
            release = release_root / version
            return (*load_model(release, device), release)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            return None
    candidates: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories
                          if name not in {"self_taught", "learning-work", "releases"}
                          and not (current_path / name).is_symlink()]
        if "artifact-manifest.json" in files:
            candidates.append(current_path)
        checkpoint = current_path / "arachne_small.pt"
        if "arachne_small.pt" in files and not checkpoint.is_symlink():
            candidates.append(checkpoint)
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        try:
            model, tokenizer = load_model(candidate, device)
            return model, tokenizer, candidate
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            continue
    return None
