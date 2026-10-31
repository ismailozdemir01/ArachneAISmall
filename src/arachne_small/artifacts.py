import json
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
    }


def load_model(artifact_dir: str | Path, device: str | torch.device = "cpu") -> tuple[ArachneSmall, spm.SentencePieceProcessor]:
    """Load a verified inference artifact; this allocates the full model in memory."""
    summary = verify_artifact(artifact_dir)
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
