from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    intermediate_size: int = 5632
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        dimensions = (self.vocab_size, self.d_model, self.n_layers, self.n_heads,
                      self.n_kv_heads, self.max_seq_len, self.intermediate_size)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive integers.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even.")
        if not math.isfinite(self.rope_theta) or not math.isfinite(self.norm_eps) or self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be finite and positive.")
        if not isinstance(self.tie_embeddings, bool) or not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("tie_embeddings and gradient_checkpointing must be booleans.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameter_count(self) -> int:
        d, f = self.d_model, self.intermediate_size
        kv_width = self.n_kv_heads * self.head_dim
        attention = 2 * d * d + 2 * d * kv_width
        mlp = 3 * d * f
        layer_norms = 2 * d
        embeddings = self.vocab_size * d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        return embeddings + output + self.n_layers * (attention + mlp + layer_norms) + d

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    save_interval: int = 500
    precision: str = "bf16"
    num_workers: int = 2
    output_dir: str = "artifacts/run"

    def validate(self) -> None:
        integer_values = (self.batch_size, self.gradient_accumulation_steps, self.max_steps,
                          self.eval_interval, self.save_interval)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integer_values):
            raise ValueError("Batch, accumulation, step and interval values must be positive integers.")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps or self.num_workers < 0 or self.seed < 0:
            raise ValueError("warmup_steps must be between zero and max_steps; num_workers and seed must be non-negative.")
        numeric_values = (self.learning_rate, self.min_learning_rate, self.weight_decay, self.grad_clip)
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("Training numeric settings must be finite.")
        if self.learning_rate <= 0 or self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("Learning rates are invalid.")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip must be positive.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")

@dataclass(frozen=True)
class Config:
    model: ModelConfig
    training: TrainingConfig

def load_config(path: str | Path) -> Config:
    source = Path(path)
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict) or not isinstance(raw.get("training"), dict):
        raise ValueError(f"Configuration must contain model and training mappings: {source}")
    config = Config(ModelConfig(**raw["model"]), TrainingConfig(**raw["training"]))
    config.model.validate()
    config.training.validate()
    return config
