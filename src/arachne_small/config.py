from dataclasses import dataclass
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

    def validate(self) -> None:
        if min(self.vocab_size, self.d_model, self.n_layers, self.n_heads,
               self.n_kv_heads, self.max_seq_len, self.intermediate_size) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even.")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameter_count(self) -> int:
        # Exact count for this implementation, including norms and embeddings.
        h, kv, d, f = self.n_heads, self.n_kv_heads, self.d_model, self.intermediate_size
        head_dim = d // h
        attention = d * (h * head_dim + 2 * kv * head_dim + h * head_dim) + d + (h * head_dim + 2 * kv * head_dim + h * head_dim)
        mlp = 3 * d * f + 3 * f
        norms = 2 * d + d
        layer = attention + mlp + norms
        embedding = self.vocab_size * d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        return embedding + output + self.n_layers * layer + d

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
        if min(self.batch_size, self.gradient_accumulation_steps, self.max_steps,
               self.eval_interval, self.save_interval) <= 0:
            raise ValueError("Batch, accumulation, step and interval values must be positive.")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("Learning rates are invalid.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")

@dataclass(frozen=True)
class Config:
    model: ModelConfig
    training: TrainingConfig

def load_config(path: str | Path) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = Config(ModelConfig(**raw["model"]), TrainingConfig(**raw["training"]))
    config.model.validate()
    config.training.validate()
    return config
