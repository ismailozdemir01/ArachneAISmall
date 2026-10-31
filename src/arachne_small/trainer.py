import math, random, json, os
from pathlib import Path
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from .config import Config
from .model import ArachneSmall
from .data import TokenBlockDataset, sha256_file
from safetensors.torch import save_file
import shutil

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))

def _atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

@torch.no_grad()
def evaluate(model, loader, device, precision: str = "fp32", max_batches: int = 50) -> float:
    was_training = model.training
    model.eval()
    losses = []
    try:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                result = model(x, y, activation_dtype=amp_dtype if precision != "fp32" else None)
            loss = result["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Validation produced a non-finite loss.")
            losses.append((float(loss.item()), x.size(0)))
    finally:
        model.train(was_training)
    if not losses:
        raise RuntimeError("Validation dataset produced no batches.")
    sample_count = sum(batch_size for _, batch_size in losses)
    return sum(loss * batch_size for loss, batch_size in losses) / sample_count

def _truncate_training_log(path: Path, checkpoint_step: int) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return
    temporary = path.with_name(path.name + ".tmp")
    try:
        with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            index = 0
            while True:
                line = source.readline()
                if not line:
                    break
                index += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        break
                    raise ValueError(f"Corrupt training log at {path}:{index}")
                if not isinstance(record, dict) or not isinstance(record.get("step"), int):
                    raise ValueError(f"Invalid training log record at {path}:{index}")
                if record["step"] <= checkpoint_step:
                    target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_safetensors_save(state: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        # Clone tensors so tied embedding/output weights have independent storage in the file.
        save_file({name: tensor.detach().cpu().contiguous().clone() for name, tensor in state.items()}, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train(config: Config, train_path: str, validation_path: str,
          tokenizer_path: str, resume: str | None = None) -> Path:
    config.model.validate()
    config.training.validate()
    _seed_everything(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.training.precision
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but this CUDA device does not support it.")
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 training requires CUDA; use fp32 or supported bf16 on CPU.")
    train_ds = TokenBlockDataset(train_path, config.model.max_seq_len)
    val_ds = TokenBlockDataset(validation_path, config.model.max_seq_len)
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Train and validation datasets must each contain at least one full token block.")
    train_metadata_path = Path(train_path).with_name(Path(train_path).name + ".json")
    validation_metadata_path = Path(validation_path).with_name(Path(validation_path).name + ".json")
    if not train_metadata_path.is_file() or not validation_metadata_path.is_file():
        raise FileNotFoundError("Both token streams require matching .json provenance sidecars generated by prepare-data.")
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    validation_metadata = json.loads(validation_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(train_metadata, dict) or not isinstance(validation_metadata, dict):
        raise ValueError("Token dataset sidecars must contain JSON objects.")
    for token_path, metadata, dataset in (
        (Path(train_path), train_metadata, train_ds),
        (Path(validation_path), validation_metadata, val_ds),
    ):
        if metadata.get("format_version") != 1 or metadata.get("token_format") != "little-endian-uint32":
            raise ValueError(f"Unsupported token dataset format: {token_path}")
        if metadata.get("token_count") != dataset.tokens.numel():
            raise ValueError(f"Token count does not match provenance metadata: {token_path}")
        if metadata.get("token_file_sha256") != sha256_file(token_path):
            raise ValueError(f"Token stream checksum mismatch: {token_path}")
        if metadata.get("tokenizer_vocab_size") != config.model.vocab_size:
            raise ValueError(f"Tokenizer vocabulary does not match model config: {token_path}")
    tokenizer_file = Path(tokenizer_path)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(tokenizer_file)
    tokenizer_sha256 = train_metadata.get("tokenizer_sha256")
    if not tokenizer_sha256 or tokenizer_sha256 != validation_metadata.get("tokenizer_sha256"):
        raise ValueError("Train and validation streams must use the exact same tokenizer model.")
    if train_metadata.get("eos_id") != validation_metadata.get("eos_id"):
        raise ValueError("Train and validation streams must use the same EOS token ID.")
    special_tokens = train_metadata.get("special_tokens")
    if not isinstance(special_tokens, dict) or special_tokens != validation_metadata.get("special_tokens"):
        raise ValueError("Train and validation streams must use identical tokenizer special-token IDs.")
    if special_tokens.get("eos_id") != train_metadata.get("eos_id"):
        raise ValueError("Dataset EOS metadata does not match the tokenizer special-token metadata.")
    if sha256_file(tokenizer_file) != tokenizer_sha256:
        raise ValueError("Provided tokenizer file does not match the token dataset provenance.")
    tokenizer_processor = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    actual_special_tokens = {
        "pad_id": tokenizer_processor.pad_id(),
        "unk_id": tokenizer_processor.unk_id(),
        "bos_id": tokenizer_processor.bos_id(),
        "eos_id": tokenizer_processor.eos_id(),
    }
    if tokenizer_processor.vocab_size() != config.model.vocab_size:
        raise ValueError("SentencePiece model vocabulary does not match the model configuration.")
    if actual_special_tokens != special_tokens:
        raise ValueError("Token dataset special-token metadata does not match the actual tokenizer model.")
    train_max = int(train_ds.tokens.max().item())
    validation_max = int(val_ds.tokens.max().item())
    if min(int(train_ds.tokens.min().item()), int(val_ds.tokens.min().item())) < 0:
        raise ValueError("Token files cannot contain negative token IDs.")
    if max(train_max, validation_max) >= config.model.vocab_size:
        raise ValueError(
            f"Token ID {max(train_max, validation_max)} exceeds model vocabulary "
            f"size {config.model.vocab_size}; train tokenizer and model config must match."
        )
    data_generator = torch.Generator()
    data_generator.manual_seed(config.training.seed)
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                              generator=data_generator,
                              num_workers=config.training.num_workers,
                              drop_last=(len(train_ds) >= config.training.batch_size),
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False,
                            num_workers=config.training.num_workers, pin_memory=device.type == "cuda")
    model = ArachneSmall(config.model).to(device)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 or "norm" in name else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.training.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    start_step = 0
    tokens_seen = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("artifact_version") != 1:
            raise ValueError("Unsupported or invalid training checkpoint format.")
        if checkpoint.get("model_config") != config.model.__dict__:
            raise ValueError("Checkpoint model configuration does not match the active configuration.")
        previous_training = checkpoint.get("training_config", {})
        immutable_training_fields = (
            "seed", "batch_size", "gradient_accumulation_steps", "learning_rate",
            "min_learning_rate", "warmup_steps", "weight_decay", "grad_clip", "precision",
        )
        if any(previous_training.get(field) != getattr(config.training, field)
               for field in immutable_training_fields):
            raise ValueError("Optimizer-critical training settings differ from the checkpoint.")
        if checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
            raise ValueError("Checkpoint tokenizer does not match the active training data.")
        if not {"model", "optimizer", "step"}.issubset(checkpoint):
            raise ValueError("Checkpoint is missing model, optimizer or step state.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        if tokens_seen < 0:
            raise ValueError("Checkpoint token counter cannot be negative.")
        if start_step < 0 or start_step > config.training.max_steps:
            raise ValueError("Checkpoint step is outside the configured training schedule.")
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    out = Path(config.training.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out / "training.jsonl"
    if log_path.exists() and not resume:
        raise FileExistsError(f"Training log already exists: {log_path}. Choose a new output_dir or resume a checkpoint.")
    if resume:
        _truncate_training_log(log_path, start_step)
    if (out / "model.safetensors").exists() and not resume:
        raise FileExistsError(f"Model artifact already exists in {out}; choose a new output_dir.")
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step, config.training.max_steps):
        lr = _lr(step, config.training.max_steps, config.training.warmup_steps,
                 config.training.learning_rate, config.training.min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        step_tokens = 0
        for _ in range(config.training.gradient_accumulation_steps):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            step_tokens += x.numel()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                raw_loss = model(x, y, activation_dtype=amp_dtype if precision != "fp32" else None)["loss"]
                if raw_loss is None or not torch.isfinite(raw_loss):
                    raise RuntimeError(f"Training produced a non-finite loss at step {step + 1}.")
                loss = raw_loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated += float(loss.detach().item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Gradient norm became non-finite at step {step + 1}.")
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += step_tokens
        record = {"step": step + 1, "tokens_seen": tokens_seen,
                  "tokens_this_step": step_tokens, "loss": accumulated, "learning_rate": lr,
                  "gradient_norm": float(grad_norm.item())}
        if (step + 1) % config.training.eval_interval == 0 or step + 1 == config.training.max_steps:
            record["validation_loss"] = evaluate(model, val_loader, device, precision=precision)
            record["validation_perplexity"] = math.exp(min(80.0, record["validation_loss"]))
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
        if (step + 1) % config.training.save_interval == 0 or step + 1 == config.training.max_steps:
            path = out / "checkpoints" / f"step-{step + 1:08d}.pt"
            _atomic_torch_save({"artifact_version": 1, "step": step + 1,
                                "tokens_seen": tokens_seen, "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                "model_config": config.model.__dict__,
                                "training_config": config.training.__dict__,
                                "tokenizer_sha256": tokenizer_sha256}, path)
    final_validation_loss = evaluate(model, val_loader, device, precision=precision)
    final_validation_perplexity = math.exp(min(80.0, final_validation_loss))
    model_path = out / "model.safetensors"
    tokenizer_output = out / "tokenizer.model"
    config_output = out / "config.json"
    manifest_output = out / "artifact-manifest.json"
    _atomic_safetensors_save(model.state_dict(), model_path)
    tokenizer_temporary = tokenizer_output.with_name(tokenizer_output.name + ".tmp")
    try:
        shutil.copyfile(tokenizer_file, tokenizer_temporary)
        os.replace(tokenizer_temporary, tokenizer_output)
    finally:
        tokenizer_temporary.unlink(missing_ok=True)
    model_config = config.model.__dict__.copy()
    model_config.pop("gradient_checkpointing", None)
    artifact_config = {
        "format": "arachne-small",
        "format_version": 1,
        "architecture": "decoder-only-transformer",
        "model": model_config,
        "parameter_count": config.model.parameter_count(),
        "tokenizer": {"file": tokenizer_output.name, "sha256": tokenizer_sha256,
                      "vocab_size": config.model.vocab_size, **special_tokens},
        "weights": {"file": model_path.name, "format": "safetensors"},
    }
    _atomic_json_save(artifact_config, config_output)
    artifact_manifest = {
        "format": "arachne-small-artifact",
        "format_version": 1,
        "training_steps_completed": config.training.max_steps,
        "training_tokens_processed": tokens_seen,
        "final_validation_loss": final_validation_loss,
        "final_validation_perplexity": final_validation_perplexity,
        "training": {
            "seed": config.training.seed,
            "precision": config.training.precision,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "max_steps": config.training.max_steps,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        },
        "parameter_count": config.model.parameter_count(),
        "model_config_sha256": sha256_file(config_output),
        "weights_sha256": sha256_file(model_path),
        "tokenizer_sha256": sha256_file(tokenizer_output),
        "tokenizer_special_tokens": special_tokens,
        "train_tokens_sha256": train_metadata["token_file_sha256"],
        "validation_tokens_sha256": validation_metadata["token_file_sha256"],
        "training_log": log_path.name,
    }
    _atomic_json_save(artifact_manifest, manifest_output)
    return model_path
