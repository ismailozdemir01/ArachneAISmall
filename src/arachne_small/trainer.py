import math, random, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config
from .model import ArachneSmall
from .data import TokenBlockDataset

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

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        result = model(x.to(device), y.to(device))
        losses.append(float(result["loss"].item()))
    if not losses:
        raise RuntimeError("Validation dataset produced no batches.")
    model.train()
    return sum(losses) / len(losses)

def train(config: Config, train_path: str, validation_path: str,
          resume: str | None = None) -> Path:
    _seed_everything(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.training.precision
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but this CUDA device does not support it.")
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 training requires CUDA; use fp32 on CPU.")
    train_ds = TokenBlockDataset(train_path, config.model.max_seq_len)
    val_ds = TokenBlockDataset(validation_path, config.model.max_seq_len)
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Train and validation datasets must each contain at least one full token block.")
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                              num_workers=config.training.num_workers, drop_last=False,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False,
                            num_workers=config.training.num_workers)
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
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != config.model.__dict__:
            raise ValueError("Checkpoint model configuration does not match the active configuration.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    out = Path(config.training.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out / "training.jsonl"
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step, config.training.max_steps):
        lr = _lr(step, config.training.max_steps, config.training.warmup_steps,
                 config.training.learning_rate, config.training.min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                loss = model(x, y)["loss"] / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated += float(loss.detach().item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        record = {"step": step + 1, "loss": accumulated, "learning_rate": lr}
        if (step + 1) % config.training.eval_interval == 0:
            record["validation_loss"] = evaluate(model, val_loader, device)
            record["validation_perplexity"] = math.exp(min(20.0, record["validation_loss"]))
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        if (step + 1) % config.training.save_interval == 0 or step + 1 == config.training.max_steps:
            path = out / "checkpoints" / f"step-{step + 1:08d}.pt"
            torch.save({"step": step + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "model_config": config.model.__dict__,
                        "training_config": config.training.__dict__}, path)
    final = out / "model.pt"
    torch.save({"model": model.state_dict(), "model_config": config.model.__dict__}, final)
    return final
