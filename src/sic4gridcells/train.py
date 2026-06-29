from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sic4gridcells.config import Config, load_config, resolve_device, save_effective_config
from sic4gridcells.data import make_sic_batch
from sic4gridcells.losses import sic_losses
from sic4gridcells.model import VelocityConditionedRNN


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    final_step: int
    checkpoint_path: Path


def train(config_path: str | Path, resume_checkpoint: str | Path | None = None) -> RunResult:
    cfg = load_config(config_path)
    return train_with_config(cfg, resume_checkpoint=resume_checkpoint)


def train_with_config(
    cfg: Config,
    resume_checkpoint: str | Path | None = None,
) -> RunResult:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    resume_path = Path(resume_checkpoint) if resume_checkpoint is not None else None
    checkpoint = None
    start_step = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        start_step = int(checkpoint["step"])
        _validate_resume_config(cfg, checkpoint["config"], resume_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(cfg, output_dir / "config.yaml")

    model = VelocityConditionedRNN(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    metrics_path = output_dir / "metrics.jsonl"
    latest_checkpoint = checkpoint_dir / "step_0.pt"
    if checkpoint is not None and resume_path is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "generator_state" in checkpoint:
            generator.set_state(checkpoint["generator_state"].cpu())
        latest_checkpoint = resume_path
    if resume_path is not None:
        _trim_metrics_file_to_step(metrics_path, start_step)
    if start_step >= cfg.train.max_optimizer_steps:
        return RunResult(
            output_dir=output_dir,
            final_step=start_step,
            checkpoint_path=latest_checkpoint,
        )
    writer_kwargs = {}
    if resume_path is not None:
        writer_kwargs["purge_step"] = start_step + 1
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), **writer_kwargs)
    try:
        metrics_mode = "a" if resume_path is not None and metrics_path.exists() else "w"
        with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
            for step in range(start_step + 1, cfg.train.max_optimizer_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                micro_metrics: list[dict[str, float]] = []
                for _ in range(cfg.train.accumulate_grad_batches):
                    batch = make_sic_batch(cfg, generator, device)
                    initial_positions = (
                        batch.initial_positions
                        if cfg.model.initial_position_encoding != "none"
                        else None
                    )
                    rollout = model(batch.velocities, initial_positions=initial_positions)
                    losses = sic_losses(batch, rollout, cfg)
                    (losses["loss/total"] / cfg.train.accumulate_grad_batches).backward()
                    micro_metrics.append(
                        _metrics_to_float_dict(losses, rollout.zero_norm_fraction)
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.grad_clip_norm,
                )
                optimizer.step()
                log_row = _aggregate_metric_dicts(micro_metrics)
                log_row["step"] = float(step)
                log_row["grad_norm"] = float(grad_norm.detach().cpu())
                monitor_value = log_row[cfg.train.scheduler_monitor]
                scheduler.step(monitor_value)
                log_row["lr"] = float(optimizer.param_groups[0]["lr"])

                should_log = (
                    step == 1
                    or step == cfg.train.max_optimizer_steps
                    or step % cfg.train.log_every == 0
                )
                if should_log:
                    _write_metrics(metrics_file, writer, step, log_row)
                if step % cfg.train.checkpoint_every == 0 or step == cfg.train.max_optimizer_steps:
                    latest_checkpoint = checkpoint_dir / f"step_{step}.pt"
                    _save_checkpoint(
                        latest_checkpoint,
                        step,
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        generator,
                    )
    finally:
        writer.close()
    return RunResult(
        output_dir=output_dir,
        final_step=cfg.train.max_optimizer_steps,
        checkpoint_path=latest_checkpoint,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics_to_float_dict(
    losses: dict[str, torch.Tensor],
    zero_norm_fraction: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in losses.items():
        metrics[key] = float(value.detach().cpu())
    metrics["stats/zero_norm_fraction"] = float(zero_norm_fraction.detach().cpu())
    return metrics


def _aggregate_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    metrics: dict[str, float] = {}
    for key in keys:
        total = sum(row[key] for row in rows)
        if key.endswith("_pairs") or key.endswith("_steps"):
            metrics[key] = total
        else:
            metrics[key] = total / len(rows)
    return metrics


def _write_metrics(
    metrics_file,
    writer: SummaryWriter,
    step: int,
    metrics: dict[str, float],
) -> None:
    json_metrics = {
        key: _json_metric_value(value)
        for key, value in metrics.items()
    }
    metrics_file.write(json.dumps(json_metrics, sort_keys=True, allow_nan=False) + "\n")
    metrics_file.flush()
    for key, value in metrics.items():
        if key != "step" and math.isfinite(value):
            writer.add_scalar(key, value, step)


def _json_metric_value(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return value


def _save_checkpoint(
    path: Path,
    step: int,
    cfg: Config,
    model: VelocityConditionedRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "generator_state": generator.get_state(),
        },
        path,
    )


def _validate_resume_config(
    cfg: Config,
    checkpoint_config: dict,
    checkpoint_path: Path,
) -> None:
    current = asdict(cfg)
    expected = json.loads(json.dumps(checkpoint_config))
    allowed_differences = {
        ("train", "max_optimizer_steps"),
    }
    differences = _config_differences(current, expected)
    blocking = [
        ".".join(path)
        for path in differences
        if path not in allowed_differences
    ]
    if blocking:
        joined = ", ".join(sorted(blocking))
        raise ValueError(
            f"Cannot resume {checkpoint_path}: config differs from checkpoint in {joined}"
        )
    current_max_steps = int(current["train"]["max_optimizer_steps"])
    checkpoint_max_steps = int(expected["train"]["max_optimizer_steps"])
    if current_max_steps < checkpoint_max_steps:
        raise ValueError(
            "Cannot resume "
            f"{checkpoint_path}: train.max_optimizer_steps cannot decrease "
            f"from {checkpoint_max_steps} to {current_max_steps}"
        )


def _config_differences(
    left: dict,
    right: dict,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    differences: list[tuple[str, ...]] = []
    for key in sorted(set(left) | set(right)):
        path = prefix + (str(key),)
        if key not in left or key not in right:
            differences.append(path)
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            differences.extend(_config_differences(left_value, right_value, path))
        elif left_value != right_value:
            differences.append(path)
    return differences


def _trim_metrics_file_to_step(metrics_path: Path, step: int) -> None:
    if not metrics_path.exists():
        return
    kept_lines: list[str] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_step = row.get("step")
            if isinstance(row_step, (int, float)) and row_step <= step:
                kept_lines.append(line)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for line in kept_lines:
            handle.write(line + "\n")
