from __future__ import annotations

import json
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


def train(config_path: str | Path) -> RunResult:
    cfg = load_config(config_path)
    return train_with_config(cfg)


def train_with_config(cfg: Config) -> RunResult:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
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
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    latest_checkpoint = checkpoint_dir / "step_0.pt"
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for step in range(1, cfg.train.max_optimizer_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                micro_metrics: list[dict[str, float]] = []
                for _ in range(cfg.train.accumulate_grad_batches):
                    batch = make_sic_batch(cfg, generator, device)
                    rollout = model(batch.velocities)
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
    metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
    metrics_file.flush()
    for key, value in metrics.items():
        if key != "step":
            writer.add_scalar(key, value, step)


def _save_checkpoint(
    path: Path,
    step: int,
    cfg: Config,
    model: VelocityConditionedRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )
