from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sic4gridcells.config import Config, load_config, validate_config
from sic4gridcells.logging_utils import to_jsonable
from sic4gridcells.train import train_with_config


@dataclass(frozen=True)
class ProfileSummary:
    output_dir: str
    config_path: str
    requested_steps: int
    final_step: int
    checkpoint_path: str
    checkpoint_size_mb: float | None
    estimated_checkpoint_count: int | None
    estimated_checkpoint_storage_mb: float | None
    mean_step_seconds: float | None
    last_step_seconds: float | None
    estimated_seconds_for_config_steps: float | None
    estimated_hours_for_config_steps: float | None
    metrics_rows: int
    last_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_training_run(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 20,
    device: str | None = None,
    overwrite_output: bool = False,
) -> ProfileSummary:
    if steps <= 0:
        raise ValueError("steps must be positive")
    cfg = load_config(config_path)
    target_steps = cfg.train.max_optimizer_steps
    pilot_cfg = _pilot_config(
        cfg,
        output_dir=output_dir,
        steps=steps,
        device=device,
    )
    result = train_with_config(
        pilot_cfg,
        overwrite_output=overwrite_output,
        config_path=config_path,
    )
    summary = summarize_profile_run(
        output_dir=result.output_dir,
        config_path=config_path,
        requested_steps=steps,
        target_steps=target_steps,
        final_step=result.final_step,
        checkpoint_path=result.checkpoint_path,
        checkpoint_every=cfg.train.checkpoint_every,
    )
    write_profile_summary(summary, result.output_dir / "profile_summary.json")
    return summary


def summarize_profile_run(
    *,
    output_dir: str | Path,
    config_path: str | Path,
    requested_steps: int,
    target_steps: int,
    final_step: int,
    checkpoint_path: str | Path,
    checkpoint_every: int | None = None,
) -> ProfileSummary:
    out_dir = Path(output_dir)
    checkpoint = Path(checkpoint_path)
    rows = _load_metrics_rows(out_dir / "metrics.jsonl")
    step_seconds = [
        row["perf/step_seconds"]
        for row in rows
        if _is_finite_number(row.get("perf/step_seconds"))
    ]
    mean_step_seconds = _mean(step_seconds)
    last_step_seconds = step_seconds[-1] if step_seconds else None
    estimate_seconds = (
        mean_step_seconds * target_steps
        if mean_step_seconds is not None
        else None
    )
    checkpoint_size_mb = (
        checkpoint.stat().st_size / (1024 ** 2)
        if checkpoint.exists()
        else None
    )
    checkpoint_count = _checkpoint_count(target_steps, checkpoint_every)
    checkpoint_storage_mb = (
        checkpoint_size_mb * checkpoint_count
        if checkpoint_size_mb is not None and checkpoint_count is not None
        else None
    )
    return ProfileSummary(
        output_dir=str(out_dir),
        config_path=str(config_path),
        requested_steps=requested_steps,
        final_step=final_step,
        checkpoint_path=str(checkpoint),
        checkpoint_size_mb=checkpoint_size_mb,
        estimated_checkpoint_count=checkpoint_count,
        estimated_checkpoint_storage_mb=checkpoint_storage_mb,
        mean_step_seconds=mean_step_seconds,
        last_step_seconds=last_step_seconds,
        estimated_seconds_for_config_steps=estimate_seconds,
        estimated_hours_for_config_steps=(
            estimate_seconds / 3600.0 if estimate_seconds is not None else None
        ),
        metrics_rows=len(rows),
        last_metrics=rows[-1] if rows else {},
    )


def write_profile_summary(summary: ProfileSummary, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary.to_dict()), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _pilot_config(
    cfg: Config,
    *,
    output_dir: str | Path,
    steps: int,
    device: str | None,
) -> Config:
    data = asdict(cfg)
    data["output_dir"] = str(output_dir)
    if device is not None:
        data["device"] = device
    data["train"]["max_optimizer_steps"] = steps
    data["train"]["checkpoint_every"] = steps
    data["train"]["log_every"] = 1
    pilot_cfg = Config(
        seed=int(data["seed"]),
        device=str(data["device"]),
        output_dir=str(data["output_dir"]),
        data=type(cfg.data)(**data["data"]),
        model=type(cfg.model)(**data["model"]),
        loss=type(cfg.loss)(**data["loss"]),
        train=type(cfg.train)(**data["train"]),
        assumptions=[str(item) for item in data.get("assumptions", [])],
    )
    validate_config(pilot_cfg)
    return pilot_cfg


def _load_metrics_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    str(key): float(value)
                    for key, value in row.items()
                    if _is_finite_number(value)
                }
            )
    return rows


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _checkpoint_count(target_steps: int, checkpoint_every: int | None) -> int | None:
    if checkpoint_every is None or checkpoint_every <= 0 or target_steps <= 0:
        return None
    regular = target_steps // checkpoint_every
    if target_steps % checkpoint_every == 0:
        return regular
    return regular + 1
