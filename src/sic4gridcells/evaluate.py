from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sic4gridcells.analysis import GridScorer
from sic4gridcells.config import (
    Config,
    load_config,
    resolve_device,
    save_effective_config,
    validate_config,
)
from sic4gridcells.model import VelocityConditionedRNN
from sic4gridcells.plotting import save_ratemap_pdf, save_sac_pdf, save_summary_figure

ZERO_RESPONSE_EPS = 1e-12
EVAL_STEP_SCALE_FRACTION = 0.15


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    checkpoint_path: Path
    arena_dirs: dict[float, Path]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    device: str = "auto",
    arena_sizes: tuple[float, ...] = (2.0, 3.0, 4.0),
    nbins: int = 32,
    n_trajectories: int = 32,
    steps_per_trajectory: int = 256,
    start_mode: str = "origin",
) -> EvaluationResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = _config_from_checkpoint(checkpoint["config"])
    device_t = resolve_device(device)
    model = VelocityConditionedRNN(cfg).to(device_t)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if start_mode not in {"origin", "uniform"}:
        raise ValueError("start_mode must be 'origin' or 'uniform'")
    if start_mode == "uniform" and cfg.model.initial_position_encoding != "additive_mlp":
        raise ValueError(
            "start_mode='uniform' requires a checkpoint trained with "
            "model.initial_position_encoding='additive_mlp'"
        )
    if start_mode == "uniform" and cfg.data.initial_position_mode != "uniform_box":
        raise ValueError(
            "start_mode='uniform' requires a checkpoint trained with "
            "data.initial_position_mode='uniform_box'"
        )

    out_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(cfg, out_dir / "config.yaml")
    (out_dir / "arena_summaries").mkdir(exist_ok=True)

    result_dirs: dict[float, Path] = {}
    summary_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for arena_size in arena_sizes:
            arena_dir = out_dir / f"arena_{_format_arena_size(arena_size)}"
            arena_dir.mkdir(parents=True, exist_ok=True)
            result_dirs[arena_size] = arena_dir
            positions, hidden_states = _run_bounded_random_walks(
                model,
                device_t,
                arena_size=arena_size,
                n_trajectories=n_trajectories,
                steps_per_trajectory=steps_per_trajectory,
                start_mode=start_mode,
            )
            scorer = GridScorer(
                nbins=nbins,
                coords_range=[[-arena_size / 2, arena_size / 2], [-arena_size / 2, arena_size / 2]],
            )
            ratemaps, occupancy_counts = _accumulate_ratemaps(
                positions,
                hidden_states,
                arena_size=arena_size,
                nbins=nbins,
            )
            coverage_summary = _summarize_coverage(occupancy_counts)
            unit_response_stats = _summarize_unit_responses(ratemaps, occupancy_counts)
            scores = scorer.get_scores_batch(ratemaps)
            scales, peak_counts = scorer.calculate_grid_scales(scores.sacs)
            _write_arena_artifacts(
                arena_dir,
                ratemaps=ratemaps,
                occupancy_counts=occupancy_counts,
                sacs=scores.sacs,
                score_60=scores.score_60,
                score_90=scores.score_90,
                scales=scales,
                peak_counts=peak_counts,
                mask_60=scores.mask_60,
                mask_90=scores.mask_90,
                unit_response_stats=unit_response_stats,
            )
            summary_rows.append(
                {
                    "arena_size": arena_size,
                    "mean_grid_score_60": _json_float(np.nanmean(scores.score_60)),
                    "mean_grid_score_90": _json_float(np.nanmean(scores.score_90)),
                    "mean_scale": _json_float(_safe_nanmean(scales)),
                    **coverage_summary,
                    **_unit_response_counts(unit_response_stats),
                }
            )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(Path(checkpoint_path)),
                "config": asdict(cfg),
                "arena_summaries": summary_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return EvaluationResult(output_dir=out_dir, checkpoint_path=Path(checkpoint_path), arena_dirs=result_dirs)


def _run_bounded_random_walks(
    model: VelocityConditionedRNN,
    device: torch.device,
    *,
    arena_size: float,
    n_trajectories: int,
    steps_per_trajectory: int,
    start_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    batch_positions = []
    batch_hidden = []
    for _ in range(n_trajectories):
        positions, velocities = _sample_bounded_random_walk(
            steps_per_trajectory,
            arena_size,
            device=device,
            start_mode=start_mode,
        )
        initial_positions = positions.new_zeros((1, 2))
        if start_mode == "uniform":
            initial_positions = (positions[0] - velocities[0]).unsqueeze(0)
        rollout = model(
            velocities.unsqueeze(0),
            initial_positions=(
                initial_positions
                if model.initial_position_encoding != "none"
                else None
            ),
        )
        batch_positions.append(positions.unsqueeze(0))
        batch_hidden.append(rollout.hidden_states.cpu())
    return torch.cat(batch_positions, dim=0).cpu().numpy(), torch.cat(batch_hidden, dim=0).cpu().numpy()


def _sample_bounded_random_walk(
    steps: int,
    arena_size: float,
    *,
    device: torch.device,
    start_mode: str = "origin",
) -> tuple[torch.Tensor, torch.Tensor]:
    half = arena_size / 2.0
    positions = torch.empty(steps, 2, device=device)
    velocities = torch.empty(steps, 2, device=device)
    if start_mode == "origin":
        current = torch.zeros(2, device=device)
    elif start_mode == "uniform":
        current = torch.empty(2, device=device).uniform_(-half, half)
    else:
        raise ValueError("start_mode must be 'origin' or 'uniform'")
    step_scale = _evaluation_step_scale(arena_size)
    for index in range(steps):
        direction = torch.rand((), device=device) * (2.0 * torch.pi)
        speed = torch.rand((), device=device) * step_scale
        step = torch.stack((torch.cos(direction), torch.sin(direction))) * speed
        next_position = current + step
        next_position = _reflect_into_box(next_position, half)
        velocities[index] = next_position - current
        positions[index] = next_position
        current = next_position
    return positions, velocities


def _evaluation_step_scale(arena_size: float) -> float:
    return arena_size * EVAL_STEP_SCALE_FRACTION


def _reflect_into_box(position: torch.Tensor, half_width: float) -> torch.Tensor:
    reflected = position.clone()
    over = reflected > half_width
    reflected = torch.where(over, 2.0 * half_width - reflected, reflected)
    under = reflected < -half_width
    reflected = torch.where(under, -2.0 * half_width - reflected, reflected)
    return reflected


def _accumulate_ratemaps(
    positions: np.ndarray,
    hidden_states: np.ndarray,
    *,
    arena_size: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_positions = positions.reshape(-1, 2)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    sums = np.zeros((flat_hidden.shape[-1], nbins, nbins), dtype=np.float64)
    counts = np.zeros((nbins, nbins), dtype=np.int64)
    scorer = GridScorer(nbins=nbins, coords_range=[[-arena_size / 2, arena_size / 2], [-arena_size / 2, arena_size / 2]])
    x_idx, y_idx, valid = scorer._digitize_positions(flat_positions[:, 0], flat_positions[:, 1])
    if not np.any(valid):
        return np.full_like(sums, np.nan), counts
    valid_act = flat_hidden[valid]
    np.add.at(counts, (x_idx, y_idx), 1)
    for unit in range(valid_act.shape[1]):
        np.add.at(sums[unit], (x_idx, y_idx), valid_act[:, unit])
    ratemaps = np.full_like(sums, np.nan)
    np.divide(sums, counts[None, :, :], out=ratemaps, where=counts[None, :, :] > 0)
    return ratemaps, counts


def _summarize_coverage(occupancy_counts: np.ndarray) -> dict[str, int | float]:
    visited = occupancy_counts > 0
    visited_bins = int(visited.sum())
    total_bins = int(occupancy_counts.size)
    return {
        "visited_bins": visited_bins,
        "unvisited_bins": total_bins - visited_bins,
        "total_bins": total_bins,
        "coverage_fraction": visited_bins / total_bins if total_bins else 0.0,
    }


def _summarize_unit_responses(
    ratemaps: np.ndarray,
    occupancy_counts: np.ndarray,
) -> list[dict[str, object]]:
    visited = occupancy_counts > 0
    if not np.any(visited):
        return [
            {
                "response_status": "no_coverage",
                "max_abs_response": None,
                "zero_response": False,
                "invalid_response": False,
            }
            for _ in range(ratemaps.shape[0])
        ]
    response_stats = []
    visited_ratemaps = ratemaps[:, visited]
    for unit_values in visited_ratemaps:
        finite = np.isfinite(unit_values)
        if not finite.all():
            response_stats.append(
                {
                    "response_status": "invalid",
                    "max_abs_response": None,
                    "zero_response": False,
                    "invalid_response": True,
                }
            )
            continue
        max_abs_response = float(np.max(np.abs(unit_values)))
        zero_response = max_abs_response <= ZERO_RESPONSE_EPS
        response_stats.append(
            {
                "response_status": "zero" if zero_response else "active",
                "max_abs_response": _json_float(max_abs_response),
                "zero_response": bool(zero_response),
                "invalid_response": False,
            }
        )
    return response_stats


def _unit_response_counts(unit_response_stats: list[dict[str, object]]) -> dict[str, int]:
    return {
        "units_without_coverage": _count_response_status(unit_response_stats, "no_coverage"),
        "zero_response_units": _count_response_status(unit_response_stats, "zero"),
        "invalid_response_units": _count_response_status(unit_response_stats, "invalid"),
        "active_units": _count_response_status(unit_response_stats, "active"),
    }


def _count_response_status(
    unit_response_stats: list[dict[str, object]],
    status: str,
) -> int:
    return sum(1 for row in unit_response_stats if row["response_status"] == status)


def _write_arena_artifacts(
    arena_dir: Path,
    *,
    ratemaps: np.ndarray,
    occupancy_counts: np.ndarray,
    sacs: np.ndarray,
    score_60: np.ndarray,
    score_90: np.ndarray,
    scales: np.ndarray,
    peak_counts: np.ndarray,
    mask_60: list[tuple[float, float]],
    mask_90: list[tuple[float, float]],
    unit_response_stats: list[dict[str, object]],
) -> None:
    np.savez_compressed(arena_dir / "ratemaps.npz", ratemaps=ratemaps)
    np.savez_compressed(arena_dir / "occupancy.npz", occupancy_counts=occupancy_counts)
    np.savez_compressed(arena_dir / "sacs.npz", sacs=sacs)
    save_ratemap_pdf(ratemaps, arena_dir / "ratemaps.pdf", "Ratemaps")
    save_sac_pdf(sacs, arena_dir / "sacs.pdf", "Spatial autocorrelograms")
    save_summary_figure(ratemaps, sacs, arena_dir / "summary.png", "SIC evaluation summary")
    rows = []
    with (arena_dir / "grid_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit",
                "score_60",
                "score_90",
                "scale",
                "peak_count",
                "response_status",
                "max_abs_response",
                "zero_response",
                "invalid_response",
                "mask_60_min",
                "mask_60_max",
                "mask_90_min",
                "mask_90_max",
            ],
        )
        writer.writeheader()
        for index in range(ratemaps.shape[0]):
            row = {
                "unit": index,
                "score_60": _json_float(score_60[index]),
                "score_90": _json_float(score_90[index]),
                "scale": _json_float(scales[index]),
                "peak_count": int(peak_counts[index]),
                **unit_response_stats[index],
                "mask_60_min": _json_float(mask_60[index][0]),
                "mask_60_max": _json_float(mask_60[index][1]),
                "mask_90_min": _json_float(mask_90[index][0]),
                "mask_90_max": _json_float(mask_90[index][1]),
            }
            writer.writerow(row)
            rows.append(row)
    with (arena_dir / "grid_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True, allow_nan=False)


def _config_from_checkpoint(config_dict: dict) -> Config:
    return load_config_from_dict(config_dict)


def load_config_from_dict(config_dict: dict) -> Config:
    from sic4gridcells.config import DataConfig, LossConfig, ModelConfig, TrainConfig

    cfg = Config(
        seed=int(config_dict["seed"]),
        device=str(config_dict["device"]),
        output_dir=str(config_dict["output_dir"]),
        data=DataConfig(**config_dict["data"]),
        model=ModelConfig(**config_dict["model"]),
        loss=LossConfig(**config_dict["loss"]),
        train=TrainConfig(**config_dict["train"]),
        assumptions=[str(item) for item in config_dict.get("assumptions", [])],
    )
    validate_config(cfg)
    return cfg


def _format_arena_size(value: float) -> str:
    return str(value).replace(".", "p")


def _safe_nanmean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _json_float(value: float | np.floating) -> float | None:
    as_float = float(value)
    if not np.isfinite(as_float):
        return None
    return as_float
