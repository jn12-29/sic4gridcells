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
from sic4gridcells.plotting import (
    save_metric_histogram,
    save_pairwise_distance_plot,
    save_ratemap_pdf,
    save_sac_pdf,
    save_summary_figure,
)

ZERO_RESPONSE_EPS = 1e-12
EVAL_STEP_SCALE_FRACTION = 0.15
EVAL_TURN_STD_RADIANS = 0.35
GRID_MODULE_SCORE_THRESHOLD = 0.0
GRID_MODULE_SCALE_RATIO = 1.2
PAIRWISE_DISTANCE_SAMPLE_PAIRS = 20000


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
    seed: int | None = None,
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
    eval_seed = cfg.seed if seed is None else int(seed)
    generator = _make_torch_generator(device_t, eval_seed)

    result_dirs: dict[float, Path] = {}
    summary_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for arena_index, arena_size in enumerate(arena_sizes):
            arena_dir = out_dir / f"arena_{_format_arena_size(arena_size)}"
            arena_dir.mkdir(parents=True, exist_ok=True)
            result_dirs[arena_size] = arena_dir
            positions, hidden_states, velocities = _run_bounded_random_walks(
                model,
                device_t,
                arena_size=arena_size,
                n_trajectories=n_trajectories,
                steps_per_trajectory=steps_per_trajectory,
                start_mode=start_mode,
                generator=generator,
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
            grid_metrics = scorer.calculate_grid_metrics(scores.sacs)
            scale_meters = _scale_pixels_to_meters(
                grid_metrics.scale_pixels,
                arena_size=arena_size,
                nbins=nbins,
            )
            module_ids = _assign_scale_modules(
                scale_meters,
                scores.score_60,
                unit_response_stats,
            )
            module_summary = _summarize_modules(
                module_ids,
                scale_meters,
                grid_metrics.orientation_degrees,
                scores.score_60,
            )
            trajectory_stats = _summarize_trajectory_stats(positions, velocities)
            pairwise_stats = _summarize_pairwise_neural_distances(
                positions,
                hidden_states,
                sigma_x=cfg.loss.sigma_x,
                seed=eval_seed + arena_index,
            )
            fourier_stats = _summarize_fourier_structure(
                ratemaps,
                scale_meters,
                module_ids,
                arena_size=arena_size,
            )
            phase_summary = _summarize_phase_tiling(
                ratemaps,
                scale_meters,
                module_ids,
                arena_size=arena_size,
            )
            state_space_summary, state_space_arrays = _summarize_state_space(
                hidden_states,
                module_ids,
            )
            _write_arena_artifacts(
                arena_dir,
                ratemaps=ratemaps,
                occupancy_counts=occupancy_counts,
                sacs=scores.sacs,
                score_60=scores.score_60,
                score_90=scores.score_90,
                scale_pixels=grid_metrics.scale_pixels,
                scale_meters=scale_meters,
                orientation_degrees=grid_metrics.orientation_degrees,
                peak_counts=grid_metrics.peak_counts,
                module_ids=module_ids,
                module_summary=module_summary,
                trajectory_stats=trajectory_stats,
                pairwise_stats=pairwise_stats,
                fourier_stats=fourier_stats,
                phase_summary=phase_summary,
                state_space_summary=state_space_summary,
                state_space_arrays=state_space_arrays,
                mask_60=scores.mask_60,
                mask_90=scores.mask_90,
                unit_response_stats=unit_response_stats,
            )
            summary_rows.append(
                {
                    "arena_size": arena_size,
                    "mean_grid_score_60": _json_float(np.nanmean(scores.score_60)),
                    "mean_grid_score_90": _json_float(np.nanmean(scores.score_90)),
                    "mean_scale": _json_float(_safe_nanmean(grid_metrics.scale_pixels)),
                    "mean_scale_pixels": _json_float(_safe_nanmean(grid_metrics.scale_pixels)),
                    "mean_scale_meters": _json_float(_safe_nanmean(scale_meters)),
                    "mean_orientation_degrees": _json_float(
                        _circular_nanmean(grid_metrics.orientation_degrees, period=60.0)
                    ),
                    "detected_modules": int(len(module_summary)),
                    "state_space_modules": int(len(state_space_summary)),
                    "near_spatial_pair_count": int(
                        pairwise_stats["summary"]["near_spatial_pair_count"]
                    ),
                    "near_spatial_mean_neural_distance": _json_float(
                        pairwise_stats["summary"]["near_spatial_mean_neural_distance"]
                    ),
                    **coverage_summary,
                    **_unit_response_counts(unit_response_stats),
                }
            )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(Path(checkpoint_path)),
                "config": asdict(cfg),
                "evaluation_seed": eval_seed,
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
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_positions = []
    batch_hidden = []
    batch_velocities = []
    for _ in range(n_trajectories):
        positions, velocities = _sample_bounded_random_walk(
            steps_per_trajectory,
            arena_size,
            device=device,
            start_mode=start_mode,
            generator=generator,
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
        batch_velocities.append(velocities.unsqueeze(0))
        batch_hidden.append(rollout.hidden_states.cpu())
    return (
        torch.cat(batch_positions, dim=0).cpu().numpy(),
        torch.cat(batch_hidden, dim=0).cpu().numpy(),
        torch.cat(batch_velocities, dim=0).cpu().numpy(),
    )


def _sample_bounded_random_walk(
    steps: int,
    arena_size: float,
    *,
    device: torch.device,
    start_mode: str = "origin",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = arena_size / 2.0
    positions = torch.empty(steps, 2, device=device)
    velocities = torch.empty(steps, 2, device=device)
    if start_mode == "origin":
        current = torch.zeros(2, device=device)
    elif start_mode == "uniform":
        current = torch.empty(2, device=device).uniform_(-half, half, generator=generator)
    else:
        raise ValueError("start_mode must be 'origin' or 'uniform'")
    step_scale = _evaluation_step_scale(arena_size)
    heading = torch.rand((), device=device, generator=generator) * (2.0 * torch.pi)
    for index in range(steps):
        if index > 0:
            heading = (
                heading
                + torch.randn((), device=device, generator=generator) * EVAL_TURN_STD_RADIANS
            )
        speed = (0.5 + 0.5 * torch.rand((), device=device, generator=generator)) * step_scale
        step = torch.stack((torch.cos(heading), torch.sin(heading))) * speed
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


def _scale_pixels_to_meters(
    scale_pixels: np.ndarray,
    *,
    arena_size: float,
    nbins: int,
) -> np.ndarray:
    return np.asarray(scale_pixels, dtype=np.float64) * (arena_size / nbins)


def _assign_scale_modules(
    scale_meters: np.ndarray,
    score_60: np.ndarray,
    unit_response_stats: list[dict[str, object]],
    *,
    score_threshold: float = GRID_MODULE_SCORE_THRESHOLD,
    scale_ratio: float = GRID_MODULE_SCALE_RATIO,
) -> np.ndarray:
    module_ids = np.full(scale_meters.shape[0], -1, dtype=np.int64)
    active = np.asarray([row["response_status"] == "active" for row in unit_response_stats])
    eligible = (
        active
        & np.isfinite(scale_meters)
        & (scale_meters > 0.0)
        & np.isfinite(score_60)
        & (score_60 > score_threshold)
    )
    order = np.argsort(scale_meters[eligible])
    eligible_indices = np.flatnonzero(eligible)[order]
    current_module = -1
    previous_scale: float | None = None
    for index in eligible_indices:
        scale = float(scale_meters[index])
        if previous_scale is None or scale / previous_scale > scale_ratio:
            current_module += 1
        module_ids[index] = current_module
        previous_scale = scale
    return module_ids


def _summarize_modules(
    module_ids: np.ndarray,
    scale_meters: np.ndarray,
    orientation_degrees: np.ndarray,
    score_60: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for module_id in sorted(int(value) for value in np.unique(module_ids) if value >= 0):
        member = module_ids == module_id
        rows.append(
            {
                "module_id": module_id,
                "unit_count": int(member.sum()),
                "mean_scale_meters": _json_float(_safe_nanmean(scale_meters[member])),
                "median_scale_meters": _json_float(_safe_nanmedian(scale_meters[member])),
                "mean_orientation_degrees": _json_float(
                    _circular_nanmean(orientation_degrees[member], period=60.0)
                ),
                "mean_grid_score_60": _json_float(_safe_nanmean(score_60[member])),
            }
        )
    return rows


def _summarize_trajectory_stats(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, object]:
    del positions
    speeds = np.linalg.norm(velocities.reshape(-1, 2), axis=1)
    turn_angles = _turn_angles_degrees(velocities)
    return {
        "trajectory_count": int(velocities.shape[0]),
        "steps_per_trajectory": int(velocities.shape[1]),
        "mean_speed": _json_float(_safe_nanmean(speeds)),
        "max_speed": _json_float(float(np.max(speeds)) if speeds.size else float("nan")),
        "mean_abs_turn_degrees": _json_float(_safe_nanmean(np.abs(turn_angles))),
        "turn_std_radians": EVAL_TURN_STD_RADIANS,
    }


def _turn_angles_degrees(velocities: np.ndarray) -> np.ndarray:
    if velocities.shape[1] < 2:
        return np.empty(0, dtype=np.float64)
    prev = velocities[:, :-1, :]
    curr = velocities[:, 1:, :]
    prev_norm = np.linalg.norm(prev, axis=-1)
    curr_norm = np.linalg.norm(curr, axis=-1)
    valid = (prev_norm > 1e-12) & (curr_norm > 1e-12)
    if not np.any(valid):
        return np.empty(0, dtype=np.float64)
    dot = np.sum(prev * curr, axis=-1)
    cos_angle = np.divide(dot, prev_norm * curr_norm, out=np.ones_like(dot), where=valid)
    cos_angle = np.clip(cos_angle[valid], -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _summarize_pairwise_neural_distances(
    positions: np.ndarray,
    hidden_states: np.ndarray,
    *,
    sigma_x: float,
    seed: int,
    max_pairs: int = PAIRWISE_DISTANCE_SAMPLE_PAIRS,
) -> dict[str, object]:
    flat_positions = positions.reshape(-1, 2)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    point_count = flat_positions.shape[0]
    if point_count < 2:
        return {
            "summary": {
                "sampled_pair_count": 0,
                "near_spatial_pair_count": 0,
                "near_spatial_mean_neural_distance": None,
            },
            "rows": [],
        }
    rng = np.random.default_rng(seed)
    sample_count = int(min(max_pairs, point_count * max(point_count - 1, 1)))
    left = rng.integers(0, point_count, size=sample_count)
    right = rng.integers(0, point_count, size=sample_count)
    keep = left != right
    left = left[keep]
    right = right[keep]
    if left.size == 0:
        return {
            "summary": {
                "sampled_pair_count": 0,
                "near_spatial_pair_count": 0,
                "near_spatial_mean_neural_distance": None,
            },
            "rows": [],
        }
    spatial_distance = np.linalg.norm(flat_positions[left] - flat_positions[right], axis=1)
    neural_distance = np.linalg.norm(flat_hidden[left] - flat_hidden[right], axis=1)
    trajectory_count, steps_per_trajectory = positions.shape[:2]
    trajectory_ids = np.repeat(np.arange(trajectory_count), steps_per_trajectory)
    step_ids = np.tile(np.arange(steps_per_trajectory), trajectory_count)
    same_trajectory = trajectory_ids[left] == trajectory_ids[right]
    temporal_separation = np.abs(step_ids[left] - step_ids[right])
    spatial_edges = np.asarray([0.0, sigma_x, 2 * sigma_x, 4 * sigma_x, 8 * sigma_x, np.inf])
    temporal_edges = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf])
    rows = _binned_neural_distance_rows(
        "spatial",
        spatial_distance,
        neural_distance,
        spatial_edges,
        spatial_values=spatial_distance,
    )
    rows.extend(
        _binned_neural_distance_rows(
            "temporal",
            temporal_separation[same_trajectory].astype(np.float64),
            neural_distance[same_trajectory],
            temporal_edges,
            spatial_values=spatial_distance[same_trajectory],
        )
    )
    near_spatial = spatial_distance <= sigma_x
    near_mean = float(np.mean(neural_distance[near_spatial])) if np.any(near_spatial) else float("nan")
    return {
        "summary": {
            "sampled_pair_count": int(left.size),
            "near_spatial_pair_count": int(near_spatial.sum()),
            "near_spatial_mean_neural_distance": _json_float(near_mean),
        },
        "rows": rows,
    }


def _binned_neural_distance_rows(
    kind: str,
    values: np.ndarray,
    neural_distance: np.ndarray,
    edges: np.ndarray,
    *,
    spatial_values: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        if np.isinf(high):
            mask = values >= low
        else:
            mask = (values >= low) & (values < high)
        count = int(mask.sum())
        rows.append(
            {
                "kind": kind,
                "bin_low": _json_float(low),
                "bin_high": None if np.isinf(high) else _json_float(high),
                "count": count,
                "mean_neural_distance": _json_float(
                    float(np.mean(neural_distance[mask])) if count else float("nan")
                ),
                "mean_spatial_distance": _json_float(
                    float(np.mean(spatial_values[mask])) if count else float("nan")
                ),
                "mean_bin_value": _json_float(float(np.mean(values[mask])) if count else float("nan")),
            }
        )
    return rows


def _summarize_fourier_structure(
    ratemaps: np.ndarray,
    scale_meters: np.ndarray,
    module_ids: np.ndarray,
    *,
    arena_size: float,
) -> list[dict[str, object]]:
    nbins = ratemaps.shape[-1]
    bin_width = arena_size / nbins
    frequencies = np.fft.fftshift(np.fft.fftfreq(nbins, d=bin_width))
    freq_y, freq_x = np.meshgrid(frequencies, frequencies, indexing="ij")
    center = nbins // 2
    rows = []
    for unit, ratemap in enumerate(ratemaps):
        finite = np.isfinite(ratemap)
        if not np.any(finite):
            rows.append(_empty_fourier_row(unit, int(module_ids[unit])))
            continue
        filled = np.where(finite, ratemap, np.nanmean(ratemap[finite]))
        centered = filled - np.mean(filled)
        power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
        power[center, center] = 0.0
        total_power = float(np.sum(power))
        if total_power <= 0.0:
            rows.append(_empty_fourier_row(unit, int(module_ids[unit])))
            continue
        peak_index = np.unravel_index(int(np.argmax(power)), power.shape)
        fx = float(freq_x[peak_index])
        fy = float(freq_y[peak_index])
        frequency = float(np.hypot(fx, fy))
        period = 1.0 / frequency if frequency > 0.0 else float("nan")
        rows.append(
            {
                "unit": unit,
                "module_id": int(module_ids[unit]),
                "dominant_frequency_cycles_per_meter": _json_float(frequency),
                "dominant_period_meters": _json_float(period),
                "dominant_orientation_degrees": _json_float(
                    float(np.mod(np.degrees(np.arctan2(fy, fx)), 180.0))
                ),
                "dominant_power_fraction": _json_float(float(power[peak_index] / total_power)),
                "scale_meters": _json_float(scale_meters[unit]),
            }
        )
    return rows


def _empty_fourier_row(unit: int, module_id: int) -> dict[str, object]:
    return {
        "unit": unit,
        "module_id": module_id,
        "dominant_frequency_cycles_per_meter": None,
        "dominant_period_meters": None,
        "dominant_orientation_degrees": None,
        "dominant_power_fraction": None,
        "scale_meters": None,
    }


def _summarize_phase_tiling(
    ratemaps: np.ndarray,
    scale_meters: np.ndarray,
    module_ids: np.ndarray,
    *,
    arena_size: float,
) -> list[dict[str, object]]:
    nbins = ratemaps.shape[-1]
    edges = np.linspace(-arena_size / 2.0, arena_size / 2.0, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows = []
    for unit, ratemap in enumerate(ratemaps):
        scale = float(scale_meters[unit])
        finite = np.isfinite(ratemap)
        if not np.any(finite) or not np.isfinite(scale) or scale <= 0.0:
            rows.append(
                {
                    "unit": unit,
                    "module_id": int(module_ids[unit]),
                    "peak_x_meters": None,
                    "peak_y_meters": None,
                    "phase_x": None,
                    "phase_y": None,
                    "scale_meters": _json_float(scale),
                }
            )
            continue
        filled = np.where(finite, ratemap, -np.inf)
        peak_y, peak_x = np.unravel_index(int(np.argmax(filled)), filled.shape)
        peak_x_m = float(centers[peak_x])
        peak_y_m = float(centers[peak_y])
        rows.append(
            {
                "unit": unit,
                "module_id": int(module_ids[unit]),
                "peak_x_meters": _json_float(peak_x_m),
                "peak_y_meters": _json_float(peak_y_m),
                "phase_x": _json_float(float(np.mod(peak_x_m + arena_size / 2.0, scale) / scale)),
                "phase_y": _json_float(float(np.mod(peak_y_m + arena_size / 2.0, scale) / scale)),
                "scale_meters": _json_float(scale),
            }
        )
    return rows


def _summarize_state_space(
    hidden_states: np.ndarray,
    module_ids: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for module_id in sorted(int(value) for value in np.unique(module_ids) if value >= 0):
        unit_indices = np.flatnonzero(module_ids == module_id)
        if unit_indices.size < 2:
            continue
        module_activity = flat_hidden[:, unit_indices].astype(np.float64, copy=False)
        finite_samples = np.isfinite(module_activity).all(axis=1)
        module_activity = module_activity[finite_samples]
        if module_activity.shape[0] < 2:
            rows.append(
                {
                    "module_id": module_id,
                    "unit_count": int(unit_indices.size),
                    "sample_count": int(module_activity.shape[0]),
                    "dropped_nonfinite_samples": int((~finite_samples).sum()),
                    "pca_components": 0,
                    "top3_explained_variance": None,
                    "top6_explained_variance": None,
                    "has_six_units": bool(unit_indices.size >= 6),
                }
            )
            continue
        centered = module_activity - module_activity.mean(axis=0, keepdims=True)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        total_variance = float(np.sum(singular_values**2))
        if total_variance > 0.0:
            explained = (singular_values**2) / total_variance
        else:
            explained = np.zeros_like(singular_values)
        projection = np.zeros((centered.shape[0], 3), dtype=np.float64)
        component_count = min(3, vh.shape[0])
        if component_count:
            projection[:, :component_count] = centered @ vh[:component_count].T
        prefix = f"module_{module_id}"
        arrays[f"{prefix}_unit_indices"] = unit_indices.astype(np.int64)
        arrays[f"{prefix}_projection3"] = projection.astype(np.float32)
        arrays[f"{prefix}_explained_variance_ratio"] = explained.astype(np.float64)
        rows.append(
            {
                "module_id": module_id,
                "unit_count": int(unit_indices.size),
                "sample_count": int(module_activity.shape[0]),
                "dropped_nonfinite_samples": int((~finite_samples).sum()),
                "pca_components": int(explained.size),
                "top3_explained_variance": _json_float(float(np.sum(explained[:3]))),
                "top6_explained_variance": _json_float(float(np.sum(explained[:6]))),
                "has_six_units": bool(unit_indices.size >= 6),
            }
        )
    return rows, arrays


def _write_arena_artifacts(
    arena_dir: Path,
    *,
    ratemaps: np.ndarray,
    occupancy_counts: np.ndarray,
    sacs: np.ndarray,
    score_60: np.ndarray,
    score_90: np.ndarray,
    scale_pixels: np.ndarray,
    scale_meters: np.ndarray,
    orientation_degrees: np.ndarray,
    peak_counts: np.ndarray,
    module_ids: np.ndarray,
    module_summary: list[dict[str, object]],
    trajectory_stats: dict[str, object],
    pairwise_stats: dict[str, object],
    fourier_stats: list[dict[str, object]],
    phase_summary: list[dict[str, object]],
    state_space_summary: list[dict[str, object]],
    state_space_arrays: dict[str, np.ndarray],
    mask_60: list[tuple[float, float]],
    mask_90: list[tuple[float, float]],
    unit_response_stats: list[dict[str, object]],
) -> None:
    np.savez_compressed(arena_dir / "ratemaps.npz", ratemaps=ratemaps)
    np.savez_compressed(arena_dir / "occupancy.npz", occupancy_counts=occupancy_counts)
    np.savez_compressed(arena_dir / "sacs.npz", sacs=sacs)
    np.savez_compressed(
        arena_dir / "grid_metrics.npz",
        score_60=score_60,
        score_90=score_90,
        scale_pixels=scale_pixels,
        scale_meters=scale_meters,
        orientation_degrees=orientation_degrees,
        peak_counts=peak_counts,
        module_ids=module_ids,
    )
    save_ratemap_pdf(ratemaps, arena_dir / "ratemaps.pdf", "Ratemaps")
    save_sac_pdf(sacs, arena_dir / "sacs.pdf", "Spatial autocorrelograms")
    save_summary_figure(ratemaps, sacs, arena_dir / "summary.png", "SIC evaluation summary")
    save_metric_histogram(score_60, arena_dir / "grid_score_60_histogram.png", "Grid score 60")
    save_metric_histogram(scale_meters, arena_dir / "scale_meters_histogram.png", "Grid scale (m)")
    rows = []
    with (arena_dir / "grid_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit",
                "score_60",
                "score_90",
                "scale",
                "scale_pixels",
                "scale_meters",
                "orientation_degrees",
                "module_id",
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
                "scale": _json_float(scale_pixels[index]),
                "scale_pixels": _json_float(scale_pixels[index]),
                "scale_meters": _json_float(scale_meters[index]),
                "orientation_degrees": _json_float(orientation_degrees[index]),
                "module_id": int(module_ids[index]),
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
    _write_json(arena_dir / "module_summary.json", module_summary)
    _write_csv(
        arena_dir / "module_summary.csv",
        module_summary,
        [
            "module_id",
            "unit_count",
            "mean_scale_meters",
            "median_scale_meters",
            "mean_orientation_degrees",
            "mean_grid_score_60",
        ],
    )
    _write_json(arena_dir / "trajectory_stats.json", trajectory_stats)
    _write_json(arena_dir / "pairwise_distance_stats.json", pairwise_stats)
    save_pairwise_distance_plot(
        list(pairwise_stats["rows"]),
        arena_dir / "pairwise_distance.png",
        "Neural distance by separation",
    )
    _write_csv(
        arena_dir / "pairwise_distance_stats.csv",
        list(pairwise_stats["rows"]),
        [
            "kind",
            "bin_low",
            "bin_high",
            "count",
            "mean_neural_distance",
            "mean_spatial_distance",
            "mean_bin_value",
        ],
    )
    _write_json(arena_dir / "fourier_stats.json", fourier_stats)
    _write_csv(
        arena_dir / "fourier_stats.csv",
        fourier_stats,
        [
            "unit",
            "module_id",
            "dominant_frequency_cycles_per_meter",
            "dominant_period_meters",
            "dominant_orientation_degrees",
            "dominant_power_fraction",
            "scale_meters",
        ],
    )
    _write_json(arena_dir / "phase_summary.json", phase_summary)
    _write_csv(
        arena_dir / "phase_summary.csv",
        phase_summary,
        [
            "unit",
            "module_id",
            "peak_x_meters",
            "peak_y_meters",
            "phase_x",
            "phase_y",
            "scale_meters",
        ],
    )
    _write_json(arena_dir / "state_space_summary.json", state_space_summary)
    _write_csv(
        arena_dir / "state_space_summary.csv",
        state_space_summary,
        [
            "module_id",
            "unit_count",
            "sample_count",
            "dropped_nonfinite_samples",
            "pca_components",
            "top3_explained_variance",
            "top6_explained_variance",
            "has_six_units",
        ],
    )
    np.savez_compressed(arena_dir / "state_space_modules.npz", **state_space_arrays)


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device: torch.device | str = device if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _safe_nanmean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _safe_nanmedian(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.median(values[finite]))


def _circular_nanmean(values: np.ndarray, *, period: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    angles = values[finite] / period * 2.0 * np.pi
    mean = np.angle(np.mean(np.exp(1j * angles)))
    return float(np.mod(mean / (2.0 * np.pi) * period, period))


def _json_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    as_float = float(value)
    if not np.isfinite(as_float):
        return None
    return as_float
