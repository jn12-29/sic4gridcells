from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from sic4gridcells.runtime import atomic_json_dump


DEFAULT_MIN_GRID_SCORE_60 = 0.0
DEFAULT_MIN_MODULE_UNITS = 3
DEFAULT_MAX_SCALE_RATIO_WITHIN_MODULE = 1.2
DEFAULT_PATH_SAMPLE_PAIRS = 20000
DEFAULT_STATE_SPACE_SAMPLES = 512


@dataclass(frozen=True)
class AnalysisResult:
    eval_dir: Path
    output_dir: Path
    manifest_path: Path
    summary_path: Path


def analyze_evaluation_output(
    eval_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    run_id: str | None = None,
    seed: int | None = None,
    variant: str = "baseline",
    diagnostic_only: bool = False,
    min_grid_score_60: float = DEFAULT_MIN_GRID_SCORE_60,
    min_module_units: int = DEFAULT_MIN_MODULE_UNITS,
    max_scale_ratio_within_module: float = DEFAULT_MAX_SCALE_RATIO_WITHIN_MODULE,
    max_path_pairs: int = DEFAULT_PATH_SAMPLE_PAIRS,
    max_state_space_samples: int = DEFAULT_STATE_SPACE_SAMPLES,
) -> AnalysisResult:
    """Build figure-ready analysis artifacts from an existing evaluation directory."""

    eval_path = Path(eval_dir)
    out_dir = Path(output_dir) if output_dir is not None else eval_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "summary_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json_object(eval_path / "summary.json")
    effective_run_id = run_id or eval_path.parent.name or eval_path.name
    effective_seed = _resolve_seed(summary, seed)
    arena_infos = _arena_infos(eval_path, summary)

    all_unit_rows: list[dict[str, Any]] = []
    all_module_rows: list[dict[str, Any]] = []
    all_fourier_rows: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []
    all_state_rows: list[dict[str, Any]] = []
    path_summary_rows: list[dict[str, Any]] = []
    path_bin_rows: list[dict[str, Any]] = []
    dependencies: list[str] = [str(eval_path / "summary.json")]

    for arena_size, arena_dir in arena_infos:
        arena_out = out_dir / f"arena_{_format_arena_size(arena_size)}"
        arena_out.mkdir(parents=True, exist_ok=True)
        grid_rows = _load_json_list(arena_dir / "grid_stats.json")
        dependencies.append(str(arena_dir / "grid_stats.json"))

        module_result = robust_module_detection(
            grid_rows,
            run_id=effective_run_id,
            seed=effective_seed,
            variant=variant,
            arena_size=arena_size,
            min_grid_score_60=min_grid_score_60,
            min_module_units=min_module_units,
            max_scale_ratio_within_module=max_scale_ratio_within_module,
        )
        unit_rows = module_result["unit_rows"]
        module_rows = module_result["module_rows"]
        all_unit_rows.extend(unit_rows)
        all_module_rows.extend(module_rows)
        _write_rows(arena_out / "unit_modules", unit_rows, UNIT_MODULE_FIELDS)
        _write_rows(arena_out / "module_summary", module_rows, MODULE_SUMMARY_FIELDS)

        ratemap_path = arena_dir / "ratemaps.npz"
        if ratemap_path.exists():
            dependencies.append(str(ratemap_path))
            with np.load(ratemap_path) as ratemap_data:
                ratemaps = np.asarray(ratemap_data["ratemaps"], dtype=np.float64)
            fourier_rows = module_fourier_lattice_vectors(
                ratemaps,
                unit_rows,
                run_id=effective_run_id,
                seed=effective_seed,
                variant=variant,
                arena_size=arena_size,
            )
            phase_rows = phase_tiling_table(
                ratemaps,
                unit_rows,
                run_id=effective_run_id,
                seed=effective_seed,
                variant=variant,
                arena_size=arena_size,
            )
        else:
            if not diagnostic_only:
                raise FileNotFoundError(f"Required analysis artifact is missing: {ratemap_path}")
            fourier_rows = []
            phase_rows = []
        all_fourier_rows.extend(fourier_rows)
        all_phase_rows.extend(phase_rows)
        _write_rows(arena_out / "fourier_lattice_vectors", fourier_rows, FOURIER_FIELDS)
        _write_rows(arena_out / "phase_tiling", phase_rows, PHASE_FIELDS)

        rollout_path = arena_dir / "rollout_arrays.npz"
        if rollout_path.exists():
            dependencies.append(str(rollout_path))
            with np.load(rollout_path) as rollout_data:
                positions = np.asarray(rollout_data["positions"], dtype=np.float64)
                hidden_states = np.asarray(rollout_data["hidden_states"], dtype=np.float64)
            path_summary, path_bins = path_invariance_probe(
                positions,
                hidden_states,
                run_id=effective_run_id,
                seed=effective_seed,
                variant=variant,
                arena_size=arena_size,
                ratemap_nbins=_ratemap_nbins(arena_dir),
                max_pairs=max_path_pairs,
            )
            state_rows, state_arrays = state_space_artifacts(
                hidden_states,
                unit_rows,
                run_id=effective_run_id,
                seed=effective_seed,
                variant=variant,
                arena_size=arena_size,
                max_samples=max_state_space_samples,
            )
            np.savez_compressed(arena_out / "state_space_ext.npz", **state_arrays)
        else:
            if not diagnostic_only:
                raise FileNotFoundError(f"Required analysis artifact is missing: {rollout_path}")
            path_summary = _empty_path_summary(
                run_id=effective_run_id,
                seed=effective_seed,
                variant=variant,
                arena_size=arena_size,
            )
            path_bins = []
            state_rows = []
            np.savez_compressed(arena_out / "state_space_ext.npz")
        path_summary_rows.append(path_summary)
        path_bin_rows.extend(path_bins)
        all_state_rows.extend(state_rows)
        _write_rows(arena_out / "path_invariance_bins", path_bins, PATH_BIN_FIELDS)
        _write_rows(arena_out / "state_space_summary", state_rows, STATE_SPACE_FIELDS)

    cross_unit_rows, cross_module_rows = cross_arena_aggregation(all_unit_rows)
    summary_payload = {
        "run_id": effective_run_id,
        "seed": effective_seed,
        "variant": variant,
        "diagnostic_only": diagnostic_only,
        "eval_dir": str(eval_path),
        "output_dir": str(out_dir),
        "arena_count": len(arena_infos),
        "unit_rows": len(all_unit_rows),
        "module_rows": len(all_module_rows),
        "path_invariance_rows": len(path_summary_rows),
        "cross_arena_unit_rows": len(cross_unit_rows),
        "cross_arena_module_rows": len(cross_module_rows),
        "parameters": {
            "min_grid_score_60": min_grid_score_60,
            "min_module_units": min_module_units,
            "max_scale_ratio_within_module": max_scale_ratio_within_module,
            "max_path_pairs": max_path_pairs,
            "max_state_space_samples": max_state_space_samples,
        },
    }

    _write_rows(tables_dir / "unit_modules", all_unit_rows, UNIT_MODULE_FIELDS)
    _write_rows(tables_dir / "module_summary", all_module_rows, MODULE_SUMMARY_FIELDS)
    _write_rows(
        tables_dir / "cross_arena_unit_metrics",
        cross_unit_rows,
        CROSS_ARENA_UNIT_FIELDS,
    )
    _write_rows(
        tables_dir / "cross_arena_module_stability",
        cross_module_rows,
        CROSS_ARENA_MODULE_FIELDS,
    )
    _write_rows(tables_dir / "path_invariance_summary", path_summary_rows, PATH_SUMMARY_FIELDS)
    _write_rows(tables_dir / "path_invariance_bins", path_bin_rows, PATH_BIN_FIELDS)
    _write_rows(tables_dir / "fourier_lattice_vectors", all_fourier_rows, FOURIER_FIELDS)
    _write_rows(tables_dir / "phase_tiling", all_phase_rows, PHASE_FIELDS)
    _write_rows(tables_dir / "state_space_summary", all_state_rows, STATE_SPACE_FIELDS)
    atomic_json_dump(summary_payload, out_dir / "summary.json")
    manifest = {
        **summary_payload,
        "dependencies": sorted(set(dependencies)),
        "outputs": {
            "summary": str(out_dir / "summary.json"),
            "summary_tables": str(tables_dir),
            "manifest": str(out_dir / "analysis_manifest.json"),
        },
    }
    atomic_json_dump(manifest, out_dir / "analysis_manifest.json")
    return AnalysisResult(
        eval_dir=eval_path,
        output_dir=out_dir,
        manifest_path=out_dir / "analysis_manifest.json",
        summary_path=out_dir / "summary.json",
    )


def robust_module_detection(
    grid_rows: list[dict[str, Any]],
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
    min_grid_score_60: float = DEFAULT_MIN_GRID_SCORE_60,
    min_module_units: int = DEFAULT_MIN_MODULE_UNITS,
    max_scale_ratio_within_module: float = DEFAULT_MAX_SCALE_RATIO_WITHIN_MODULE,
) -> dict[str, list[dict[str, Any]]]:
    parsed = sorted(grid_rows, key=lambda row: int(row.get("unit", 0)))
    base_rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in parsed:
        unit = int(row["unit"])
        score_60 = _optional_float(row.get("score_60"))
        score_90 = _optional_float(row.get("score_90"))
        scale = _optional_float(row.get("scale_meters"))
        orientation = _optional_float(row.get("orientation_degrees"))
        response_status = str(row.get("response_status", "unknown"))
        reason = _rejection_reason(
            response_status=response_status,
            score_60=score_60,
            scale_meters=scale,
            min_grid_score_60=min_grid_score_60,
        )
        base = {
            "run_id": run_id,
            "seed": seed,
            "variant": variant,
            "arena_size": arena_size,
            "unit": unit,
            "module_id": -1,
            "source_module_id": _optional_int(row.get("module_id")),
            "module_confidence": None,
            "rejection_reason": reason,
            "response_status": response_status,
            "score_60": score_60,
            "score_90": score_90,
            "scale_meters": scale,
            "orientation_degrees": orientation,
        }
        base_rows.append(base)
        if reason is None:
            eligible.append(base)

    eligible.sort(key=lambda row: float(row["scale_meters"]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    group_min_scale: float | None = None
    for row in eligible:
        scale = float(row["scale_meters"])
        if (
            group_min_scale is None
            or scale / group_min_scale <= max_scale_ratio_within_module
        ):
            current.append(row)
            group_min_scale = scale if group_min_scale is None else group_min_scale
        else:
            groups.append(current)
            current = [row]
            group_min_scale = scale
    if current:
        groups.append(current)

    module_rows: list[dict[str, Any]] = []
    next_module_id = 0
    for group in groups:
        if len(group) < min_module_units:
            for row in group:
                row["rejection_reason"] = "module_too_small"
            continue
        module_id = next_module_id
        next_module_id += 1
        scales = np.asarray([float(row["scale_meters"]) for row in group], dtype=np.float64)
        scores = np.asarray([float(row["score_60"]) for row in group], dtype=np.float64)
        orientations = np.asarray(
            [_optional_float(row["orientation_degrees"]) for row in group],
            dtype=np.float64,
        )
        scale_ratio = float(np.max(scales) / np.min(scales)) if scales.size else float("nan")
        confidence = _module_confidence(
            unit_count=len(group),
            min_module_units=min_module_units,
            scores=scores,
            min_grid_score_60=min_grid_score_60,
            scale_ratio=scale_ratio,
            max_scale_ratio_within_module=max_scale_ratio_within_module,
        )
        for row in group:
            row["module_id"] = module_id
            row["module_confidence"] = confidence
            row["rejection_reason"] = None
        module_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "variant": variant,
                "arena_size": arena_size,
                "module_id": module_id,
                "unit_count": len(group),
                "mean_scale_meters": _json_float(_finite_mean(scales)),
                "median_scale_meters": _json_float(_finite_median(scales)),
                "scale_ratio": _json_float(scale_ratio),
                "mean_grid_score_60": _json_float(_finite_mean(scores)),
                "mean_orientation_degrees": _json_float(_circular_mean(orientations, period=60.0)),
                "module_confidence": confidence,
                "unit_ids": " ".join(str(row["unit"]) for row in group),
            }
        )
    return {"unit_rows": base_rows, "module_rows": module_rows}


def cross_arena_aggregation(
    unit_rows: list[dict[str, Any]],
    *,
    max_scale_ratio: float = DEFAULT_MAX_SCALE_RATIO_WITHIN_MODULE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not unit_rows:
        return [], []
    arena_sizes = sorted({float(row["arena_size"]) for row in unit_rows})
    if len(arena_sizes) < 2:
        return [], []
    reference_size = arena_sizes[0]
    rows_by_unit_arena = {
        (int(row["unit"]), float(row["arena_size"])): row for row in unit_rows
    }
    units = sorted({int(row["unit"]) for row in unit_rows})
    unit_output: list[dict[str, Any]] = []
    for unit in units:
        reference = rows_by_unit_arena.get((unit, reference_size))
        if reference is None:
            continue
        for arena_size in arena_sizes[1:]:
            current = rows_by_unit_arena.get((unit, arena_size))
            if current is None:
                continue
            ref_scale = _optional_float(reference.get("scale_meters"))
            cur_scale = _optional_float(current.get("scale_meters"))
            scale_ratio = _symmetric_ratio(ref_scale, cur_scale)
            orientation_delta = _orientation_delta(
                _optional_float(reference.get("orientation_degrees")),
                _optional_float(current.get("orientation_degrees")),
                period=60.0,
            )
            stable = (
                scale_ratio is not None
                and scale_ratio <= max_scale_ratio
                and int(reference.get("module_id", -1)) >= 0
                and int(current.get("module_id", -1)) >= 0
            )
            unit_output.append(
                {
                    "run_id": current["run_id"],
                    "seed": current["seed"],
                    "variant": current["variant"],
                    "unit": unit,
                    "reference_arena_size": reference_size,
                    "arena_size": arena_size,
                    "reference_module_id": reference.get("module_id"),
                    "module_id": current.get("module_id"),
                    "reference_grid_score_60": reference.get("score_60"),
                    "grid_score_60": current.get("score_60"),
                    "reference_scale_meters": ref_scale,
                    "scale_meters": cur_scale,
                    "scale_ratio": scale_ratio,
                    "orientation_delta_degrees": orientation_delta,
                    "stable": stable,
                }
            )

    module_output: list[dict[str, Any]] = []
    module_ids = sorted(
        {
            int(row["module_id"])
            for row in unit_rows
            if _optional_int(row.get("module_id")) is not None and int(row["module_id"]) >= 0
        }
    )
    for module_id in module_ids:
        ref_members = [
            row
            for row in unit_rows
            if float(row["arena_size"]) == reference_size and int(row["module_id"]) == module_id
        ]
        if not ref_members:
            continue
        for arena_size in arena_sizes[1:]:
            members = [
                row
                for row in unit_rows
                if float(row["arena_size"]) == arena_size and int(row["module_id"]) == module_id
            ]
            ref_scale = _finite_mean(
                np.asarray([_optional_float(row.get("scale_meters")) for row in ref_members])
            )
            cur_scale = _finite_mean(
                np.asarray([_optional_float(row.get("scale_meters")) for row in members])
            )
            scale_ratio = _symmetric_ratio(ref_scale, cur_scale)
            module_output.append(
                {
                    "run_id": ref_members[0]["run_id"],
                    "seed": ref_members[0]["seed"],
                    "variant": ref_members[0]["variant"],
                    "module_id": module_id,
                    "reference_arena_size": reference_size,
                    "arena_size": arena_size,
                    "reference_unit_count": len(ref_members),
                    "unit_count": len(members),
                    "reference_mean_scale_meters": _json_float(ref_scale),
                    "mean_scale_meters": _json_float(cur_scale),
                    "scale_ratio": scale_ratio,
                    "stable_unit_fraction": _stable_unit_fraction(
                        unit_output,
                        module_id=module_id,
                        arena_size=arena_size,
                    ),
                }
            )
    return unit_output, module_output


def path_invariance_probe(
    positions: np.ndarray,
    hidden_states: np.ndarray,
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
    ratemap_nbins: int | None,
    max_pairs: int = DEFAULT_PATH_SAMPLE_PAIRS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat_positions = positions.reshape(-1, positions.shape[-1])
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    point_count = flat_positions.shape[0]
    if point_count < 2:
        return (
            _empty_path_summary(run_id=run_id, seed=seed, variant=variant, arena_size=arena_size),
            [],
        )
    rng = np.random.default_rng(0 if seed is None else seed)
    sample_count = min(max_pairs, point_count * max(point_count - 1, 1))
    left = rng.integers(0, point_count, size=sample_count)
    right = rng.integers(0, point_count, size=sample_count)
    keep = left != right
    left = left[keep]
    right = right[keep]
    if left.size == 0:
        return (
            _empty_path_summary(run_id=run_id, seed=seed, variant=variant, arena_size=arena_size),
            [],
        )
    spatial_distance = np.linalg.norm(flat_positions[left] - flat_positions[right], axis=1)
    neural_distance = np.linalg.norm(flat_hidden[left] - flat_hidden[right], axis=1)
    trajectory_count, steps = positions.shape[:2]
    trajectory_ids = np.repeat(np.arange(trajectory_count), steps)
    different_path = trajectory_ids[left] != trajectory_ids[right]
    radius = arena_size / ratemap_nbins if ratemap_nbins and ratemap_nbins > 0 else arena_size * 0.05
    same_position = (spatial_distance <= radius) & different_path
    control = (spatial_distance >= 4.0 * radius) & (spatial_distance <= 8.0 * radius)
    same_mean = _finite_mean(neural_distance[same_position])
    control_mean = _finite_mean(neural_distance[control])
    score = None
    if math.isfinite(same_mean) and math.isfinite(control_mean) and control_mean > 0.0:
        score = float(max(0.0, min(1.0, 1.0 - same_mean / control_mean)))
    bin_rows = _distance_bin_rows(
        spatial_distance,
        neural_distance,
        run_id=run_id,
        seed=seed,
        variant=variant,
        arena_size=arena_size,
    )
    decorrelation_length = _decorrelation_length(bin_rows)
    summary = {
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "arena_size": arena_size,
        "same_position_radius_meters": radius,
        "same_position_pair_count": int(same_position.sum()),
        "same_position_mean_neural_distance": _json_float(same_mean),
        "spatial_control_pair_count": int(control.sum()),
        "spatial_control_mean_neural_distance": _json_float(control_mean),
        "path_invariance_score": _json_float(score),
        "decorrelation_length_meters": _json_float(decorrelation_length),
        "decorrelation_fit_method": "threshold_632",
        "sampled_pair_count": int(left.size),
    }
    return summary, bin_rows


def module_fourier_lattice_vectors(
    ratemaps: np.ndarray,
    unit_rows: list[dict[str, Any]],
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
) -> list[dict[str, Any]]:
    del ratemaps
    rows: list[dict[str, Any]] = []
    for module_id in _module_ids(unit_rows):
        members = [row for row in unit_rows if int(row["module_id"]) == module_id]
        scales = np.asarray([_optional_float(row.get("scale_meters")) for row in members])
        orientations = np.asarray([_optional_float(row.get("orientation_degrees")) for row in members])
        period = _finite_median(scales)
        if not math.isfinite(period) or period <= 0.0:
            continue
        frequency = 1.0 / period
        base_orientation = _circular_mean(orientations, period=60.0)
        if not math.isfinite(base_orientation):
            base_orientation = 0.0
        for vector_index, offset in enumerate((0.0, 60.0, 120.0)):
            orientation = float((base_orientation + offset) % 180.0)
            radians = math.radians(orientation)
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "variant": variant,
                    "arena_size": arena_size,
                    "module_id": module_id,
                    "vector_index": vector_index,
                    "frequency_cycles_per_meter": _json_float(frequency),
                    "period_meters": _json_float(period),
                    "orientation_degrees": _json_float(orientation),
                    "vector_x": _json_float(frequency * math.cos(radians)),
                    "vector_y": _json_float(frequency * math.sin(radians)),
                    "unit_count": len(members),
                }
            )
    return rows


def phase_tiling_table(
    ratemaps: np.ndarray,
    unit_rows: list[dict[str, Any]],
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
) -> list[dict[str, Any]]:
    nbins = ratemaps.shape[-1]
    centers = (np.linspace(-arena_size / 2.0, arena_size / 2.0, nbins + 1)[:-1] + arena_size / nbins / 2.0)
    rows: list[dict[str, Any]] = []
    by_unit = {int(row["unit"]): row for row in unit_rows}
    for unit, ratemap in enumerate(ratemaps):
        meta = by_unit.get(unit)
        if meta is None:
            continue
        scale = _optional_float(meta.get("scale_meters"))
        finite = np.isfinite(ratemap)
        peak_x = None
        peak_y = None
        phase_x = None
        phase_y = None
        if np.any(finite):
            filled = np.where(finite, ratemap, -np.inf)
            iy, ix = np.unravel_index(int(np.argmax(filled)), filled.shape)
            peak_x = float(centers[ix])
            peak_y = float(centers[iy])
            if scale is not None and scale > 0.0:
                phase_x = float(np.mod(peak_x + arena_size / 2.0, scale) / scale)
                phase_y = float(np.mod(peak_y + arena_size / 2.0, scale) / scale)
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "variant": variant,
                "arena_size": arena_size,
                "unit": unit,
                "module_id": meta.get("module_id", -1),
                "peak_x_meters": _json_float(peak_x),
                "peak_y_meters": _json_float(peak_y),
                "phase_x": _json_float(phase_x),
                "phase_y": _json_float(phase_y),
                "scale_meters": scale,
            }
        )
    return rows


def state_space_artifacts(
    hidden_states: np.ndarray,
    unit_rows: list[dict[str, Any]],
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
    max_samples: int = DEFAULT_STATE_SPACE_SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if flat_hidden.shape[0] > max_samples:
        indices = np.linspace(0, flat_hidden.shape[0] - 1, max_samples, dtype=np.int64)
        flat_hidden = flat_hidden[indices]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for module_id in _module_ids(unit_rows):
        units = np.asarray(
            [int(row["unit"]) for row in unit_rows if int(row["module_id"]) == module_id],
            dtype=np.int64,
        )
        if units.size < 2:
            continue
        activity = flat_hidden[:, units].astype(np.float64, copy=False)
        finite = np.isfinite(activity).all(axis=1)
        activity = activity[finite]
        if activity.shape[0] < 2:
            continue
        pca_projection, explained = _pca(activity, components=6)
        embedding = _isomap_like_embedding(activity, components=3)
        prefix = f"module_{module_id}"
        arrays[f"{prefix}_unit_indices"] = units
        arrays[f"{prefix}_pca6_projection"] = pca_projection.astype(np.float32)
        arrays[f"{prefix}_spectral3_embedding"] = embedding.astype(np.float32)
        arrays[f"{prefix}_explained_variance_ratio"] = explained.astype(np.float64)
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "variant": variant,
                "arena_size": arena_size,
                "module_id": module_id,
                "unit_count": int(units.size),
                "sample_count": int(activity.shape[0]),
                "pca_components": int(explained.size),
                "top3_explained_variance": _json_float(float(np.sum(explained[:3]))),
                "top6_explained_variance": _json_float(float(np.sum(explained[:6]))),
                "embedding_components": int(embedding.shape[1]),
                "dropped_nonfinite_samples": int((~finite).sum()),
            }
        )
    return rows, arrays


UNIT_MODULE_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "unit",
    "module_id",
    "source_module_id",
    "module_confidence",
    "rejection_reason",
    "response_status",
    "score_60",
    "score_90",
    "scale_meters",
    "orientation_degrees",
]
MODULE_SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "module_id",
    "unit_count",
    "mean_scale_meters",
    "median_scale_meters",
    "scale_ratio",
    "mean_grid_score_60",
    "mean_orientation_degrees",
    "module_confidence",
    "unit_ids",
]
CROSS_ARENA_UNIT_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "unit",
    "reference_arena_size",
    "arena_size",
    "reference_module_id",
    "module_id",
    "reference_grid_score_60",
    "grid_score_60",
    "reference_scale_meters",
    "scale_meters",
    "scale_ratio",
    "orientation_delta_degrees",
    "stable",
]
CROSS_ARENA_MODULE_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "module_id",
    "reference_arena_size",
    "arena_size",
    "reference_unit_count",
    "unit_count",
    "reference_mean_scale_meters",
    "mean_scale_meters",
    "scale_ratio",
    "stable_unit_fraction",
]
PATH_SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "same_position_radius_meters",
    "same_position_pair_count",
    "same_position_mean_neural_distance",
    "spatial_control_pair_count",
    "spatial_control_mean_neural_distance",
    "path_invariance_score",
    "decorrelation_length_meters",
    "decorrelation_fit_method",
    "sampled_pair_count",
]
PATH_BIN_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "bin_low",
    "bin_high",
    "pair_count",
    "mean_spatial_distance",
    "mean_neural_distance",
]
FOURIER_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "module_id",
    "vector_index",
    "frequency_cycles_per_meter",
    "period_meters",
    "orientation_degrees",
    "vector_x",
    "vector_y",
    "unit_count",
]
PHASE_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "unit",
    "module_id",
    "peak_x_meters",
    "peak_y_meters",
    "phase_x",
    "phase_y",
    "scale_meters",
]
STATE_SPACE_FIELDS = [
    "run_id",
    "seed",
    "variant",
    "arena_size",
    "module_id",
    "unit_count",
    "sample_count",
    "pca_components",
    "top3_explained_variance",
    "top6_explained_variance",
    "embedding_components",
    "dropped_nonfinite_samples",
]


def _rejection_reason(
    *,
    response_status: str,
    score_60: float | None,
    scale_meters: float | None,
    min_grid_score_60: float,
) -> str | None:
    if response_status != "active":
        return "inactive_or_invalid_response"
    if scale_meters is None or scale_meters <= 0.0:
        return "missing_or_nonpositive_scale"
    if score_60 is None or score_60 < min_grid_score_60:
        return "low_grid_score_60"
    return None


def _module_confidence(
    *,
    unit_count: int,
    min_module_units: int,
    scores: np.ndarray,
    min_grid_score_60: float,
    scale_ratio: float,
    max_scale_ratio_within_module: float,
) -> float:
    size_conf = min(1.0, unit_count / max(min_module_units, 1))
    score_conf = float(np.mean(scores >= min_grid_score_60)) if scores.size else 0.0
    scale_conf = 1.0 if scale_ratio <= max_scale_ratio_within_module else 0.0
    return round(float((size_conf + score_conf + scale_conf) / 3.0), 6)


def _distance_bin_rows(
    spatial_distance: np.ndarray,
    neural_distance: np.ndarray,
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
) -> list[dict[str, Any]]:
    finite = np.isfinite(spatial_distance) & np.isfinite(neural_distance)
    if not np.any(finite):
        return []
    distances = spatial_distance[finite]
    neural = neural_distance[finite]
    max_distance = float(np.max(distances))
    if max_distance <= 0.0:
        edges = np.asarray([0.0, 1e-9])
    else:
        edges = np.linspace(0.0, max_distance, 9)
    rows: list[dict[str, Any]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (distances >= low) & (distances < high)
        if high == edges[-1]:
            mask = (distances >= low) & (distances <= high)
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "variant": variant,
                "arena_size": arena_size,
                "bin_low": _json_float(float(low)),
                "bin_high": _json_float(float(high)),
                "pair_count": int(mask.sum()),
                "mean_spatial_distance": _json_float(_finite_mean(distances[mask])),
                "mean_neural_distance": _json_float(_finite_mean(neural[mask])),
            }
        )
    return rows


def _decorrelation_length(bin_rows: list[dict[str, Any]]) -> float:
    values = [
        (
            _optional_float(row.get("mean_spatial_distance")),
            _optional_float(row.get("mean_neural_distance")),
        )
        for row in bin_rows
    ]
    values = [(x, y) for x, y in values if x is not None and y is not None]
    if len(values) < 2:
        return float("nan")
    y_values = np.asarray([item[1] for item in values], dtype=np.float64)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_max <= y_min:
        return float("nan")
    target = y_min + (1.0 - math.exp(-1.0)) * (y_max - y_min)
    for x_value, y_value in values:
        if y_value >= target:
            return float(x_value)
    return float(values[-1][0])


def _pca(activity: np.ndarray, *, components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = activity - activity.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular_values**2))
    explained = (singular_values**2) / total if total > 0.0 else np.zeros_like(singular_values)
    component_count = min(components, vh.shape[0])
    projection = np.zeros((centered.shape[0], components), dtype=np.float64)
    if component_count:
        projection[:, :component_count] = centered @ vh[:component_count].T
    return projection, explained[:components]


def _isomap_like_embedding(activity: np.ndarray, *, components: int) -> np.ndarray:
    sample_count = activity.shape[0]
    if sample_count < 2:
        return np.zeros((sample_count, components), dtype=np.float64)
    diff = activity[:, None, :] - activity[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    neighbors = min(8, sample_count - 1)
    graph = np.full_like(distances, np.inf)
    np.fill_diagonal(graph, 0.0)
    nearest = np.argpartition(distances, kth=neighbors, axis=1)[:, : neighbors + 1]
    row_indices = np.arange(sample_count)[:, None]
    graph[row_indices, nearest] = distances[row_indices, nearest]
    graph = np.minimum(graph, graph.T)
    shortest = scipy.sparse.csgraph.shortest_path(
        scipy.sparse.csr_matrix(graph),
        directed=False,
        unweighted=False,
    )
    finite = np.isfinite(shortest)
    if not np.all(finite):
        fallback = float(np.max(shortest[finite])) if np.any(finite) else 0.0
        shortest = np.where(finite, shortest, fallback)
    return _classical_mds(shortest, components=components)


def _classical_mds(distances: np.ndarray, *, components: int) -> np.ndarray:
    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, components), dtype=np.float64)
    squared = distances**2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    selected = order[:components]
    clipped = np.maximum(eigenvalues[selected], 0.0)
    embedding = eigenvectors[:, selected] * np.sqrt(clipped)[None, :]
    if embedding.shape[1] < components:
        padded = np.zeros((n, components), dtype=np.float64)
        padded[:, : embedding.shape[1]] = embedding
        return padded
    return embedding


def _arena_infos(eval_path: Path, summary: dict[str, Any]) -> list[tuple[float, Path]]:
    infos = []
    for arena in summary.get("arena_summaries", []):
        if not isinstance(arena, dict):
            continue
        arena_size = _optional_float(arena.get("arena_size"))
        if arena_size is None:
            continue
        infos.append((arena_size, eval_path / f"arena_{_format_arena_size(arena_size)}"))
    if infos:
        return infos
    for arena_dir in sorted(eval_path.glob("arena_*")):
        arena_size = _parse_arena_dir_size(arena_dir.name)
        if arena_size is not None:
            infos.append((arena_size, arena_dir))
    return infos


def _parse_arena_dir_size(name: str) -> float | None:
    if not name.startswith("arena_"):
        return None
    try:
        return float(name.removeprefix("arena_").replace("p", "."))
    except ValueError:
        return None


def _resolve_seed(summary: dict[str, Any], seed: int | None) -> int | None:
    if seed is not None:
        return int(seed)
    value = summary.get("evaluation_seed")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _format_arena_size(value: float) -> str:
    return str(value).replace(".", "p")


def _ratemap_nbins(arena_dir: Path) -> int | None:
    path = arena_dir / "ratemaps.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        ratemaps = data["ratemaps"]
        return int(ratemaps.shape[-1])


def _module_ids(unit_rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["module_id"]) for row in unit_rows if int(row["module_id"]) >= 0})


def _stable_unit_fraction(
    unit_rows: list[dict[str, Any]],
    *,
    module_id: int,
    arena_size: float,
) -> float | None:
    rows = [
        row
        for row in unit_rows
        if float(row["arena_size"]) == arena_size
        and int(row.get("reference_module_id", -1)) == module_id
    ]
    if not rows:
        return None
    return float(sum(1 for row in rows if row.get("stable")) / len(rows))


def _empty_path_summary(
    *,
    run_id: str,
    seed: int | None,
    variant: str,
    arena_size: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "arena_size": arena_size,
        "same_position_radius_meters": None,
        "same_position_pair_count": 0,
        "same_position_mean_neural_distance": None,
        "spatial_control_pair_count": 0,
        "spatial_control_mean_neural_distance": None,
        "path_invariance_score": None,
        "decorrelation_length_meters": None,
        "decorrelation_fit_method": "threshold_632",
        "sampled_pair_count": 0,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(row) for row in data]


def _write_rows(path_prefix: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(rows, path_prefix.with_suffix(".json"))
    with path_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.median(values[finite]))


def _circular_mean(values: np.ndarray, *, period: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    angles = values[finite] / period * 2.0 * np.pi
    mean = np.angle(np.mean(np.exp(1j * angles)))
    return float(np.mod(mean / (2.0 * np.pi) * period, period))


def _symmetric_ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left <= 0.0 or right <= 0.0:
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return float(max(left, right) / min(left, right))


def _orientation_delta(left: float | None, right: float | None, *, period: float) -> float | None:
    if left is None or right is None:
        return None
    delta = abs((right - left + period / 2.0) % period - period / 2.0)
    return float(delta)
