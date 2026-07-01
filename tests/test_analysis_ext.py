import json
from pathlib import Path

import numpy as np
import pytest

from sic4gridcells.analysis_ext import (
    analyze_evaluation_output,
    cross_arena_aggregation,
    path_invariance_probe,
    robust_module_detection,
    state_space_artifacts,
)


def test_robust_module_detection_filters_and_groups_units() -> None:
    result = robust_module_detection(
        _grid_rows(),
        run_id="run-a",
        seed=3,
        variant="baseline",
        arena_size=2.0,
    )

    module_rows = result["module_rows"]
    unit_rows = result["unit_rows"]

    assert len(module_rows) == 1
    assert module_rows[0]["unit_count"] == 3
    assert module_rows[0]["module_confidence"] == 1.0
    assert [row["module_id"] for row in unit_rows[:3]] == [0, 0, 0]
    assert unit_rows[3]["rejection_reason"] == "module_too_small"
    assert unit_rows[4]["rejection_reason"] == "inactive_or_invalid_response"


def test_cross_arena_aggregation_matches_units_and_modules() -> None:
    first = robust_module_detection(
        _grid_rows(scale_multiplier=1.0),
        run_id="run-a",
        seed=3,
        variant="baseline",
        arena_size=2.0,
    )["unit_rows"]
    second = robust_module_detection(
        _grid_rows(scale_multiplier=1.05),
        run_id="run-a",
        seed=3,
        variant="baseline",
        arena_size=3.0,
    )["unit_rows"]

    unit_rows, module_rows = cross_arena_aggregation(first + second)

    assert any(row["stable"] is True for row in unit_rows)
    assert module_rows[0]["module_id"] == 0
    assert module_rows[0]["stable_unit_fraction"] == 1.0


def test_path_invariance_probe_reports_same_position_pairs() -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0], [0.05, 0.0], [0.6, 0.0], [0.65, 0.0]],
            [[0.0, 0.0], [0.05, 0.0], [0.6, 0.0], [0.65, 0.0]],
        ],
        dtype=np.float64,
    )
    hidden_states = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.1], [1.0, 0.0], [1.0, 0.1]],
            [[0.0, 0.0], [0.0, 0.1], [1.0, 0.0], [1.0, 0.1]],
        ],
        dtype=np.float64,
    )

    summary, rows = path_invariance_probe(
        positions,
        hidden_states,
        run_id="run-a",
        seed=5,
        variant="baseline",
        arena_size=2.0,
        ratemap_nbins=20,
        max_pairs=200,
    )

    assert summary["same_position_pair_count"] > 0
    assert summary["path_invariance_score"] is not None
    assert rows


def test_state_space_artifacts_have_finite_shapes() -> None:
    hidden_states = np.arange(2 * 5 * 4, dtype=np.float64).reshape(2, 5, 4)
    unit_rows = [
        {"unit": 0, "module_id": 0},
        {"unit": 1, "module_id": 0},
        {"unit": 2, "module_id": 0},
        {"unit": 3, "module_id": -1},
    ]

    rows, arrays = state_space_artifacts(
        hidden_states,
        unit_rows,
        run_id="run-a",
        seed=3,
        variant="baseline",
        arena_size=2.0,
    )

    assert rows[0]["module_id"] == 0
    assert rows[0]["top6_explained_variance"] <= 1.0
    assert arrays["module_0_pca6_projection"].shape == (10, 6)
    assert arrays["module_0_spectral3_embedding"].shape == (10, 3)
    assert np.isfinite(arrays["module_0_pca6_projection"]).all()


def test_analyze_evaluation_output_writes_strict_json_and_tables(tmp_path: Path) -> None:
    eval_dir = _write_eval_fixture(tmp_path)

    result = analyze_evaluation_output(eval_dir, run_id="run-a", variant="baseline")

    assert result.manifest_path.exists()
    assert (result.output_dir / "summary_tables" / "module_summary.csv").exists()
    assert (result.output_dir / "summary_tables" / "cross_arena_unit_metrics.csv").exists()
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text, parse_constant=_fail_parse_constant)
    assert manifest["run_id"] == "run-a"
    assert manifest["arena_count"] == 2
    with np.load(result.output_dir / "arena_2p0" / "state_space_ext.npz") as data:
        assert "module_0_pca6_projection" in data.files


def test_analyze_evaluation_output_fails_loudly_for_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    eval_dir = _write_eval_fixture(tmp_path)
    (eval_dir / "arena_2p0" / "ratemaps.npz").unlink()

    with pytest.raises(FileNotFoundError, match="ratemaps.npz"):
        analyze_evaluation_output(eval_dir, run_id="run-a", variant="baseline")


def test_analyze_evaluation_output_allows_missing_artifacts_for_diagnostic_runs(
    tmp_path: Path,
) -> None:
    eval_dir = _write_eval_fixture(tmp_path)
    (eval_dir / "arena_2p0" / "rollout_arrays.npz").unlink()

    result = analyze_evaluation_output(
        eval_dir,
        run_id="run-a",
        variant="baseline",
        diagnostic_only=True,
    )

    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["diagnostic_only"] is True


def _write_eval_fixture(tmp_path: Path) -> Path:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "evaluation_seed": 3,
                "arena_summaries": [
                    {"arena_size": 2.0},
                    {"arena_size": 3.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    for arena_size, scale_multiplier in ((2.0, 1.0), (3.0, 1.05)):
        arena_dir = eval_dir / f"arena_{str(arena_size).replace('.', 'p')}"
        arena_dir.mkdir()
        (arena_dir / "grid_stats.json").write_text(
            json.dumps(_grid_rows(scale_multiplier=scale_multiplier)),
            encoding="utf-8",
        )
        ratemaps = np.zeros((5, 6, 6), dtype=np.float64)
        for unit in range(5):
            ratemaps[unit, unit % 6, (unit * 2) % 6] = float(unit + 1)
        ratemaps[:, 0, 0] = np.nan
        np.savez_compressed(arena_dir / "ratemaps.npz", ratemaps=ratemaps)
        positions = np.asarray(
            [
                [[0.0, 0.0], [0.05, 0.0], [0.6, 0.0], [0.65, 0.0]],
                [[0.0, 0.0], [0.05, 0.0], [0.6, 0.0], [0.65, 0.0]],
            ],
            dtype=np.float32,
        )
        hidden = np.zeros((2, 4, 5), dtype=np.float32)
        hidden[:, :, :3] = np.asarray(
            [[[0.0, 0.0, 0.1], [0.0, 0.1, 0.1], [1.0, 0.0, 0.2], [1.0, 0.1, 0.2]]],
            dtype=np.float32,
        )
        np.savez_compressed(
            arena_dir / "rollout_arrays.npz",
            positions=positions,
            velocities=np.zeros_like(positions),
            hidden_states=hidden,
        )
    return eval_dir


def _grid_rows(*, scale_multiplier: float = 1.0) -> list[dict[str, object]]:
    rows = []
    for unit, scale in enumerate([0.2, 0.21, 0.22, 0.5, None]):
        active = scale is not None
        rows.append(
            {
                "unit": unit,
                "response_status": "active" if active else "zero",
                "score_60": 0.2 if active else 0.0,
                "score_90": 0.0,
                "scale_meters": None if scale is None else scale * scale_multiplier,
                "orientation_degrees": 10.0 + unit,
                "module_id": 0 if unit < 3 else -1,
            }
        )
    return rows


def _fail_parse_constant(value: str) -> None:
    raise AssertionError(f"non-strict JSON constant: {value}")
