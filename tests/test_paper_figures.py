import json
from pathlib import Path

import numpy as np
import pytest

import scripts.build_paper_figures as build_paper_figures_script
from sic4gridcells.analysis_ext import analyze_evaluation_output
from sic4gridcells.figure_data import FigureDataError, TABLE_FIELDNAMES
from sic4gridcells.paper_figures import build_paper_figures


def test_build_paper_figures_writes_stable_outputs_and_manifest(tmp_path: Path) -> None:
    suite_dir = _write_suite_with_analysis(tmp_path)
    output_dir = tmp_path / "figures"

    result = build_paper_figures(suite_dir, output_dir)

    assert result.manifest_path.exists()
    for figure_name in [
        "fig_grid_modules",
        "fig_arena_generalization",
        "fig_path_invariance",
        "fig_fourier_phase_state",
        "fig_ablations",
    ]:
        assert (output_dir / f"{figure_name}.png").exists()
        assert (output_dir / f"{figure_name}.pdf").exists()
    assert (output_dir / "summary_tables" / "module_summary.csv").exists()
    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8"),
        parse_constant=_fail_parse_constant,
    )
    assert len(manifest["figures"]) == 5
    assert manifest["figures"][0]["dependencies"]


def test_build_paper_figures_fails_on_non_diagnostic_validation_blocker(tmp_path: Path) -> None:
    suite_dir = _write_suite_with_analysis(tmp_path, validation_passed=False)

    with pytest.raises(FigureDataError, match="validation blockers"):
        build_paper_figures(suite_dir, tmp_path / "figures")


def test_build_paper_figures_ignores_skipped_runs_without_analysis(tmp_path: Path) -> None:
    suite_dir = _write_suite_with_analysis(tmp_path)
    manifest_path = suite_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"].append(
        {
            "run_id": "disabled",
            "variant": "baseline",
            "seed": 1,
            "status": "skipped",
            "diagnostic_only": False,
            "reason": "disabled for test",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = build_paper_figures(suite_dir, tmp_path / "figures")

    assert result.manifest_path.exists()


def test_build_paper_figures_fails_when_required_analysis_table_is_missing(
    tmp_path: Path,
) -> None:
    suite_dir = _write_suite_with_analysis(tmp_path)
    missing = suite_dir / "runs" / "baseline" / "analysis" / "summary_tables" / "phase_tiling.csv"
    missing.unlink()
    (missing.with_suffix(".json")).unlink()

    with pytest.raises(FigureDataError, match="phase_tiling"):
        build_paper_figures(suite_dir, tmp_path / "figures")


def test_build_paper_figures_preserves_empty_table_headers(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite-empty"
    tables_dir = suite_dir / "analysis" / "summary_tables"
    tables_dir.mkdir(parents=True)
    for name, fields in TABLE_FIELDNAMES.items():
        (tables_dir / f"{name}.csv").write_text(",".join(fields) + "\n", encoding="utf-8")

    output_dir = tmp_path / "figures-empty"
    build_paper_figures(suite_dir, output_dir)

    header = (output_dir / "summary_tables" / "unit_modules.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert header.startswith("run_id,seed,variant,arena_size,unit,module_id")


def test_build_paper_figures_cli_prints_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_dir = _write_suite_with_analysis(tmp_path)
    output_dir = tmp_path / "cli-figures"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_paper_figures.py",
            "--suite-dir",
            str(suite_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    build_paper_figures_script.main()

    output = capsys.readouterr().out
    assert f"figures={output_dir}" in output
    assert f"manifest={output_dir / 'figure_manifest.json'}" in output


def _write_suite_with_analysis(
    tmp_path: Path,
    *,
    validation_passed: bool = True,
    diagnostic_only: bool = False,
) -> Path:
    suite_dir = tmp_path / "suite"
    eval_dir = suite_dir / "runs" / "baseline" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "evaluation_seed": 7,
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
    analysis = analyze_evaluation_output(
        eval_dir,
        suite_dir / "runs" / "baseline" / "analysis",
        run_id="suite-baseline",
        seed=7,
        variant="baseline",
        diagnostic_only=diagnostic_only,
    )
    validation_report = suite_dir / "runs" / "baseline" / "validation_report.json"
    validation_report.write_text(
        json.dumps({"passed": validation_passed}, indent=2),
        encoding="utf-8",
    )
    (suite_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "suite-baseline",
                        "variant": "baseline",
                        "seed": 7,
                        "status": "finished",
                        "diagnostic_only": diagnostic_only,
                        "eval_output_dir": str(eval_dir),
                        "analysis_output_dir": str(analysis.output_dir),
                        "validation_report_path": str(validation_report),
                        "validation_passed": validation_passed,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return suite_dir


def _grid_rows(*, scale_multiplier: float) -> list[dict[str, object]]:
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
