import json
from pathlib import Path

import pytest

import scripts.validate_eval as validate_eval_script
from sic4gridcells.validation import (
    REQUIRED_ARENA_ARTIFACTS,
    ValidationThresholds,
    validate_evaluation_output,
    write_validation_report,
)


def test_validate_evaluation_output_passes_complete_fixture(tmp_path: Path) -> None:
    output_dir = _write_eval_fixture(tmp_path)

    report = validate_evaluation_output(output_dir)

    assert report.passed is True
    assert report.blocker_count == 0
    assert report.arenas[0].qualifying_modules == 1


def test_validate_evaluation_output_reports_quality_blockers(tmp_path: Path) -> None:
    output_dir = _write_eval_fixture(
        tmp_path,
        coverage_fraction=0.25,
        active_units=0,
        invalid_response_units=1,
        module_unit_count=1,
    )
    (output_dir / "arena_2p0" / "sacs.pdf").unlink()

    report = validate_evaluation_output(output_dir)

    codes = {issue.code for issue in report.issues}
    assert report.passed is False
    assert {
        "low_coverage",
        "low_active_units",
        "invalid_response_units",
        "insufficient_modules",
        "missing_artifact",
    } <= codes


def test_validate_evaluation_output_checks_required_arena_sizes(tmp_path: Path) -> None:
    output_dir = _write_eval_fixture(tmp_path)

    report = validate_evaluation_output(
        output_dir,
        ValidationThresholds(required_arena_sizes=(2.0, 3.0)),
    )

    assert report.passed is False
    assert any(issue.code == "missing_required_arena" for issue in report.issues)


def test_validate_evaluation_output_reports_invalid_summary_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text("{", encoding="utf-8")

    report = validate_evaluation_output(output_dir)

    assert report.passed is False
    assert [issue.code for issue in report.issues] == ["invalid_summary_json"]


def test_validate_evaluation_output_reports_unreadable_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "summary.json").write_bytes(b"\xff")

    report = validate_evaluation_output(output_dir)

    assert report.passed is False
    assert [issue.code for issue in report.issues] == ["unreadable_summary"]


def test_validate_evaluation_output_reports_invalid_arena_entries(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps({"arena_summaries": ["bad", {"coverage_fraction": 1.0}]}),
        encoding="utf-8",
    )

    report = validate_evaluation_output(output_dir)

    assert report.passed is False
    assert {issue.code for issue in report.issues} == {
        "invalid_arena_summary",
        "invalid_arena_size",
    }


def test_validate_evaluation_output_keeps_bad_numbers_json_serializable(tmp_path: Path) -> None:
    output_dir = _write_eval_fixture(tmp_path)
    summary = {
        "arena_summaries": [
            {
                "arena_size": 2.0,
                "coverage_fraction": float("inf"),
                "active_units": float("inf"),
                "invalid_response_units": float("inf"),
            }
        ]
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = validate_evaluation_output(
        output_dir,
        ValidationThresholds(required_arena_sizes=(float("inf"),)),
    )
    report_path = tmp_path / "validation.json"
    write_validation_report(report, report_path)

    assert report.passed is False
    assert "invalid_required_arena_size" in {issue.code for issue in report.issues}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False


def test_validate_eval_cli_writes_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = _write_eval_fixture(tmp_path)
    report_path = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_eval.py",
            "--output-dir",
            str(output_dir),
            "--json-output",
            str(report_path),
        ],
    )

    exit_code = validate_eval_script.main()

    assert exit_code == 0
    assert "validation passed" in capsys.readouterr().out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_validate_eval_cli_returns_failure_for_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = _write_eval_fixture(tmp_path, coverage_fraction=0.1)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_eval.py",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert validate_eval_script.main() == 1


def _write_eval_fixture(
    tmp_path: Path,
    *,
    coverage_fraction: float = 0.9,
    active_units: int = 8,
    invalid_response_units: int = 0,
    module_unit_count: int = 3,
) -> Path:
    output_dir = tmp_path / "eval"
    arena_dir = output_dir / "arena_2p0"
    arena_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "arena_summaries": [
                    {
                        "arena_size": 2.0,
                        "coverage_fraction": coverage_fraction,
                        "active_units": active_units,
                        "invalid_response_units": invalid_response_units,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name in REQUIRED_ARENA_ARTIFACTS:
        (arena_dir / name).write_text("artifact\n", encoding="utf-8")
    (arena_dir / "module_summary.csv").write_text(
        "module_id,unit_count,mean_scale_meters,median_scale_meters,mean_orientation_degrees,mean_grid_score_60\n"
        f"0,{module_unit_count},0.2,0.2,10.0,0.1\n",
        encoding="utf-8",
    )
    return output_dir
