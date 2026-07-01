from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sic4gridcells.logging_utils import to_jsonable
from sic4gridcells.runtime import atomic_json_dump
from sic4gridcells.analysis_ext import (
    CROSS_ARENA_MODULE_FIELDS,
    CROSS_ARENA_UNIT_FIELDS,
    FOURIER_FIELDS,
    MODULE_SUMMARY_FIELDS,
    PATH_BIN_FIELDS,
    PATH_SUMMARY_FIELDS,
    PHASE_FIELDS,
    STATE_SPACE_FIELDS,
    UNIT_MODULE_FIELDS,
)


TABLE_NAMES = (
    "unit_modules",
    "module_summary",
    "cross_arena_unit_metrics",
    "cross_arena_module_stability",
    "path_invariance_summary",
    "path_invariance_bins",
    "fourier_lattice_vectors",
    "phase_tiling",
    "state_space_summary",
)

TABLE_FIELDNAMES = {
    "unit_modules": UNIT_MODULE_FIELDS,
    "module_summary": MODULE_SUMMARY_FIELDS,
    "cross_arena_unit_metrics": CROSS_ARENA_UNIT_FIELDS,
    "cross_arena_module_stability": CROSS_ARENA_MODULE_FIELDS,
    "path_invariance_summary": PATH_SUMMARY_FIELDS,
    "path_invariance_bins": PATH_BIN_FIELDS,
    "fourier_lattice_vectors": FOURIER_FIELDS,
    "phase_tiling": PHASE_FIELDS,
    "state_space_summary": STATE_SPACE_FIELDS,
}


@dataclass(frozen=True)
class FigureRun:
    run_id: str
    variant: str
    seed: int | None
    diagnostic_only: bool
    status: str | None
    checkpoint_path: Path | None
    eval_output_dir: Path | None
    analysis_output_dir: Path | None
    validation_report_path: Path | None
    validation_passed: bool | None


@dataclass
class FigureDataBundle:
    suite_dir: Path
    runs: list[FigureRun]
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    diagnostic_runs: list[str] = field(default_factory=list)

    def table(self, name: str) -> list[dict[str, Any]]:
        return self.tables.get(name, [])


class FigureDataError(RuntimeError):
    """Raised when required figure inputs are missing or not paper-claim-ready."""


def load_figure_data(suite_dir: str | Path) -> FigureDataBundle:
    suite_path = Path(suite_dir)
    runs = discover_figure_runs(suite_path)
    tables = {name: [] for name in TABLE_NAMES}
    dependencies = {name: [] for name in TABLE_NAMES}
    blockers: list[str] = []
    diagnostic_runs: list[str] = []

    for run in runs:
        if run.diagnostic_only:
            diagnostic_runs.append(run.run_id)
        if run.validation_passed is False and not run.diagnostic_only:
            blockers.append(
                f"run {run.run_id} has validation blockers and is not diagnostic-only"
            )
        analysis_dir = _resolve_analysis_dir(run)
        if analysis_dir is None or not (analysis_dir / "summary_tables").exists():
            message = f"run {run.run_id} is missing analysis summary_tables"
            if run.status == "skipped":
                continue
            if run.diagnostic_only:
                continue
            blockers.append(message)
            continue
        for name in TABLE_NAMES:
            csv_path = analysis_dir / "summary_tables" / f"{name}.csv"
            json_path = analysis_dir / "summary_tables" / f"{name}.json"
            if csv_path.exists():
                rows = _load_csv_rows(csv_path)
                dependencies[name].append(str(csv_path))
            elif json_path.exists():
                rows = _load_json_rows(json_path)
                dependencies[name].append(str(json_path))
            else:
                if not run.diagnostic_only:
                    blockers.append(
                        f"run {run.run_id} is missing required analysis table {name}"
                    )
                rows = []
            tables[name].extend(rows)

    if blockers:
        raise FigureDataError("; ".join(blockers))
    return FigureDataBundle(
        suite_dir=suite_path,
        runs=runs,
        tables=tables,
        dependencies=dependencies,
        diagnostic_runs=diagnostic_runs,
    )


def discover_figure_runs(suite_dir: Path) -> list[FigureRun]:
    manifest_path = _first_existing(
        suite_dir / "manifest.json",
        suite_dir / "suite_manifest.json",
        suite_dir / "paper_suite_manifest.json",
    )
    if manifest_path is not None:
        manifest = _load_json(manifest_path)
        rows = _run_rows_from_manifest(manifest)
        return [_figure_run_from_row(row, suite_dir=suite_dir) for row in rows]

    summary_path = suite_dir / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        rows = summary if isinstance(summary, list) else _run_rows_from_manifest(summary)
        return [_figure_run_from_row(row, suite_dir=suite_dir) for row in rows]

    if (suite_dir / "analysis" / "summary_tables").exists():
        return [
            FigureRun(
                run_id=suite_dir.name,
                variant="baseline",
                seed=None,
                diagnostic_only=False,
                status="finished",
                checkpoint_path=None,
                eval_output_dir=None,
                analysis_output_dir=suite_dir / "analysis",
                validation_report_path=None,
                validation_passed=None,
            )
        ]
    if (suite_dir / "summary_tables").exists():
        return [
            FigureRun(
                run_id=suite_dir.name,
                variant="baseline",
                seed=None,
                diagnostic_only=False,
                status="finished",
                checkpoint_path=None,
                eval_output_dir=None,
                analysis_output_dir=suite_dir,
                validation_report_path=None,
                validation_passed=None,
            )
        ]
    raise FigureDataError(
        f"Could not discover suite manifest or analysis summary_tables under {suite_dir}"
    )


def write_summary_tables(
    bundle: FigureDataBundle,
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    out_dir = Path(output_dir)
    tables_dir = out_dir / "summary_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    for name in TABLE_NAMES:
        rows = bundle.table(name)
        csv_path = tables_dir / f"{name}.csv"
        json_path = tables_dir / f"{name}.json"
        _write_csv_rows(csv_path, rows, fieldnames=TABLE_FIELDNAMES.get(name))
        atomic_json_dump(rows, json_path)
        outputs[name] = {"csv": str(csv_path), "json": str(json_path)}
    return outputs


def _run_rows_from_manifest(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return [dict(row) for row in manifest if isinstance(row, dict)]
    if not isinstance(manifest, dict):
        raise FigureDataError("Suite manifest must be a JSON object or list")
    for key in ("runs", "run_records", "results"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            return [dict(row) for row in value.values() if isinstance(row, dict)]
    if "analysis_output_dir" in manifest or "analysis_dir" in manifest:
        return [manifest]
    raise FigureDataError("Suite manifest does not contain runs, run_records, or results")


def _figure_run_from_row(row: dict[str, Any], *, suite_dir: Path) -> FigureRun:
    name = _string_or_none(row.get("name"))
    run_id = _string_or_none(row.get("run_id")) or name or "run"
    variant = _string_or_none(row.get("variant")) or name or run_id
    validation_report_path = _path_or_none(
        row.get("validation_report_path") or row.get("validation_report"),
        base=suite_dir,
    )
    validation_passed = _optional_bool(row.get("validation_passed"))
    if validation_passed is None and validation_report_path is not None and validation_report_path.exists():
        validation_passed = _validation_report_passed(validation_report_path)
    diagnostic_only = bool(row.get("diagnostic_only", False))
    return FigureRun(
        run_id=run_id,
        variant=variant,
        seed=_optional_int(row.get("seed")),
        diagnostic_only=diagnostic_only,
        status=_string_or_none(row.get("status")),
        checkpoint_path=_path_or_none(
            row.get("checkpoint_path") or row.get("checkpoint"),
            base=suite_dir,
        ),
        eval_output_dir=_path_or_none(
            row.get("eval_output_dir")
            or row.get("evaluation_output_dir")
            or row.get("evaluation_dir"),
            base=suite_dir,
        ),
        analysis_output_dir=_path_or_none(
            row.get("analysis_output_dir") or row.get("analysis_dir"),
            base=suite_dir,
        ),
        validation_report_path=validation_report_path,
        validation_passed=validation_passed,
    )


def _resolve_analysis_dir(run: FigureRun) -> Path | None:
    if run.analysis_output_dir is not None:
        return run.analysis_output_dir
    if run.eval_output_dir is not None:
        candidate = run.eval_output_dir / "analysis"
        if candidate.exists():
            return candidate
    return None


def _validation_report_passed(path: Path) -> bool | None:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if isinstance(payload, dict) and isinstance(payload.get("passed"), bool):
        return bool(payload["passed"])
    return None


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [_decode_csv_row(row) for row in csv.DictReader(handle)]


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise FigureDataError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = fieldnames or _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_jsonable(row.get(field)) for field in output_fieldnames})


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names or ["empty"]


def _decode_csv_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: _decode_scalar(value) for key, value in row.items()}


def _decode_scalar(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _path_or_none(value: Any, *, base: Path) -> Path | None:
    text = _string_or_none(value)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        if path.exists():
            return path
        path = base / path
    return path


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {None, ""}:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None
