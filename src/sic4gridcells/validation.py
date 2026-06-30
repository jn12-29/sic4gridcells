from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any


REQUIRED_ARENA_ARTIFACTS = (
    "rollout_arrays.npz",
    "ratemaps.npz",
    "occupancy.npz",
    "sacs.npz",
    "grid_metrics.npz",
    "grid_stats.csv",
    "grid_stats.json",
    "module_summary.csv",
    "module_summary.json",
    "trajectory_stats.json",
    "pairwise_distance_stats.csv",
    "pairwise_distance_stats.json",
    "pairwise_distance.png",
    "fourier_stats.csv",
    "fourier_stats.json",
    "phase_summary.csv",
    "phase_summary.json",
    "state_space_summary.csv",
    "state_space_summary.json",
    "state_space_modules.npz",
    "grid_score_60_histogram.png",
    "scale_meters_histogram.png",
    "summary.png",
    "ratemaps.pdf",
    "sacs.pdf",
)


@dataclass(frozen=True)
class ValidationThresholds:
    min_coverage_fraction: float = 0.8
    min_active_units: int = 1
    max_invalid_response_units: int = 0
    min_module_count: int = 1
    min_module_units: int = 3
    min_module_mean_grid_score_60: float = 0.0
    required_arena_sizes: tuple[float, ...] = ()
    require_artifacts: bool = True


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    arena_size: float | None = None
    path: str | None = None


@dataclass(frozen=True)
class ArenaValidation:
    arena_size: float
    arena_dir: str
    coverage_fraction: float | None
    active_units: int | None
    invalid_response_units: int | None
    qualifying_modules: int
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationReport:
    output_dir: str
    passed: bool
    blocker_count: int
    warning_count: int
    arenas: list[ArenaValidation]
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_evaluation_output(
    output_dir: str | Path,
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    threshold_cfg = thresholds or ValidationThresholds()
    out_dir = Path(output_dir)
    issues: list[ValidationIssue] = []
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="missing_summary",
                message="Evaluation output is missing summary.json.",
                path=str(summary_path),
            )
        )
        return _report(out_dir, [], issues)
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="unreadable_summary",
                message=f"summary.json could not be read: {type(exc).__name__}.",
                path=str(summary_path),
            )
        )
        return _report(out_dir, [], issues)
    try:
        summary = json.loads(raw_summary)
    except JSONDecodeError as exc:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="invalid_summary_json",
                message=f"summary.json is not valid JSON: {exc.msg}.",
                path=str(summary_path),
            )
        )
        return _report(out_dir, [], issues)
    if not isinstance(summary, dict):
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="invalid_summary",
                message="summary.json must contain a JSON object.",
                path=str(summary_path),
            )
        )
        return _report(out_dir, [], issues)
    arena_summaries = summary.get("arena_summaries")
    if not isinstance(arena_summaries, list) or not arena_summaries:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="missing_arena_summaries",
                message="summary.json does not contain non-empty arena_summaries.",
                path=str(summary_path),
            )
        )
        return _report(out_dir, [], issues)
    arenas = []
    observed_sizes: set[float] = set()
    for index, arena_summary in enumerate(arena_summaries):
        if not isinstance(arena_summary, dict):
            issues.append(
                ValidationIssue(
                    severity="blocker",
                    code="invalid_arena_summary",
                    message=f"arena_summaries[{index}] must contain a JSON object.",
                    path=str(summary_path),
                )
            )
            continue
        arena_size = _optional_float(arena_summary.get("arena_size"))
        if arena_size is None:
            issues.append(
                ValidationIssue(
                    severity="blocker",
                    code="invalid_arena_size",
                    message=f"arena_summaries[{index}].arena_size must be finite.",
                    path=str(summary_path),
                )
            )
            continue
        arena = _validate_arena(out_dir, arena_summary, arena_size, threshold_cfg)
        arenas.append(arena)
        issues.extend(arena.issues)
        observed_sizes.add(arena.arena_size)
    for index, required_value in enumerate(threshold_cfg.required_arena_sizes):
        required_size = _optional_float(required_value)
        if required_size is None:
            issues.append(
                ValidationIssue(
                    severity="blocker",
                    code="invalid_required_arena_size",
                    message=f"required_arena_sizes[{index}] must be finite.",
                    path=str(summary_path),
                )
            )
            continue
        if not any(abs(required_size - observed_size) < 1e-9 for observed_size in observed_sizes):
            issues.append(
                ValidationIssue(
                    severity="blocker",
                    code="missing_required_arena",
                    message=f"Required arena size {required_size} is missing from summary.json.",
                    arena_size=required_size,
                    path=str(summary_path),
                )
            )
    return _report(out_dir, arenas, issues)


def write_validation_report(report: ValidationReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_arena(
    output_dir: Path,
    arena_summary: dict[str, Any],
    arena_size: float,
    thresholds: ValidationThresholds,
) -> ArenaValidation:
    arena_dir = output_dir / f"arena_{_format_arena_size(arena_size)}"
    issues: list[ValidationIssue] = []
    if not arena_dir.exists():
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="missing_arena_dir",
                message="Arena directory is missing.",
                arena_size=arena_size,
                path=str(arena_dir),
            )
        )
        return ArenaValidation(
            arena_size=arena_size,
            arena_dir=str(arena_dir),
            coverage_fraction=_optional_float(arena_summary.get("coverage_fraction")),
            active_units=_optional_int(arena_summary.get("active_units")),
            invalid_response_units=_optional_int(arena_summary.get("invalid_response_units")),
            qualifying_modules=0,
            issues=issues,
        )
    if thresholds.require_artifacts:
        issues.extend(_artifact_issues(arena_dir, arena_size))
    coverage = _optional_float(arena_summary.get("coverage_fraction"))
    if coverage is None or coverage < thresholds.min_coverage_fraction:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="low_coverage",
                message=(
                    f"coverage_fraction={coverage} is below "
                    f"{thresholds.min_coverage_fraction}."
                ),
                arena_size=arena_size,
                path=str(output_dir / "summary.json"),
            )
        )
    active_units = _optional_int(arena_summary.get("active_units"))
    if active_units is None or active_units < thresholds.min_active_units:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="low_active_units",
                message=f"active_units={active_units} is below {thresholds.min_active_units}.",
                arena_size=arena_size,
                path=str(output_dir / "summary.json"),
            )
        )
    invalid_units = _optional_int(arena_summary.get("invalid_response_units"))
    if invalid_units is None or invalid_units > thresholds.max_invalid_response_units:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="invalid_response_units",
                message=(
                    f"invalid_response_units={invalid_units} exceeds "
                    f"{thresholds.max_invalid_response_units}."
                ),
                arena_size=arena_size,
                path=str(output_dir / "summary.json"),
            )
        )
    qualifying_modules = _count_qualifying_modules(arena_dir, thresholds)
    if qualifying_modules < thresholds.min_module_count:
        issues.append(
            ValidationIssue(
                severity="blocker",
                code="insufficient_modules",
                message=(
                    f"qualifying_modules={qualifying_modules} is below "
                    f"{thresholds.min_module_count}."
                ),
                arena_size=arena_size,
                path=str(arena_dir / "module_summary.csv"),
            )
        )
    return ArenaValidation(
        arena_size=arena_size,
        arena_dir=str(arena_dir),
        coverage_fraction=coverage,
        active_units=active_units,
        invalid_response_units=invalid_units,
        qualifying_modules=qualifying_modules,
        issues=issues,
    )


def _artifact_issues(arena_dir: Path, arena_size: float) -> list[ValidationIssue]:
    issues = []
    for name in REQUIRED_ARENA_ARTIFACTS:
        path = arena_dir / name
        if not path.exists():
            issues.append(
                ValidationIssue(
                    severity="blocker",
                    code="missing_artifact",
                    message=f"Required arena artifact is missing: {name}.",
                    arena_size=arena_size,
                    path=str(path),
                )
            )
    return issues


def _count_qualifying_modules(arena_dir: Path, thresholds: ValidationThresholds) -> int:
    module_path = arena_dir / "module_summary.csv"
    if not module_path.exists():
        return 0
    count = 0
    try:
        with module_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                unit_count = _optional_int(row.get("unit_count")) or 0
                mean_score = _optional_float(row.get("mean_grid_score_60"))
                if (
                    unit_count >= thresholds.min_module_units
                    and mean_score is not None
                    and mean_score >= thresholds.min_module_mean_grid_score_60
                ):
                    count += 1
    except (OSError, UnicodeError, csv.Error):
        return 0
    return count


def _report(
    output_dir: Path,
    arenas: list[ArenaValidation],
    issues: list[ValidationIssue],
) -> ValidationReport:
    blocker_count = sum(1 for issue in issues if issue.severity == "blocker")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return ValidationReport(
        output_dir=str(output_dir),
        passed=blocker_count == 0,
        blocker_count=blocker_count,
        warning_count=warning_count,
        arenas=arenas,
        issues=issues,
    )


def _format_arena_size(arena_size: float) -> str:
    return str(arena_size).replace(".", "p")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
