from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.validation import (
    ValidationThresholds,
    validate_evaluation_output,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SIC evaluation artifacts.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--json-output", help="Optional path for the validation report JSON.")
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-active-units", type=int, default=1)
    parser.add_argument("--max-invalid-response-units", type=int, default=0)
    parser.add_argument("--min-module-count", type=int, default=1)
    parser.add_argument("--min-module-units", type=int, default=3)
    parser.add_argument("--min-module-grid-score-60", type=float, default=0.0)
    parser.add_argument(
        "--arena-sizes",
        default="",
        help="Optional comma-separated arena sizes that must be present.",
    )
    parser.add_argument(
        "--no-artifact-check",
        action="store_true",
        help="Skip per-arena required artifact existence checks.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Return exit code 0 even when validation blockers are found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = ValidationThresholds(
        min_coverage_fraction=args.min_coverage,
        min_active_units=args.min_active_units,
        max_invalid_response_units=args.max_invalid_response_units,
        min_module_count=args.min_module_count,
        min_module_units=args.min_module_units,
        min_module_mean_grid_score_60=args.min_module_grid_score_60,
        required_arena_sizes=_parse_arena_sizes(args.arena_sizes),
        require_artifacts=not args.no_artifact_check,
    )
    report = validate_evaluation_output(args.output_dir, thresholds)
    if args.json_output:
        write_validation_report(report, args.json_output)
    status = "passed" if report.passed else "failed"
    print(
        f"validation {status} output_dir={report.output_dir} "
        f"blockers={report.blocker_count} warnings={report.warning_count}"
    )
    for issue in report.issues[:20]:
        suffix = f" arena={issue.arena_size}" if issue.arena_size is not None else ""
        path = f" path={issue.path}" if issue.path else ""
        print(f"{issue.severity} {issue.code}:{suffix}{path} {issue.message}")
    if len(report.issues) > 20:
        print(f"... {len(report.issues) - 20} more issues")
    if report.passed or args.allow_fail:
        return 0
    return 1


def _parse_arena_sizes(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


if __name__ == "__main__":
    raise SystemExit(main())
