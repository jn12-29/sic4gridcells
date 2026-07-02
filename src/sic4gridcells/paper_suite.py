from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sic4gridcells.config import Config, load_config
from sic4gridcells.evaluate import EvaluationResult, evaluate_checkpoint
from sic4gridcells.logging_utils import (
    JsonlEventLogger,
    elapsed_seconds,
    log_file_context,
)
from sic4gridcells.profiling import ProfileSummary, profile_training_run
from sic4gridcells.runtime import (
    discover_latest_checkpoint,
    is_output_completed,
    prepare_output_dir,
)
from sic4gridcells.train import RunResult, train
from sic4gridcells.validation import (
    ValidationThresholds,
    validate_evaluation_output,
    write_validation_report,
)

logger = logging.getLogger("sic4gridcells.paper_suite")

PAPER_SUITE_OUTPUT_MARKERS = (
    "run.log",
    "paper_suite_events.jsonl",
    "manifest.json",
    "summary.json",
    "summary.csv",
)


@dataclass(frozen=True)
class PaperSuiteRun:
    name: str
    config_path: Path
    enabled: bool
    variant: str
    seed: int | None
    diagnostic_only: bool
    reason: str | None = None


@dataclass(frozen=True)
class PaperSuiteResult:
    status: str
    config_path: Path
    variant: str
    seed: int | None
    diagnostic_only: bool
    reason: str | None = None
    profile_summary: ProfileSummary | None = None
    run_result: RunResult | None = None
    evaluation_result: EvaluationResult | None = None
    validation_report_path: Path | None = None
    validation_passed: bool | None = None
    analysis_output_dir: Path | None = None
    figure_output_dir: Path | None = None
    error_type: str | None = None
    error_message: str | None = None


def run_paper_suite(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    resume_existing: bool = False,
    skip_completed: bool = False,
    overwrite_output: bool = False,
    command_line_args: dict[str, Any] | None = None,
) -> dict[str, PaperSuiteResult]:
    plan = load_paper_suite_plan(config_path)
    suite_dir = _suite_dir(plan)
    prepare_output_dir(
        suite_dir,
        resume=resume_existing or skip_completed,
        overwrite=overwrite_output and not (resume_existing or skip_completed),
        markers=PAPER_SUITE_OUTPUT_MARKERS,
    )
    runs = materialize_paper_suite_configs(plan)
    results: dict[str, PaperSuiteResult] = {}
    continue_on_error = bool(plan.get("continue_on_error", False))
    git_commit = _git_commit()
    start_time = time.perf_counter()
    error_to_raise: Exception | None = None
    detailed_run_names: set[str] = set()

    with log_file_context(suite_dir / "run.log", logger_names=("sic4gridcells",), mode="w"):
        with JsonlEventLogger(suite_dir / "paper_suite_events.jsonl", mode="w") as events:
            events.emit(
                "paper_suite_start",
                status="started",
                config_path=Path(config_path),
                suite_dir=suite_dir,
                dry_run=dry_run,
                resume_existing=resume_existing,
                skip_completed=skip_completed,
                overwrite_output=overwrite_output,
                run_count=len(runs),
            )
            for suite_run in runs:
                if not suite_run.enabled:
                    results[suite_run.name] = PaperSuiteResult(
                        status="skipped",
                        config_path=suite_run.config_path,
                        variant=suite_run.variant,
                        seed=suite_run.seed,
                        diagnostic_only=suite_run.diagnostic_only,
                        reason=suite_run.reason,
                    )
                    events.emit(
                        "paper_suite_run_skipped",
                        status="skipped",
                        name=suite_run.name,
                        reason=suite_run.reason,
                    )
                    continue
                run_start = time.perf_counter()
                try:
                    cfg = load_config(suite_run.config_path)
                    detailed_logging = cfg.logging.detail_level == "detailed"
                    if detailed_logging:
                        detailed_run_names.add(suite_run.name)
                    if dry_run:
                        results[suite_run.name] = PaperSuiteResult(
                            status="validated",
                            config_path=suite_run.config_path,
                            variant=suite_run.variant,
                            seed=suite_run.seed,
                            diagnostic_only=suite_run.diagnostic_only,
                        )
                        events.emit(
                            "paper_suite_run_validated",
                            status="validated",
                            name=suite_run.name,
                            config_path=suite_run.config_path,
                        )
                        continue
                    if skip_completed and is_output_completed(
                        cfg.output_dir,
                        cfg.train.max_optimizer_steps,
                    ):
                        latest = discover_latest_checkpoint(cfg.output_dir)
                        results[suite_run.name] = _completed_skip_result(
                            plan,
                            suite_run,
                            cfg,
                            latest=latest,
                        )
                        continue
                    profile_start = None
                    if detailed_logging and _section_enabled(plan, "profile"):
                        profile_start = _emit_stage_start(
                            events,
                            suite_run.name,
                            "profile",
                            config_path=suite_run.config_path,
                        )
                    profile_summary = _maybe_profile_run(
                        plan,
                        suite_run,
                        suite_dir=suite_dir,
                        overwrite_output=overwrite_output,
                    )
                    if detailed_logging and profile_start is not None:
                        _emit_stage_finished(
                            events,
                            suite_run.name,
                            "profile",
                            profile_start,
                            output_dir=(
                                None
                                if profile_summary is None
                                else profile_summary.output_dir
                            ),
                        )
                    resume_checkpoint = None
                    if resume_existing:
                        latest = discover_latest_checkpoint(cfg.output_dir)
                        if latest is not None:
                            resume_checkpoint = latest.path
                    train_kwargs: dict[str, Any] = {}
                    if resume_checkpoint is not None:
                        train_kwargs["resume_checkpoint"] = resume_checkpoint
                    if overwrite_output and resume_checkpoint is None:
                        train_kwargs["overwrite_output"] = True
                    train_start = None
                    if detailed_logging:
                        train_start = _emit_stage_start(
                            events,
                            suite_run.name,
                            "train",
                            config_path=suite_run.config_path,
                            resume_checkpoint=resume_checkpoint,
                        )
                    run_result = train(suite_run.config_path, **train_kwargs)
                    if detailed_logging and train_start is not None:
                        _emit_stage_finished(
                            events,
                            suite_run.name,
                            "train",
                            train_start,
                            output_dir=run_result.output_dir,
                            checkpoint_path=run_result.checkpoint_path,
                            final_step=run_result.final_step,
                        )
                    eval_start = None
                    if detailed_logging and _section_enabled(plan, "evaluation"):
                        eval_start = _emit_stage_start(
                            events,
                            suite_run.name,
                            "eval",
                            checkpoint_path=run_result.checkpoint_path,
                        )
                    evaluation_result = _maybe_evaluate_run(
                        plan,
                        suite_run,
                        run_result,
                        overwrite_output=overwrite_output,
                    )
                    if detailed_logging and eval_start is not None:
                        _emit_stage_finished(
                            events,
                            suite_run.name,
                            "eval",
                            eval_start,
                            output_dir=(
                                None
                                if evaluation_result is None
                                else evaluation_result.output_dir
                            ),
                        )
                    validation_report_path = None
                    validation_passed = None
                    if evaluation_result is not None:
                        validation_start = None
                        if detailed_logging and _section_enabled(plan, "validation"):
                            validation_start = _emit_stage_start(
                                events,
                                suite_run.name,
                                "validation",
                                evaluation_output_dir=evaluation_result.output_dir,
                            )
                        validation_report_path, validation_passed = _maybe_validate_run(
                            plan,
                            suite_run,
                            evaluation_result,
                        )
                        if detailed_logging and validation_start is not None:
                            _emit_stage_finished(
                                events,
                                suite_run.name,
                                "validation",
                                validation_start,
                                report_path=validation_report_path,
                                validation_passed=validation_passed,
                            )
                    analysis_start = None
                    if (
                        detailed_logging
                        and evaluation_result is not None
                        and _section_enabled(plan, "analysis")
                    ):
                        analysis_start = _emit_stage_start(
                            events,
                            suite_run.name,
                            "analysis",
                            evaluation_output_dir=evaluation_result.output_dir,
                        )
                    analysis_output_dir = _maybe_analyze_run(
                        plan,
                        suite_run,
                        evaluation_result,
                    )
                    if detailed_logging and analysis_start is not None:
                        _emit_stage_finished(
                            events,
                            suite_run.name,
                            "analysis",
                            analysis_start,
                            output_dir=analysis_output_dir,
                        )
                    results[suite_run.name] = PaperSuiteResult(
                        status="finished",
                        config_path=suite_run.config_path,
                        variant=suite_run.variant,
                        seed=suite_run.seed,
                        diagnostic_only=suite_run.diagnostic_only,
                        profile_summary=profile_summary,
                        run_result=run_result,
                        evaluation_result=evaluation_result,
                        validation_report_path=validation_report_path,
                        validation_passed=validation_passed,
                        analysis_output_dir=analysis_output_dir,
                    )
                    events.emit(
                        "paper_suite_run_finished",
                        status="finished",
                        name=suite_run.name,
                        output_dir=run_result.output_dir,
                        checkpoint_path=run_result.checkpoint_path,
                        duration_seconds=elapsed_seconds(run_start),
                    )
                except Exception as exc:
                    results[suite_run.name] = PaperSuiteResult(
                        status="failed",
                        config_path=suite_run.config_path,
                        variant=suite_run.variant,
                        seed=suite_run.seed,
                        diagnostic_only=suite_run.diagnostic_only,
                        reason=str(exc),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    events.emit(
                        "paper_suite_run_failed",
                        status="failed",
                        name=suite_run.name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        duration_seconds=elapsed_seconds(run_start),
                    )
                    if not continue_on_error:
                        error_to_raise = exc
                        break
            _write_suite_outputs(
                plan,
                results,
                suite_dir=suite_dir,
                config_path=Path(config_path),
                dry_run=dry_run,
                resume_existing=resume_existing,
                skip_completed=skip_completed,
                overwrite_output=overwrite_output,
                command_line_args=command_line_args or {},
                git_commit=git_commit,
                suite_figure_output_dir=None,
            )
            suite_figure_output_dir = None
            if not dry_run and error_to_raise is None and _can_build_suite_figures(results):
                figure_start = None
                if detailed_run_names and _section_enabled(plan, "figures"):
                    figure_start = _emit_stage_start(
                        events,
                        None,
                        "figure",
                        suite_dir=suite_dir,
                        run_names=sorted(detailed_run_names),
                    )
                suite_figure_output_dir = _maybe_build_figures(plan, suite_dir=suite_dir)
                if figure_start is not None:
                    _emit_stage_finished(
                        events,
                        None,
                        "figure",
                        figure_start,
                        output_dir=suite_figure_output_dir,
                    )
                if suite_figure_output_dir is not None:
                    _write_suite_outputs(
                        plan,
                        results,
                        suite_dir=suite_dir,
                        config_path=Path(config_path),
                        dry_run=dry_run,
                        resume_existing=resume_existing,
                        skip_completed=skip_completed,
                        overwrite_output=overwrite_output,
                        command_line_args=command_line_args or {},
                        git_commit=git_commit,
                        suite_figure_output_dir=suite_figure_output_dir,
                    )
            events.emit(
                "paper_suite_summary_written",
                status="written",
                suite_dir=suite_dir,
                manifest_path=suite_dir / "manifest.json",
                summary_json=suite_dir / "summary.json",
                summary_csv=suite_dir / "summary.csv",
            )
            events.emit(
                "paper_suite_finished" if error_to_raise is None else "paper_suite_failed",
                status="finished" if error_to_raise is None else "failed",
                suite_dir=suite_dir,
                duration_seconds=elapsed_seconds(start_time),
            )
    if error_to_raise is not None:
        raise error_to_raise
    return results


def load_paper_suite_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle) or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Paper suite config must contain a YAML mapping: {path}")
    if "runs" not in plan or not isinstance(plan["runs"], list):
        raise ValueError("Paper suite config requires a 'runs' list")
    return plan


def materialize_paper_suite_configs(plan: dict[str, Any]) -> list[PaperSuiteRun]:
    suite_dir = _suite_dir(plan)
    config_dir = Path(str(plan.get("config_dir", suite_dir / "configs")))
    config_dir.mkdir(parents=True, exist_ok=True)
    runs: list[PaperSuiteRun] = []
    seen: set[str] = set()
    for row in plan["runs"]:
        if not isinstance(row, dict):
            raise ValueError("Each paper suite run must be a mapping")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("Each paper suite run requires a non-empty name")
        if name in seen:
            raise ValueError(f"Duplicate paper suite run name: {name}")
        seen.add(name)
        source_config = row.get("config")
        if source_config is None:
            raise ValueError(f"Paper suite run '{name}' requires a config path")
        cfg_dict = asdict(load_config(source_config))
        if "seed" in row:
            cfg_dict["seed"] = int(row["seed"])
        overrides = row.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Paper suite run '{name}' overrides must be a mapping")
        _deep_update(cfg_dict, overrides, context="paper suite override")
        cfg_dict["output_dir"] = str(suite_dir / "runs" / name / "train")
        config_path = config_dir / f"{name}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg_dict, handle, sort_keys=False)
        runs.append(
            PaperSuiteRun(
                name=name,
                config_path=config_path,
                enabled=bool(row.get("enabled", True)),
                variant=str(row.get("variant", name)),
                seed=int(row["seed"]) if "seed" in row else int(cfg_dict["seed"]),
                diagnostic_only=bool(row.get("diagnostic_only", False)),
                reason=row.get("reason"),
            )
        )
    return runs


def _section_enabled(plan: dict[str, Any], name: str) -> bool:
    section = plan.get(name, {})
    return isinstance(section, dict) and bool(section.get("enabled", False))


def _emit_stage_start(
    events: JsonlEventLogger,
    run_name: str | None,
    stage: str,
    **fields: Any,
) -> float:
    start_time = time.perf_counter()
    events.emit(
        "paper_suite_stage_start",
        status="started",
        name=run_name,
        stage=stage,
        **fields,
    )
    return start_time


def _emit_stage_finished(
    events: JsonlEventLogger,
    run_name: str | None,
    stage: str,
    start_time: float,
    **fields: Any,
) -> None:
    events.emit(
        "paper_suite_stage_finished",
        status="finished",
        name=run_name,
        stage=stage,
        duration_seconds=elapsed_seconds(start_time),
        **fields,
    )


def _maybe_profile_run(
    plan: dict[str, Any],
    suite_run: PaperSuiteRun,
    *,
    suite_dir: Path,
    overwrite_output: bool,
) -> ProfileSummary | None:
    profile_cfg = plan.get("profile", {})
    if not isinstance(profile_cfg, dict) or not bool(profile_cfg.get("enabled", False)):
        return None
    output_dir = Path(str(profile_cfg.get("output_dir_name", "profile")))
    if not output_dir.is_absolute():
        output_dir = suite_dir / "runs" / suite_run.name / output_dir
    return profile_training_run(
        suite_run.config_path,
        output_dir,
        steps=int(profile_cfg.get("steps", 20)),
        device=profile_cfg.get("device"),
        overwrite_output=overwrite_output,
    )


def _maybe_evaluate_run(
    plan: dict[str, Any],
    suite_run: PaperSuiteRun,
    run_result: RunResult,
    *,
    overwrite_output: bool,
) -> EvaluationResult | None:
    eval_cfg = plan.get("evaluation", {})
    if not isinstance(eval_cfg, dict) or not bool(eval_cfg.get("enabled", False)):
        return None
    seed_value = eval_cfg.get("seed")
    seed = None if seed_value is None else int(seed_value)
    return evaluate_checkpoint(
        run_result.checkpoint_path,
        Path(run_result.output_dir).parent / str(eval_cfg.get("output_dir_name", "eval")),
        device=str(eval_cfg.get("device", "auto")),
        arena_sizes=_parse_arena_sizes(eval_cfg.get("arena_sizes", [2.0, 3.0, 4.0])),
        nbins=int(eval_cfg.get("nbins", 32)),
        n_trajectories=int(eval_cfg.get("trajectories", 32)),
        steps_per_trajectory=int(eval_cfg.get("steps", 256)),
        start_mode=str(eval_cfg.get("start_mode", "origin")),
        trajectory_mode=str(eval_cfg.get("trajectory_mode", "reflect")),
        seed=seed,
        overwrite_output=overwrite_output,
    )


def _maybe_validate_run(
    plan: dict[str, Any],
    suite_run: PaperSuiteRun,
    evaluation_result: EvaluationResult,
) -> tuple[Path | None, bool | None]:
    validation_cfg = plan.get("validation", {})
    if not isinstance(validation_cfg, dict) or not bool(validation_cfg.get("enabled", False)):
        return None, None
    thresholds = ValidationThresholds(
        min_coverage_fraction=float(validation_cfg.get("min_coverage", 0.8)),
        min_active_units=int(validation_cfg.get("min_active_units", 1)),
        max_invalid_response_units=int(validation_cfg.get("max_invalid_response_units", 0)),
        min_module_count=int(validation_cfg.get("min_module_count", 1)),
        min_module_units=int(validation_cfg.get("min_module_units", 3)),
        min_module_mean_grid_score_60=float(validation_cfg.get("min_module_grid_score_60", 0.0)),
        required_arena_sizes=_parse_arena_sizes(validation_cfg.get("arena_sizes", [])),
        require_artifacts=not bool(validation_cfg.get("no_artifact_check", False)),
    )
    report = validate_evaluation_output(evaluation_result.output_dir, thresholds)
    report_path = (
        Path(evaluation_result.output_dir).parent
        / str(validation_cfg.get("output_name", "validation_report.json"))
    )
    write_validation_report(report, report_path)
    if not report.passed and not suite_run.diagnostic_only and not bool(validation_cfg.get("allow_fail", False)):
        raise RuntimeError(
            f"validation blockers for {suite_run.name}: {report.blocker_count}"
        )
    return report_path, report.passed


def _maybe_analyze_run(
    plan: dict[str, Any],
    suite_run: PaperSuiteRun,
    evaluation_result: EvaluationResult | None,
) -> Path | None:
    analysis_cfg = plan.get("analysis", {})
    if (
        evaluation_result is None
        or not isinstance(analysis_cfg, dict)
        or not bool(analysis_cfg.get("enabled", False))
    ):
        return None
    from sic4gridcells.analysis_ext import analyze_evaluation_output

    output_dir = Path(evaluation_result.output_dir).parent / str(
        analysis_cfg.get("output_dir_name", "analysis")
    )
    result = analyze_evaluation_output(
        evaluation_result.output_dir,
        output_dir,
        run_id=suite_run.name,
        seed=suite_run.seed,
        variant=suite_run.variant,
        diagnostic_only=suite_run.diagnostic_only,
        min_grid_score_60=float(analysis_cfg.get("min_grid_score_60", 0.0)),
        min_module_units=int(analysis_cfg.get("min_module_units", 3)),
        max_scale_ratio_within_module=float(
            analysis_cfg.get("max_scale_ratio_within_module", 1.2)
        ),
        max_path_pairs=int(analysis_cfg.get("max_path_pairs", 20000)),
        max_state_space_samples=int(analysis_cfg.get("max_state_space_samples", 512)),
    )
    return result.output_dir


def _completed_skip_result(
    plan: dict[str, Any],
    suite_run: PaperSuiteRun,
    cfg: Config,
    *,
    latest: Any,
) -> PaperSuiteResult:
    run_root = Path(cfg.output_dir).parent
    checkpoint_path = None if latest is None else latest.path
    checkpoint_step = cfg.train.max_optimizer_steps if latest is None else latest.step
    run_result = (
        None
        if checkpoint_path is None
        else RunResult(
            output_dir=Path(cfg.output_dir),
            final_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
        )
    )
    evaluation_result = _existing_evaluation_result(plan, run_root, checkpoint_path)
    validation_report_path, validation_passed = _existing_validation_report(plan, run_root)
    analysis_output_dir = _existing_analysis_output_dir(plan, run_root)
    return PaperSuiteResult(
        status="skipped",
        config_path=suite_run.config_path,
        variant=suite_run.variant,
        seed=suite_run.seed,
        diagnostic_only=suite_run.diagnostic_only,
        reason=(
            "already completed"
            if latest is None
            else f"already completed at step {latest.step}"
        ),
        run_result=run_result,
        evaluation_result=evaluation_result,
        validation_report_path=validation_report_path,
        validation_passed=validation_passed,
        analysis_output_dir=analysis_output_dir,
    )


def _existing_evaluation_result(
    plan: dict[str, Any],
    run_root: Path,
    checkpoint_path: Path | None,
) -> EvaluationResult | None:
    eval_cfg = plan.get("evaluation", {})
    if not isinstance(eval_cfg, dict) or not bool(eval_cfg.get("enabled", False)):
        return None
    eval_dir = run_root / str(eval_cfg.get("output_dir_name", "eval"))
    if not (eval_dir / "summary.json").exists():
        return None
    return EvaluationResult(
        output_dir=eval_dir,
        checkpoint_path=checkpoint_path or Path(""),
        arena_dirs={},
    )


def _existing_validation_report(
    plan: dict[str, Any],
    run_root: Path,
) -> tuple[Path | None, bool | None]:
    validation_cfg = plan.get("validation", {})
    if not isinstance(validation_cfg, dict) or not bool(validation_cfg.get("enabled", False)):
        return None, None
    report_path = run_root / str(validation_cfg.get("output_name", "validation_report.json"))
    if not report_path.exists():
        return None, None
    return report_path, _validation_report_passed(report_path)


def _existing_analysis_output_dir(plan: dict[str, Any], run_root: Path) -> Path | None:
    analysis_cfg = plan.get("analysis", {})
    if not isinstance(analysis_cfg, dict) or not bool(analysis_cfg.get("enabled", False)):
        return None
    analysis_dir = run_root / str(analysis_cfg.get("output_dir_name", "analysis"))
    if (analysis_dir / "summary_tables").exists():
        return analysis_dir
    return None


def _maybe_build_figures(plan: dict[str, Any], *, suite_dir: Path) -> Path | None:
    figures_cfg = plan.get("figures", {})
    if not isinstance(figures_cfg, dict) or not bool(figures_cfg.get("enabled", False)):
        return None
    from sic4gridcells.paper_figures import build_paper_figures

    output_dir = Path(str(figures_cfg.get("output_dir", "figures")))
    if not output_dir.is_absolute():
        output_dir = suite_dir / output_dir
    return build_paper_figures(suite_dir, output_dir).output_dir


def _can_build_suite_figures(results: dict[str, PaperSuiteResult]) -> bool:
    return all(result.status != "failed" for result in results.values()) and any(
        result.analysis_output_dir is not None for result in results.values()
    )


def _validation_report_passed(path: Path) -> bool | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if isinstance(payload, dict) and isinstance(payload.get("passed"), bool):
        return bool(payload["passed"])
    return None


def _write_suite_outputs(
    plan: dict[str, Any],
    results: dict[str, PaperSuiteResult],
    *,
    suite_dir: Path,
    config_path: Path,
    dry_run: bool,
    resume_existing: bool,
    skip_completed: bool,
    overwrite_output: bool,
    command_line_args: dict[str, Any],
    git_commit: str | None,
    suite_figure_output_dir: Path | None,
) -> None:
    rows = [_result_row(name, result) for name, result in results.items()]
    manifest = {
        "suite_dir": str(suite_dir),
        "config_path": str(config_path),
        "run_id": str(plan.get("run_id", suite_dir.name)),
        "git_commit": git_commit,
        "command_line_args": command_line_args,
        "execution": {
            "dry_run": dry_run,
            "resume_existing": resume_existing,
            "skip_completed": skip_completed,
            "overwrite_output": overwrite_output,
        },
        "figure_output_dir": None if suite_figure_output_dir is None else str(suite_figure_output_dir),
        "runs": rows,
    }
    _write_json(suite_dir / "manifest.json", manifest)
    _write_json(suite_dir / "summary.json", rows)
    with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "run_id",
            "variant",
            "seed",
            "status",
            "diagnostic_only",
            "config_path",
            "checkpoint_path",
            "eval_output_dir",
            "analysis_output_dir",
            "figure_output_dir",
            "validation_report_path",
            "validation_passed",
            "error_type",
            "error_message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _result_row(name: str, result: PaperSuiteResult) -> dict[str, Any]:
    return {
        "name": name,
        "run_id": name,
        "variant": result.variant,
        "seed": result.seed,
        "status": result.status,
        "diagnostic_only": result.diagnostic_only,
        "reason": result.reason,
        "config_path": str(result.config_path),
        "profile_output_dir": (
            None if result.profile_summary is None else result.profile_summary.output_dir
        ),
        "checkpoint_path": (
            None if result.run_result is None else str(result.run_result.checkpoint_path)
        ),
        "output_dir": None if result.run_result is None else str(result.run_result.output_dir),
        "eval_output_dir": (
            None
            if result.evaluation_result is None
            else str(result.evaluation_result.output_dir)
        ),
        "evaluation_output_dir": (
            None
            if result.evaluation_result is None
            else str(result.evaluation_result.output_dir)
        ),
        "analysis_output_dir": (
            None if result.analysis_output_dir is None else str(result.analysis_output_dir)
        ),
        "figure_output_dir": (
            None if result.figure_output_dir is None else str(result.figure_output_dir)
        ),
        "validation_report_path": (
            None if result.validation_report_path is None else str(result.validation_report_path)
        ),
        "validation_passed": result.validation_passed,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def _suite_dir(plan: dict[str, Any]) -> Path:
    output_root = Path(str(plan.get("output_root", "results/paper_suite")))
    run_id = str(plan.get("run_id", "default"))
    return output_root / run_id


def _parse_arena_sizes(value: Any) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(float(item) for item in value.split(",") if item)
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    raise ValueError("arena_sizes must be a list or comma-separated string")


def _deep_update(
    base: dict[str, Any],
    updates: dict[str, Any],
    *,
    context: str,
    path: str = "",
) -> None:
    for key, value in updates.items():
        full_key = f"{path}.{key}" if path else str(key)
        if key not in base:
            raise ValueError(f"Unknown {context} key: {full_key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value, context=context, path=full_key)
        else:
            base[key] = value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None
