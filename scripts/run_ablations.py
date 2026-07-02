from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.config import Config, load_config
from sic4gridcells.evaluate import EvaluationResult, evaluate_checkpoint
from sic4gridcells.logging_utils import (
    JsonlEventLogger,
    VALID_LOG_LEVELS,
    cli_logging_context,
    elapsed_seconds,
    log_file_context,
)
from sic4gridcells.runtime import (
    ABLATION_OUTPUT_MARKERS,
    discover_latest_checkpoint,
    is_output_completed,
    prepare_output_dir,
)
from sic4gridcells.train import RunResult, train

logger = logging.getLogger("sic4gridcells.run_ablations")


@dataclass(frozen=True)
class AblationRun:
    name: str
    config_path: Path
    enabled: bool
    reason: str | None = None


@dataclass(frozen=True)
class AblationResult:
    status: str
    config_path: Path
    run_result: RunResult | None = None
    evaluation_result: EvaluationResult | None = None
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured SIC ablations.")
    parser.add_argument("--config", required=True, help="Path to an ablation YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and validate per-run configs without launching training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip configured post-training evaluation.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume each variant from its latest checkpoint when one exists.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip variants whose latest checkpoint already reached max_optimizer_steps.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow fresh variant runs and aggregate logs to reuse existing output directories.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=VALID_LOG_LEVELS,
        help="Console log level for stderr logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with cli_logging_context(args.log_level):
        results = run_ablations(
            args.config,
            dry_run=args.dry_run,
            skip_evaluation=args.skip_eval,
            resume_existing=args.resume_existing,
            skip_completed=args.skip_completed,
            overwrite_output=args.overwrite_output,
        )
    for name, result in results.items():
        if result.status == "validated":
            print(f"validated {name}: {result.config_path}")
        elif result.status == "skipped":
            suffix = f": {result.reason}" if result.reason else ""
            print(f"skipped {name}{suffix}")
        elif result.status == "failed":
            print(f"failed {name}")
        elif result.status == "finished" and result.run_result is not None:
            run_result = result.run_result
            print(f"finished {name} step={run_result.final_step} output_dir={run_result.output_dir}")
            print(f"checkpoint={run_result.checkpoint_path}")
            if result.evaluation_result is not None:
                print(f"evaluation={result.evaluation_result.output_dir}")


def run_ablations(
    config_path: str | Path,
    dry_run: bool = False,
    skip_evaluation: bool = False,
    resume_existing: bool = False,
    skip_completed: bool = False,
    overwrite_output: bool = False,
) -> dict[str, AblationResult]:
    plan = load_ablation_plan(config_path)
    results: dict[str, AblationResult] = {}
    continue_on_error = bool(plan.get("continue_on_error", False))
    evaluation_cfg = plan.get("evaluation", {})
    evaluation_enabled = (
        isinstance(evaluation_cfg, dict)
        and bool(evaluation_cfg.get("enabled", False))
        and not skip_evaluation
    )
    output_root = prepare_output_dir(
        str(plan.get("output_root", "results/ablations")),
        resume=resume_existing or skip_completed,
        overwrite=overwrite_output and not (resume_existing or skip_completed),
        markers=ABLATION_OUTPUT_MARKERS,
    )
    runs = materialize_run_configs(plan)
    start_time = time.perf_counter()
    error_to_raise: Exception | None = None
    with log_file_context(output_root / "run.log", logger_names=("sic4gridcells",), mode="w"):
        with JsonlEventLogger(output_root / "ablation_events.jsonl", mode="w") as events:
            logger.info(
                "ablation run started output_root=%s dry_run=%s skip_evaluation=%s",
                output_root,
                dry_run,
                skip_evaluation,
            )
            events.emit(
                "ablation_start",
                status="started",
                config_path=Path(config_path),
                output_root=output_root,
                dry_run=dry_run,
                skip_evaluation=skip_evaluation,
                resume_existing=resume_existing,
                skip_completed=skip_completed,
                overwrite_output=overwrite_output,
                variant_count=len(runs),
                continue_on_error=continue_on_error,
            )
            for run in runs:
                logger.info("variant scheduled name=%s enabled=%s", run.name, run.enabled)
                if not run.enabled:
                    results[run.name] = AblationResult(
                        status="skipped",
                        config_path=run.config_path,
                        reason=run.reason,
                    )
                    logger.info("variant skipped name=%s reason=%s", run.name, run.reason)
                    events.emit(
                        "variant_skipped",
                        status="skipped",
                        name=run.name,
                        config_path=run.config_path,
                        reason=run.reason,
                    )
                    continue
                run_start = time.perf_counter()
                try:
                    cfg = load_config(run.config_path)
                    logger.info("variant config validated name=%s path=%s", run.name, run.config_path)
                    events.emit(
                        "variant_validated",
                        status="validated",
                        name=run.name,
                        config_path=run.config_path,
                    )
                    detailed_logging = cfg.logging.detail_level == "detailed"
                    if detailed_logging:
                        events.emit(
                            "variant_config_materialized",
                            status="written",
                            name=run.name,
                            config_path=run.config_path,
                            output_dir=cfg.output_dir,
                            logging_detail_level=cfg.logging.detail_level,
                        )
                    if dry_run:
                        results[run.name] = AblationResult(
                            status="validated",
                            config_path=run.config_path,
                        )
                        continue
                    if skip_completed and is_output_completed(
                        cfg.output_dir,
                        cfg.train.max_optimizer_steps,
                    ):
                        latest = discover_latest_checkpoint(cfg.output_dir)
                        reason = (
                            "already completed"
                            if latest is None
                            else f"already completed at step {latest.step}"
                        )
                        results[run.name] = AblationResult(
                            status="skipped",
                            config_path=run.config_path,
                            reason=reason,
                        )
                        logger.info("variant skipped name=%s reason=%s", run.name, reason)
                        events.emit(
                            "variant_skipped",
                            status="skipped",
                            name=run.name,
                            config_path=run.config_path,
                            reason=reason,
                        )
                        continue
                    resume_checkpoint = None
                    if resume_existing:
                        latest = discover_latest_checkpoint(cfg.output_dir)
                        if latest is not None:
                            resume_checkpoint = latest.path
                    if detailed_logging:
                        events.emit(
                            "variant_resume_decision",
                            status="selected",
                            name=run.name,
                            output_dir=cfg.output_dir,
                            resume_existing=resume_existing,
                            resume_checkpoint=resume_checkpoint,
                        )
                    logger.info("variant training started name=%s", run.name)
                    events.emit(
                        "variant_train_start",
                        status="started",
                        name=run.name,
                        config_path=run.config_path,
                        resume_checkpoint=resume_checkpoint,
                    )
                    train_kwargs = {}
                    if resume_checkpoint is not None:
                        train_kwargs["resume_checkpoint"] = resume_checkpoint
                    if overwrite_output and resume_checkpoint is None:
                        train_kwargs["overwrite_output"] = True
                    train_start = time.perf_counter() if detailed_logging else None
                    run_result = train(run.config_path, **train_kwargs)
                    if train_start is not None:
                        events.emit(
                            "variant_train_finished",
                            status="finished",
                            name=run.name,
                            output_dir=run_result.output_dir,
                            final_step=run_result.final_step,
                            checkpoint_path=run_result.checkpoint_path,
                            duration_seconds=elapsed_seconds(train_start),
                        )
                    evaluation_result = None
                    if evaluation_enabled:
                        logger.info(
                            "variant evaluation started name=%s checkpoint=%s",
                            run.name,
                            run_result.checkpoint_path,
                        )
                        events.emit(
                            "variant_eval_start",
                            status="started",
                            name=run.name,
                            checkpoint_path=run_result.checkpoint_path,
                        )
                        eval_start = time.perf_counter() if detailed_logging else None
                        evaluation_result = _evaluate_ablation_run(
                            run_result,
                            evaluation_cfg,
                            overwrite_output=overwrite_output,
                        )
                        logger.info(
                            "variant evaluation finished name=%s output_dir=%s",
                            run.name,
                            evaluation_result.output_dir,
                        )
                        eval_finished_event = {
                            "status": "finished",
                            "name": run.name,
                            "evaluation_output_dir": evaluation_result.output_dir,
                        }
                        if eval_start is not None:
                            eval_finished_event["duration_seconds"] = elapsed_seconds(
                                eval_start
                            )
                        events.emit("variant_eval_finished", **eval_finished_event)
                    results[run.name] = AblationResult(
                        status="finished",
                        config_path=run.config_path,
                        run_result=run_result,
                        evaluation_result=evaluation_result,
                    )
                    logger.info(
                        "variant finished name=%s step=%s output_dir=%s",
                        run.name,
                        run_result.final_step,
                        run_result.output_dir,
                    )
                    events.emit(
                        "variant_finished",
                        status="finished",
                        name=run.name,
                        final_step=run_result.final_step,
                        output_dir=run_result.output_dir,
                        checkpoint_path=run_result.checkpoint_path,
                        evaluation_output_dir=(
                            None
                            if evaluation_result is None
                            else evaluation_result.output_dir
                        ),
                        duration_seconds=elapsed_seconds(run_start),
                    )
                except Exception as exc:
                    logger.exception("variant failed name=%s", run.name)
                    results[run.name] = AblationResult(
                        status="failed",
                        config_path=run.config_path,
                        reason=str(exc),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    events.emit(
                        "variant_failed",
                        status="failed",
                        name=run.name,
                        config_path=run.config_path,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        duration_seconds=elapsed_seconds(run_start),
                    )
                    if not continue_on_error:
                        error_to_raise = exc
                        break
            _write_ablation_summary(plan, results)
            logger.info("ablation summary written output_root=%s", output_root)
            events.emit(
                "ablation_summary_written",
                status="written",
                output_root=output_root,
                summary_json=output_root / "summary.json",
                summary_csv=output_root / "summary.csv",
                variant_count=len(results),
            )
            failed_variant_count = sum(
                1 for result in results.values() if result.status == "failed"
            )
            if error_to_raise is None and failed_variant_count == 0:
                logger.info(
                    "ablation finished output_root=%s variant_count=%s",
                    output_root,
                    len(results),
                )
                events.emit(
                    "ablation_finished",
                    status="finished",
                    output_root=output_root,
                    variant_count=len(results),
                    duration_seconds=elapsed_seconds(start_time),
                )
            else:
                logger.error(
                    "ablation failed output_root=%s failed_variant_count=%s",
                    output_root,
                    failed_variant_count,
                )
                events.emit(
                    "ablation_failed",
                    status="failed",
                    output_root=output_root,
                    variant_count=len(results),
                    failed_variant_count=failed_variant_count,
                    duration_seconds=elapsed_seconds(start_time),
                    error_type=(
                        None if error_to_raise is None else type(error_to_raise).__name__
                    ),
                    error_message=None if error_to_raise is None else str(error_to_raise),
                )
    if error_to_raise is not None:
        raise error_to_raise
    return results


def load_ablation_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle) or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Ablation config must contain a YAML mapping: {path}")
    if "base" not in plan or not isinstance(plan["base"], dict):
        raise ValueError("Ablation config requires a 'base' mapping")
    if "variants" not in plan or not isinstance(plan["variants"], list):
        raise ValueError("Ablation config requires a 'variants' list")
    return plan


def materialize_run_configs(plan: dict[str, Any]) -> list[AblationRun]:
    output_root = Path(str(plan.get("output_root", "results/ablations")))
    config_dir = Path(str(plan.get("config_dir", output_root / "configs")))
    config_dir.mkdir(parents=True, exist_ok=True)
    runs: list[AblationRun] = []
    seen_names: set[str] = set()
    for variant in plan["variants"]:
        if not isinstance(variant, dict):
            raise ValueError("Each ablation variant must be a mapping")
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("Each ablation variant requires a non-empty name")
        if name in seen_names:
            raise ValueError(f"Duplicate ablation variant name: {name}")
        seen_names.add(name)
        run_cfg = asdict(Config())
        _deep_update(run_cfg, plan["base"], context="ablation base")
        overrides = variant.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Ablation variant '{name}' overrides must be a mapping")
        _deep_update(run_cfg, overrides, context="ablation override")
        run_cfg["output_dir"] = str(output_root / name)
        run_config_path = config_dir / f"{name}.yaml"
        with run_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(run_cfg, handle, sort_keys=False)
        runs.append(
            AblationRun(
                name=name,
                config_path=run_config_path,
                enabled=bool(variant.get("enabled", True)),
                reason=variant.get("reason"),
            )
        )
    return runs


def _evaluate_ablation_run(
    run_result: RunResult,
    evaluation_cfg: dict[str, Any],
    *,
    overwrite_output: bool = False,
) -> EvaluationResult:
    output_dir_name = str(evaluation_cfg.get("output_dir_name", "eval"))
    seed_value = evaluation_cfg.get("seed")
    seed = None if seed_value is None else int(seed_value)
    return evaluate_checkpoint(
        run_result.checkpoint_path,
        Path(run_result.output_dir) / output_dir_name,
        device=str(evaluation_cfg.get("device", "auto")),
        arena_sizes=_parse_arena_sizes(evaluation_cfg.get("arena_sizes", [2.0, 3.0, 4.0])),
        nbins=int(evaluation_cfg.get("nbins", 32)),
        n_trajectories=int(evaluation_cfg.get("trajectories", 32)),
        steps_per_trajectory=int(evaluation_cfg.get("steps", 256)),
        start_mode=str(evaluation_cfg.get("start_mode", "origin")),
        trajectory_mode=str(evaluation_cfg.get("trajectory_mode", "reflect")),
        seed=seed,
        overwrite_output=overwrite_output,
    )


def _parse_arena_sizes(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item) for item in value.split(",") if item)
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    raise ValueError("evaluation.arena_sizes must be a list or comma-separated string")


def _write_ablation_summary(
    plan: dict[str, Any],
    results: dict[str, AblationResult],
) -> None:
    output_root = Path(str(plan.get("output_root", "results/ablations")))
    output_root.mkdir(parents=True, exist_ok=True)
    json_rows = []
    csv_rows = []
    for name, result in results.items():
        row = {
            "name": name,
            "status": result.status,
            "config_path": str(result.config_path),
            "reason": result.reason,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "output_dir": None,
            "checkpoint_path": None,
            "evaluation_output_dir": None,
            "arena_summaries": [],
        }
        if result.run_result is not None:
            row["output_dir"] = str(result.run_result.output_dir)
            row["checkpoint_path"] = str(result.run_result.checkpoint_path)
        if result.evaluation_result is not None:
            row["evaluation_output_dir"] = str(result.evaluation_result.output_dir)
            summary_path = result.evaluation_result.output_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["arena_summaries"] = summary.get("arena_summaries", [])
        json_rows.append(row)
        arena_summaries = row["arena_summaries"] or [None]
        for arena_summary in arena_summaries:
            csv_row = {
                "name": name,
                "status": result.status,
                "config_path": str(result.config_path),
                "output_dir": row["output_dir"],
                "checkpoint_path": row["checkpoint_path"],
                "evaluation_output_dir": row["evaluation_output_dir"],
                "error_type": result.error_type,
                "error_message": result.error_message,
                "arena_size": None,
                "mean_grid_score_60": None,
                "mean_grid_score_90": None,
                "mean_scale_meters": None,
                "detected_modules": None,
                "coverage_fraction": None,
                "units_without_coverage": None,
                "zero_response_units": None,
                "invalid_response_units": None,
                "active_units": None,
            }
            if isinstance(arena_summary, dict):
                csv_row.update(
                    {
                        "arena_size": arena_summary.get("arena_size"),
                        "mean_grid_score_60": arena_summary.get("mean_grid_score_60"),
                        "mean_grid_score_90": arena_summary.get("mean_grid_score_90"),
                        "mean_scale_meters": arena_summary.get("mean_scale_meters"),
                        "detected_modules": arena_summary.get("detected_modules"),
                        "coverage_fraction": arena_summary.get("coverage_fraction"),
                        "units_without_coverage": arena_summary.get("units_without_coverage"),
                        "zero_response_units": arena_summary.get("zero_response_units"),
                        "invalid_response_units": arena_summary.get("invalid_response_units"),
                        "active_units": arena_summary.get("active_units"),
                    }
                )
            csv_rows.append(csv_row)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_rows, handle, indent=2, sort_keys=True, allow_nan=False)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "status",
            "config_path",
            "output_dir",
            "checkpoint_path",
            "evaluation_output_dir",
            "error_type",
            "error_message",
            "arena_size",
            "mean_grid_score_60",
            "mean_grid_score_90",
            "mean_scale_meters",
            "detected_modules",
            "coverage_fraction",
            "units_without_coverage",
            "zero_response_units",
            "invalid_response_units",
            "active_units",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


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


if __name__ == "__main__":
    main()
