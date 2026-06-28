from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.config import Config, load_config
from sic4gridcells.evaluate import EvaluationResult, evaluate_checkpoint
from sic4gridcells.train import RunResult, train


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_ablations(args.config, dry_run=args.dry_run, skip_evaluation=args.skip_eval)
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
) -> dict[str, AblationResult]:
    plan = load_ablation_plan(config_path)
    runs = materialize_run_configs(plan)
    results: dict[str, AblationResult] = {}
    continue_on_error = bool(plan.get("continue_on_error", False))
    evaluation_cfg = plan.get("evaluation", {})
    evaluation_enabled = (
        isinstance(evaluation_cfg, dict)
        and bool(evaluation_cfg.get("enabled", False))
        and not skip_evaluation
    )
    for run in runs:
        if not run.enabled:
            results[run.name] = AblationResult(
                status="skipped",
                config_path=run.config_path,
                reason=run.reason,
            )
            continue
        load_config(run.config_path)
        if dry_run:
            results[run.name] = AblationResult(status="validated", config_path=run.config_path)
            continue
        try:
            run_result = train(run.config_path)
            evaluation_result = None
            if evaluation_enabled:
                evaluation_result = _evaluate_ablation_run(run_result, evaluation_cfg)
            results[run.name] = AblationResult(
                status="finished",
                config_path=run.config_path,
                run_result=run_result,
                evaluation_result=evaluation_result,
            )
        except Exception:
            if not continue_on_error:
                raise
            results[run.name] = AblationResult(status="failed", config_path=run.config_path)
    _write_ablation_summary(plan, results)
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
        seed=seed,
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
                "arena_size": None,
                "mean_grid_score_60": None,
                "mean_scale_meters": None,
                "detected_modules": None,
                "active_units": None,
            }
            if isinstance(arena_summary, dict):
                csv_row.update(
                    {
                        "arena_size": arena_summary.get("arena_size"),
                        "mean_grid_score_60": arena_summary.get("mean_grid_score_60"),
                        "mean_scale_meters": arena_summary.get("mean_scale_meters"),
                        "detected_modules": arena_summary.get("detected_modules"),
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
            "arena_size",
            "mean_grid_score_60",
            "mean_scale_meters",
            "detected_modules",
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
