import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.run_paper_suite as run_paper_suite_script
from sic4gridcells.config import load_config
from sic4gridcells.evaluate import EvaluationResult
from sic4gridcells.paper_suite import (
    load_paper_suite_plan,
    materialize_paper_suite_configs,
    run_paper_suite,
)
from sic4gridcells.train import RunResult


def test_paper_suite_dry_run_writes_manifest_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(tmp_path)

    def fail_train(*args, **kwargs) -> RunResult:
        raise AssertionError("dry-run should not train")

    monkeypatch.setattr("sic4gridcells.paper_suite.train", fail_train)

    results = run_paper_suite(suite_config, dry_run=True)

    assert results["baseline_seed0"].status == "validated"
    assert results["disabled"].status == "skipped"
    suite_dir = tmp_path / "paper_suite" / "smoke"
    manifest = json.loads(
        (suite_dir / "manifest.json").read_text(encoding="utf-8"),
        parse_constant=_fail_parse_constant,
    )
    assert manifest["execution"]["dry_run"] is True
    assert manifest["runs"][0]["status"] == "validated"
    assert manifest["runs"][0]["checkpoint_path"] is None
    assert (suite_dir / "summary.csv").exists()
    assert load_config(tmp_path / "paper_suite" / "smoke" / "configs" / "baseline_seed0.yaml").seed == 0


def test_paper_suite_materializes_overrides_and_rejects_unknown_keys(tmp_path: Path) -> None:
    source_config = _write_training_config(tmp_path)
    plan = {
        "output_root": str(tmp_path / "paper_suite"),
        "run_id": "smoke",
        "runs": [
            {
                "name": "baseline",
                "config": str(source_config),
                "seed": 4,
                "overrides": {"model": {"n_units": 6}},
            }
        ],
    }

    runs = materialize_paper_suite_configs(plan)

    assert runs[0].seed == 4
    cfg = load_config(runs[0].config_path)
    assert cfg.seed == 4
    assert cfg.model.n_units == 6
    assert cfg.logging.detail_level == "detailed"
    assert cfg.output_dir == str(tmp_path / "paper_suite" / "smoke" / "runs" / "baseline" / "train")

    plan["runs"][0]["overrides"] = {"logging": {"detail_level": "standard"}}
    runs = materialize_paper_suite_configs(plan)
    assert load_config(runs[0].config_path).logging.detail_level == "standard"

    plan["runs"][0]["overrides"] = {"bad": True}
    with pytest.raises(ValueError, match="Unknown paper suite override key: bad"):
        materialize_paper_suite_configs(plan)

    plan["runs"][0]["overrides"] = {"logging": {"verbose": True}}
    with pytest.raises(ValueError, match="Unknown paper suite override key: logging.verbose"):
        materialize_paper_suite_configs(plan)


def test_paper_suite_real_execution_calls_pipeline_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(
        tmp_path,
        profile_enabled=True,
        evaluation_enabled=True,
        validation_enabled=True,
        analysis_enabled=True,
    )
    calls: list[str] = []

    def fake_profile_training_run(*args, **kwargs) -> SimpleNamespace:
        calls.append("profile")
        output_dir = tmp_path / "profile"
        output_dir.mkdir(exist_ok=True)
        return SimpleNamespace(output_dir=str(output_dir))

    def fake_train(config_path: str | Path, **kwargs) -> RunResult:
        calls.append("train")
        cfg = load_config(config_path)
        output_dir = Path(cfg.output_dir)
        checkpoint = output_dir / "checkpoints" / "step_1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return RunResult(output_dir=output_dir, final_step=1, checkpoint_path=checkpoint)

    def fake_evaluate_checkpoint(checkpoint_path: str | Path, output_dir: str | Path, **kwargs) -> EvaluationResult:
        calls.append("eval")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps({"arena_summaries": []}), encoding="utf-8")
        return EvaluationResult(output_dir=output, checkpoint_path=Path(checkpoint_path), arena_dirs={})

    def fake_validate_evaluation_output(*args, **kwargs) -> SimpleNamespace:
        calls.append("validate")
        return SimpleNamespace(passed=True, blocker_count=0)

    def fake_write_validation_report(report: SimpleNamespace, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"passed": report.passed}), encoding="utf-8")

    def fake_analyze_evaluation_output(*args, **kwargs) -> SimpleNamespace:
        calls.append("analysis")
        output_dir = Path(args[1])
        (output_dir / "summary_tables").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output_dir)

    monkeypatch.setattr("sic4gridcells.paper_suite.profile_training_run", fake_profile_training_run)
    monkeypatch.setattr("sic4gridcells.paper_suite.train", fake_train)
    monkeypatch.setattr("sic4gridcells.paper_suite.evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr("sic4gridcells.paper_suite.validate_evaluation_output", fake_validate_evaluation_output)
    monkeypatch.setattr("sic4gridcells.paper_suite.write_validation_report", fake_write_validation_report)
    monkeypatch.setattr("sic4gridcells.analysis_ext.analyze_evaluation_output", fake_analyze_evaluation_output)

    results = run_paper_suite(suite_config)

    assert calls == ["profile", "train", "eval", "validate", "analysis"]
    assert results["baseline_seed0"].status == "finished"
    suite_dir = tmp_path / "paper_suite" / "smoke"
    manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["validation_passed"] is True
    assert manifest["runs"][0]["analysis_output_dir"].endswith("/analysis")
    events = _load_jsonl(suite_dir / "paper_suite_events.jsonl")
    stage_pairs = {
        (row["event"], row.get("stage"))
        for row in events
        if row["event"].startswith("paper_suite_stage_")
    }
    for stage in ("profile", "train", "eval", "validation", "analysis"):
        assert ("paper_suite_stage_start", stage) in stage_pairs
        assert ("paper_suite_stage_finished", stage) in stage_pairs


def test_paper_suite_standard_logging_suppresses_stage_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(tmp_path, logging_detail_level="standard")

    def fake_train(config_path: str | Path, **kwargs) -> RunResult:
        cfg = load_config(config_path)
        checkpoint = Path(cfg.output_dir) / "checkpoints" / "step_1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return RunResult(
            output_dir=Path(cfg.output_dir),
            final_step=1,
            checkpoint_path=checkpoint,
        )

    monkeypatch.setattr("sic4gridcells.paper_suite.train", fake_train)

    run_paper_suite(suite_config)

    events = _load_jsonl(tmp_path / "paper_suite" / "smoke" / "paper_suite_events.jsonl")
    assert "paper_suite_run_finished" in {row["event"] for row in events}
    assert all(not row["event"].startswith("paper_suite_stage_") for row in events)


def test_paper_suite_does_not_build_figures_after_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(
        tmp_path,
        evaluation_enabled=True,
        validation_enabled=True,
    )

    def fake_train(config_path: str | Path, **kwargs) -> RunResult:
        cfg = load_config(config_path)
        output_dir = Path(cfg.output_dir)
        checkpoint = output_dir / "checkpoints" / "step_1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return RunResult(output_dir=output_dir, final_step=1, checkpoint_path=checkpoint)

    def fake_evaluate_checkpoint(checkpoint_path: str | Path, output_dir: str | Path, **kwargs) -> EvaluationResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return EvaluationResult(output_dir=output, checkpoint_path=Path(checkpoint_path), arena_dirs={})

    def fake_validate_evaluation_output(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(passed=False, blocker_count=1)

    def fake_write_validation_report(report: SimpleNamespace, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"passed": report.passed}), encoding="utf-8")

    def fail_build_figures(*args, **kwargs) -> None:
        raise AssertionError("figures must not build after validation failure")

    monkeypatch.setattr("sic4gridcells.paper_suite.train", fake_train)
    monkeypatch.setattr("sic4gridcells.paper_suite.evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr("sic4gridcells.paper_suite.validate_evaluation_output", fake_validate_evaluation_output)
    monkeypatch.setattr("sic4gridcells.paper_suite.write_validation_report", fake_write_validation_report)
    monkeypatch.setattr("sic4gridcells.paper_figures.build_paper_figures", fail_build_figures)

    with pytest.raises(RuntimeError, match="validation blockers"):
        run_paper_suite(suite_config)


def test_paper_suite_does_not_build_figures_when_continue_on_error_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(
        tmp_path,
        evaluation_enabled=True,
        validation_enabled=True,
        continue_on_error=True,
        figures_enabled=True,
    )

    def fake_train(config_path: str | Path, **kwargs) -> RunResult:
        cfg = load_config(config_path)
        output_dir = Path(cfg.output_dir)
        checkpoint = output_dir / "checkpoints" / "step_1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return RunResult(output_dir=output_dir, final_step=1, checkpoint_path=checkpoint)

    def fake_evaluate_checkpoint(checkpoint_path: str | Path, output_dir: str | Path, **kwargs) -> EvaluationResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return EvaluationResult(output_dir=output, checkpoint_path=Path(checkpoint_path), arena_dirs={})

    def fake_validate_evaluation_output(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(passed=False, blocker_count=1)

    def fake_write_validation_report(report: SimpleNamespace, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"passed": report.passed}), encoding="utf-8")

    def fail_build_figures(*args, **kwargs) -> None:
        raise AssertionError("figures must not build when any run failed")

    monkeypatch.setattr("sic4gridcells.paper_suite.train", fake_train)
    monkeypatch.setattr("sic4gridcells.paper_suite.evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr("sic4gridcells.paper_suite.validate_evaluation_output", fake_validate_evaluation_output)
    monkeypatch.setattr("sic4gridcells.paper_suite.write_validation_report", fake_write_validation_report)
    monkeypatch.setattr("sic4gridcells.paper_figures.build_paper_figures", fail_build_figures)

    results = run_paper_suite(suite_config)

    assert results["baseline_seed0"].status == "failed"
    suite_dir = tmp_path / "paper_suite" / "smoke"
    manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure_output_dir"] is None


def test_paper_suite_skip_completed_reuses_existing_analysis_for_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_config = _write_suite_config(
        tmp_path,
        evaluation_enabled=True,
        validation_enabled=True,
        analysis_enabled=True,
        figures_enabled=True,
    )
    run_root = tmp_path / "paper_suite" / "smoke" / "runs" / "baseline_seed0"
    train_dir = run_root / "train"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_1.pt").write_bytes(b"checkpoint")
    eval_dir = run_root / "eval"
    eval_dir.mkdir()
    (eval_dir / "summary.json").write_text(
        json.dumps({"arena_summaries": []}),
        encoding="utf-8",
    )
    validation_report = run_root / "validation_report.json"
    validation_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    analysis_dir = run_root / "analysis"
    (analysis_dir / "summary_tables").mkdir(parents=True)

    def fail_train(*args, **kwargs) -> RunResult:
        raise AssertionError("completed run should be skipped")

    figure_calls: list[tuple[Path, Path]] = []

    def fake_build_paper_figures(suite_dir: str | Path, output_dir: str | Path) -> SimpleNamespace:
        figure_calls.append((Path(suite_dir), Path(output_dir)))
        return SimpleNamespace(output_dir=Path(output_dir))

    monkeypatch.setattr("sic4gridcells.paper_suite.train", fail_train)
    monkeypatch.setattr("sic4gridcells.paper_figures.build_paper_figures", fake_build_paper_figures)

    results = run_paper_suite(suite_config, skip_completed=True)

    assert results["baseline_seed0"].status == "skipped"
    assert results["baseline_seed0"].analysis_output_dir == analysis_dir
    assert figure_calls == [
        (tmp_path / "paper_suite" / "smoke", tmp_path / "paper_suite" / "smoke" / "figures")
    ]
    manifest = json.loads(
        (tmp_path / "paper_suite" / "smoke" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runs"][0]["analysis_output_dir"] == str(analysis_dir)
    assert manifest["figure_output_dir"] == str(tmp_path / "paper_suite" / "smoke" / "figures")
    events = _load_jsonl(tmp_path / "paper_suite" / "smoke" / "paper_suite_events.jsonl")
    figure_events = [
        row
        for row in events
        if row["event"].startswith("paper_suite_stage_") and row.get("stage") == "figure"
    ]
    assert {row["event"] for row in figure_events} == {
        "paper_suite_stage_start",
        "paper_suite_stage_finished",
    }
    assert figure_events[0]["name"] is None


def test_run_paper_suite_cli_prints_validated_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_config = _write_suite_config(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_paper_suite.py", "--config", str(suite_config), "--dry-run"],
    )

    run_paper_suite_script.main()

    output = capsys.readouterr().out
    assert "validated baseline_seed0:" in output
    assert "skipped disabled: disabled for test" in output
    manifest = json.loads(
        (tmp_path / "paper_suite" / "smoke" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["command_line_args"]["log_level"] == "INFO"


def test_repo_paper_suite_smoke_config_loads() -> None:
    plan = load_paper_suite_plan("configs/paper_suite_smoke.yaml")

    assert plan["run_id"] == "smoke"
    assert plan["runs"][0]["config"] == "configs/smoke.yaml"


def _write_suite_config(
    tmp_path: Path,
    *,
    profile_enabled: bool = False,
    evaluation_enabled: bool = False,
    validation_enabled: bool = False,
    analysis_enabled: bool = False,
    figures_enabled: bool = False,
    continue_on_error: bool = False,
    logging_detail_level: str = "detailed",
) -> Path:
    source_config = _write_training_config(tmp_path, logging_detail_level=logging_detail_level)
    suite_config = tmp_path / "paper_suite.yaml"
    suite_config.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "paper_suite"),
                "run_id": "smoke",
                "config_dir": str(tmp_path / "paper_suite" / "smoke" / "configs"),
                "continue_on_error": continue_on_error,
                "runs": [
                    {
                        "name": "baseline_seed0",
                        "variant": "baseline",
                        "seed": 0,
                        "config": str(source_config),
                    },
                    {
                        "name": "disabled",
                        "enabled": False,
                        "reason": "disabled for test",
                        "variant": "baseline",
                        "seed": 1,
                        "config": str(source_config),
                    },
                ],
                "profile": {"enabled": profile_enabled, "steps": 1, "device": "cpu"},
                "evaluation": {
                    "enabled": evaluation_enabled,
                    "output_dir_name": "eval",
                    "device": "cpu",
                    "arena_sizes": [1.0],
                    "nbins": 4,
                    "trajectories": 1,
                    "steps": 2,
                    "start_mode": "origin",
                    "trajectory_mode": "reflect",
                },
                "validation": {
                    "enabled": validation_enabled,
                    "min_coverage": 0.0,
                    "min_active_units": 0,
                    "min_module_count": 0,
                    "arena_sizes": [1.0],
                },
                "analysis": {"enabled": analysis_enabled, "output_dir_name": "analysis"},
                "figures": {"enabled": figures_enabled},
            }
        ),
        encoding="utf-8",
    )
    return suite_config


def _write_training_config(
    tmp_path: Path,
    *,
    logging_detail_level: str = "detailed",
) -> Path:
    config_path = tmp_path / "train.yaml"
    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "train"),
        "data": {
            "batch_size": 2,
            "trajectory_length": 3,
            "velocity_low": -0.05,
            "velocity_high": 0.05,
            "initial_position_mode": "zero",
            "initial_position_low": 0.0,
            "initial_position_high": 0.0,
        },
        "model": {
            "n_units": 4,
            "mlp_layers": 1,
            "mlp_hidden_width": 8,
            "trainable_initial_state": True,
            "initial_position_encoding": "none",
            "initial_position_hidden_width": 4,
        },
        "loss": {
            "sigma_x": 0.05,
            "sigma_g": 0.4,
            "lambda_sep": 1.0,
            "lambda_inv": 0.1,
            "lambda_cap": 0.5,
            "lambda_coniso": 1.0,
            "pairwise_reduction": "mean",
            "chunk_size": 4,
        },
        "train": {
            "optimizer": "adamw",
            "scheduler": "reduce_on_plateau",
            "scheduler_monitor": "loss/total",
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "lr": 0.00002,
            "weight_decay": 0.0,
            "grad_clip_norm": 0.1,
            "accumulate_grad_batches": 1,
            "max_optimizer_steps": 1,
            "checkpoint_every": 1,
            "log_every": 1,
        },
        "logging": {"detail_level": logging_detail_level},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _load_jsonl(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    return [
        json.loads(line, parse_constant=_fail_parse_constant)
        for line in raw.splitlines()
        if line.strip()
    ]


def _fail_parse_constant(value: str) -> None:
    raise AssertionError(f"non-strict JSON constant: {value}")
