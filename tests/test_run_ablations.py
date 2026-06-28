import json
from pathlib import Path

import pytest
import yaml

import scripts.run_ablations as run_ablations_script
from sic4gridcells.evaluate import EvaluationResult
from sic4gridcells.config import load_config
from sic4gridcells.train import RunResult


def test_repo_ablation_config_materializes_valid_training_configs(tmp_path: Path) -> None:
    plan = run_ablations_script.load_ablation_plan("configs/ablations.yaml")
    assert str(plan["config_dir"]).startswith("results/")
    plan["output_root"] = str(tmp_path / "runs")
    plan["config_dir"] = str(tmp_path / "configs")

    runs = run_ablations_script.materialize_run_configs(plan)

    assert [run.name for run in runs] == [
        "baseline",
        "no_capacity",
        "reduced_sigma_g",
        "no_separation",
        "no_invariance",
        "no_conformal_isometry",
        "no_permutation_augmentation",
    ]
    for run in runs:
        cfg = load_config(run.config_path)
        assert cfg.output_dir == str(tmp_path / "runs" / run.name)
        assert cfg.data.batch_size == 16
        assert cfg.data.trajectory_length == 30
        assert cfg.model.n_units == 64
    assert load_config(tmp_path / "configs" / "no_capacity.yaml").loss.lambda_cap == 0.0
    assert load_config(tmp_path / "configs" / "reduced_sigma_g.yaml").loss.sigma_g == 0.2
    assert load_config(tmp_path / "configs" / "no_separation.yaml").loss.lambda_sep == 0.0
    assert load_config(tmp_path / "configs" / "no_invariance.yaml").loss.lambda_inv == 0.0
    assert (
        load_config(tmp_path / "configs" / "no_conformal_isometry.yaml").loss.lambda_coniso
        == 0.0
    )
    no_permutation = next(run for run in runs if run.name == "no_permutation_augmentation")
    assert no_permutation.enabled is True
    assert no_permutation.reason is None
    assert load_config(no_permutation.config_path).data.augmentation_mode == "identity"


def test_run_ablations_invokes_train_for_enabled_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ablation_path = tmp_path / "ablations.yaml"
    ablation_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "runs"),
                "config_dir": str(tmp_path / "configs"),
                "base": _tiny_base_config(tmp_path / "base"),
                "variants": [
                    {"name": "baseline", "overrides": {}},
                    {
                        "name": "no_capacity",
                        "overrides": {"loss": {"lambda_cap": 0.0}},
                    },
                    {
                        "name": "declared_only",
                        "enabled": False,
                        "reason": "not supported by current config schema",
                        "overrides": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_train(config_path: str | Path) -> RunResult:
        path = Path(config_path)
        calls.append(path)
        cfg = load_config(path)
        return RunResult(
            output_dir=Path(cfg.output_dir),
            final_step=cfg.train.max_optimizer_steps,
            checkpoint_path=Path(cfg.output_dir) / "checkpoints" / "step_1.pt",
        )

    monkeypatch.setattr(run_ablations_script, "train", fake_train)

    results = run_ablations_script.run_ablations(ablation_path)

    assert list(results) == ["baseline", "no_capacity", "declared_only"]
    assert [path.name for path in calls] == ["baseline.yaml", "no_capacity.yaml"]
    assert results["baseline"].status == "finished"
    assert results["baseline"].run_result is not None
    assert results["baseline"].run_result.final_step == 1
    assert results["declared_only"].status == "skipped"
    assert results["declared_only"].reason == "not supported by current config schema"
    assert load_config(tmp_path / "configs" / "no_capacity.yaml").loss.lambda_cap == 0.0
    assert (tmp_path / "runs" / "summary.json").exists()
    assert (tmp_path / "runs" / "summary.csv").exists()


def test_run_ablations_dry_run_validates_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ablation_path = tmp_path / "ablations.yaml"
    ablation_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "runs"),
                "config_dir": str(tmp_path / "configs"),
                "base": _tiny_base_config(tmp_path / "base"),
                "variants": [{"name": "baseline", "overrides": {}}],
            }
        ),
        encoding="utf-8",
    )

    def fail_train(config_path: str | Path) -> RunResult:
        raise AssertionError(f"dry-run should not train {config_path}")

    monkeypatch.setattr(run_ablations_script, "train", fail_train)

    results = run_ablations_script.run_ablations(ablation_path, dry_run=True)

    assert results["baseline"].status == "validated"
    assert results["baseline"].run_result is None
    assert load_config(tmp_path / "configs" / "baseline.yaml").train.max_optimizer_steps == 1
    assert (tmp_path / "runs" / "summary.json").exists()


def test_main_dry_run_does_not_report_validated_runs_as_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ablation_path = tmp_path / "ablations.yaml"
    ablation_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "runs"),
                "config_dir": str(tmp_path / "configs"),
                "base": _tiny_base_config(tmp_path / "base"),
                "variants": [
                    {"name": "baseline", "overrides": {}},
                    {
                        "name": "declared_only",
                        "enabled": False,
                        "reason": "not supported by current config schema",
                        "overrides": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_ablations.py", "--config", str(ablation_path), "--dry-run"],
    )

    run_ablations_script.main()

    output = capsys.readouterr().out
    assert "validated baseline:" in output
    assert "skipped baseline" not in output
    assert "skipped declared_only: not supported by current config schema" in output


def test_ablation_override_can_target_defaulted_config_fields(tmp_path: Path) -> None:
    base = _tiny_base_config(tmp_path / "base")
    del base["data"]["initial_position_mode"]
    del base["data"]["initial_position_low"]
    del base["data"]["initial_position_high"]
    del base["model"]["initial_position_encoding"]
    del base["model"]["initial_position_hidden_width"]
    plan = {
        "output_root": str(tmp_path / "runs"),
        "config_dir": str(tmp_path / "configs"),
        "base": base,
        "variants": [
            {
                "name": "position_encoder",
                "overrides": {
                    "data": {
                        "initial_position_mode": "uniform_box",
                        "initial_position_low": -0.5,
                        "initial_position_high": 0.5,
                    },
                    "model": {
                        "initial_position_encoding": "additive_mlp",
                        "initial_position_hidden_width": 8,
                    },
                },
            }
        ],
    }

    run_ablations_script.materialize_run_configs(plan)
    cfg = load_config(tmp_path / "configs" / "position_encoder.yaml")

    assert cfg.data.initial_position_mode == "uniform_box"
    assert cfg.data.initial_position_low == -0.5
    assert cfg.data.initial_position_high == 0.5
    assert cfg.model.initial_position_encoding == "additive_mlp"
    assert cfg.model.initial_position_hidden_width == 8


def test_run_ablations_can_evaluate_and_summarize_finished_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ablation_path = tmp_path / "ablations.yaml"
    ablation_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "runs"),
                "config_dir": str(tmp_path / "configs"),
                "evaluation": {
                    "enabled": True,
                    "output_dir_name": "eval",
                    "device": "cpu",
                    "arena_sizes": [1.0],
                    "nbins": 4,
                    "trajectories": 1,
                    "steps": 2,
                    "seed": 7,
                },
                "base": _tiny_base_config(tmp_path / "base"),
                "variants": [{"name": "baseline", "overrides": {}}],
            }
        ),
        encoding="utf-8",
    )

    def fake_train(config_path: str | Path) -> RunResult:
        cfg = load_config(config_path)
        output_dir = Path(cfg.output_dir)
        return RunResult(
            output_dir=output_dir,
            final_step=1,
            checkpoint_path=output_dir / "checkpoints" / "step_1.pt",
        )

    def fake_evaluate_checkpoint(*args, **kwargs) -> EvaluationResult:
        output_dir = Path(args[1])
        output_dir.mkdir(parents=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "arena_summaries": [
                        {
                            "arena_size": 1.0,
                            "mean_grid_score_60": 0.25,
                            "mean_scale_meters": 0.5,
                            "detected_modules": 2,
                            "active_units": 3,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return EvaluationResult(output_dir=output_dir, checkpoint_path=Path(args[0]), arena_dirs={1.0: output_dir / "arena_1p0"})

    monkeypatch.setattr(run_ablations_script, "train", fake_train)
    monkeypatch.setattr(run_ablations_script, "evaluate_checkpoint", fake_evaluate_checkpoint)

    results = run_ablations_script.run_ablations(ablation_path)

    assert results["baseline"].evaluation_result is not None
    summary = yaml.safe_load((tmp_path / "runs" / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["evaluation_output_dir"] == str(tmp_path / "runs" / "baseline" / "eval")
    assert summary[0]["arena_summaries"][0]["detected_modules"] == 2


def test_unknown_ablation_override_key_fails(tmp_path: Path) -> None:
    plan = {
        "output_root": str(tmp_path / "runs"),
        "config_dir": str(tmp_path / "configs"),
        "base": _tiny_base_config(tmp_path / "base"),
        "variants": [{"name": "bad", "overrides": {"unknown": True}}],
    }

    with pytest.raises(ValueError, match="Unknown ablation override key: unknown"):
        run_ablations_script.materialize_run_configs(plan)


def test_unknown_nested_ablation_override_key_fails(tmp_path: Path) -> None:
    plan = {
        "output_root": str(tmp_path / "runs"),
        "config_dir": str(tmp_path / "configs"),
        "base": _tiny_base_config(tmp_path / "base"),
        "variants": [{"name": "bad", "overrides": {"model": {"not_a_field": True}}}],
    }

    with pytest.raises(ValueError, match="Unknown ablation override key: model.not_a_field"):
        run_ablations_script.materialize_run_configs(plan)


def _tiny_base_config(output_dir: Path) -> dict:
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(output_dir),
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
            "mlp_layers": 2,
            "mlp_hidden_width": 8,
            "trainable_initial_state": True,
            "initial_position_encoding": "none",
            "initial_position_hidden_width": 8,
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
    }
