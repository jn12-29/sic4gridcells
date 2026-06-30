import json
import math
from pathlib import Path

import pytest
import yaml

import scripts.profile_train as profile_cli
from sic4gridcells.profiling import profile_training_run, summarize_profile_run


def test_profile_training_run_writes_summary(tmp_path: Path) -> None:
    config_path = _write_profile_config(
        tmp_path,
        max_optimizer_steps=5,
        checkpoint_every=3,
    )
    output_dir = tmp_path / "profile"

    summary = profile_training_run(
        config_path,
        output_dir,
        steps=2,
        device="cpu",
    )

    assert summary.final_step == 2
    assert summary.requested_steps == 2
    assert summary.metrics_rows >= 1
    assert summary.mean_step_seconds is not None
    assert summary.mean_step_seconds > 0
    assert summary.estimated_seconds_for_config_steps is not None
    assert math.isclose(
        summary.estimated_seconds_for_config_steps,
        summary.mean_step_seconds * 5,
    )
    assert summary.estimated_checkpoint_count == 2
    assert summary.estimated_checkpoint_storage_mb is not None
    assert summary.estimated_checkpoint_storage_mb > 0
    assert (output_dir / "profile_summary.json").exists()
    payload = json.loads((output_dir / "profile_summary.json").read_text(encoding="utf-8"))
    assert payload["final_step"] == 2
    assert payload["last_metrics"]["perf/step_seconds"] > 0
    effective = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    assert effective["train"]["max_optimizer_steps"] == 2


def test_summarize_profile_run_handles_missing_metrics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    summary = summarize_profile_run(
        output_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        requested_steps=3,
        target_steps=100,
        final_step=3,
        checkpoint_path=checkpoint,
        checkpoint_every=20,
    )

    assert summary.mean_step_seconds is None
    assert summary.estimated_hours_for_config_steps is None
    assert summary.checkpoint_size_mb is not None
    assert summary.estimated_checkpoint_count == 5


def test_profile_train_cli_prints_summary_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_profile_config(tmp_path, max_optimizer_steps=3)
    output_dir = tmp_path / "cli-profile"
    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_train.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
    )

    profile_cli.main()

    output = capsys.readouterr().out
    assert "profile finished step=1" in output
    assert f"profile_summary={output_dir / 'profile_summary.json'}" in output


def test_profile_training_run_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    config_path = _write_profile_config(tmp_path)
    output_dir = tmp_path / "profile-collision"
    output_dir.mkdir()
    (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        profile_training_run(config_path, output_dir, steps=1, device="cpu")

    summary = profile_training_run(
        config_path,
        output_dir,
        steps=1,
        device="cpu",
        overwrite_output=True,
    )
    assert summary.final_step == 1


def test_profile_training_run_rejects_non_positive_steps(tmp_path: Path) -> None:
    config_path = _write_profile_config(tmp_path)

    with pytest.raises(ValueError, match="steps must be positive"):
        profile_training_run(config_path, tmp_path / "profile", steps=0)


def _write_profile_config(
    tmp_path: Path,
    *,
    max_optimizer_steps: int = 4,
    checkpoint_every: int = 2,
) -> Path:
    config_path = tmp_path / "profile.yaml"
    config = {
        "seed": 9,
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
            "scheduler_patience": 2,
            "lr": 0.00002,
            "weight_decay": 0.0,
            "grad_clip_norm": 0.1,
            "accumulate_grad_batches": 1,
            "max_optimizer_steps": max_optimizer_steps,
            "checkpoint_every": checkpoint_every,
            "log_every": 1,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
