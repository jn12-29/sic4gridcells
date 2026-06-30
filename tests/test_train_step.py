import json
import math
from io import StringIO
from pathlib import Path

import torch
import pytest
import yaml

import scripts.train_sic as train_cli
from sic4gridcells.train import train, _write_metrics


def test_train_smoke_writes_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "run"
    config = {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(output_dir),
        "data": {
            "batch_size": 3,
            "trajectory_length": 4,
            "velocity_low": -0.05,
            "velocity_high": 0.05,
            "initial_position_mode": "zero",
            "initial_position_low": 0.0,
            "initial_position_high": 0.0,
        },
        "model": {
            "n_units": 8,
            "mlp_layers": 2,
            "mlp_hidden_width": 16,
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
            "scheduler_patience": 2,
            "lr": 0.00002,
            "weight_decay": 0.0,
            "grad_clip_norm": 0.1,
            "accumulate_grad_batches": 2,
            "max_optimizer_steps": 2,
            "checkpoint_every": 1,
            "log_every": 1,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train(config_path)

    assert result.final_step == 2
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "run.log").exists()
    assert (output_dir / "train_events.jsonl").exists()
    assert (output_dir / "checkpoints" / "step_2.pt").exists()
    assert any((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    checkpoint = torch.load(output_dir / "checkpoints" / "step_2.pt", map_location="cpu")
    assert checkpoint["step"] == 2
    assert checkpoint["config"]["train"]["max_optimizer_steps"] == 2
    assert "model_state_dict" in checkpoint
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert rows[-1]["step"] == 2.0
    expected_metrics = {
        "loss/total",
        "loss/separation",
        "loss/invariance",
        "loss/capacity",
        "loss/conformal_isometry",
        "lr",
        "grad_norm",
        "stats/zero_norm_fraction",
        "stats/separation_pairs",
        "stats/invariance_pairs",
        "stats/conformal_isometry_steps",
    }
    assert expected_metrics <= set(rows[-1])
    for key in expected_metrics:
        assert math.isfinite(rows[-1][key])
    assert rows[-1]["stats/separation_pairs"] >= 0
    events = _load_jsonl(output_dir / "train_events.jsonl")
    assert {"train_start", "train_config_saved", "tensorboard_started", "checkpoint_saved", "tensorboard_closed", "train_finished"} <= {row["event"] for row in events}
    assert all("timestamp" in row for row in events)


def test_train_logs_on_requested_cadence(tmp_path: Path) -> None:
    config_path = tmp_path / "cadence.yaml"
    output_dir = tmp_path / "cadence-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=5)
    config["train"]["log_every"] = 2
    config["train"]["checkpoint_every"] = 5
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train(config_path)

    metric_rows = _load_jsonl(output_dir / "metrics.jsonl")
    event_rows = _load_jsonl(output_dir / "train_events.jsonl")
    assert [row["step"] for row in metric_rows] == [1.0, 2.0, 4.0, 5.0]
    assert [row["step"] for row in event_rows if row["event"] == "train_metrics"] == [1, 2, 4, 5]


def test_fresh_early_failure_replaces_previous_event_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "early-failure.yaml"
    output_dir = tmp_path / "early-failure-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train(config_path)

    def fail_resolve_device(device: str) -> torch.device:
        raise RuntimeError(f"cannot resolve {device}")

    monkeypatch.setattr("sic4gridcells.train.resolve_device", fail_resolve_device)

    with pytest.raises(RuntimeError, match="cannot resolve"):
        train(config_path)

    events = _load_jsonl(output_dir / "train_events.jsonl")
    assert [row["event"] for row in events] == ["train_failed"]
    assert events[0]["error_type"] == "RuntimeError"


def test_train_with_initial_position_encoder(tmp_path: Path) -> None:
    config_path = tmp_path / "position-encoder.yaml"
    output_dir = tmp_path / "position-run"
    config = {
        "seed": 8,
        "device": "cpu",
        "output_dir": str(output_dir),
        "data": {
            "batch_size": 2,
            "trajectory_length": 3,
            "velocity_low": -0.05,
            "velocity_high": 0.05,
            "initial_position_mode": "uniform_box",
            "initial_position_low": -0.5,
            "initial_position_high": 0.5,
        },
        "model": {
            "n_units": 4,
            "mlp_layers": 1,
            "mlp_hidden_width": 8,
            "trainable_initial_state": True,
            "initial_position_encoding": "additive_mlp",
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
            "chunk_size": 2,
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
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train(config_path)

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["data"]["initial_position_mode"] == "uniform_box"
    assert checkpoint["config"]["model"]["initial_position_encoding"] == "additive_mlp"


def test_train_can_resume_from_checkpoint_to_larger_max_step(tmp_path: Path) -> None:
    config_path = tmp_path / "resume.yaml"
    output_dir = tmp_path / "resume-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    first_result = train(config_path)

    config["train"]["max_optimizer_steps"] = 3
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    resumed_result = train(config_path, resume_checkpoint=first_result.checkpoint_path)

    assert resumed_result.final_step == 3
    assert resumed_result.checkpoint_path == output_dir / "checkpoints" / "step_3.pt"
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1.0, 2.0, 3.0]
    events = _load_jsonl(output_dir / "train_events.jsonl")
    assert [row["step"] for row in events if row["event"] == "train_metrics"] == [1, 2, 3]
    assert any(row["event"] == "train_resume_loaded" for row in events)
    checkpoint = torch.load(resumed_result.checkpoint_path, map_location="cpu")
    assert checkpoint["step"] == 3
    assert "generator_state" in checkpoint


def test_resume_rejects_non_step_config_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "resume.yaml"
    output_dir = tmp_path / "resume-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first_result = train(config_path)

    config["train"]["max_optimizer_steps"] = 2
    config["loss"]["lambda_sep"] = 0.0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="loss.lambda_sep"):
        train(config_path, resume_checkpoint=first_result.checkpoint_path)
    effective_config = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    assert effective_config["loss"]["lambda_sep"] == 1.0


def test_resume_rejects_lower_max_optimizer_steps(tmp_path: Path) -> None:
    config_path = tmp_path / "resume.yaml"
    output_dir = tmp_path / "resume-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=2)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first_result = train(config_path)

    config["train"]["max_optimizer_steps"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="train.max_optimizer_steps cannot decrease"):
        train(config_path, resume_checkpoint=first_result.checkpoint_path)


def test_resume_trims_metrics_to_checkpoint_step(tmp_path: Path) -> None:
    config_path = tmp_path / "resume.yaml"
    output_dir = tmp_path / "resume-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=3)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train(config_path)

    step_1_checkpoint = output_dir / "checkpoints" / "step_1.pt"
    resumed_result = train(config_path, resume_checkpoint=step_1_checkpoint)

    assert resumed_result.final_step == 3
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1.0, 2.0, 3.0]
    events = _load_jsonl(output_dir / "train_events.jsonl")
    assert [row["step"] for row in events if row["event"] == "train_metrics"] == [1, 2, 3]


def test_noop_resume_trims_metrics_to_checkpoint_step(tmp_path: Path) -> None:
    config_path = tmp_path / "resume.yaml"
    output_dir = tmp_path / "resume-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first_result = train(config_path)

    config["train"]["max_optimizer_steps"] = 3
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train(config_path, resume_checkpoint=first_result.checkpoint_path)

    config["train"]["max_optimizer_steps"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = train(config_path, resume_checkpoint=first_result.checkpoint_path)

    assert result.final_step == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1.0]
    events = _load_jsonl(output_dir / "train_events.jsonl")
    finished_events = [row for row in events if row["event"] == "train_finished"]
    assert finished_events[-1]["status"] == "already_complete"
    assert finished_events[-1]["final_step"] == 1
    assert [row["step"] for row in events if row["event"] == "train_metrics"] == [1]
    assert [row["step"] for row in events if row["event"] == "checkpoint_saved"] == [1]
    assert len([row for row in events if row["event"] == "train_resume_loaded"]) == 1
    assert len([row for row in events if row["event"] == "train_start"]) == 2
    assert len([row for row in events if row["event"] == "tensorboard_started"]) == 1
    assert len([row for row in events if row["event"] == "tensorboard_closed"]) == 1


def test_write_metrics_serializes_non_finite_values_as_strict_json() -> None:
    metrics_file = StringIO()
    writer = _DummyWriter()

    _write_metrics(
        metrics_file,
        writer,
        3,
        {
            "step": 3.0,
            "loss/total": float("nan"),
            "lr": 0.1,
            "grad_norm": float("inf"),
        },
    )

    raw = metrics_file.getvalue()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    row = json.loads(raw)
    assert row["loss/total"] is None
    assert row["grad_norm"] is None
    assert row["lr"] == 0.1
    assert writer.scalars == [("lr", 0.1, 3)]


class _DummyWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, value, step))


def _tiny_training_config(output_dir: Path, max_optimizer_steps: int) -> dict:
    return {
        "seed": 11,
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
            "chunk_size": 2,
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
            "max_optimizer_steps": max_optimizer_steps,
            "checkpoint_every": 1,
            "log_every": 1,
        },
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_train_cli_keeps_stdout_completion_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "cli.yaml"
    output_dir = tmp_path / "cli-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["train_sic.py", "--config", str(config_path), "--log-level", "INFO"],
    )

    train_cli.main()

    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith("finished step=1 output_dir=")
    assert output[1].startswith("checkpoint=")


def test_train_cli_log_level_debug_reaches_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "cli-debug.yaml"
    output_dir = tmp_path / "cli-debug-run"
    config = _tiny_training_config(output_dir, max_optimizer_steps=1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["train_sic.py", "--config", str(config_path), "--log-level", "DEBUG"],
    )

    train_cli.main()

    stderr = capsys.readouterr().err
    assert "DEBUG sic4gridcells.train: training metrics step=1" in stderr
