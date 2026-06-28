import json
import math
from io import StringIO
from pathlib import Path

import torch
import yaml

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
