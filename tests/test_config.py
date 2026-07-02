from pathlib import Path

import pytest
import yaml

from sic4gridcells.config import Config, load_config, save_effective_config


def test_load_smoke_config() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.data.batch_size == 4
    assert cfg.data.trajectory_length == 8
    assert cfg.data.augmentation_mode == "permutation"
    assert cfg.data.initial_position_mode == "zero"
    assert cfg.model.n_units == 16
    assert cfg.model.initial_position_encoding == "none"
    assert cfg.loss.pairwise_reduction == "mean"
    assert cfg.train.max_optimizer_steps == 10
    assert cfg.logging.detail_level == "detailed"


def test_load_paper_config_records_assumptions() -> None:
    cfg = load_config("configs/sic_paper.yaml")
    assert cfg.data.batch_size == 130
    assert cfg.data.trajectory_length == 60
    assert cfg.loss.pairwise_reduction == "sum"
    assert any("lambda_coniso" in item for item in cfg.assumptions)
    assert any("mlp_hidden_width" in item for item in cfg.assumptions)


def test_load_medium_config_is_sanity_scale() -> None:
    cfg = load_config("configs/medium.yaml")
    assert cfg.data.batch_size == 16
    assert cfg.data.trajectory_length == 30
    assert cfg.model.n_units == 64
    assert cfg.model.mlp_hidden_width == 128
    assert cfg.loss.pairwise_reduction == "mean"
    assert cfg.train.max_optimizer_steps == 5000
    assert cfg.train.max_optimizer_steps < load_config("configs/sic_paper.yaml").train.max_optimizer_steps


def test_unknown_config_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(path)


def test_logging_detail_level_defaults_to_detailed() -> None:
    assert Config().logging.detail_level == "detailed"
    assert load_config().logging.detail_level == "detailed"


def test_logging_detail_level_standard_loads(tmp_path: Path) -> None:
    path = tmp_path / "standard-logging.yaml"
    path.write_text(
        yaml.safe_dump({"logging": {"detail_level": "standard"}}),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.logging.detail_level == "standard"


def test_invalid_logging_detail_level_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-logging.yaml"
    path.write_text(
        yaml.safe_dump({"logging": {"detail_level": "verbose"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="logging.detail_level"):
        load_config(path)


def test_invalid_initial_position_modes_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad-initial-position.yaml"
    path.write_text(
        yaml.safe_dump({"data": {"initial_position_mode": "random"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.initial_position_mode"):
        load_config(path)


def test_invalid_augmentation_mode_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-augmentation.yaml"
    path.write_text(
        yaml.safe_dump({"data": {"augmentation_mode": "shuffle_once"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.augmentation_mode"):
        load_config(path)


def test_invalid_initial_position_encoder_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-initial-position-encoder.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"initial_position_encoding": "bad"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.initial_position_encoding"):
        load_config(path)


def test_invalid_scheduler_monitor_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-scheduler-monitor.yaml"
    path.write_text(
        yaml.safe_dump({"train": {"scheduler_monitor": "stats/zero_norm_fraction"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.scheduler_monitor"):
        load_config(path)


def test_scheduler_options_load(tmp_path: Path) -> None:
    for scheduler in ("none", "reduce_on_plateau", "cosine"):
        path = tmp_path / f"{scheduler}.yaml"
        path.write_text(
            yaml.safe_dump({"train": {"scheduler": scheduler, "min_lr": 0.000001}}),
            encoding="utf-8",
        )
        cfg = load_config(path)
        assert cfg.train.scheduler == scheduler
        assert cfg.train.min_lr == 0.000001


def test_invalid_scheduler_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-scheduler.yaml"
    path.write_text(
        yaml.safe_dump({"train": {"scheduler": "linear"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.scheduler"):
        load_config(path)


def test_invalid_min_lr_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad-min-lr.yaml"
    path.write_text(
        yaml.safe_dump({"train": {"lr": 0.00002, "min_lr": 0.00003}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.min_lr"):
        load_config(path)


def test_save_effective_config_roundtrip(tmp_path: Path) -> None:
    cfg = Config()
    assert cfg.output_dir == "results/smoke"
    output = tmp_path / "config.yaml"
    save_effective_config(cfg, output)
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded["data"]["batch_size"] == 130
    assert loaded["logging"]["detail_level"] == "detailed"
    assert "assumptions" in loaded
