from pathlib import Path

import pytest
import yaml

from sic4gridcells.config import Config, load_config, save_effective_config


def test_load_smoke_config() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.data.batch_size == 4
    assert cfg.data.trajectory_length == 8
    assert cfg.model.n_units == 16
    assert cfg.loss.pairwise_reduction == "mean"
    assert cfg.train.max_optimizer_steps == 10


def test_load_paper_config_records_assumptions() -> None:
    cfg = load_config("configs/sic_paper.yaml")
    assert cfg.data.batch_size == 130
    assert cfg.data.trajectory_length == 60
    assert cfg.loss.pairwise_reduction == "sum"
    assert any("lambda_coniso" in item for item in cfg.assumptions)
    assert any("mlp_hidden_width" in item for item in cfg.assumptions)


def test_unknown_config_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(path)


def test_save_effective_config_roundtrip(tmp_path: Path) -> None:
    cfg = Config()
    assert cfg.output_dir == "results/smoke"
    output = tmp_path / "config.yaml"
    save_effective_config(cfg, output)
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded["data"]["batch_size"] == 130
    assert "assumptions" in loaded
