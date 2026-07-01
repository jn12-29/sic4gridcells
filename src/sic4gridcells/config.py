from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml


@dataclass
class DataConfig:
    batch_size: int = 130
    trajectory_length: int = 60
    velocity_low: float = -0.15
    velocity_high: float = 0.15
    augmentation_mode: str = "permutation"
    initial_position_mode: str = "zero"
    initial_position_low: float = 0.0
    initial_position_high: float = 0.0


@dataclass
class ModelConfig:
    n_units: int = 128
    mlp_layers: int = 3
    mlp_hidden_width: int = 256
    trainable_initial_state: bool = True
    initial_position_encoding: str = "none"
    initial_position_hidden_width: int = 64


@dataclass
class LossConfig:
    sigma_x: float = 0.05
    sigma_g: float = 0.4
    lambda_sep: float = 1.0
    lambda_inv: float = 0.1
    lambda_cap: float = 0.5
    lambda_coniso: float = 1.0
    pairwise_reduction: str = "sum"
    chunk_size: int = 512


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    scheduler: str = "reduce_on_plateau"
    scheduler_monitor: str = "loss/total"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1000
    lr: float = 0.00002
    min_lr: float = 0.0
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.1
    accumulate_grad_batches: int = 2
    max_optimizer_steps: int = 2000000
    checkpoint_every: int = 1000
    log_every: int = 10


DEFAULT_ASSUMPTIONS = [
    "loss.lambda_coniso defaults to 1.0 because the paper appendix table omits it.",
    "model.mlp_hidden_width defaults to 256 because the paper source omits it.",
    "model.trainable_initial_state defaults to true because the paper only states shared g0.",
    "data.augmentation_mode defaults to permutation because SIC uses random velocity permutations.",
    "model.initial_position_encoding defaults to none to preserve the paper-style shared g0 baseline.",
    "data.initial_position_mode defaults to zero to preserve the paper-style shared origin baseline.",
    "train.scheduler_factor defaults to 0.5 because the paper source omits it.",
    "train.scheduler_patience defaults to 1000 optimizer steps because the paper source omits it.",
    "NormReLU maps all-zero post-ReLU vectors to all-zero vectors.",
]

VALID_SCHEDULER_MONITORS = {
    "loss/total",
    "loss/separation",
    "loss/invariance",
    "loss/capacity",
    "loss/conformal_isometry",
}


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "results/smoke"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    assumptions: list[str] = field(default_factory=lambda: list(DEFAULT_ASSUMPTIONS))


T = TypeVar("T")


def load_config(path: str | Path | None = None) -> Config:
    config_dict = asdict(Config())
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            overrides = yaml.safe_load(handle) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        _deep_update(config_dict, overrides)
    cfg = _config_from_dict(config_dict)
    validate_config(cfg)
    return cfg


def load_config_from_dict(data: dict[str, Any]) -> Config:
    config_dict = asdict(Config())
    _deep_update(config_dict, data)
    cfg = _config_from_dict(config_dict)
    validate_config(cfg)
    return cfg


def save_effective_config(cfg: Config, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, sort_keys=False)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in {"cpu", "cuda"}:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Config requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device(device)
    raise ValueError(f"Unsupported device: {device}")


def validate_config(cfg: Config) -> None:
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if cfg.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if cfg.data.trajectory_length <= 0:
        raise ValueError("data.trajectory_length must be positive")
    if cfg.data.velocity_low >= cfg.data.velocity_high:
        raise ValueError("data.velocity_low must be less than data.velocity_high")
    if cfg.data.augmentation_mode not in {"permutation", "identity"}:
        raise ValueError("data.augmentation_mode must be 'permutation' or 'identity'")
    if cfg.data.initial_position_mode not in {"zero", "uniform_box"}:
        raise ValueError("data.initial_position_mode must be 'zero' or 'uniform_box'")
    if cfg.data.initial_position_mode == "uniform_box":
        if cfg.data.initial_position_low >= cfg.data.initial_position_high:
            raise ValueError(
                "data.initial_position_low must be less than initial_position_high "
                "when data.initial_position_mode='uniform_box'"
            )
    if cfg.model.n_units <= 0:
        raise ValueError("model.n_units must be positive")
    if cfg.model.mlp_layers <= 0:
        raise ValueError("model.mlp_layers must be positive")
    if cfg.model.mlp_hidden_width <= 0:
        raise ValueError("model.mlp_hidden_width must be positive")
    if cfg.model.initial_position_encoding not in {"none", "additive_mlp"}:
        raise ValueError("model.initial_position_encoding must be 'none' or 'additive_mlp'")
    if cfg.model.initial_position_hidden_width <= 0:
        raise ValueError("model.initial_position_hidden_width must be positive")
    if cfg.loss.sigma_x <= 0:
        raise ValueError("loss.sigma_x must be positive")
    if cfg.loss.sigma_g <= 0:
        raise ValueError("loss.sigma_g must be positive")
    if cfg.loss.pairwise_reduction not in {"sum", "mean"}:
        raise ValueError("loss.pairwise_reduction must be 'sum' or 'mean'")
    if cfg.loss.chunk_size <= 0:
        raise ValueError("loss.chunk_size must be positive")
    if cfg.train.optimizer != "adamw":
        raise ValueError("Only train.optimizer='adamw' is implemented")
    if cfg.train.scheduler not in {"none", "reduce_on_plateau", "cosine"}:
        raise ValueError("train.scheduler must be one of: none, reduce_on_plateau, cosine")
    if cfg.train.scheduler_monitor not in VALID_SCHEDULER_MONITORS:
        raise ValueError(
            "train.scheduler_monitor must be one of: "
            + ", ".join(sorted(VALID_SCHEDULER_MONITORS))
        )
    if cfg.train.scheduler_factor <= 0 or cfg.train.scheduler_factor >= 1:
        raise ValueError("train.scheduler_factor must be in (0, 1)")
    if cfg.train.scheduler_patience < 0:
        raise ValueError("train.scheduler_patience must be non-negative")
    if cfg.train.lr <= 0:
        raise ValueError("train.lr must be positive")
    if cfg.train.min_lr < 0:
        raise ValueError("train.min_lr must be non-negative")
    if cfg.train.min_lr > cfg.train.lr:
        raise ValueError("train.min_lr must be less than or equal to train.lr")
    if cfg.train.weight_decay < 0:
        raise ValueError("train.weight_decay must be non-negative")
    if cfg.train.grad_clip_norm <= 0:
        raise ValueError("train.grad_clip_norm must be positive")
    if cfg.train.accumulate_grad_batches <= 0:
        raise ValueError("train.accumulate_grad_batches must be positive")
    if cfg.train.max_optimizer_steps <= 0:
        raise ValueError("train.max_optimizer_steps must be positive")
    if cfg.train.checkpoint_every <= 0:
        raise ValueError("train.checkpoint_every must be positive")
    if cfg.train.log_every <= 0:
        raise ValueError("train.log_every must be positive")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _config_from_dict(data: dict[str, Any]) -> Config:
    return Config(
        seed=int(data["seed"]),
        device=str(data["device"]),
        output_dir=str(data["output_dir"]),
        data=_dataclass_from_dict(DataConfig, data["data"]),
        model=_dataclass_from_dict(ModelConfig, data["model"]),
        loss=_dataclass_from_dict(LossConfig, data["loss"]),
        train=_dataclass_from_dict(TrainConfig, data["train"]),
        assumptions=[str(item) for item in data["assumptions"]],
    )


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls}")
    allowed = {field.name for field in fields(cls)}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(extra)}")
    return cls(**data)
