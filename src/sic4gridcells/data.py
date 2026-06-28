from __future__ import annotations

from dataclasses import dataclass

import torch

from sic4gridcells.config import Config, DataConfig


@dataclass(frozen=True)
class SicBatch:
    base_velocities: torch.Tensor
    permutations: torch.Tensor
    initial_positions: torch.Tensor
    velocities: torch.Tensor
    positions: torch.Tensor


def sample_base_velocities(
    cfg: Config | DataConfig,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    data_cfg = _data_cfg(cfg)
    values = torch.empty(data_cfg.trajectory_length, 2)
    values.uniform_(data_cfg.velocity_low, data_cfg.velocity_high, generator=generator)
    return values.to(device)


def sample_permutations(
    batch_size: int,
    trajectory_length: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    perms = [torch.randperm(trajectory_length, generator=generator) for _ in range(batch_size)]
    return torch.stack(perms, dim=0).to(device)


def make_sic_batch(
    cfg: Config | DataConfig,
    generator: torch.Generator,
    device: torch.device | str,
) -> SicBatch:
    data_cfg = _data_cfg(cfg)
    base_velocities = sample_base_velocities(data_cfg, generator, device)
    permutations = sample_permutations(
        data_cfg.batch_size,
        data_cfg.trajectory_length,
        generator,
        device,
    )
    initial_positions = sample_initial_positions(data_cfg, generator, device).expand(
        data_cfg.batch_size,
        -1,
    )
    velocities = base_velocities[permutations]
    positions = initial_positions.unsqueeze(1) + velocities.cumsum(dim=1)
    return SicBatch(
        base_velocities=base_velocities,
        permutations=permutations,
        initial_positions=initial_positions,
        velocities=velocities,
        positions=positions,
    )


def sample_initial_positions(
    cfg: Config | DataConfig,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    data_cfg = _data_cfg(cfg)
    if data_cfg.initial_position_mode == "zero":
        return torch.zeros(2, device=device)
    if data_cfg.initial_position_mode == "uniform_box":
        values = torch.empty(2)
        values.uniform_(
            data_cfg.initial_position_low,
            data_cfg.initial_position_high,
            generator=generator,
        )
        return values.to(device)
    raise ValueError(f"Unsupported initial_position_mode: {data_cfg.initial_position_mode}")


def _data_cfg(cfg: Config | DataConfig) -> DataConfig:
    return cfg.data if isinstance(cfg, Config) else cfg
