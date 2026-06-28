from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sic4gridcells.config import Config, ModelConfig


@dataclass(frozen=True)
class RNNRollout:
    initial_state: torch.Tensor
    hidden_states: torch.Tensor
    zero_norm_fraction: torch.Tensor


def norm_relu(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = torch.relu(x)
    norms = y.norm(dim=-1, keepdim=True)
    return torch.where(norms > eps, y / norms.clamp_min(eps), torch.zeros_like(y))


class VelocityConditionedRNN(nn.Module):
    def __init__(self, cfg: Config | ModelConfig):
        super().__init__()
        model_cfg = cfg.model if isinstance(cfg, Config) else cfg
        self.n_units = model_cfg.n_units
        self.initial_position_encoding = model_cfg.initial_position_encoding
        self.transition_mlp = self._build_mlp(model_cfg)
        self.position_encoder = self._build_position_encoder(model_cfg)
        initial_state = torch.randn(model_cfg.n_units)
        if model_cfg.trainable_initial_state:
            self.g0_raw = nn.Parameter(initial_state)
        else:
            self.register_buffer("g0_raw", initial_state)

    def forward(
        self,
        velocities: torch.Tensor,
        initial_positions: torch.Tensor | None = None,
    ) -> RNNRollout:
        if velocities.ndim != 3 or velocities.shape[-1] != 2:
            raise ValueError("velocities must have shape (batch, time, 2)")
        batch_size, trajectory_length, _ = velocities.shape
        g = self._initial_state(batch_size, velocities.device, initial_positions)
        initial_state = g
        states: list[torch.Tensor] = []
        zero_count = velocities.new_zeros(())
        total_count = velocities.new_zeros(())
        for t in range(trajectory_length):
            weights = self.transition_mlp(velocities[:, t]).view(
                batch_size,
                self.n_units,
                self.n_units,
            )
            raw = torch.bmm(weights, g.unsqueeze(-1)).squeeze(-1)
            post_relu = torch.relu(raw)
            zero_count = zero_count + (post_relu.norm(dim=-1) <= 1e-8).sum()
            total_count = total_count + post_relu.shape[0]
            g = norm_relu(raw)
            states.append(g)
        hidden_states = torch.stack(states, dim=1)
        zero_norm_fraction = zero_count / total_count.clamp_min(1)
        return RNNRollout(
            initial_state=initial_state,
            hidden_states=hidden_states,
            zero_norm_fraction=zero_norm_fraction,
        )

    @staticmethod
    def _build_mlp(cfg: ModelConfig) -> nn.Sequential:
        output_width = cfg.n_units * cfg.n_units
        layers: list[nn.Module] = []
        in_width = 2
        for _ in range(max(cfg.mlp_layers - 1, 0)):
            layers.append(nn.Linear(in_width, cfg.mlp_hidden_width))
            layers.append(nn.ReLU())
            in_width = cfg.mlp_hidden_width
        layers.append(nn.Linear(in_width, output_width))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_position_encoder(cfg: ModelConfig) -> nn.Module | None:
        if cfg.initial_position_encoding == "none":
            return None
        layers = nn.Sequential(
            nn.Linear(2, cfg.initial_position_hidden_width),
            nn.ReLU(),
            nn.Linear(cfg.initial_position_hidden_width, cfg.n_units),
        )
        final = layers[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        return layers

    def _initial_state(
        self,
        batch_size: int,
        device: torch.device,
        initial_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.initial_position_encoding == "none":
            if initial_positions is not None:
                raise ValueError(
                    "initial_positions requires model.initial_position_encoding='additive_mlp'"
                )
            return norm_relu(self.g0_raw).expand(batch_size, -1)
        if initial_positions is None:
            raise ValueError(
                "initial_positions must be provided when "
                "model.initial_position_encoding='additive_mlp'"
            )
        if initial_positions.shape != (batch_size, 2):
            raise ValueError("initial_positions must have shape (batch, 2)")
        if self.position_encoder is None:
            raise RuntimeError("position_encoder is missing for additive initial-position mode")
        encoded = self.position_encoder(initial_positions.to(device=device, dtype=self.g0_raw.dtype))
        return norm_relu(self.g0_raw.unsqueeze(0) + encoded)
