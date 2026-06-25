from __future__ import annotations

from typing import Any

import torch

from sic4gridcells.config import Config, LossConfig
from sic4gridcells.data import SicBatch
from sic4gridcells.model import RNNRollout


def sic_losses(batch: SicBatch, rollout: RNNRollout, cfg: Config) -> dict[str, torch.Tensor]:
    pairwise = pairwise_sic_losses(batch.positions, rollout.hidden_states, cfg.loss)
    coniso, coniso_count = _conformal_isometry_loss_with_count(
        batch.velocities,
        rollout.initial_state,
        rollout.hidden_states,
        cfg.loss.sigma_x,
    )
    total = (
        cfg.loss.lambda_sep * pairwise["loss/separation"]
        + cfg.loss.lambda_inv * pairwise["loss/invariance"]
        + cfg.loss.lambda_cap * pairwise["loss/capacity"]
        + cfg.loss.lambda_coniso * coniso
    )
    return {
        "loss/total": total,
        "loss/separation": pairwise["loss/separation"],
        "loss/invariance": pairwise["loss/invariance"],
        "loss/capacity": pairwise["loss/capacity"],
        "loss/conformal_isometry": coniso,
        "stats/separation_pairs": pairwise["stats/separation_pairs"],
        "stats/invariance_pairs": pairwise["stats/invariance_pairs"],
        "stats/conformal_isometry_steps": coniso_count,
    }


def pairwise_sic_losses(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    cfg: Config | LossConfig,
) -> dict[str, torch.Tensor]:
    loss_cfg = _loss_cfg(cfg)
    pos_all, g_all = _flatten_pairwise_inputs(positions, hidden_states)
    sep_total = g_all.new_zeros(())
    inv_total = g_all.new_zeros(())
    sep_count = torch.zeros((), dtype=torch.long, device=g_all.device)
    inv_count = torch.zeros((), dtype=torch.long, device=g_all.device)
    g_sq_all = g_all.square().sum(dim=1)
    for start in range(0, pos_all.shape[0], loss_cfg.chunk_size):
        end = min(start + loss_cfg.chunk_size, pos_all.shape[0])
        pos_chunk = pos_all[start:end]
        g_chunk = g_all[start:end]
        x_dist = torch.cdist(pos_chunk, pos_all)
        g_dist_sq = _pairwise_squared_distances(g_chunk, g_all, g_sq_all)
        sep_mask = x_dist > loss_cfg.sigma_x
        inv_mask = x_dist < loss_cfg.sigma_x
        sep_total = sep_total + torch.exp(
            -g_dist_sq[sep_mask] / (2.0 * loss_cfg.sigma_g**2)
        ).sum()
        inv_total = inv_total + g_dist_sq[inv_mask].sum()
        sep_count = sep_count + sep_mask.sum()
        inv_count = inv_count + inv_mask.sum()
    separation = _reduce_pairwise(sep_total, sep_count, loss_cfg.pairwise_reduction)
    invariance = _reduce_pairwise(inv_total, inv_count, loss_cfg.pairwise_reduction)
    capacity = -g_all.mean(dim=0).square().sum()
    return {
        "loss/separation": separation,
        "loss/invariance": invariance,
        "loss/capacity": capacity,
        "stats/separation_pairs": sep_count,
        "stats/invariance_pairs": inv_count,
    }


def naive_pairwise_sic_losses(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    cfg: Config | LossConfig,
) -> dict[str, torch.Tensor]:
    loss_cfg = _loss_cfg(cfg)
    pos_all, g_all = _flatten_pairwise_inputs(positions, hidden_states)
    x_dist = torch.cdist(pos_all, pos_all)
    g_dist_sq = torch.cdist(g_all, g_all).square()
    sep_mask = x_dist > loss_cfg.sigma_x
    inv_mask = x_dist < loss_cfg.sigma_x
    sep_count = sep_mask.sum()
    inv_count = inv_mask.sum()
    sep_total = torch.exp(-g_dist_sq[sep_mask] / (2.0 * loss_cfg.sigma_g**2)).sum()
    inv_total = g_dist_sq[inv_mask].sum()
    return {
        "loss/separation": _reduce_pairwise(
            sep_total,
            sep_count,
            loss_cfg.pairwise_reduction,
        ),
        "loss/invariance": _reduce_pairwise(
            inv_total,
            inv_count,
            loss_cfg.pairwise_reduction,
        ),
        "loss/capacity": -g_all.mean(dim=0).square().sum(),
        "stats/separation_pairs": sep_count,
        "stats/invariance_pairs": inv_count,
    }


def conformal_isometry_loss(
    velocities: torch.Tensor,
    initial_state: torch.Tensor,
    hidden_states: torch.Tensor,
    sigma_x: float,
) -> torch.Tensor:
    loss, _ = _conformal_isometry_loss_with_count(
        velocities,
        initial_state,
        hidden_states,
        sigma_x,
    )
    return loss


def _conformal_isometry_loss_with_count(
    velocities: torch.Tensor,
    initial_state: torch.Tensor,
    hidden_states: torch.Tensor,
    sigma_x: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.cat([initial_state.unsqueeze(1), hidden_states[:, :-1]], dim=1)
    velocity_norm = velocities.norm(dim=-1)
    valid = (velocity_norm > 0) & (velocity_norm < sigma_x)
    count = valid.sum()
    if count == 0:
        return hidden_states.new_zeros(()), count
    code_step = (hidden_states - previous).norm(dim=-1)
    ratios = code_step[valid] / velocity_norm[valid]
    return ratios.var(unbiased=False), count


def _flatten_pairwise_inputs(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (batch, time, 2)")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (batch, time, units)")
    if positions.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("positions and hidden_states must share batch/time dimensions")
    return positions.reshape(-1, 2), hidden_states.reshape(-1, hidden_states.shape[-1])


def _pairwise_squared_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    right_sq: torch.Tensor,
) -> torch.Tensor:
    left_sq = left.square().sum(dim=1, keepdim=True)
    distances = left_sq + right_sq.unsqueeze(0) - 2.0 * left @ right.T
    return distances.clamp_min(0.0)


def _reduce_pairwise(total: torch.Tensor, count: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return total
    if reduction == "mean":
        if count == 0:
            return total.new_zeros(())
        return total / count.to(dtype=total.dtype)
    raise ValueError(f"Unsupported pairwise reduction: {reduction}")


def _loss_cfg(cfg: Config | LossConfig) -> LossConfig:
    return cfg.loss if isinstance(cfg, Config) else cfg
