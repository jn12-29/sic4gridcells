import torch

from sic4gridcells.config import Config, DataConfig, LossConfig, ModelConfig, TrainConfig
from sic4gridcells.data import SicBatch
from sic4gridcells.losses import (
    conformal_isometry_loss,
    naive_pairwise_sic_losses,
    pairwise_sic_losses,
    sic_losses,
)
from sic4gridcells.model import RNNRollout, norm_relu


def test_chunked_pairwise_matches_naive_mean_reduction() -> None:
    loss_cfg = LossConfig(pairwise_reduction="mean", chunk_size=2, sigma_x=0.2)
    positions = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.0], [0.4, 0.0]],
            [[0.0, 0.0], [0.0, 0.1], [0.4, 0.1]],
        ]
    )
    hidden_states = norm_relu(torch.randn(2, 3, 5))
    chunked = pairwise_sic_losses(positions, hidden_states, loss_cfg)
    naive = naive_pairwise_sic_losses(positions, hidden_states, loss_cfg)
    for key in ["loss/separation", "loss/invariance", "loss/capacity"]:
        assert torch.allclose(chunked[key], naive[key], atol=1e-6)
    assert torch.equal(chunked["stats/separation_pairs"], naive["stats/separation_pairs"])
    assert torch.equal(chunked["stats/invariance_pairs"], naive["stats/invariance_pairs"])


def test_conformal_isometry_returns_zero_without_valid_steps() -> None:
    velocities = torch.zeros(2, 3, 2)
    initial_state = norm_relu(torch.randn(2, 4))
    hidden_states = norm_relu(torch.randn(2, 3, 4))
    loss = conformal_isometry_loss(velocities, initial_state, hidden_states, sigma_x=0.05)
    assert torch.equal(loss, torch.tensor(0.0))


def test_sic_losses_are_finite_and_capacity_is_non_positive() -> None:
    cfg = Config(
        data=DataConfig(batch_size=2, trajectory_length=3),
        model=ModelConfig(n_units=5, mlp_hidden_width=8),
        loss=LossConfig(pairwise_reduction="mean", chunk_size=2),
        train=TrainConfig(max_optimizer_steps=1),
    )
    base_velocities = torch.randn(3, 2) * 0.01
    permutations = torch.tensor([[0, 1, 2], [1, 0, 2]])
    initial_positions = torch.zeros(2, 2)
    velocities = base_velocities[permutations]
    positions = velocities.cumsum(dim=1)
    hidden_states = norm_relu(torch.randn(2, 3, 5))
    rollout = RNNRollout(
        initial_state=norm_relu(torch.randn(2, 5)),
        hidden_states=hidden_states,
        zero_norm_fraction=torch.tensor(0.0),
    )
    losses = sic_losses(
        SicBatch(base_velocities, permutations, initial_positions, velocities, positions),
        rollout,
        cfg,
    )
    for value in losses.values():
        assert torch.isfinite(value)
    assert losses["loss/capacity"] <= 0
