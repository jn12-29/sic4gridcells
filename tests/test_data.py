import torch

from sic4gridcells.config import DataConfig
from sic4gridcells.data import make_sic_batch


def test_sic_batch_shapes_and_endpoints() -> None:
    cfg = DataConfig(batch_size=5, trajectory_length=7)
    generator = torch.Generator().manual_seed(1)
    batch = make_sic_batch(cfg, generator, torch.device("cpu"))
    assert batch.base_velocities.shape == (7, 2)
    assert batch.permutations.shape == (5, 7)
    assert batch.initial_positions.shape == (5, 2)
    assert batch.velocities.shape == (5, 7, 2)
    assert batch.positions.shape == (5, 7, 2)
    expected_endpoint = batch.base_velocities.sum(dim=0)
    assert torch.allclose(batch.positions[:, -1], expected_endpoint.expand(5, -1))
    assert torch.equal(batch.initial_positions, torch.zeros_like(batch.initial_positions))


def test_each_trajectory_is_permutation_of_base_velocities() -> None:
    cfg = DataConfig(batch_size=4, trajectory_length=6)
    generator = torch.Generator().manual_seed(2)
    batch = make_sic_batch(cfg, generator, "cpu")
    for row in range(cfg.batch_size):
        assert torch.equal(torch.sort(batch.permutations[row]).values, torch.arange(6))
        assert torch.allclose(batch.velocities[row], batch.base_velocities[batch.permutations[row]])
    assert torch.allclose(
        batch.positions,
        batch.initial_positions.unsqueeze(1) + batch.velocities.cumsum(dim=1),
    )


def test_uniform_initial_position_is_shared_within_batch() -> None:
    cfg = DataConfig(
        batch_size=3,
        trajectory_length=5,
        initial_position_mode="uniform_box",
        initial_position_low=-1.0,
        initial_position_high=1.0,
    )
    generator = torch.Generator().manual_seed(3)
    batch = make_sic_batch(cfg, generator, "cpu")
    assert torch.allclose(batch.initial_positions, batch.initial_positions[:1].expand(3, -1))
    assert torch.all(batch.initial_positions >= -1.0)
    assert torch.all(batch.initial_positions <= 1.0)
    expected_endpoint = batch.initial_positions[0] + batch.base_velocities.sum(dim=0)
    assert torch.allclose(batch.positions[:, -1], expected_endpoint.expand(3, -1))
