import pytest
import torch

from sic4gridcells.config import ModelConfig
from sic4gridcells.model import VelocityConditionedRNN, norm_relu


def test_norm_relu_zero_and_unit_norm_behavior() -> None:
    x = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
    y = norm_relu(x)
    assert torch.equal(y[0], torch.zeros(2))
    assert torch.all(y[1] >= 0)
    assert torch.allclose(y[1].norm(), torch.tensor(1.0))


def test_velocity_conditioned_rnn_shapes_and_gradients() -> None:
    cfg = ModelConfig(n_units=8, mlp_layers=3, mlp_hidden_width=16)
    model = VelocityConditionedRNN(cfg)
    velocities = torch.randn(3, 4, 2)
    rollout = model(velocities)
    assert rollout.initial_state.shape == (3, 8)
    assert rollout.hidden_states.shape == (3, 4, 8)
    assert torch.all(rollout.hidden_states >= 0)
    norms = rollout.hidden_states.norm(dim=-1)
    assert torch.all((torch.isclose(norms, torch.ones_like(norms))) | (norms == 0))
    loss = rollout.hidden_states.sum()
    loss.backward()
    grad_norms = [
        parameter.grad.norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert grad_norms
    assert all(torch.isfinite(value) for value in grad_norms)


def test_initial_positions_rejected_without_encoder() -> None:
    cfg = ModelConfig(n_units=4, mlp_layers=1)
    model = VelocityConditionedRNN(cfg)
    velocities = torch.randn(2, 3, 2)
    initial_positions = torch.randn(2, 2)
    with pytest.raises(ValueError, match="initial_position_encoding"):
        model(velocities, initial_positions=initial_positions)


def test_additive_initial_position_encoder_shapes_and_gradients() -> None:
    cfg = ModelConfig(
        n_units=6,
        mlp_layers=2,
        mlp_hidden_width=8,
        initial_position_encoding="additive_mlp",
        initial_position_hidden_width=5,
    )
    model = VelocityConditionedRNN(cfg)
    velocities = torch.randn(3, 4, 2)
    initial_positions = torch.randn(3, 2)

    rollout = model(velocities, initial_positions=initial_positions)

    assert rollout.initial_state.shape == (3, 6)
    assert rollout.hidden_states.shape == (3, 4, 6)
    loss = rollout.hidden_states.sum()
    loss.backward()
    encoder_grads = [
        parameter.grad
        for parameter in model.position_encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads
    assert all(torch.isfinite(value).all() for value in encoder_grads)
