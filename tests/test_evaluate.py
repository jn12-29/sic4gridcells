from __future__ import annotations

import json
from dataclasses import asdict
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import torch

from sic4gridcells.config import Config, DataConfig, LossConfig, ModelConfig, TrainConfig
from sic4gridcells.evaluate import (
    evaluate_checkpoint,
    _accumulate_ratemaps,
    _evaluation_step_scale,
    _run_bounded_random_walks,
    _sample_bounded_random_walk,
    _summarize_coverage,
    _summarize_unit_responses,
    _unit_response_counts,
)
from sic4gridcells.model import RNNRollout, VelocityConditionedRNN


def test_evaluate_checkpoint_writes_artifacts(tmp_path: Path) -> None:
    cfg = Config(
        seed=0,
        device="cpu",
        output_dir=str(tmp_path / "train"),
        data=DataConfig(batch_size=2, trajectory_length=4, velocity_low=-0.05, velocity_high=0.05),
        model=ModelConfig(n_units=4, mlp_layers=2, mlp_hidden_width=8, trainable_initial_state=True),
        loss=LossConfig(chunk_size=2, pairwise_reduction="mean"),
        train=TrainConfig(max_optimizer_steps=1),
    )
    model = VelocityConditionedRNN(cfg)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 1,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )
    result = evaluate_checkpoint(
        checkpoint_path,
        tmp_path / "eval",
        device="cpu",
        arena_sizes=(1.0,),
        nbins=6,
        n_trajectories=3,
        steps_per_trajectory=8,
    )
    arena_dir = result.arena_dirs[1.0]
    assert (arena_dir / "ratemaps.npz").exists()
    assert (arena_dir / "occupancy.npz").exists()
    assert (arena_dir / "sacs.npz").exists()
    assert (arena_dir / "grid_stats.csv").exists()
    assert (arena_dir / "summary.png").exists()
    assert (arena_dir / "ratemaps.pdf").exists()
    assert (arena_dir / "sacs.pdf").exists()
    assert (result.output_dir / "summary.json").exists()
    summary = _load_strict_json(result.output_dir / "summary.json")
    grid_stats = _load_strict_json(arena_dir / "grid_stats.json")
    occupancy = np.load(arena_dir / "occupancy.npz")["occupancy_counts"]
    assert occupancy.shape == (6, 6)
    assert occupancy.sum() > 0
    assert all("scale" in row for row in grid_stats)
    assert all(
        {
            "response_status",
            "max_abs_response",
            "zero_response",
            "invalid_response",
        }
        <= set(row)
        for row in grid_stats
    )
    arena_summary = summary["arena_summaries"][0]
    assert "dead_units" not in arena_summary
    assert arena_summary["visited_bins"] == int((occupancy > 0).sum())
    assert arena_summary["total_bins"] == 36
    assert arena_summary["unvisited_bins"] == 36 - arena_summary["visited_bins"]
    assert 0.0 <= arena_summary["coverage_fraction"] <= 1.0
    assert {
        "units_without_coverage",
        "zero_response_units",
        "invalid_response_units",
        "active_units",
    } <= set(arena_summary)
    assert "mean_scale" in arena_summary


def test_accumulate_ratemaps_returns_occupancy_and_preserves_nan_for_unvisited_bins() -> None:
    positions = np.array(
        [
            [
                [-0.5, -0.5],
                [0.5, 0.5],
            ]
        ],
        dtype=np.float64,
    )
    hidden_states = np.array(
        [
            [
                [0.0, 2.0],
                [0.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )

    ratemaps, occupancy_counts = _accumulate_ratemaps(
        positions,
        hidden_states,
        arena_size=2.0,
        nbins=2,
    )

    assert occupancy_counts.tolist() == [[1, 0], [0, 1]]
    assert ratemaps.shape == (2, 2, 2)
    assert ratemaps[0, 0, 0] == 0.0
    assert ratemaps[0, 1, 1] == 0.0
    assert ratemaps[1, 0, 0] == 2.0
    assert ratemaps[1, 1, 1] == 4.0
    assert np.isnan(ratemaps[:, 0, 1]).all()
    assert np.isnan(ratemaps[:, 1, 0]).all()


def test_unit_response_summary_distinguishes_coverage_zero_invalid_and_active() -> None:
    occupancy_counts = np.array([[1, 0], [0, 1]], dtype=np.int64)
    ratemaps = np.array(
        [
            [[0.0, np.nan], [np.nan, 0.0]],
            [[2.0, np.nan], [np.nan, 0.0]],
            [[np.nan, np.nan], [np.nan, 0.0]],
        ],
        dtype=np.float64,
    )

    stats = _summarize_unit_responses(ratemaps, occupancy_counts)

    assert stats[0]["response_status"] == "zero"
    assert stats[0]["zero_response"] is True
    assert stats[0]["max_abs_response"] == 0.0
    assert stats[1]["response_status"] == "active"
    assert stats[1]["zero_response"] is False
    assert stats[1]["max_abs_response"] == 2.0
    assert stats[2]["response_status"] == "invalid"
    assert stats[2]["invalid_response"] is True
    assert stats[2]["max_abs_response"] is None
    assert _unit_response_counts(stats) == {
        "units_without_coverage": 0,
        "zero_response_units": 1,
        "invalid_response_units": 1,
        "active_units": 1,
    }


def test_unit_response_summary_reports_no_coverage_separately() -> None:
    ratemaps = np.full((3, 2, 2), np.nan, dtype=np.float64)
    occupancy_counts = np.zeros((2, 2), dtype=np.int64)

    coverage = _summarize_coverage(occupancy_counts)
    stats = _summarize_unit_responses(ratemaps, occupancy_counts)

    assert coverage == {
        "visited_bins": 0,
        "unvisited_bins": 4,
        "total_bins": 4,
        "coverage_fraction": 0.0,
    }
    assert [row["response_status"] for row in stats] == ["no_coverage"] * 3
    assert _unit_response_counts(stats) == {
        "units_without_coverage": 3,
        "zero_response_units": 0,
        "invalid_response_units": 0,
        "active_units": 0,
    }


def test_evaluate_checkpoint_supports_single_unit_summary_figure(tmp_path: Path) -> None:
    cfg = Config(
        seed=0,
        device="cpu",
        output_dir=str(tmp_path / "train"),
        data=DataConfig(batch_size=1, trajectory_length=2, velocity_low=-0.05, velocity_high=0.05),
        model=ModelConfig(n_units=1, mlp_layers=1, mlp_hidden_width=4, trainable_initial_state=True),
        loss=LossConfig(chunk_size=1, pairwise_reduction="mean"),
        train=TrainConfig(max_optimizer_steps=1),
    )
    model = VelocityConditionedRNN(cfg)
    checkpoint_path = tmp_path / "single_unit.pt"
    torch.save(
        {
            "step": 1,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )

    result = evaluate_checkpoint(
        checkpoint_path,
        tmp_path / "eval-single",
        device="cpu",
        arena_sizes=(1.0,),
        nbins=4,
        n_trajectories=1,
        steps_per_trajectory=4,
    )

    assert (result.arena_dirs[1.0] / "summary.png").exists()


def test_origin_random_walk_starts_at_reset_origin() -> None:
    positions, velocities = _sample_bounded_random_walk(
        4,
        1.0,
        device=torch.device("cpu"),
        start_mode="origin",
    )
    assert torch.allclose(positions[0] - velocities[0], torch.zeros(2))


def test_evaluation_step_scale_does_not_shrink_with_sample_count() -> None:
    arena_size = 2.0

    assert "steps" not in signature(_evaluation_step_scale).parameters
    assert _evaluation_step_scale(arena_size) > arena_size / 256 * 0.75


def test_random_walk_step_size_does_not_shrink_with_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_index = 0

    def fake_rand(*args, **kwargs):
        nonlocal call_index
        value = 0.0 if call_index % 2 == 0 else 1.0
        call_index += 1
        return torch.tensor(value, device=kwargs.get("device"))

    monkeypatch.setattr("torch.rand", fake_rand)

    short_positions, _ = _sample_bounded_random_walk(
        2,
        2.0,
        device=torch.device("cpu"),
        start_mode="origin",
    )
    long_positions, _ = _sample_bounded_random_walk(
        256,
        2.0,
        device=torch.device("cpu"),
        start_mode="origin",
    )

    expected_first_step = torch.tensor([_evaluation_step_scale(2.0), 0.0])
    assert torch.allclose(short_positions[0], expected_first_step)
    assert torch.allclose(long_positions[0], expected_first_step)


def test_random_walk_velocity_scale_is_bounded_by_arena_scale() -> None:
    positions, velocities = _sample_bounded_random_walk(
        32,
        2.0,
        device=torch.device("cpu"),
        start_mode="origin",
    )

    assert positions.shape == (32, 2)
    assert torch.linalg.norm(velocities, dim=1).max() <= _evaluation_step_scale(2.0) + 1e-6


def test_uniform_start_requires_initial_position_encoder(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(n_units=4, mlp_layers=1, mlp_hidden_width=8),
    )

    with pytest.raises(ValueError, match="requires a checkpoint trained"):
        evaluate_checkpoint(
            checkpoint_path,
            tmp_path / "eval",
            device="cpu",
            arena_sizes=(1.0,),
            nbins=4,
            n_trajectories=1,
            steps_per_trajectory=4,
            start_mode="uniform",
        )


def test_uniform_start_with_initial_position_encoder_writes_artifacts(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(
            n_units=4,
            mlp_layers=1,
            mlp_hidden_width=8,
            initial_position_encoding="additive_mlp",
            initial_position_hidden_width=4,
        ),
        data_cfg=DataConfig(
            batch_size=1,
            trajectory_length=2,
            velocity_low=-0.05,
            velocity_high=0.05,
            initial_position_mode="uniform_box",
            initial_position_low=-0.5,
            initial_position_high=0.5,
        ),
    )

    result = evaluate_checkpoint(
        checkpoint_path,
        tmp_path / "eval-uniform",
        device="cpu",
        arena_sizes=(1.0,),
        nbins=4,
        n_trajectories=1,
        steps_per_trajectory=4,
        start_mode="uniform",
    )

    assert (result.arena_dirs[1.0] / "grid_stats.json").exists()


def test_uniform_random_walk_passes_sampled_start_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    positions = torch.tensor(
        [
            [0.25, -0.10],
            [0.30, -0.08],
        ]
    )
    velocities = torch.tensor(
        [
            [0.05, 0.02],
            [0.05, 0.02],
        ]
    )
    captured: list[torch.Tensor | None] = []

    def fake_walk(*args, **kwargs):
        return positions, velocities

    monkeypatch.setattr("sic4gridcells.evaluate._sample_bounded_random_walk", fake_walk)
    model = _CapturingModel(captured)

    _run_bounded_random_walks(
        model,
        torch.device("cpu"),
        arena_size=1.0,
        n_trajectories=1,
        steps_per_trajectory=2,
        start_mode="uniform",
    )

    assert len(captured) == 1
    assert captured[0] is not None
    assert torch.allclose(captured[0], torch.tensor([[0.20, -0.12]]))


def test_uniform_start_requires_uniform_box_training(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(
            n_units=4,
            mlp_layers=1,
            mlp_hidden_width=8,
            initial_position_encoding="additive_mlp",
            initial_position_hidden_width=4,
        ),
    )

    with pytest.raises(ValueError, match="data.initial_position_mode"):
        evaluate_checkpoint(
            checkpoint_path,
            tmp_path / "eval-uniform-zero-trained",
            device="cpu",
            arena_sizes=(1.0,),
            nbins=4,
            n_trajectories=1,
            steps_per_trajectory=4,
            start_mode="uniform",
        )


def test_checkpoint_config_is_validated(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(n_units=4, mlp_layers=1, mlp_hidden_width=8),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["config"]["model"]["initial_position_encoding"] = "future_encoder"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="model.initial_position_encoding"):
        evaluate_checkpoint(
            checkpoint_path,
            tmp_path / "eval-invalid-config",
            device="cpu",
            arena_sizes=(1.0,),
            nbins=4,
            n_trajectories=1,
            steps_per_trajectory=4,
        )


def test_eval_auto_device_uses_current_hardware_not_checkpoint_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(n_units=4, mlp_layers=1, mlp_hidden_width=8),
        device="cuda",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = evaluate_checkpoint(
        checkpoint_path,
        tmp_path / "eval-auto-cpu",
        device="auto",
        arena_sizes=(1.0,),
        nbins=4,
        n_trajectories=1,
        steps_per_trajectory=2,
    )

    assert (result.output_dir / "summary.json").exists()
    summary = _load_strict_json(result.output_dir / "summary.json")
    assert summary["config"]["device"] == "cuda"


def test_eval_explicit_cuda_still_requires_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        ModelConfig(n_units=4, mlp_layers=1, mlp_hidden_width=8),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        evaluate_checkpoint(
            checkpoint_path,
            tmp_path / "eval-cuda",
            device="cuda",
            arena_sizes=(1.0,),
            nbins=4,
            n_trajectories=1,
            steps_per_trajectory=2,
        )


def _load_strict_json(path: Path):
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    return json.loads(
        raw,
        parse_constant=lambda value: pytest.fail(
            f"non-standard JSON constant {value!r} in {path}"
        ),
    )


class _CapturingModel:
    initial_position_encoding = "additive_mlp"
    n_units = 2

    def __init__(self, captured: list[torch.Tensor | None]) -> None:
        self.captured = captured

    def __call__(
        self,
        velocities: torch.Tensor,
        *,
        initial_positions: torch.Tensor | None,
    ) -> RNNRollout:
        self.captured.append(
            None if initial_positions is None else initial_positions.detach().cpu().clone()
        )
        batch_size, steps, _ = velocities.shape
        hidden_states = torch.ones(batch_size, steps, self.n_units)
        return RNNRollout(
            initial_state=torch.zeros(batch_size, self.n_units),
            hidden_states=hidden_states,
            zero_norm_fraction=torch.tensor(0.0),
        )


def _write_checkpoint(
    tmp_path: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig | None = None,
    device: str = "cpu",
) -> Path:
    cfg = Config(
        seed=0,
        device=device,
        output_dir=str(tmp_path / "train"),
        data=data_cfg
        or DataConfig(batch_size=1, trajectory_length=2, velocity_low=-0.05, velocity_high=0.05),
        model=model_cfg,
        loss=LossConfig(chunk_size=1, pairwise_reduction="mean"),
        train=TrainConfig(max_optimizer_steps=1),
    )
    model = VelocityConditionedRNN(cfg)
    checkpoint_path = tmp_path / f"{model_cfg.initial_position_encoding}.pt"
    torch.save(
        {
            "step": 1,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )
    return checkpoint_path
