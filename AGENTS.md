# AGENTS.md

Repo-specific instructions for agents working in this repository.

## Scope

- Repository root: `/mnt/sata2/xh/ai4neuron/gridCells/sic4gridcells`.
- Project package: `src/sic4gridcells`.
- Human-facing setup and usage live in `README.md`.
- Reproduction planning details live in `docs/sic-reproduction-plan.md` and `docs/sic-implementation-plan.md`.

## Environment

- Use `.venv/bin/python` for Python commands.
- Install into the local environment with `uv pip install --python .venv/bin/python ...`.
- Do not use bare `uv pip install ...`; on this machine it can target the active conda environment.
- Keep generated outputs under ignored paths: `results/`, `data/`, `outputs/`, or `.work/`.

## Commands

| Task | Command |
| --- | --- |
| Install editable package | `uv pip install --python .venv/bin/python -e .` |
| Run all tests | `.venv/bin/python -m pytest` |
| Run config/data/model/loss tests | `.venv/bin/python -m pytest tests/test_config.py tests/test_data.py tests/test_model.py tests/test_losses.py` |
| Run train smoke test | `.venv/bin/python -m pytest tests/test_train_step.py` |
| Show training CLI help | `.venv/bin/python scripts/train_sic.py --help` |
| Run smoke training | `.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml` |
| Run evaluation tests | `.venv/bin/python -m pytest tests/test_analysis.py tests/test_evaluate.py` |
| Show evaluation CLI help | `.venv/bin/python scripts/eval_checkpoint.py --help` |
| Evaluate smoke checkpoint | `.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16` |
| Show ablation CLI help | `.venv/bin/python scripts/run_ablations.py --help` |
| Dry-run ablations | `.venv/bin/python scripts/run_ablations.py --config configs/ablations.yaml --dry-run` |

## Current Contracts

- Do not add supervised place-cell or head-direction target training; SIC is self-supervised from velocity permutations and relative spatial separation.
- Keep the entry point `scripts/train_sic.py` thin: parse args and call `sic4gridcells.train.train`.
- `train.max_optimizer_steps` counts optimizer steps, not gradient-accumulation microbatches.
- `VelocityConditionedRNN.forward` returns `RNNRollout(initial_state, hidden_states, zero_norm_fraction)`.
- `RNNRollout.initial_state` has shape `(B, N)` and is used for conformal-isometry loss at `t=0`.
- `RNNRollout.hidden_states` has shape `(B, T, N)`.
- `sic_losses(batch, rollout, cfg)` returns `loss/total`, individual loss terms, and pair/step counts.
- Checkpoints must remain loadable with default `torch.load(path, map_location="cpu")`; store config data as built-in containers, not custom dataclass objects.
- `scripts/eval_checkpoint.py` reloads the training config from the checkpoint; it does not take a separate `--config`.
- Evaluation trajectories are bounded random walks and must not reuse supervised place-cell or head-direction targets.
- `scripts/eval_checkpoint.py --start-mode origin` is the default for no-encoder checkpoints; `--start-mode uniform` requires `model.initial_position_encoding: additive_mlp` and `data.initial_position_mode: uniform_box`.
- Ratemap empty bins remain `NaN`; visited zero responses remain `0.0`. Evaluation writes `occupancy.npz` for coverage and reports `units_without_coverage`, `zero_response_units`, `invalid_response_units`, and `active_units` instead of using `dead_units` as a coverage proxy.
- SAC/grid scoring must use finite ratemap bins as its overlap mask; evaluation walk step scale must not shrink as `--steps` increases.
- Evaluation artifacts live under caller-selected `results/` output directories.

## File Ownership

- `configs/smoke.yaml` is the fast local workflow config.
- `configs/medium.yaml` is the medium sanity-run config.
- `configs/sic_paper.yaml` is the paper-scale config and should keep paper values unless the implementation plan changes.
- `configs/ablations.yaml` is the ablation orchestration plan; `no_permutation_augmentation` uses `data.augmentation_mode: identity`.
- `docs/sic-implementation-plan.md` owns implementation contracts and phase boundaries.
- `docs/sic-reproduction-plan.md` owns paper-material rationale and reproduction scope.
- `docs/runbook.md` owns medium, paper-scale, and ablation run commands plus artifact checklists.
- `.work/sic-reproduction/source/` is paper source material only; do not import it as project code.

## Finish Checks

After changing code, configs, scripts, or docs that mention commands, run the narrowest relevant checks. For broad core changes, run:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml
```

Before finishing docs changes, verify referenced paths exist and grep the changed docs for stale placeholders or claims about unimplemented evaluation features.
