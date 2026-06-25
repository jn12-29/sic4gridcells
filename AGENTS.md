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

## Current Contracts

- Do not add supervised place-cell or head-direction target training; SIC is self-supervised from velocity permutations and relative spatial separation.
- Keep the entry point `scripts/train_sic.py` thin: parse args and call `sic4gridcells.train.train`.
- `train.max_optimizer_steps` counts optimizer steps, not gradient-accumulation microbatches.
- `VelocityConditionedRNN.forward` returns `RNNRollout(initial_state, hidden_states, zero_norm_fraction)`.
- `RNNRollout.initial_state` has shape `(B, N)` and is used for conformal-isometry loss at `t=0`.
- `RNNRollout.hidden_states` has shape `(B, T, N)`.
- `sic_losses(batch, rollout, cfg)` returns `loss/total`, individual loss terms, and pair/step counts.
- Checkpoints must remain loadable with default `torch.load(path, map_location="cpu")`; store config data as built-in containers, not custom dataclass objects.

## File Ownership

- `configs/smoke.yaml` is the fast local workflow config.
- `configs/sic_paper.yaml` is the paper-scale config and should keep paper values unless the implementation plan changes.
- `docs/sic-implementation-plan.md` owns implementation contracts and phase boundaries.
- `docs/sic-reproduction-plan.md` owns paper-material rationale and reproduction scope.
- `.work/sic-reproduction/source/` is paper source material only; do not import it as project code.

## Finish Checks

After changing code, configs, scripts, or docs that mention commands, run the narrowest relevant checks. For broad core changes, run:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml
```

Before finishing docs changes, verify referenced paths exist and grep the changed docs for stale placeholders or claims about unimplemented evaluation features.

