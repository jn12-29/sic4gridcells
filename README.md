# sic4gridcells

Plain PyTorch reproduction scaffold for the SIC grid-cell model from Schaeffer et al., "Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells".

The current repository implements a runnable SIC reproduction slice: SIC velocity permutation batches, velocity-conditioned RNN rollout, separation/invariance/capacity/conformal-isometry losses, training, checkpoint evaluation, ratemaps, SAC/grid scoring, smoke/medium/paper configs, ablation orchestration, and unit tests. See `docs/runbook.md` for longer run commands, and `docs/sic-implementation-plan.md` plus `docs/sic-reproduction-plan.md` for reproduction scope and remaining paper-level work.

## Setup

Use the local uv-managed environment in this repository.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

If dependencies need to be installed or refreshed, keep targeting the local environment explicitly:

```bash
uv pip install --python .venv/bin/python -e .
```

Do not use bare `uv pip install ...` in this workspace; on this machine it can install into the active conda environment instead of `.venv`.

## Quick Start

Run the test suite:

```bash
.venv/bin/python -m pytest
```

Run the 10-step smoke training job:

```bash
.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml
```

Expected CLI output:

```text
finished step=10 output_dir=results/smoke
checkpoint=results/smoke/checkpoints/step_10.pt
```

The smoke run writes:

- `results/smoke/config.yaml`
- `results/smoke/metrics.jsonl`
- `results/smoke/tensorboard/`
- `results/smoke/checkpoints/step_5.pt`
- `results/smoke/checkpoints/step_10.pt`

`results/` is ignored by git.

Resume an interrupted run with the same config. To extend beyond the
checkpoint config, change only `train.max_optimizer_steps`:

```bash
.venv/bin/python scripts/train_sic.py --config configs/medium.yaml --resume results/medium/checkpoints/step_500.pt
```

Evaluate the smoke checkpoint:

```bash
.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16 --seed 0
```

The evaluation writes `summary.json`, `config.yaml`, and per-arena ratemap, SAC, grid-stat, trajectory, pairwise-distance, Fourier, phase, state-space, PDF, and PNG artifacts under `results/smoke/eval/`. See `docs/runbook.md` for the artifact checklist used for medium, paper-scale, and ablation runs.

Evaluation defaults to `--start-mode origin`, which keeps reset model state aligned with position bins. `--start-mode uniform` is only valid for checkpoints trained with `model.initial_position_encoding: additive_mlp` and `data.initial_position_mode: uniform_box`. If `--seed` is omitted, evaluation uses the checkpoint config seed.
Evaluation also defaults to `--trajectory-mode reflect` for backward-compatible bounded walks. Use `--trajectory-mode smooth_avoid_walls` for smoother wall-avoiding diagnostic trajectories.

## Configs

- `configs/smoke.yaml`: small CPU smoke run for tests and workflow checks.
- `configs/medium.yaml`: medium sanity-run profile (`B=16`, `T=30`, `N=64`).
- `configs/sic_paper.yaml`: paper-scale training hyperparameters from the reproduction plan.
- `configs/ablations.yaml`: ablation orchestration plan for `scripts/run_ablations.py`.

Important training semantics:

- `train.max_optimizer_steps` counts optimizer steps.
- `train.accumulate_grad_batches` controls microbatches inside each optimizer step.
- `data.augmentation_mode: permutation` is the SIC default; `identity` disables velocity permutation augmentation for the configured ablation.
- `loss.pairwise_reduction: sum` is closer to the paper formulas; `mean` is useful for smaller smoke runs.

## Project Layout

```text
configs/                  YAML training and ablation configs
docs/                     runbook, reproduction plan, and implementation plan
scripts/train_sic.py      thin training CLI entry point
scripts/eval_checkpoint.py checkpoint evaluation CLI
scripts/run_ablations.py  ablation orchestration CLI
src/sic4gridcells/        package source
tests/                    pytest suite
```

Key modules:

- `sic4gridcells.config`: YAML loading, defaults, validation, effective-config saving.
- `sic4gridcells.data`: SIC base velocity sampling and random permutation batches.
- `sic4gridcells.model`: NormReLU and velocity-conditioned RNN.
- `sic4gridcells.losses`: SIC losses and tiny naive pairwise implementation for tests.
- `sic4gridcells.train`: training loop, metrics, TensorBoard logging, checkpointing.
- `sic4gridcells.evaluate`: checkpoint reload, bounded random-walk evaluation, artifact writing.
- `sic4gridcells.analysis`: ratemap, SAC, grid score, and grid-scale utilities.
- `sic4gridcells.plotting`: PDF and PNG evaluation figures.
- `docs/runbook.md`: medium, paper-scale, and ablation command sequence.

## Current Limits

This is not yet a full paper reproduction. The current code verifies the core training and evaluation contracts, but these paper-level pieces still require longer runs and analysis:

- medium-scale training has a config but has not been run to completion here
- paper-scale training and multi-seed sweeps
- full paper figure reproduction from trained results
- toroidal manifold confirmation beyond the preliminary state-space PCA summaries
