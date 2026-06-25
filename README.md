# sic4gridcells

Plain PyTorch reproduction scaffold for the SIC grid-cell model from Schaeffer et al., "Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells".

The current repository implements the first runnable training slice: SIC velocity permutation batches, velocity-conditioned RNN rollout, separation/invariance/capacity/conformal-isometry losses, a training loop, smoke and paper configs, and unit tests. Evaluation, ratemaps, grid scoring, and ablation orchestration are planned but not implemented yet; see `docs/sic-implementation-plan.md` and `docs/sic-reproduction-plan.md`.

## Setup

Use the local uv-managed environment in this repository.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

If dependencies need to be installed or refreshed, keep targeting the local environment explicitly:

```bash
uv pip install --python .venv/bin/python torch numpy pyyaml tqdm tensorboard pytest
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

## Configs

- `configs/smoke.yaml`: small CPU smoke run for tests and workflow checks.
- `configs/sic_paper.yaml`: paper-scale training hyperparameters from the reproduction plan.

Important training semantics:

- `train.max_optimizer_steps` counts optimizer steps.
- `train.accumulate_grad_batches` controls microbatches inside each optimizer step.
- `loss.pairwise_reduction: sum` is closer to the paper formulas; `mean` is useful for smaller smoke runs.

## Project Layout

```text
configs/                  YAML training configs
docs/                     reproduction and implementation plans
scripts/train_sic.py      thin CLI entry point
src/sic4gridcells/        package source
tests/                    pytest suite
```

Key modules:

- `sic4gridcells.config`: YAML loading, defaults, validation, effective-config saving.
- `sic4gridcells.data`: SIC base velocity sampling and random permutation batches.
- `sic4gridcells.model`: NormReLU and velocity-conditioned RNN.
- `sic4gridcells.losses`: SIC losses and tiny naive pairwise implementation for tests.
- `sic4gridcells.train`: training loop, metrics, TensorBoard logging, checkpointing.

## Current Limits

This is not yet a full paper reproduction. The current code verifies the core training contracts and smoke execution. The following are not implemented in the package yet:

- ratemap generation
- grid score and module detection
- checkpoint evaluation CLI
- medium-scale sanity config
- paper-figure plotting
- ablation runner

