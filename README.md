# sic4gridcells

Plain PyTorch reproduction scaffold for the SIC grid-cell model from Schaeffer et al., "Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells".

The current repository implements a runnable SIC reproduction slice: SIC velocity permutation batches, velocity-conditioned RNN rollout, separation/invariance/capacity/conformal-isometry losses, training, checkpoint evaluation, evaluation validation, ratemaps, SAC/grid scoring, smoke/medium/paper configs, ablation orchestration, paper-suite analysis/figure generation, and unit tests. See `docs/runbook.md` for longer run commands, and `docs/sic-implementation-plan.md` plus `docs/sic-reproduction-plan.md` for reproduction scope and remaining paper-level work.

## Setup

Use the local uv-managed environment in this repository.

```bash
uv venv --python 3.12
uv pip install -e .
```

If dependencies need to be installed or refreshed, run the install command from the repository root so uv uses the local `.venv`:

```bash
uv pip install -e .
```

## Quick Start

Run the test suite:

```bash
uv run python -m pytest
```

Run the 10-step smoke training job:

```bash
uv run python scripts/train_sic.py --config configs/smoke.yaml
```

Expected CLI output:

```text
finished step=10 output_dir=results/smoke
checkpoint=results/smoke/checkpoints/step_10.pt
```

The CLI also accepts `--log-level` for stderr logging; runtime logs are written to `results/smoke/run.log` and structured events to `results/smoke/train_events.jsonl`.

The smoke run writes:

- `results/smoke/config.yaml`
- `results/smoke/run.log`
- `results/smoke/train_events.jsonl`
- `results/smoke/metrics.jsonl`
- `results/smoke/tensorboard/`
- `results/smoke/checkpoints/step_5.pt`
- `results/smoke/checkpoints/step_10.pt`
- `results/smoke/checkpoints/latest.pt`
- `results/smoke/checkpoints/checkpoint_manifest.json`

`results/` is ignored by git.

Resume an interrupted run with the same config. To extend beyond the
checkpoint config, change only `train.max_optimizer_steps`:

```bash
uv run python scripts/train_sic.py --config configs/medium.yaml --resume results/medium/checkpoints/step_500.pt
```

Fresh training runs refuse to reuse an output directory that already contains
files. Use `--resume` for interrupted runs, or pass `--overwrite-output` only
when intentionally rerunning into the same directory. Training metrics include
step timing, throughput, disk free space, and CUDA memory fields when CUDA is
active.

Run a short training profile before committing to a longer config:

```bash
uv run python scripts/profile_train.py --config configs/smoke.yaml --output-dir results/smoke-profile --steps 2 --device cpu
```

The profile command runs a short pilot through the same training loop and writes
`profile_summary.json` with observed step time, checkpoint size, and rough
runtime/checkpoint-storage estimates for the profiled config. Treat it as
planning evidence, not as completed medium or paper training/evaluation.

Evaluate the smoke checkpoint:

```bash
uv run python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16 --seed 0
```

The evaluation writes `summary.json`, `config.yaml`, and per-arena ratemap, SAC, grid-stat, trajectory, pairwise-distance, Fourier, phase, state-space, PDF, and PNG artifacts under `results/smoke/eval/`. See `docs/runbook.md` for the artifact checklist used for medium, paper-scale, and ablation runs.
It also writes `run.log` and `eval_events.jsonl` alongside `summary.json`.

Evaluation defaults to `--start-mode origin`, which keeps reset model state aligned with position bins. `--start-mode uniform` is only valid for checkpoints trained with `model.initial_position_encoding: additive_mlp` and `data.initial_position_mode: uniform_box`. If `--seed` is omitted, evaluation uses the checkpoint config seed.
Evaluation also defaults to `--trajectory-mode reflect` for backward-compatible bounded walks. Use `--trajectory-mode smooth_avoid_walls` for smoother wall-avoiding diagnostic trajectories.

The evaluation CLI also accepts `--log-level`; runtime logs go to `run.log` and structured events go to `eval_events.jsonl` beside the evaluation summary.
Evaluation also refuses to reuse an existing output directory unless
`--overwrite-output` is passed.

Validate the smoke evaluation artifacts with relaxed quality thresholds:

```bash
uv run python scripts/validate_eval.py --output-dir results/smoke/eval --arena-sizes 1.0 --min-coverage 0.0 --min-active-units 0 --min-module-count 0
```

The validation CLI checks required artifacts plus coverage, active-unit, invalid-response, and module evidence thresholds. Default thresholds are conservative and are intended for claim gates; validation blockers mean the evaluation output is incomplete or insufficient evidence, not that training or evaluation crashed.

Dry-run the paper-suite orchestrator without launching training:

```bash
uv run python scripts/run_paper_suite.py --config configs/paper_suite_smoke.yaml --dry-run --overwrite-output
```

Build paper-result figures from an existing suite directory:

```bash
uv run python scripts/build_paper_figures.py --suite-dir results/paper_suite/smoke --output-dir results/paper_suite/smoke/figures
```

The figure builder only reads existing suite, evaluation, and analysis artifacts. It writes stable `fig_*.png`, `fig_*.pdf`, `summary_tables/`, and `figure_manifest.json` outputs.

## Configs

- `configs/smoke.yaml`: small CPU smoke run for tests and workflow checks.
- `configs/medium.yaml`: medium sanity-run profile (`B=16`, `T=30`, `N=64`).
- `configs/sic_paper.yaml`: paper-scale training hyperparameters from the reproduction plan.
- `configs/ablations.yaml`: ablation orchestration plan for `scripts/run_ablations.py`.
- `configs/paper_suite_smoke.yaml`: smoke paper-suite orchestration plan for dry-runs and local workflow checks.
- `configs/paper_suite.yaml`: paper-suite orchestration plan for paper-scale baseline seeds.
- `scripts/run_ablations.py` accepts `--log-level`; the ablation root writes `run.log` and `ablation_events.jsonl`.
- `scripts/run_ablations.py` accepts `--resume-existing` to resume variants from their latest checkpoints, `--skip-completed` to skip variants that already reached `train.max_optimizer_steps`, and `--overwrite-output` for intentional fresh reruns into existing directories.
- `scripts/run_paper_suite.py` accepts `--dry-run`, `--resume-existing`, `--skip-completed`, `--overwrite-output`, and `--log-level`; suite outputs live under `results/paper_suite/<run_id>/`.

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
scripts/profile_train.py  short training profile CLI
scripts/eval_checkpoint.py checkpoint evaluation CLI
scripts/validate_eval.py evaluation artifact and quality validation CLI
scripts/run_ablations.py  ablation orchestration CLI
scripts/run_paper_suite.py paper-suite orchestration CLI
scripts/build_paper_figures.py paper-result figure builder CLI
src/sic4gridcells/        package source
tests/                    pytest suite
```

Key modules:

- `sic4gridcells.config`: YAML loading, defaults, validation, effective-config saving.
- `sic4gridcells.data`: SIC base velocity sampling and random permutation batches.
- `sic4gridcells.model`: NormReLU and velocity-conditioned RNN.
- `sic4gridcells.losses`: SIC losses and tiny naive pairwise implementation for tests.
- `sic4gridcells.train`: training loop, metrics, TensorBoard logging, checkpointing.
- `sic4gridcells.profiling`: short-run profiling summaries for long-run planning.
- `sic4gridcells.evaluate`: checkpoint reload, bounded random-walk evaluation, artifact writing.
- `sic4gridcells.validation`: evaluation artifact completeness and quality-gate reporting.
- `sic4gridcells.analysis`: ratemap, SAC, grid score, and grid-scale utilities.
- `sic4gridcells.analysis_ext`: paper-suite analysis tables for modules, cross-arena stability, path invariance, Fourier/phase, and state-space summaries.
- `sic4gridcells.figure_data`: figure-ready table discovery and manifest dependency loading.
- `sic4gridcells.paper_figures`: paper-result PNG/PDF rendering from existing artifacts.
- `sic4gridcells.paper_suite`: suite manifest, dry-run, and orchestration helpers.
- `sic4gridcells.plotting`: PDF and PNG evaluation figures.
- `sic4gridcells.runtime`: output safety, atomic checkpoint writes, latest-checkpoint discovery, and runtime diagnostics.
- `docs/runbook.md`: medium, paper-scale, and ablation command sequence.

## Current Limits

This is not yet a completed paper reproduction result. The current code verifies the core training, evaluation, validation, analysis, and figure-generation contracts, but these paper-level pieces still require completed long runs and review:

- medium-scale training has a config but has not been run to completion here
- paper-scale training and multi-seed sweeps
- paper-claim figures built from trained, validation-passing results
- toroidal manifold confirmation beyond the state-space summary artifacts
