# SIC runbook

Use `uv run python` for all Python commands in this repository. Keep generated artifacts under `results/`.

## Smoke

```bash
uv run python -m pytest
uv run python scripts/train_sic.py --config configs/smoke.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16 --seed 0
```

The smoke evaluation writes `summary.json`, `config.yaml`, `run.log`, `eval_events.jsonl`, and one per-arena directory such as `arena_1p0/`. Per-arena artifacts include `rollout_arrays.npz`, `ratemaps.npz`, `occupancy.npz`, `sacs.npz`, `grid_metrics.npz`, `grid_stats.csv`, `grid_stats.json`, `module_summary.csv`, `module_summary.json`, `trajectory_stats.json`, `pairwise_distance_stats.csv`, `pairwise_distance_stats.json`, `pairwise_distance.png`, `fourier_stats.csv`, `fourier_stats.json`, `phase_summary.csv`, `phase_summary.json`, `state_space_summary.csv`, `state_space_summary.json`, `state_space_modules.npz`, `grid_score_60_histogram.png`, `scale_meters_histogram.png`, `summary.png`, `ratemaps.pdf`, and `sacs.pdf`.

Unvisited ratemap bins are stored as `NaN`; coverage is determined from `occupancy_counts`; visited zero responses remain `0.0`; non-finite visited responses are reported as invalid in JSON summaries. SAC/grid scoring uses finite ratemap bins as its overlap mask, grid scale is reported in both SAC pixels and meters, and random-walk step scale is arena-size based rather than shrinking with `--steps`.

Evaluation defaults to `--start-mode origin`, which keeps reset model state aligned with position bins. `--start-mode uniform` is only valid for checkpoints trained with `model.initial_position_encoding: additive_mlp` and `data.initial_position_mode: uniform_box`. If `--seed` is omitted, evaluation uses the checkpoint config seed.
Evaluation defaults to `--trajectory-mode reflect`, the original bounded random-walk sampler. Use `--trajectory-mode smooth_avoid_walls` for smoother wall-avoiding diagnostic trajectories closer to the paper's evaluation description.
The training, evaluation, and ablation CLIs also accept `--log-level` for stderr logging; stdout still prints the final completion lines used by the smoke commands above.
Each workflow writes a human-readable `run.log` plus a strict JSONL event file: `train_events.jsonl`, `eval_events.jsonl`, or `ablation_events.jsonl` at the corresponding output root.

## Medium sanity run

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sic.py --config configs/medium.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/medium/checkpoints/step_5000.pt --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
```

If training is interrupted, resume from the latest checkpoint with the same
config. To extend beyond the checkpoint config, change only
`train.max_optimizer_steps`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sic.py --config configs/medium.yaml --resume results/medium/checkpoints/step_500.pt
```

Check `results/medium/eval/summary.json`, each `arena_*/module_summary.csv`, `arena_*/grid_stats.csv`, `arena_*/pairwise_distance_stats.csv`, `arena_*/pairwise_distance.png`, `arena_*/fourier_stats.csv`, `arena_*/phase_summary.csv`, `arena_*/state_space_summary.csv`, `arena_*/grid_score_60_histogram.png`, and `arena_*/scale_meters_histogram.png`.

## Paper-scale run

```bash
CUDA_VISIBLE_DEVICES=<id> uv run python scripts/train_sic.py --config configs/sic_paper.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/sic_paper/checkpoints/step_2000000.pt --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
```

Record GPU model, CUDA device id, peak memory, throughput, final checkpoint path, and evaluation command beside the run output.

## Ablations

```bash
uv run python scripts/run_ablations.py --config configs/ablations.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablations.py --config configs/ablations.yaml
```

The ablation runner writes per-variant configs under `results/ablations/configs/`, train outputs under `results/ablations/<variant>/`, evaluation outputs under `results/ablations/<variant>/eval/`, `run.log`, `ablation_events.jsonl`, and aggregate `results/ablations/summary.csv` plus `results/ablations/summary.json`.

## Paper-claim checklist

- Multiple modules: inspect `module_summary.csv`, `scale_meters_histogram.png`, and per-unit `module_id` in `grid_stats.csv`.
- Generalization: compare `mean_scale_meters`, grid scores, and ratemaps across 2 m, 3 m, and 4 m arenas.
- Path invariance: inspect `pairwise_distance_stats.csv` and `pairwise_distance.png` for near-zero spatial distance and temporal separation bins.
- Fourier/phase/state-space: inspect `fourier_stats.csv`, `phase_summary.csv`, `state_space_summary.csv`, and `state_space_modules.npz`.
- Ablations: compare `results/ablations/summary.csv` across baseline, no capacity, reduced `sigma_g`, no separation, no invariance, no conformal isometry, and no permutation augmentation.
