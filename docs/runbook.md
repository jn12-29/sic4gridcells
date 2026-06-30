# SIC runbook

Use `uv run python` for all Python commands in this repository. Keep generated artifacts under `results/`.

## Smoke

```bash
uv run python -m pytest
uv run python scripts/train_sic.py --config configs/smoke.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16 --seed 0
uv run python scripts/validate_eval.py --output-dir results/smoke/eval --arena-sizes 1.0 --min-coverage 0.0 --min-active-units 0 --min-module-count 0
```

Fresh train, evaluation, and ablation runs refuse to reuse non-empty output
directories. Use `--resume` for interrupted training, `--resume-existing` for
ablation batches, or `--overwrite-output` only for intentional reruns.

The smoke evaluation writes `summary.json`, `config.yaml`, `run.log`, `eval_events.jsonl`, and one per-arena directory such as `arena_1p0/`. Per-arena artifacts include `rollout_arrays.npz`, `ratemaps.npz`, `occupancy.npz`, `sacs.npz`, `grid_metrics.npz`, `grid_stats.csv`, `grid_stats.json`, `module_summary.csv`, `module_summary.json`, `trajectory_stats.json`, `pairwise_distance_stats.csv`, `pairwise_distance_stats.json`, `pairwise_distance.png`, `fourier_stats.csv`, `fourier_stats.json`, `phase_summary.csv`, `phase_summary.json`, `state_space_summary.csv`, `state_space_summary.json`, `state_space_modules.npz`, `grid_score_60_histogram.png`, `scale_meters_histogram.png`, `summary.png`, `ratemaps.pdf`, and `sacs.pdf`.

`scripts/validate_eval.py` reports required-artifact blockers and quantitative
quality blockers. The smoke command relaxes quality thresholds so the short
run checks artifact completeness without implying paper-level evidence.

Unvisited ratemap bins are stored as `NaN`; coverage is determined from `occupancy_counts`; visited zero responses remain `0.0`; non-finite visited responses are reported as invalid in JSON summaries. SAC/grid scoring uses finite ratemap bins as its overlap mask, grid scale is reported in both SAC pixels and meters, and random-walk step scale is arena-size based rather than shrinking with `--steps`.

Evaluation defaults to `--start-mode origin`, which keeps reset model state aligned with position bins. `--start-mode uniform` is only valid for checkpoints trained with `model.initial_position_encoding: additive_mlp` and `data.initial_position_mode: uniform_box`. If `--seed` is omitted, evaluation uses the checkpoint config seed.
Evaluation defaults to `--trajectory-mode reflect`, the original bounded random-walk sampler. Use `--trajectory-mode smooth_avoid_walls` for smoother wall-avoiding diagnostic trajectories closer to the paper's evaluation description.
The training, evaluation, and ablation CLIs also accept `--log-level` for stderr logging; stdout still prints the final completion lines used by the smoke commands above.
Each workflow writes a human-readable `run.log` plus a strict JSONL event file: `train_events.jsonl`, `eval_events.jsonl`, or `ablation_events.jsonl` at the corresponding output root.
Training checkpoints are written atomically. In addition to `step_<step>.pt`,
the checkpoint directory contains `latest.pt` and
`checkpoint_manifest.json`; use these for recovery or ablation resumption.
Training metrics include `perf/step_seconds`, `perf/points_per_second`,
`disk/output_free_gb`, and CUDA memory metrics when CUDA is active.

## Medium sanity run

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sic.py --config configs/medium.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/medium/checkpoints/step_5000.pt --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
uv run python scripts/validate_eval.py --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --json-output results/medium/eval/validation.json --allow-fail
```

If training is interrupted, resume from the latest checkpoint with the same
config. To extend beyond the checkpoint config, change only
`train.max_optimizer_steps`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sic.py --config configs/medium.yaml --resume results/medium/checkpoints/step_500.pt
```

Check `results/medium/eval/summary.json`, each `arena_*/module_summary.csv`, `arena_*/grid_stats.csv`, `arena_*/pairwise_distance_stats.csv`, `arena_*/pairwise_distance.png`, `arena_*/fourier_stats.csv`, `arena_*/phase_summary.csv`, `arena_*/state_space_summary.csv`, `arena_*/grid_score_60_histogram.png`, and `arena_*/scale_meters_histogram.png`.
If `validation.json` contains blockers, treat the medium run as diagnostic only.

## Paper-scale run

```bash
CUDA_VISIBLE_DEVICES=<id> uv run python scripts/train_sic.py --config configs/sic_paper.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/sic_paper/checkpoints/step_2000000.pt --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
uv run python scripts/validate_eval.py --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --json-output results/sic_paper/eval/validation.json
```

Confirm `metrics.jsonl` contains throughput and CUDA peak-memory fields, and
record GPU model, CUDA device id, final checkpoint path, and evaluation command
beside the run output.

## Ablations

```bash
uv run python scripts/run_ablations.py --config configs/ablations.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablations.py --config configs/ablations.yaml
for variant in baseline no_capacity reduced_sigma_g no_separation no_invariance no_conformal_isometry no_permutation_augmentation; do
  uv run python scripts/validate_eval.py --output-dir "results/ablations/${variant}/eval" --arena-sizes 2.0,3.0,4.0 --json-output "results/ablations/${variant}/eval/validation.json" --allow-fail
done
```

The ablation runner writes per-variant configs under `results/ablations/configs/`, train outputs under `results/ablations/<variant>/`, evaluation outputs under `results/ablations/<variant>/eval/`, `run.log`, `ablation_events.jsonl`, and aggregate `results/ablations/summary.csv` plus `results/ablations/summary.json`.
After interruption, continue with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablations.py --config configs/ablations.yaml --resume-existing --skip-completed
```

The aggregate CSV includes coverage, zero/invalid/active response counts, and
both 60-degree and 90-degree grid-score means when evaluation summaries are
available.
Ablation validation commands use `--allow-fail` to keep collecting all variant
reports; variants with blockers are diagnostic only.

## Paper-claim checklist

- Multiple modules: inspect `module_summary.csv`, `scale_meters_histogram.png`, and per-unit `module_id` in `grid_stats.csv`.
- Generalization: compare `mean_scale_meters`, grid scores, and ratemaps across 2 m, 3 m, and 4 m arenas.
- Path invariance: inspect `pairwise_distance_stats.csv` and `pairwise_distance.png` for near-zero spatial distance and temporal separation bins.
- Fourier/phase/state-space: inspect `fourier_stats.csv`, `phase_summary.csv`, `state_space_summary.csv`, and `state_space_modules.npz`.
- Validation gate: inspect each `validation.json`; do not make paper-level claims while blockers remain.
- Ablations: compare `results/ablations/summary.csv` across baseline, no capacity, reduced `sigma_g`, no separation, no invariance, no conformal isometry, and no permutation augmentation.
