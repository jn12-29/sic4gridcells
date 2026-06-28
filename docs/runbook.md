# SIC runbook

Use `.venv/bin/python` for all Python commands in this repository. Keep generated artifacts under `results/`.

## Smoke

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml
.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16 --seed 0
```

## Medium sanity run

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sic.py --config configs/medium.yaml
.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/medium/checkpoints/step_5000.pt --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
```

Check `results/medium/eval/summary.json`, each `arena_*/module_summary.csv`, `arena_*/grid_stats.csv`, `arena_*/pairwise_distance_stats.csv`, `arena_*/pairwise_distance.png`, `arena_*/fourier_stats.csv`, `arena_*/phase_summary.csv`, `arena_*/state_space_summary.csv`, `arena_*/grid_score_60_histogram.png`, and `arena_*/scale_meters_histogram.png`.

## Paper-scale run

```bash
CUDA_VISIBLE_DEVICES=<id> .venv/bin/python scripts/train_sic.py --config configs/sic_paper.yaml
.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/sic_paper/checkpoints/step_2000000.pt --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
```

Record GPU model, CUDA device id, peak memory, throughput, final checkpoint path, and evaluation command beside the run output.

## Ablations

```bash
.venv/bin/python scripts/run_ablations.py --config configs/ablations.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_ablations.py --config configs/ablations.yaml
```

The ablation runner writes per-variant configs under `results/ablations/configs/`, train outputs under `results/ablations/<variant>/`, evaluation outputs under `results/ablations/<variant>/eval/`, and aggregate `results/ablations/summary.csv` plus `results/ablations/summary.json`.

## Paper-claim checklist

- Multiple modules: inspect `module_summary.csv`, `scale_meters_histogram.png`, and per-unit `module_id` in `grid_stats.csv`.
- Generalization: compare `mean_scale_meters`, grid scores, and ratemaps across 2 m, 3 m, and 4 m arenas.
- Path invariance: inspect `pairwise_distance_stats.csv` and `pairwise_distance.png` for near-zero spatial distance and temporal separation bins.
- Fourier/phase/state-space: inspect `fourier_stats.csv`, `phase_summary.csv`, `state_space_summary.csv`, and `state_space_modules.npz`.
- Ablations: compare `results/ablations/summary.csv` across baseline, no capacity, reduced `sigma_g`, no separation, no invariance, no conformal isometry, and no permutation augmentation.
