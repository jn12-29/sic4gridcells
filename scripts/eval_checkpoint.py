from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SIC checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    parser.add_argument("--output-dir", help="Directory for evaluation artifacts.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--arena-sizes", default="2.0,3.0,4.0", help="Comma-separated arena sizes.")
    parser.add_argument("--nbins", type=int, default=32)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, help="Evaluation trajectory seed. Defaults to checkpoint config seed.")
    parser.add_argument(
        "--start-mode",
        choices=("origin", "uniform"),
        default="origin",
        help="Initial position mode for evaluation trajectories.",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("reflect", "smooth_avoid_walls"),
        default="reflect",
        help="Evaluation trajectory sampler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arena_sizes = tuple(float(item) for item in args.arena_sizes.split(",") if item)
    result = evaluate_checkpoint(
        args.checkpoint,
        args.output_dir,
        device=args.device,
        arena_sizes=arena_sizes,
        nbins=args.nbins,
        n_trajectories=args.trajectories,
        steps_per_trajectory=args.steps,
        start_mode=args.start_mode,
        trajectory_mode=args.trajectory_mode,
        seed=args.seed,
    )
    print(f"finished output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
