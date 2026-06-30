from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.logging_utils import VALID_LOG_LEVELS, cli_logging_context
from sic4gridcells.profiling import profile_training_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short SIC training profile.")
    parser.add_argument("--config", required=True, help="Training config to profile.")
    parser.add_argument("--output-dir", required=True, help="Output directory for profile artifacts.")
    parser.add_argument("--steps", type=int, default=20, help="Optimizer steps for the pilot run.")
    parser.add_argument("--device", help="Override config device for the pilot run.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow the profile run to reuse an existing output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=VALID_LOG_LEVELS,
        help="Console log level for stderr logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with cli_logging_context(args.log_level):
        summary = profile_training_run(
            args.config,
            args.output_dir,
            steps=args.steps,
            device=args.device,
            overwrite_output=args.overwrite_output,
        )
    print(f"profile finished step={summary.final_step} output_dir={summary.output_dir}")
    print(f"checkpoint={summary.checkpoint_path}")
    print(f"profile_summary={Path(summary.output_dir) / 'profile_summary.json'}")
    if summary.mean_step_seconds is not None:
        print(f"mean_step_seconds={summary.mean_step_seconds:.6f}")
    if summary.estimated_hours_for_config_steps is not None:
        print(f"estimated_hours_for_config_steps={summary.estimated_hours_for_config_steps:.3f}")


if __name__ == "__main__":
    main()
