from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.logging_utils import VALID_LOG_LEVELS, cli_logging_context
from sic4gridcells.paper_suite import run_paper_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured SIC paper suite.")
    parser.add_argument("--config", required=True, help="Path to a paper suite YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and validate per-run configs without launching execution.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume suite runs from latest checkpoints when available.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs whose latest checkpoint already reached max_optimizer_steps.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow fresh suite outputs to reuse existing output directories.",
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
    command_line_args = {
        "config": args.config,
        "dry_run": args.dry_run,
        "resume_existing": args.resume_existing,
        "skip_completed": args.skip_completed,
        "overwrite_output": args.overwrite_output,
        "log_level": args.log_level,
    }
    with cli_logging_context(args.log_level):
        results = run_paper_suite(
            args.config,
            dry_run=args.dry_run,
            resume_existing=args.resume_existing,
            skip_completed=args.skip_completed,
            overwrite_output=args.overwrite_output,
            command_line_args=command_line_args,
        )
    for name, result in results.items():
        if result.status == "validated":
            print(f"validated {name}: {result.config_path}")
        elif result.status == "skipped":
            suffix = f": {result.reason}" if result.reason else ""
            print(f"skipped {name}{suffix}")
        elif result.status == "failed":
            print(f"failed {name}")
        elif result.status == "finished" and result.run_result is not None:
            print(
                f"finished {name} step={result.run_result.final_step} "
                f"output_dir={result.run_result.output_dir}"
            )
            print(f"checkpoint={result.run_result.checkpoint_path}")
            if result.evaluation_result is not None:
                print(f"evaluation={result.evaluation_result.output_dir}")
            if result.analysis_output_dir is not None:
                print(f"analysis={result.analysis_output_dir}")


if __name__ == "__main__":
    main()
