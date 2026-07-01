from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.logging_utils import VALID_LOG_LEVELS, cli_logging_context
from sic4gridcells.paper_figures import build_paper_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SIC paper-result figures.")
    parser.add_argument("--suite-dir", required=True, help="Paper suite output directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for figure artifacts.")
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
        result = build_paper_figures(args.suite_dir, args.output_dir)
    print(f"figures={result.output_dir}")
    print(f"manifest={result.manifest_path}")


if __name__ == "__main__":
    main()
