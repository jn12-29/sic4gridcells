from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic4gridcells.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SIC grid cell model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume",
        help="Path to a checkpoint to resume from. Only train.max_optimizer_steps may differ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args.config, resume_checkpoint=args.resume)
    print(f"finished step={result.final_step} output_dir={result.output_dir}")
    print(f"checkpoint={result.checkpoint_path}")


if __name__ == "__main__":
    main()
