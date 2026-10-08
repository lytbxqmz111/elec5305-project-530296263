"""Evaluate fitted handcrafted and MERT systems on the frozen balanced test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.balanced_evaluation import run_balanced_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "balanced_evaluation.yaml",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    report = run_balanced_evaluation(ROOT, config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
