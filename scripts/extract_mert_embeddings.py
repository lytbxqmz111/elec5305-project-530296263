"""CLI for frozen MERT inference, mean pooling, caching, and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.mert import extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract resumable frozen MERT embeddings for selected NSynth families."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mert_embeddings.yaml",
    )
    parser.add_argument("--cache-tag", default="ordinary")
    parser.add_argument(
        "--limit-per-split-family",
        type=int,
        help="Pilot only: restrict every split/family to this many rows.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing shards and manifest coverage without inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = extract_embeddings(
        args.config.resolve(),
        cache_tag=args.cache_tag,
        limit_per_split_family=args.limit_per_split_family,
        validate_only=args.validate_only,
    )
    print(json.dumps(report, indent=2))
