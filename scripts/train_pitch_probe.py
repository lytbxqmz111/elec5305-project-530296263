"""Train and evaluate handcrafted and frozen-MERT coarse-pitch probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pitch_probe import run_pitch_probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pitch_probe.yaml",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    report = run_pitch_probes(ROOT, config)
    compact = {
        "target": report["target"],
        "pitch_range": report["pitch_range"],
        "balanced_test_rows": report["balanced_test_rows"],
        "representations": {
            name: {
                "selected_C": result["selected_C"],
                "ordinary_test": result["metrics"]["ordinary_test"],
                "balanced_test": result["metrics"]["balanced_test"],
            }
            for name, result in report["representations"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
