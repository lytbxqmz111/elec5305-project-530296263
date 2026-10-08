"""Train the reproducible ordinary handcrafted-feature baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.handcrafted import feature_names
from src.models.handcrafted_baseline import load_feature_cache, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "handcrafted_baseline.yaml",
    )
    parser.add_argument("--cache-tag", default="ordinary")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    decision = json.loads(
        (ROOT / config["selection_decision_path"]).read_text(encoding="utf-8")
    )
    ordinary_pitch_range = (
        int(decision["selection"]["pitch_low"]),
        int(decision["selection"]["pitch_high"]),
    )
    cache_dir = ROOT / config["feature_cache_dir"] / args.cache_tag
    features = load_feature_cache(cache_dir)
    columns = feature_names(int(config["features"]["n_mfcc"]))
    suffix = Path() if args.cache_tag == "ordinary" else Path(args.cache_tag)
    report = train_and_evaluate(
        features,
        columns,
        config["classifier"],
        ROOT / config["model_dir"] / suffix,
        ROOT / config["results_dir"] / suffix,
        ROOT / config["figures_dir"] / suffix,
        ordinary_test_pitch_range=ordinary_pitch_range,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
