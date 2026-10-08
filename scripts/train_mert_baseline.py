"""Train the reproducible frozen-MERT LinearSVC ordinary baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mert_baseline import load_embedding_matrix, train_mert_ordinary_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mert_baseline.yaml",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    decision = json.loads(
        (ROOT / config["selection_decision_path"]).read_text(encoding="utf-8")
    )
    pitch_range = (
        int(decision["selection"]["pitch_low"]),
        int(decision["selection"]["pitch_high"]),
    )
    print("Loading and validating frozen MERT embedding shards...", flush=True)
    index, embeddings = load_embedding_matrix(
        ROOT, ROOT / config["embedding_index"]
    )
    print(
        f"Loaded {len(index):,} embeddings with dimension {embeddings.shape[1]}.",
        flush=True,
    )
    report = train_mert_ordinary_baseline(
        index,
        embeddings,
        config["classifier"],
        pitch_range,
        ROOT / config["model_dir"],
        ROOT / config["results_dir"],
        ROOT / config["figures_dir"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
