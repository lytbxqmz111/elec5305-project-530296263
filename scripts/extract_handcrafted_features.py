"""Extract resumable handcrafted feature-cache parts for the ordinary split."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.handcrafted import extract_file, feature_names


IDENTITY_COLUMNS = [
    "item_id",
    "split",
    "instrument_family_str",
    "instrument_str",
    "pitch",
    "velocity",
    "audio_relpath",
]


def _extract_task(task: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    row, settings, root_string = task
    path = Path(root_string) / row["audio_relpath"]
    values = extract_file(path, settings)
    return {**row, **values}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_metadata(config: dict[str, Any]) -> pd.DataFrame:
    metadata = pd.read_parquet(ROOT / config["metadata_path"])
    if config["dataset"].get("use_selected_families", True):
        decision = json.loads(
            (ROOT / config["selection_decision_path"]).read_text(encoding="utf-8")
        )
        families = decision["selection"]["families"]
        metadata = metadata.loc[
            metadata["instrument_family_str"].isin(families)
        ].copy()
    return metadata.sort_values(["split", "item_id"]).reset_index(drop=True)


def extract_cache(
    config_path: Path,
    workers: int | None = None,
    limit_per_split: int | None = None,
    cache_tag: str = "ordinary",
) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = selected_metadata(config)
    if limit_per_split is not None:
        metadata = (
            metadata.groupby(
                ["split", "instrument_family_str"],
                group_keys=False,
                observed=True,
            )
            .head(limit_per_split)
            .reset_index(drop=True)
        )
    settings = config["features"]
    worker_count = workers or int(config["extraction"]["workers"])
    rows_per_part = int(config["extraction"]["rows_per_part"])
    cache_dir = ROOT / config["feature_cache_dir"] / cache_tag
    parts_dir = cache_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    written = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for split in ("train", "valid", "test"):
            split_frame = metadata.loc[metadata["split"].eq(split), IDENTITY_COLUMNS]
            part_count = math.ceil(len(split_frame) / rows_per_part)
            for part_index in range(part_count):
                output_path = parts_dir / f"{split}_{part_index:04d}.parquet"
                start = part_index * rows_per_part
                stop = min(start + rows_per_part, len(split_frame))
                expected_rows = stop - start
                if output_path.is_file():
                    cached = pd.read_parquet(output_path, columns=["item_id"])
                    if len(cached) == expected_rows:
                        skipped += expected_rows
                        continue
                    raise RuntimeError(f"Incomplete cache part: {output_path}")
                records = split_frame.iloc[start:stop].to_dict("records")
                tasks = ((row, settings, str(ROOT)) for row in records)
                results = list(executor.map(_extract_task, tasks, chunksize=16))
                result_frame = pd.DataFrame(results)
                columns = IDENTITY_COLUMNS + feature_names(int(settings["n_mfcc"]))
                result_frame[columns].to_parquet(output_path, index=False)
                written += len(result_frame)
                print(
                    f"{split}: part {part_index + 1}/{part_count}, "
                    f"written={written:,}, resumed={skipped:,}",
                    flush=True,
                )

    report = {
        "cache_tag": cache_tag,
        "rows": int(len(metadata)),
        "written_this_run": written,
        "resumed_rows": skipped,
        "workers": worker_count,
        "feature_count": len(feature_names(int(settings["n_mfcc"]))),
        "elapsed_seconds": time.time() - start_time,
        "split_rows": metadata["split"].value_counts().to_dict(),
    }
    (cache_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "handcrafted_baseline.yaml",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Pilot only: maximum rows per split and family.",
    )
    parser.add_argument("--cache-tag", default="ordinary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_cache(
        args.config.resolve(),
        workers=args.workers,
        limit_per_split=args.limit_per_split,
        cache_tag=args.cache_tag,
    )
