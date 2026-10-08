from pathlib import Path

import pandas as pd

from src.data.nsynth_audit import add_register_bin, validate_manifest


def _row(item: str, split: str, family: str, instrument: str, pitch: int) -> dict:
    return {
        "item_id": item,
        "split": split,
        "instrument_family_str": family,
        "instrument_str": instrument,
        "pitch": pitch,
        "velocity": 50,
    }


def test_register_bins_are_equal_and_closed_on_integer_interval() -> None:
    frame = pd.DataFrame(
        [_row(str(pitch), "train", "a", "i1", pitch) for pitch in range(48, 72)]
    )
    result = add_register_bin(frame, 48, 71)
    assert result.groupby("pitch_bin").size().to_dict() == {
        "lower": 8,
        "middle": 8,
        "upper": 8,
    }


def test_manifest_validation_accepts_balanced_disjoint_data() -> None:
    rows = []
    for split in ("train", "valid", "test"):
        for family in ("a", "b"):
            for index in (1, 2):
                instrument = f"{split}_{family}_{index}"
                row = _row(
                    f"{split}_{family}_{index}", split, family, instrument, 60
                )
                row["pitch_bin"] = "middle"
                rows.append(row)
    manifest = pd.DataFrame(rows)
    summary = validate_manifest(
        manifest,
        {
            "families": ["a", "b"],
            "pitch_low": 48,
            "pitch_high": 71,
        },
        {"max_instrument_share_per_cell": 0.5},
    )
    assert summary["register_velocity_balance"] is True
    assert summary["maximum_observed_instrument_share_per_cell"] == 0.5
