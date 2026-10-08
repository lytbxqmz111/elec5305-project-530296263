import numpy as np
import pandas as pd
import pytest

from src.models.pitch_probe import (
    assign_pitch_bins,
    balanced_row_indices,
    pitch_metrics,
)


PITCH_BINS = {
    "lower": [43, 51],
    "middle": [52, 60],
    "upper": [61, 69],
}


def test_pitch_bin_boundaries_are_inclusive() -> None:
    labels = assign_pitch_bins(pd.Series([43, 51, 52, 60, 61, 69]), PITCH_BINS)
    assert labels.tolist() == [
        "lower",
        "lower",
        "middle",
        "middle",
        "upper",
        "upper",
    ]


def test_pitch_bin_assignment_rejects_outside_pitch() -> None:
    with pytest.raises(ValueError, match="outside configured bins"):
        assign_pitch_bins(pd.Series([42, 43]), PITCH_BINS)


def test_pitch_metrics_use_all_three_labels() -> None:
    y_true = np.asarray(["lower", "middle", "upper", "upper"])
    y_pred = np.asarray(["lower", "middle", "middle", "upper"])
    metrics = pitch_metrics(y_true, y_pred)
    assert metrics["rows"] == 4
    assert set(metrics["per_pitch_bin_recall"]) == {"lower", "middle", "upper"}
    assert metrics["per_pitch_bin_recall"]["upper"] == 0.5


def test_balanced_alignment_accepts_equivalent_integer_dtypes() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["test"],
            "item_id": ["note_1"],
            "instrument_family_str": ["bass"],
            "pitch": pd.Series([43], dtype="int64"),
            "velocity": pd.Series([75], dtype="int64"),
        }
    )
    manifest = metadata.copy()
    manifest["pitch"] = manifest["pitch"].astype("int8")
    manifest["velocity"] = manifest["velocity"].astype("int8")
    assert balanced_row_indices(metadata, manifest).tolist() == [0]
