import numpy as np
import pandas as pd
import pytest

from src.models.cross_register import align_cache, fit_source, validate_manifest


def _manifest():
    rows = []
    for split in ("train", "valid", "test"):
        for register, pitch in (("lower", 45), ("middle", 55), ("upper", 65)):
            for family in ("a", "b"):
                for number in range(2):
                    rows.append({"split": split, "item_id": f"{split}_{register}_{family}_{number}", "instrument_family_str": family, "instrument_str": f"{split}_{family}", "pitch": pitch, "pitch_bin": register, "velocity": 100})
    return pd.DataFrame(rows)


DECISION = {"selection": {"families": ["a", "b"], "pitch_bins": {"lower": [43, 51], "middle": [52, 60], "upper": [61, 69]}}}


def test_manifest_rejects_instrument_leakage():
    frame = _manifest()
    validate_manifest(frame, DECISION)
    frame.loc[frame["split"].eq("test") & frame["instrument_family_str"].eq("a"), "instrument_str"] = "valid_a"
    validate_manifest(frame, DECISION)
    frame.loc[frame["split"].eq("test") & frame["instrument_family_str"].eq("a"), "instrument_str"] = "train_a"
    with pytest.raises(ValueError, match="overlaps"):
        validate_manifest(frame, DECISION)


def test_cache_alignment_checks_metadata():
    frame = _manifest().iloc[:3].copy()
    cache = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    expected = np.array([cache.index[cache["item_id"].eq(item)][0] for item in frame["item_id"]])
    assert np.array_equal(align_cache(frame, cache), expected)
    cache.loc[0, "pitch"] = 99
    with pytest.raises(ValueError, match="pitch"):
        align_cache(frame, cache)


def test_source_training_uses_only_source_register():
    frame = validate_manifest(_manifest(), DECISION)
    matrix = np.array([[0.0 if family == "a" else 5.0, float(pitch)] for family, pitch in zip(frame["instrument_family_str"], frame["pitch"])])
    settings = {"c_values": [0.1], "max_iter": 1000, "random_seed": 1, "class_weight": "balanced"}
    _, predictions, report = fit_source(frame, matrix, "lower", settings, ["a", "b"])
    assert report["train_rows"] == 4
    assert report["valid_rows"] == 4
    assert set(predictions["pitch_bin"]) == {"lower", "upper"}
    assert len(predictions) == 8
