"""Unit tests for the NSynth metadata loader and split checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.nsynth_metadata import (
    MetadataValidationError,
    load_split,
    validate_official_split_separation,
    validate_split,
    verify_audio_inventory,
)


def _example(note_str: str, instrument_str: str, family: str) -> dict:
    return {
        "qualities": [0] * 10,
        "pitch": 60,
        "note": 1,
        "instrument_source_str": "acoustic",
        "velocity": 75,
        "instrument_str": instrument_str,
        "instrument": 1,
        "sample_rate": 16000,
        "qualities_str": [],
        "instrument_source": 0,
        "note_str": note_str,
        "instrument_family": 1,
        "instrument_family_str": family,
    }


def _write_split(root: Path, name: str, examples: dict) -> Path:
    split_dir = root / name
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True)
    (split_dir / "examples.json").write_text(
        json.dumps(examples), encoding="utf-8"
    )
    for note_str in examples:
        (audio_dir / f"{note_str}.wav").write_bytes(b"RIFF")
    return split_dir


def test_load_validate_and_audio_inventory(tmp_path: Path) -> None:
    note_str = "brass_acoustic_001-060-075"
    split_dir = _write_split(
        tmp_path,
        "nsynth-train",
        {note_str: _example(note_str, "brass_acoustic_001", "brass")},
    )

    frame = load_split(split_dir, "train", tmp_path)
    report = validate_split(frame, "train", expected_count=1)
    audio_report = verify_audio_inventory(frame, split_dir / "audio", "train")

    assert report["rows"] == 1
    assert report["sample_rates"] == [16000]
    assert audio_report["wav_files"] == 1
    assert frame.loc[0, "audio_relpath"].endswith(f"{note_str}.wav")


def test_train_overlap_is_rejected() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "instrument_str": ["shared", "shared", "test_only"],
        }
    )
    with pytest.raises(MetadataValidationError):
        validate_official_split_separation(metadata)


def test_valid_test_overlap_is_reported_but_allowed() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "instrument_str": ["train_only", "shared_eval", "shared_eval"],
        }
    )
    report = validate_official_split_separation(metadata)
    assert report["train_valid_instrument_overlap"] == 0
    assert report["train_test_instrument_overlap"] == 0
    assert report["valid_test_instrument_overlap"] == 1
