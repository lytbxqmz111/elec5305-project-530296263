import numpy as np
import pandas as pd
import pytest

from src.analysis.acoustic_pairs import (
    analyse_audio,
    midi_to_hz,
    validate_pair_selection,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "a",
                "split": "test",
                "instrument_family_str": "brass",
                "instrument_str": "brass_1",
                "instrument_source_str": "acoustic",
                "pitch": 60,
                "velocity": 100,
                "pitch_bin": "middle",
            },
            {
                "item_id": "b",
                "split": "test",
                "instrument_family_str": "string",
                "instrument_str": "string_1",
                "instrument_source_str": "acoustic",
                "pitch": 60,
                "velocity": 100,
                "pitch_bin": "middle",
            },
            {
                "item_id": "c",
                "split": "test",
                "instrument_family_str": "organ",
                "instrument_str": "organ_1",
                "instrument_source_str": "electronic",
                "pitch": 43,
                "velocity": 75,
                "pitch_bin": "lower",
            },
            {
                "item_id": "d",
                "split": "test",
                "instrument_family_str": "keyboard",
                "instrument_str": "keyboard_1",
                "instrument_source_str": "electronic",
                "pitch": 43,
                "velocity": 75,
                "pitch_bin": "lower",
            },
            {
                "item_id": "e",
                "split": "test",
                "instrument_family_str": "brass",
                "instrument_str": "brass_2",
                "instrument_source_str": "acoustic",
                "pitch": 43,
                "velocity": 50,
                "pitch_bin": "lower",
            },
            {
                "item_id": "f",
                "split": "test",
                "instrument_family_str": "brass",
                "instrument_str": "brass_2",
                "instrument_source_str": "acoustic",
                "pitch": 68,
                "velocity": 50,
                "pitch_bin": "upper",
            },
            {
                "item_id": "g",
                "split": "test",
                "instrument_family_str": "organ",
                "instrument_str": "organ_2",
                "instrument_source_str": "electronic",
                "pitch": 43,
                "velocity": 127,
                "pitch_bin": "lower",
            },
            {
                "item_id": "h",
                "split": "test",
                "instrument_family_str": "organ",
                "instrument_str": "organ_2",
                "instrument_source_str": "electronic",
                "pitch": 69,
                "velocity": 127,
                "pitch_bin": "upper",
            },
        ]
    )


def _pairs() -> list[dict[str, str]]:
    return [
        {"pair_id": "p1", "comparison_type": "same_pitch_different_family", "item_a": "a", "item_b": "b", "rationale": "test"},
        {"pair_id": "p2", "comparison_type": "same_pitch_different_family", "item_a": "c", "item_b": "d", "rationale": "test"},
        {"pair_id": "p3", "comparison_type": "same_family_different_pitch", "item_a": "e", "item_b": "f", "rationale": "test"},
        {"pair_id": "p4", "comparison_type": "same_family_different_pitch", "item_a": "g", "item_b": "h", "rationale": "test"},
    ]


def test_pair_selection_enforces_two_of_each_comparison() -> None:
    selected = validate_pair_selection(_manifest(), _pairs())
    assert len(selected) == 8
    assert selected.groupby("comparison_type")["pair_id"].nunique().to_dict() == {
        "same_family_different_pitch": 2,
        "same_pitch_different_family": 2,
    }


def test_pair_selection_rejects_unmatched_same_pitch_velocity() -> None:
    manifest = _manifest()
    manifest.loc[manifest["item_id"].eq("b"), "velocity"] = 25
    with pytest.raises(ValueError, match="does not match velocity"):
        validate_pair_selection(manifest, _pairs())


def test_audio_analysis_returns_all_plot_arrays_and_finite_metrics() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    settings = {
        "sample_rate": sample_rate,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 40,
        "n_mfcc": 13,
        "rolloff_fraction": 0.85,
        "pre_emphasis": 0.97,
        "spectrum_fft": 16384,
        "attack_plot_seconds": 1.0,
    }
    result = analyse_audio(audio, settings)
    for key in (
        "audio",
        "high_resolution_spectrum_db",
        "mel_db",
        "mel_envelope_db",
        "normalised_rms",
    ):
        assert np.isfinite(result[key]).all()
    assert result["mel_db"].shape[0] == 40
    assert 439 < midi_to_hz(69) < 441
