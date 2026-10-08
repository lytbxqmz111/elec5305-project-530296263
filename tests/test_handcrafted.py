import numpy as np
import pandas as pd

from src.features.handcrafted import extract_features, feature_names, mel_filterbank
from src.models.handcrafted_baseline import train_and_evaluate


SETTINGS = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 40,
    "n_mfcc": 13,
    "rolloff_fraction": 0.85,
    "pre_emphasis": 0.97,
}


def test_mel_filterbank_shape_and_nonnegative() -> None:
    filters = mel_filterbank(16000, 1024, 40)
    assert filters.shape == (40, 513)
    assert np.isfinite(filters).all()
    assert (filters >= 0).all()
    assert (filters.sum(axis=1) > 0).all()


def test_sine_features_match_schema_and_are_finite() -> None:
    time = np.arange(16000 * 4, dtype=np.float32) / 16000
    audio = 0.2 * np.sin(2 * np.pi * 440 * time)
    values = extract_features(audio, SETTINGS)
    assert list(values) == feature_names(13)
    assert len(values) == 76
    assert np.isfinite(list(values.values())).all()
    assert 400 < values["spectral_centroid_hz_mean"] < 1000


def test_scaler_and_classifier_are_fit_on_train_only(tmp_path) -> None:
    rows = []
    for split, offset in (("train", 0.0), ("valid", 100.0), ("test", 200.0)):
        for family_index, family in enumerate(("a", "b")):
            for index in range(8):
                rows.append(
                    {
                        "item_id": f"{split}_{family}_{index}",
                        "split": split,
                        "instrument_family_str": family,
                        "instrument_str": f"{split}_{family}_{index // 2}",
                        "pitch": 60 + family_index,
                        "velocity": 50,
                        "x": offset + family_index * 10 + index,
                    }
                )
    report = train_and_evaluate(
        pd.DataFrame(rows),
        ["x"],
        {
            "c_values": [0.1],
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_seed": 5305,
        },
        tmp_path / "model",
        tmp_path / "results",
        tmp_path / "figures",
        ordinary_test_pitch_range=(60, 61),
    )
    assert report["scaler_fit_split"] == "train"
    assert report["classifier_fit_split"] == "train"
    assert report["test_evaluations"] == 1
    assert report["ordinary_test_pitch_range"] == [60, 61]
