"""Deterministic handcrafted features for 16-kHz monophonic NSynth notes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile


SPECTRAL_NAMES = (
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "spectral_flux",
)


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Return triangular Mel filters for a one-sided power spectrum."""

    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        if center > left:
            filters[index, left:center] = np.linspace(
                0.0, 1.0, center - left, endpoint=False, dtype=np.float32
            )
        if right > center:
            filters[index, center:right] = np.linspace(
                1.0, 0.0, right - center, endpoint=False, dtype=np.float32
            )
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-12)
    filters *= enorm[:, np.newaxis].astype(np.float32)
    return filters


def read_wav_mono(path: Path, expected_sample_rate: int) -> np.ndarray:
    """Read PCM WAV, convert to mono float32, and validate sample rate."""

    sample_rate, audio = wavfile.read(path, mmap=True)
    if int(sample_rate) != expected_sample_rate:
        raise ValueError(
            f"Expected {expected_sample_rate} Hz, found {sample_rate} Hz in {path}"
        )
    values = np.asarray(audio)
    if values.ndim == 2:
        values = values.astype(np.float32).mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        dtype_info = np.iinfo(values.dtype)
        scale = float(max(abs(dtype_info.min), dtype_info.max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32)
    if values.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return values


def _frames(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frame_count = 1 + (len(audio) - n_fft) // hop_length
    shape = (frame_count, n_fft)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    return np.lib.stride_tricks.as_strided(
        audio, shape=shape, strides=strides, writeable=False
    )


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
    }


def feature_names(n_mfcc: int = 13) -> list[str]:
    names: list[str] = []
    for prefix in ("mfcc", "delta_mfcc"):
        for index in range(1, n_mfcc + 1):
            names.extend((f"{prefix}_{index:02d}_mean", f"{prefix}_{index:02d}_std"))
    for name in SPECTRAL_NAMES:
        names.extend((f"{name}_mean", f"{name}_std"))
    names.extend(
        [
            "rms_mean",
            "rms_std",
            "rms_q10",
            "rms_median",
            "rms_q90",
            "rms_max",
            "zcr_mean",
            "zcr_std",
            "crest_factor",
            "temporal_centroid_seconds",
            "attack_start_seconds",
            "attack_peak_seconds",
            "attack_duration_seconds",
            "attack_slope_per_second",
        ]
    )
    return names


def extract_features(audio: np.ndarray, settings: dict[str, Any]) -> dict[str, float]:
    """Extract fixed-length MFCC, spectral, temporal, and attack descriptors."""

    sample_rate = int(settings["sample_rate"])
    n_fft = int(settings["n_fft"])
    hop_length = int(settings["hop_length"])
    n_mels = int(settings["n_mels"])
    n_mfcc = int(settings["n_mfcc"])
    rolloff_fraction = float(settings["rolloff_fraction"])
    pre_emphasis = float(settings["pre_emphasis"])
    eps = 1e-10

    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - pre_emphasis * audio[:-1]
    raw_frames = _frames(audio, n_fft, hop_length)
    emphasized_frames = _frames(emphasized, n_fft, hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(emphasized_frames * window, n=n_fft, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = magnitude * magnitude
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate).astype(np.float32)

    mel_energies = mel_filterbank(sample_rate, n_fft, n_mels) @ power.T
    log_mel = np.log(np.maximum(mel_energies, eps))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    delta_mfcc = np.gradient(mfcc, axis=1) if mfcc.shape[1] > 1 else np.zeros_like(mfcc)

    features: dict[str, float] = {}
    for index in range(n_mfcc):
        features.update(_summary(f"mfcc_{index + 1:02d}", mfcc[index]))
    for index in range(n_mfcc):
        features.update(_summary(f"delta_mfcc_{index + 1:02d}", delta_mfcc[index]))

    magnitude_sum = np.maximum(magnitude.sum(axis=1), eps)
    centroid = (magnitude * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt(
        (magnitude * (frequencies[np.newaxis, :] - centroid[:, np.newaxis]) ** 2).sum(axis=1)
        / magnitude_sum
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff_threshold = rolloff_fraction * cumulative[:, -1]
    rolloff_index = np.argmax(cumulative >= rolloff_threshold[:, np.newaxis], axis=1)
    rolloff = frequencies[rolloff_index]
    flatness = np.exp(np.mean(np.log(np.maximum(power, eps)), axis=1)) / np.maximum(
        np.mean(power, axis=1), eps
    )
    normalised = magnitude / magnitude_sum[:, np.newaxis]
    flux = np.zeros(len(normalised), dtype=np.float32)
    if len(normalised) > 1:
        flux[1:] = np.sqrt(np.sum(np.diff(normalised, axis=0) ** 2, axis=1))

    for name, values in zip(
        SPECTRAL_NAMES, (centroid, bandwidth, rolloff, flatness, flux)
    ):
        features.update(_summary(name, values))

    rms = np.sqrt(np.mean(raw_frames.astype(np.float64) ** 2, axis=1))
    signs = np.signbit(raw_frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    features.update(
        {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_q10": float(np.quantile(rms, 0.10)),
            "rms_median": float(np.median(rms)),
            "rms_q90": float(np.quantile(rms, 0.90)),
            "rms_max": float(np.max(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "crest_factor": float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio**2)), eps)),
        }
    )

    times = (np.arange(len(rms)) * hop_length + n_fft / 2) / sample_rate
    energy = rms * rms
    features["temporal_centroid_seconds"] = float(
        np.sum(times * energy) / max(np.sum(energy), eps)
    )
    peak_index = int(np.argmax(rms))
    peak = float(rms[peak_index])
    if peak <= eps:
        start_index = peak_index
        end_index = peak_index
    else:
        before_peak = rms[: peak_index + 1]
        above_10 = np.flatnonzero(before_peak >= 0.10 * peak)
        above_90 = np.flatnonzero(before_peak >= 0.90 * peak)
        start_index = int(above_10[0]) if len(above_10) else peak_index
        end_index = int(above_90[0]) if len(above_90) else peak_index
    attack_duration = max((end_index - start_index) * hop_length / sample_rate, 0.0)
    attack_delta = float(rms[end_index] - rms[start_index])
    features.update(
        {
            "attack_start_seconds": float(start_index * hop_length / sample_rate),
            "attack_peak_seconds": float(peak_index * hop_length / sample_rate),
            "attack_duration_seconds": float(attack_duration),
            "attack_slope_per_second": float(
                attack_delta / max(attack_duration, hop_length / sample_rate)
            ),
        }
    )

    expected = feature_names(n_mfcc)
    if list(features) != expected:
        raise RuntimeError("Feature ordering does not match the declared schema")
    values = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite handcrafted feature encountered")
    return features


def extract_file(path: Path, settings: dict[str, Any]) -> dict[str, float]:
    audio = read_wav_mono(path, int(settings["sample_rate"]))
    return extract_features(audio, settings)
