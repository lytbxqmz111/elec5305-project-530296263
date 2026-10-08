"""Reproducible paired acoustic analysis for selected NSynth notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from src.features.handcrafted import (
    extract_features,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    read_wav_mono,
)


PAIR_TYPES = {"same_pitch_different_family", "same_family_different_pitch"}


def midi_to_hz(pitch: int) -> float:
    return float(440.0 * 2.0 ** ((int(pitch) - 69) / 12.0))


def validate_pair_selection(
    manifest: pd.DataFrame,
    pair_settings: list[dict[str, Any]],
) -> pd.DataFrame:
    """Validate pair controls and return one frozen row per selected item."""

    test = manifest.loc[manifest["split"].eq("test")].copy()
    if test.duplicated("item_id").any():
        raise RuntimeError("Balanced test manifest contains duplicate item IDs")
    indexed = test.set_index("item_id", drop=False)
    selected_rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for pair in pair_settings:
        pair_id = str(pair["pair_id"])
        comparison_type = str(pair["comparison_type"])
        if pair_id in seen_pairs:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        if comparison_type not in PAIR_TYPES:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
        seen_pairs.add(pair_id)
        item_ids = [str(pair["item_a"]), str(pair["item_b"])]
        missing = [item_id for item_id in item_ids if item_id not in indexed.index]
        if missing:
            raise KeyError(f"Pair {pair_id} items absent from balanced test: {missing}")
        a = indexed.loc[item_ids[0]]
        b = indexed.loc[item_ids[1]]
        if comparison_type == "same_pitch_different_family":
            if int(a["pitch"]) != int(b["pitch"]):
                raise ValueError(f"{pair_id} does not match pitch")
            if int(a["velocity"]) != int(b["velocity"]):
                raise ValueError(f"{pair_id} does not match velocity")
            if a["instrument_family_str"] == b["instrument_family_str"]:
                raise ValueError(f"{pair_id} must use different families")
            if a["instrument_source_str"] != b["instrument_source_str"]:
                raise ValueError(f"{pair_id} must match source type")
        else:
            if a["instrument_family_str"] != b["instrument_family_str"]:
                raise ValueError(f"{pair_id} must match family")
            if a["instrument_str"] != b["instrument_str"]:
                raise ValueError(f"{pair_id} must use the same source instrument")
            if int(a["velocity"]) != int(b["velocity"]):
                raise ValueError(f"{pair_id} does not match velocity")
            if int(a["pitch"]) == int(b["pitch"]):
                raise ValueError(f"{pair_id} must use different pitches")
            if {str(a["pitch_bin"]), str(b["pitch_bin"])} != {"lower", "upper"}:
                raise ValueError(f"{pair_id} must span lower and upper registers")
        for position, row in (("A", a), ("B", b)):
            selected = row.to_dict()
            selected.update(
                {
                    "pair_id": pair_id,
                    "comparison_type": comparison_type,
                    "position": position,
                    "rationale": str(pair["rationale"]),
                }
            )
            selected_rows.append(selected)
    selection = pd.DataFrame(selected_rows)
    type_counts = selection.groupby("comparison_type")["pair_id"].nunique()
    for comparison_type in PAIR_TYPES:
        if int(type_counts.get(comparison_type, 0)) < 2:
            raise RuntimeError(f"At least two {comparison_type} pairs are required")
    return selection


def _attack_indices(rms: np.ndarray) -> tuple[int, int, int]:
    peak_index = int(np.argmax(rms))
    peak = float(rms[peak_index])
    if peak <= 1e-12:
        return peak_index, peak_index, peak_index
    before_peak = rms[: peak_index + 1]
    above_10 = np.flatnonzero(before_peak >= 0.10 * peak)
    above_90 = np.flatnonzero(before_peak >= 0.90 * peak)
    start = int(above_10[0]) if len(above_10) else peak_index
    end = int(above_90[0]) if len(above_90) else peak_index
    return start, end, peak_index


def analyse_audio(audio: np.ndarray, settings: dict[str, Any]) -> dict[str, Any]:
    """Calculate arrays for all five plots plus baseline-compatible metrics."""

    sample_rate = int(settings["sample_rate"])
    n_fft = int(settings["n_fft"])
    hop_length = int(settings["hop_length"])
    n_mels = int(settings["n_mels"])
    spectrum_fft = int(settings["spectrum_fft"])
    frequencies, stft_times, stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft)
    power = magnitude**2
    mean_magnitude = np.maximum(magnitude.mean(axis=1), 1e-12)
    spectrum_db = 20.0 * np.log10(mean_magnitude / np.max(mean_magnitude))
    spectrum_frequencies = frequencies

    # A zero-padded whole-note FFT exposes harmonic spacing more clearly.
    spectrum_fft = max(
        spectrum_fft,
        1 << int(np.ceil(np.log2(max(len(audio), 1)))),
    )
    windowed = audio * np.hanning(len(audio))
    high_resolution_spectrum = np.abs(np.fft.rfft(windowed, n=spectrum_fft))
    high_resolution_spectrum = 20.0 * np.log10(
        np.maximum(high_resolution_spectrum, 1e-12)
        / np.max(np.maximum(high_resolution_spectrum, 1e-12))
    )
    high_resolution_frequencies = np.fft.rfftfreq(
        spectrum_fft, d=1.0 / sample_rate
    )

    mel_power = mel_filterbank(sample_rate, n_fft, n_mels) @ power
    mel_power = np.maximum(mel_power, 1e-12)
    mel_db = 10.0 * np.log10(mel_power / np.max(mel_power))
    mel_mean_power = np.maximum(mel_power.mean(axis=1), 1e-12)
    mel_envelope_db = 10.0 * np.log10(mel_mean_power / np.max(mel_mean_power))
    mel_points = np.linspace(
        hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2
    )
    mel_frequencies = np.asarray(mel_to_hz(mel_points[1:-1]), dtype=np.float64)

    if len(audio) < n_fft:
        framed_audio = np.pad(audio, (0, n_fft - len(audio)))[np.newaxis, :]
    else:
        frame_count = 1 + (len(audio) - n_fft) // hop_length
        framed_audio = np.lib.stride_tricks.as_strided(
            audio,
            shape=(frame_count, n_fft),
            strides=(audio.strides[0] * hop_length, audio.strides[0]),
            writeable=False,
        )
    rms = np.sqrt(np.mean(framed_audio.astype(np.float64) ** 2, axis=1))
    rms_times = np.arange(len(rms)) * hop_length / sample_rate
    start, end, peak = _attack_indices(rms)
    normalised_rms = rms / max(float(rms.max()), 1e-12)
    handcrafted = extract_features(audio, settings)
    return {
        "audio": audio,
        "times": np.arange(len(audio)) / sample_rate,
        "spectrum_db": spectrum_db,
        "spectrum_frequencies": spectrum_frequencies,
        "high_resolution_spectrum_db": high_resolution_spectrum,
        "high_resolution_frequencies": high_resolution_frequencies,
        "mel_db": mel_db,
        "stft_times": stft_times,
        "mel_mean_power": mel_mean_power,
        "mel_envelope_db": mel_envelope_db,
        "mel_frequencies": mel_frequencies,
        "rms": rms,
        "normalised_rms": normalised_rms,
        "rms_times": rms_times,
        "attack_start_index": start,
        "attack_end_index": end,
        "attack_peak_index": peak,
        "metrics": handcrafted,
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def _attack_correlation(a: dict[str, Any], b: dict[str, Any]) -> float:
    length = min(len(a["normalised_rms"]), len(b["normalised_rms"]))
    first = a["normalised_rms"][:length]
    second = b["normalised_rms"][:length]
    if length < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def pair_metrics(
    pair_rows: pd.DataFrame,
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    a_row = pair_rows.loc[pair_rows["position"].eq("A")].iloc[0]
    b_row = pair_rows.loc[pair_rows["position"].eq("B")].iloc[0]
    a = analyses[str(a_row["item_id"])]
    b = analyses[str(b_row["item_id"])]
    a_metrics = a["metrics"]
    b_metrics = b["metrics"]
    pitch_difference = int(b_row["pitch"]) - int(a_row["pitch"])
    f0_ratio = midi_to_hz(int(b_row["pitch"])) / midi_to_hz(int(a_row["pitch"]))
    centroid_ratio = float(
        b_metrics["spectral_centroid_hz_mean"]
        / max(a_metrics["spectral_centroid_hz_mean"], 1e-12)
    )
    return {
        "pair_id": str(a_row["pair_id"]),
        "comparison_type": str(a_row["comparison_type"]),
        "item_a": str(a_row["item_id"]),
        "item_b": str(b_row["item_id"]),
        "pitch_a": int(a_row["pitch"]),
        "pitch_b": int(b_row["pitch"]),
        "pitch_difference_semitones": pitch_difference,
        "expected_f0_ratio_b_over_a": f0_ratio,
        "spectral_centroid_ratio_b_over_a": centroid_ratio,
        "mel_envelope_cosine_similarity": _cosine_similarity(
            a["mel_mean_power"], b["mel_mean_power"]
        ),
        "attack_envelope_correlation": _attack_correlation(a, b),
        "attack_duration_difference_seconds_b_minus_a": float(
            b_metrics["attack_duration_seconds"]
            - a_metrics["attack_duration_seconds"]
        ),
    }


def _plot_pair(
    pair_rows: pd.DataFrame,
    analyses: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    path: Path,
) -> None:
    pair_rows = pair_rows.sort_values("position")
    figure, axes = plt.subplots(5, 2, figsize=(16, 18), constrained_layout=True)
    sample_rate = int(settings["sample_rate"])
    latest_attack = max(
        float(
            analyses[str(row["item_id"])]["rms_times"][
                analyses[str(row["item_id"])]["attack_peak_index"]
            ]
        )
        for _, row in pair_rows.iterrows()
    )
    audio_duration = min(
        float(analyses[str(row["item_id"])]["times"][-1])
        for _, row in pair_rows.iterrows()
    )
    attack_limit = min(
        audio_duration,
        max(float(settings["attack_plot_seconds"]), latest_attack + 0.20),
    )
    for column, (_, row) in enumerate(pair_rows.iterrows()):
        item_id = str(row["item_id"])
        analysis = analyses[item_id]
        title = (
            f"{row['instrument_family_str']} | {row['instrument_str']}\n"
            f"MIDI {int(row['pitch'])}, velocity {int(row['velocity'])}, "
            f"{row['instrument_source_str']}"
        )
        axes[0, column].set_title(title)
        stride = max(1, len(analysis["audio"]) // 8000)
        axes[0, column].plot(
            analysis["times"][::stride],
            analysis["audio"][::stride],
            linewidth=0.6,
        )
        axes[0, column].set_xlim(0, analysis["times"][-1])
        axes[0, column].set_ylim(-1.05, 1.05)
        axes[0, column].set_xlabel("Time (s)")

        frequency_mask = (
            analysis["high_resolution_frequencies"] >= 20
        ) & (analysis["high_resolution_frequencies"] <= sample_rate / 2)
        axes[1, column].semilogx(
            analysis["high_resolution_frequencies"][frequency_mask],
            analysis["high_resolution_spectrum_db"][frequency_mask],
            linewidth=0.7,
        )
        axes[1, column].set_xlim(20, sample_rate / 2)
        axes[1, column].set_ylim(-80, 2)
        axes[1, column].set_xlabel("Frequency (Hz, log scale)")

        axes[2, column].pcolormesh(
            analysis["stft_times"],
            analysis["mel_frequencies"],
            analysis["mel_db"],
            shading="auto",
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        axes[2, column].set_yscale("log")
        axes[2, column].set_ylim(20, sample_rate / 2)
        axes[2, column].set_xlabel("Time (s)")

        axes[3, column].semilogx(
            analysis["mel_frequencies"],
            analysis["mel_envelope_db"],
            linewidth=1.4,
        )
        axes[3, column].set_xlim(20, sample_rate / 2)
        axes[3, column].set_ylim(-80, 2)
        axes[3, column].set_xlabel("Frequency (Hz, log scale)")

        axes[4, column].plot(
            analysis["rms_times"], analysis["normalised_rms"], linewidth=1.5
        )
        for index, colour, label in (
            (analysis["attack_start_index"], "tab:green", "10%"),
            (analysis["attack_end_index"], "tab:orange", "90%"),
            (analysis["attack_peak_index"], "tab:red", "peak"),
        ):
            axes[4, column].axvline(
                analysis["rms_times"][index],
                color=colour,
                linestyle="--",
                linewidth=1,
                label=label,
            )
        axes[4, column].set_xlim(0, attack_limit)
        axes[4, column].set_ylim(0, 1.05)
        axes[4, column].set_xlabel("Time (s)")
        axes[4, column].legend(loc="best", fontsize=8)

    row_labels = (
        "Waveform amplitude",
        "Magnitude (dB)",
        "Mel frequency (Hz)",
        "Mel-smoothed envelope (dB)",
        "Normalised RMS envelope",
    )
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(label)
        if row_index not in (2,):
            axes[row_index, 1].set_ylabel(label)
    pair_id = str(pair_rows.iloc[0]["pair_id"])
    comparison = str(pair_rows.iloc[0]["comparison_type"]).replace("_", " ")
    figure.suptitle(f"{pair_id}: {comparison}", fontsize=17)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _item_metric_row(row: pd.Series, analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis["metrics"]
    f0 = midi_to_hz(int(row["pitch"]))
    centroid = float(metrics["spectral_centroid_hz_mean"])
    return {
        "pair_id": row["pair_id"],
        "comparison_type": row["comparison_type"],
        "position": row["position"],
        "item_id": row["item_id"],
        "instrument_family": row["instrument_family_str"],
        "instrument": row["instrument_str"],
        "source_type": row["instrument_source_str"],
        "midi_pitch": int(row["pitch"]),
        "f0_hz": f0,
        "velocity": int(row["velocity"]),
        "spectral_centroid_hz": centroid,
        "centroid_over_f0": centroid / f0,
        "spectral_bandwidth_hz": float(metrics["spectral_bandwidth_hz_mean"]),
        "spectral_rolloff_hz": float(metrics["spectral_rolloff_hz_mean"]),
        "spectral_flatness": float(metrics["spectral_flatness_mean"]),
        "temporal_centroid_seconds": float(metrics["temporal_centroid_seconds"]),
        "attack_duration_seconds": float(metrics["attack_duration_seconds"]),
        "attack_peak_seconds": float(metrics["attack_peak_seconds"]),
    }


def _interpret_pair(
    pair_rows: pd.DataFrame,
    item_metrics: pd.DataFrame,
    pair_metric: dict[str, Any],
) -> str:
    pair_rows = pair_rows.sort_values("position")
    items = item_metrics.loc[item_metrics["pair_id"].eq(pair_metric["pair_id"])].set_index(
        "position"
    )
    a_row, b_row = pair_rows.iloc[0], pair_rows.iloc[1]
    a, b = items.loc["A"], items.loc["B"]
    brighter = "A" if a["spectral_centroid_hz"] > b["spectral_centroid_hz"] else "B"
    faster = "A" if a["attack_duration_seconds"] < b["attack_duration_seconds"] else "B"
    if pair_metric["comparison_type"] == "same_pitch_different_family":
        return (
            f"Pitch ({int(a_row['pitch'])}) and velocity ({int(a_row['velocity'])}) are fixed, "
            f"so the visible harmonic-envelope and onset differences are not caused by register. "
            f"Sample {brighter} has the higher mean spectral centroid "
            f"({a['spectral_centroid_hz']:.0f} versus {b['spectral_centroid_hz']:.0f} Hz for A/B), "
            f"while sample {faster} has the shorter 10-90% attack "
            f"({a['attack_duration_seconds']:.3f} versus {b['attack_duration_seconds']:.3f} s). "
            f"The Mel-envelope cosine similarity is {pair_metric['mel_envelope_cosine_similarity']:.3f}; "
            "the remaining differences therefore provide direct family/timbre cues under matched pitch metadata."
        )
    pitch_ratio = pair_metric["expected_f0_ratio_b_over_a"]
    centroid_ratio = pair_metric["spectral_centroid_ratio_b_over_a"]
    attack_change = pair_metric["attack_duration_difference_seconds_b_minus_a"]
    attack_direction = "longer" if attack_change > 0 else "shorter"
    return (
        f"The source instrument and velocity are fixed while pitch rises by "
        f"{pair_metric['pitch_difference_semitones']} semitones, an expected f0 ratio of {pitch_ratio:.2f}. "
        f"The spectral centroid changes by a factor of {centroid_ratio:.2f}, and centroid/f0 changes "
        f"from {a['centroid_over_f0']:.2f} to {b['centroid_over_f0']:.2f}; this shows that the spectrum "
        "does not move as a perfectly pitch-invariant template. "
        f"The upper note's 10-90% attack is {abs(attack_change):.3f} s {attack_direction}, "
        f"while the Mel-envelope similarity is {pair_metric['mel_envelope_cosine_similarity']:.3f}. "
        "Thus register changes both harmonic placement and aspects of the apparent timbre within one source instrument."
    )


def _write_report(
    selection: pd.DataFrame,
    item_metrics: pd.DataFrame,
    pair_metrics_frame: pd.DataFrame,
    figures_dir: Path,
    path: Path,
) -> None:
    lines = [
        "# Representative Acoustic-Pair Analysis",
        "",
        "## Selection protocol",
        "",
        "All examples come from the frozen 594-note balanced test manifest. Same-pitch pairs match exact MIDI pitch, velocity, and source type while changing family. Same-family pairs match source instrument and velocity while spanning the lower and upper registers. Selection uses metadata only, not classifier correctness.",
        "",
        "Each figure contains waveform, high-resolution magnitude spectrum, log-Mel spectrogram, time-averaged Mel-smoothed spectral envelope, and normalised RMS attack envelope. Attack duration uses the same 10%-to-90% definition as the handcrafted baseline.",
        "",
        "## Selected comparisons",
        "",
        "| Pair | Type | A | B | Controlled metadata |",
        "|---|---|---|---|---|",
    ]
    for pair_id, rows in selection.groupby("pair_id", sort=False):
        rows = rows.sort_values("position")
        a, b = rows.iloc[0], rows.iloc[1]
        if a["comparison_type"] == "same_pitch_different_family":
            control = (
                f"MIDI {int(a['pitch'])}; velocity {int(a['velocity'])}; "
                f"source {a['instrument_source_str']}"
            )
        else:
            control = f"instrument {a['instrument_str']}; velocity {int(a['velocity'])}"
        lines.append(
            f"| {pair_id} | {a['comparison_type']} | {a['item_id']} | {b['item_id']} | {control} |"
        )
    lines.extend(["", "## Pair-by-pair interpretation", ""])
    metrics_by_pair = {
        row["pair_id"]: row for row in pair_metrics_frame.to_dict(orient="records")
    }
    for pair_id, rows in selection.groupby("pair_id", sort=False):
        lines.extend(
            [
                f"### {pair_id}",
                "",
                str(rows.iloc[0]["rationale"]),
                "",
                _interpret_pair(rows, item_metrics, metrics_by_pair[pair_id]),
                "",
                f"![{pair_id}](../../figures/acoustic_analysis/{pair_id}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "These examples illustrate mechanisms rather than estimate population-level effects. NSynth source type, recording chain, and synthesis design can also affect spectra and envelopes. The controlled classifier experiments provide the aggregate evidence; these plots explain acoustically plausible reasons for the observed register dependence.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python scripts\\run_acoustic_analysis.py --config configs\\acoustic_analysis.yaml",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_acoustic_analysis(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = pd.read_parquet(root / config["manifest_path"])
    selection = validate_pair_selection(manifest, config["pairs"])
    settings = config["analysis"]
    results_dir = root / config["results_dir"]
    figures_dir = root / config["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(results_dir / "selected_acoustic_pairs.csv", index=False)

    analyses: dict[str, dict[str, Any]] = {}
    item_rows: list[dict[str, Any]] = []
    for _, row in selection.iterrows():
        item_id = str(row["item_id"])
        if item_id in analyses:
            continue
        audio_path = root / str(row["audio_relpath"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        audio = read_wav_mono(audio_path, int(settings["sample_rate"]))
        analyses[item_id] = analyse_audio(audio, settings)
    for _, row in selection.iterrows():
        item_rows.append(_item_metric_row(row, analyses[str(row["item_id"])]))
    item_metrics = pd.DataFrame(item_rows)
    item_metrics.to_csv(results_dir / "acoustic_item_metrics.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for pair_id, rows in selection.groupby("pair_id", sort=False):
        metric = pair_metrics(rows, analyses)
        pair_rows.append(metric)
        _plot_pair(
            rows,
            analyses,
            settings,
            figures_dir / f"{pair_id}.png",
        )
    pair_metrics_frame = pd.DataFrame(pair_rows)
    pair_metrics_frame.to_csv(results_dir / "acoustic_pair_metrics.csv", index=False)

    report = {
        "manifest": config["manifest_path"],
        "selection_uses_predictions": False,
        "pair_count": int(selection["pair_id"].nunique()),
        "same_pitch_pair_count": int(
            selection.loc[
                selection["comparison_type"].eq("same_pitch_different_family"),
                "pair_id",
            ].nunique()
        ),
        "same_family_pair_count": int(
            selection.loc[
                selection["comparison_type"].eq("same_family_different_pitch"),
                "pair_id",
            ].nunique()
        ),
        "analysis_settings": settings,
        "pairs": pair_metrics_frame.to_dict(orient="records"),
    }
    (results_dir / "acoustic_analysis_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_report(
        selection,
        item_metrics,
        pair_metrics_frame,
        figures_dir,
        results_dir / "README.md",
    )
    return report
