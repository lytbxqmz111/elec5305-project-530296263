"""Build, validate, summarise, and visualise official NSynth metadata.

The module reads the ``examples.json`` file from each official split without
loading audio samples. It preserves the official train, valid, and test split
labels and verifies that training source instruments do not occur in valid or
test. The generated Parquet file is the canonical note-level metadata table
for later subset construction and feature extraction.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


LOGGER = logging.getLogger("nsynth_metadata")

QUALITY_NAMES = (
    "bright",
    "dark",
    "distortion",
    "fast_decay",
    "long_release",
    "multiphonic",
    "nonlinear_env",
    "percussive",
    "reverb",
    "tempo_synced",
)

SPLIT_ORDER = ("train", "valid", "test")


class MetadataValidationError(RuntimeError):
    """Raised when extracted data do not match the official NSynth structure."""


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration and validate required keys."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {
        "dataset_root",
        "split_directories",
        "expected_split_counts",
        "metadata_dir",
        "tables_dir",
        "figures_dir",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Missing configuration keys: {sorted(missing)}")
    return config


def resolve_from_root(root: Path, value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def _quality_flags(example: dict[str, Any]) -> dict[str, bool]:
    raw = list(example.get("qualities", []))
    if len(raw) != len(QUALITY_NAMES):
        raise MetadataValidationError(
            f"Expected {len(QUALITY_NAMES)} quality flags, found {len(raw)} "
            f"for {example.get('note_str', '<unknown>')}"
        )
    return {
        f"quality_{name}": bool(raw[index])
        for index, name in enumerate(QUALITY_NAMES)
    }


def load_split(
    split_dir: Path,
    split_name: str,
    root: Path,
) -> pd.DataFrame:
    """Load one official split from its examples.json file."""

    json_path = split_dir / "examples.json"
    audio_dir = split_dir / "audio"
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {json_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")

    LOGGER.info("Loading %s metadata from %s", split_name, json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, dict[str, Any]] = json.load(handle)

    records: list[dict[str, Any]] = []
    records_append = records.append
    audio_prefix = (audio_dir.relative_to(root)).as_posix()

    for item_id, example in payload.items():
        note_str = str(example["note_str"])
        record = {
            "item_id": str(item_id),
            "note": int(example["note"]),
            "note_str": note_str,
            "split": split_name,
            "audio_relpath": f"{audio_prefix}/{note_str}.wav",
            "instrument": int(example["instrument"]),
            "instrument_str": str(example["instrument_str"]),
            "instrument_family": int(example["instrument_family"]),
            "instrument_family_str": str(example["instrument_family_str"]),
            "instrument_source": int(example["instrument_source"]),
            "instrument_source_str": str(example["instrument_source_str"]),
            "pitch": int(example["pitch"]),
            "velocity": int(example["velocity"]),
            "sample_rate": int(example["sample_rate"]),
            "qualities_str": "|".join(example.get("qualities_str", [])),
        }
        record.update(_quality_flags(example))
        records_append(record)

    del payload
    gc.collect()

    frame = pd.DataFrame.from_records(records)
    del records
    gc.collect()

    for column in (
        "note",
        "instrument",
        "instrument_family",
        "instrument_source",
        "pitch",
        "velocity",
        "sample_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], downcast="integer")

    LOGGER.info("Loaded %s rows for %s", f"{len(frame):,}", split_name)
    return frame


def validate_split(
    frame: pd.DataFrame,
    split_name: str,
    expected_count: int,
) -> dict[str, Any]:
    """Validate row-level invariants for one split."""

    errors: list[str] = []

    if len(frame) != expected_count:
        errors.append(
            f"{split_name}: expected {expected_count:,} rows, found {len(frame):,}"
        )
    if frame["item_id"].duplicated().any():
        errors.append(f"{split_name}: duplicate item_id values")
    if frame["note_str"].duplicated().any():
        errors.append(f"{split_name}: duplicate note_str values")
    if not frame["item_id"].equals(frame["note_str"]):
        errors.append(f"{split_name}: JSON key does not match note_str")
    if not frame["pitch"].between(0, 127).all():
        errors.append(f"{split_name}: pitch outside MIDI range 0 to 127")
    if not frame["velocity"].between(0, 127).all():
        errors.append(f"{split_name}: velocity outside MIDI range 0 to 127")
    if set(frame["sample_rate"].unique()) != {16000}:
        errors.append(
            f"{split_name}: unexpected sample rates "
            f"{sorted(frame['sample_rate'].unique().tolist())}"
        )

    family_per_instrument = frame.groupby("instrument_str")[
        "instrument_family_str"
    ].nunique()
    if (family_per_instrument != 1).any():
        errors.append(f"{split_name}: an instrument maps to multiple families")

    source_per_instrument = frame.groupby("instrument_str")[
        "instrument_source_str"
    ].nunique()
    if (source_per_instrument != 1).any():
        errors.append(f"{split_name}: an instrument maps to multiple sources")

    if errors:
        raise MetadataValidationError("\n".join(errors))

    return {
        "rows": int(len(frame)),
        "instruments": int(frame["instrument_str"].nunique()),
        "families": int(frame["instrument_family_str"].nunique()),
        "pitch_min": int(frame["pitch"].min()),
        "pitch_max": int(frame["pitch"].max()),
        "velocity_values": sorted(int(value) for value in frame["velocity"].unique()),
        "sample_rates": sorted(int(value) for value in frame["sample_rate"].unique()),
    }


def verify_audio_inventory(
    frame: pd.DataFrame,
    audio_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    """Verify that JSON note identifiers and WAV file names match exactly."""

    LOGGER.info("Checking WAV inventory for %s", split_name)
    wav_names = {path.stem for path in audio_dir.glob("*.wav")}
    metadata_names = set(frame["note_str"])
    missing = sorted(metadata_names - wav_names)
    extra = sorted(wav_names - metadata_names)
    if missing or extra:
        raise MetadataValidationError(
            f"{split_name}: WAV inventory mismatch; "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing_sample={missing[:5]}, extra_sample={extra[:5]}"
        )
    return {
        "wav_files": len(wav_names),
        "missing_wav_files": 0,
        "extra_wav_files": 0,
    }


def validate_official_split_separation(
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Check the official train-versus-valid/test source-instrument separation."""

    instruments = {
        split: set(
            metadata.loc[metadata["split"] == split, "instrument_str"].unique()
        )
        for split in SPLIT_ORDER
    }
    train_valid = sorted(instruments["train"] & instruments["valid"])
    train_test = sorted(instruments["train"] & instruments["test"])
    valid_test = sorted(instruments["valid"] & instruments["test"])

    if train_valid or train_test:
        raise MetadataValidationError(
            "Official split leakage detected: "
            f"train-valid={len(train_valid)}, train-test={len(train_test)}"
        )

    return {
        "train_valid_instrument_overlap": 0,
        "train_test_instrument_overlap": 0,
        "valid_test_instrument_overlap": len(valid_test),
        "valid_test_overlap_sample": valid_test[:10],
    }


def write_metadata(
    metadata: pd.DataFrame,
    metadata_dir: Path,
    write_compressed_csv: bool,
) -> dict[str, str]:
    """Write the canonical metadata table and optional portable CSV copy."""

    metadata_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = metadata_dir / "nsynth_metadata.parquet"
    try:
        metadata.to_parquet(parquet_path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet requires pyarrow. Install the project requirements "
            "with: pip install -r requirements.txt"
        ) from exc

    written = {"parquet": parquet_path.as_posix()}
    if write_compressed_csv:
        csv_path = metadata_dir / "nsynth_metadata.csv.gz"
        metadata.to_csv(csv_path, index=False, compression="gzip")
        written["compressed_csv"] = csv_path.as_posix()
    return written


def write_summary_tables(metadata: pd.DataFrame, tables_dir: Path) -> list[Path]:
    """Write compact CSV summaries used for audit and family selection."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    split_summary = (
        metadata.groupby("split", observed=True)
        .agg(
            notes=("note_str", "size"),
            instruments=("instrument_str", "nunique"),
            families=("instrument_family_str", "nunique"),
            pitch_min=("pitch", "min"),
            pitch_max=("pitch", "max"),
            velocity_min=("velocity", "min"),
            velocity_max=("velocity", "max"),
        )
        .reindex(SPLIT_ORDER)
        .reset_index()
    )
    path = tables_dir / "nsynth_split_summary.csv"
    split_summary.to_csv(path, index=False)
    written.append(path)

    family_summary = (
        metadata.groupby(["split", "instrument_family_str"], observed=True)
        .agg(
            notes=("note_str", "size"),
            instruments=("instrument_str", "nunique"),
            source_types=("instrument_source_str", "nunique"),
            pitch_min=("pitch", "min"),
            pitch_q05=("pitch", lambda values: float(values.quantile(0.05))),
            pitch_median=("pitch", "median"),
            pitch_q95=("pitch", lambda values: float(values.quantile(0.95))),
            pitch_max=("pitch", "max"),
            unique_pitches=("pitch", "nunique"),
            velocity_min=("velocity", "min"),
            velocity_max=("velocity", "max"),
            unique_velocities=("velocity", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "instrument_family_str"])
    )
    path = tables_dir / "nsynth_family_summary.csv"
    family_summary.to_csv(path, index=False)
    written.append(path)

    family_pitch = (
        metadata.groupby(
            ["split", "instrument_family_str", "pitch"], observed=True
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    path = tables_dir / "nsynth_family_pitch_counts.csv"
    family_pitch.to_csv(path, index=False)
    written.append(path)

    family_velocity = (
        metadata.groupby(
            ["split", "instrument_family_str", "velocity"], observed=True
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    path = tables_dir / "nsynth_family_velocity_counts.csv"
    family_velocity.to_csv(path, index=False)
    written.append(path)

    family_source = (
        metadata.groupby(
            ["split", "instrument_family_str", "instrument_source_str"],
            observed=True,
        )
        .agg(
            notes=("note_str", "size"),
            instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    path = tables_dir / "nsynth_family_source_summary.csv"
    family_source.to_csv(path, index=False)
    written.append(path)

    return written


def _family_order(metadata: pd.DataFrame) -> list[str]:
    return sorted(metadata["instrument_family_str"].unique().tolist())


def plot_pitch_lines(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot pitch counts for every family within each official split."""

    counts = (
        metadata.groupby(
            ["split", "instrument_family_str", "pitch"], observed=True
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    families = _family_order(metadata)
    palette = dict(zip(families, sns.color_palette("tab20", len(families))))
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for axis, split in zip(axes, SPLIT_ORDER):
        subset = counts[counts["split"] == split]
        sns.lineplot(
            data=subset,
            x="pitch",
            y="notes",
            hue="instrument_family_str",
            hue_order=families,
            palette=palette,
            linewidth=1.5,
            ax=axis,
        )
        axis.set_title(f"{split.capitalize()} split")
        axis.set_ylabel("Notes")
        axis.grid(alpha=0.2)
        if axis is not axes[0] and axis.legend_ is not None:
            axis.legend_.remove()
    axes[-1].set_xlabel("MIDI pitch")
    axes[0].legend(
        title="Instrument family",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    fig.suptitle("NSynth pitch distributions by family and official split", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.84, 0.98))
    path = figures_dir / "nsynth_pitch_distribution_by_family.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family_pitch_heatmaps(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot log-scaled family-by-pitch count heatmaps for all splits."""

    families = _family_order(metadata)
    pitch_min = int(metadata["pitch"].min())
    pitch_max = int(metadata["pitch"].max())
    pitches = list(range(pitch_min, pitch_max + 1))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for axis, split in zip(axes, SPLIT_ORDER):
        subset = metadata[metadata["split"] == split]
        matrix = pd.crosstab(
            subset["instrument_family_str"], subset["pitch"]
        ).reindex(index=families, columns=pitches, fill_value=0)
        sns.heatmap(
            np.log1p(matrix),
            cmap="mako",
            cbar_kws={"label": "log(1 + note count)"},
            xticklabels=4,
            yticklabels=True,
            ax=axis,
        )
        axis.set_title(f"{split.capitalize()} split")
        axis.set_ylabel("Family")
    axes[-1].set_xlabel("MIDI pitch")
    fig.suptitle("NSynth family and pitch coverage", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = figures_dir / "nsynth_family_pitch_heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family_velocity_heatmaps(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot family-by-velocity counts for each official split."""

    families = _family_order(metadata)
    velocities = sorted(int(value) for value in metadata["velocity"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for axis, split in zip(axes, SPLIT_ORDER):
        subset = metadata[metadata["split"] == split]
        matrix = pd.crosstab(
            subset["instrument_family_str"], subset["velocity"]
        ).reindex(index=families, columns=velocities, fill_value=0)
        sns.heatmap(
            np.log1p(matrix),
            annot=True,
            fmt=".1f",
            cmap="crest",
            cbar=axis is axes[-1],
            cbar_kws={"label": "log(1 + note count)"},
            ax=axis,
        )
        axis.set_title(f"{split.capitalize()} split")
        axis.set_xlabel("MIDI velocity")
        axis.set_ylabel("Family" if axis is axes[0] else "")
    fig.suptitle("NSynth velocity coverage by family", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = figures_dir / "nsynth_family_velocity_heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family_counts(metadata: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot note and source-instrument counts by family and split."""

    summary = (
        metadata.groupby(["split", "instrument_family_str"], observed=True)
        .agg(
            notes=("note_str", "size"),
            instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    families = _family_order(metadata)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    sns.barplot(
        data=summary,
        x="instrument_family_str",
        y="notes",
        hue="split",
        hue_order=SPLIT_ORDER,
        order=families,
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Notes on log scale")
    axes[0].set_xlabel("")
    axes[0].set_title("Note counts")
    sns.barplot(
        data=summary,
        x="instrument_family_str",
        y="instruments",
        hue="split",
        hue_order=SPLIT_ORDER,
        order=families,
        ax=axes[1],
    )
    axes[1].set_ylabel("Source instruments")
    axes[1].set_xlabel("Instrument family")
    axes[1].set_title("Source-instrument counts")
    axes[1].tick_params(axis="x", rotation=35)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    axes[0].legend(title="Official split")
    fig.suptitle("NSynth family support by official split", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = figures_dir / "nsynth_family_note_and_instrument_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def create_figures(metadata: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Create the first metadata-audit figures required by the project."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    return [
        plot_pitch_lines(metadata, figures_dir),
        plot_family_pitch_heatmaps(metadata, figures_dir),
        plot_family_velocity_heatmaps(metadata, figures_dir),
        plot_family_counts(metadata, figures_dir),
    ]


def build_metadata(
    config_path: Path,
    skip_audio_verification: bool = False,
) -> dict[str, Any]:
    """Run the complete metadata build and audit pipeline."""

    root = project_root()
    config = load_config(config_path)
    dataset_root = resolve_from_root(root, config["dataset_root"])
    metadata_dir = resolve_from_root(root, config["metadata_dir"])
    tables_dir = resolve_from_root(root, config["tables_dir"])
    figures_dir = resolve_from_root(root, config["figures_dir"])

    split_frames: list[pd.DataFrame] = []
    split_reports: dict[str, Any] = {}
    verify_audio = bool(config.get("verify_audio_files", True))
    verify_audio = verify_audio and not skip_audio_verification

    for split_name in SPLIT_ORDER:
        split_folder = config["split_directories"][split_name]
        split_dir = dataset_root / split_folder
        frame = load_split(split_dir, split_name, root)
        report = validate_split(
            frame,
            split_name,
            int(config["expected_split_counts"][split_name]),
        )
        if verify_audio:
            report.update(
                verify_audio_inventory(frame, split_dir / "audio", split_name)
            )
        split_reports[split_name] = report
        split_frames.append(frame)

    metadata = pd.concat(split_frames, ignore_index=True)
    del split_frames
    gc.collect()

    split_separation = validate_official_split_separation(metadata)

    metadata_outputs = write_metadata(
        metadata,
        metadata_dir,
        bool(config.get("write_compressed_csv", False)),
    )
    table_paths = write_summary_tables(metadata, tables_dir)
    figure_paths = create_figures(metadata, figures_dir)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_path.relative_to(root).as_posix(),
        "dataset_root": dataset_root.relative_to(root).as_posix(),
        "total_rows": int(len(metadata)),
        "total_instruments": int(metadata["instrument_str"].nunique()),
        "families": sorted(metadata["instrument_family_str"].unique().tolist()),
        "split_reports": split_reports,
        "split_separation": split_separation,
        "metadata_outputs": {
            key: Path(value).relative_to(root).as_posix()
            for key, value in metadata_outputs.items()
        },
        "table_outputs": [path.relative_to(root).as_posix() for path in table_paths],
        "figure_outputs": [
            path.relative_to(root).as_posix() for path in figure_paths
        ],
    }

    report_path = metadata_dir / "nsynth_metadata_build_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Metadata build completed: %s rows", f"{len(metadata):,}")
    LOGGER.info("Canonical table: %s", metadata_outputs["parquet"])
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit metadata from official NSynth splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "nsynth_metadata.yaml",
        help="Path to the metadata YAML configuration.",
    )
    parser.add_argument(
        "--skip-audio-verification",
        action="store_true",
        help="Skip the exact JSON-versus-WAV filename inventory check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    args = parse_args(argv)
    report = build_metadata(
        args.config.resolve(),
        skip_audio_verification=args.skip_audio_verification,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
