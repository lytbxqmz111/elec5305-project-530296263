"""Audit NSynth metadata and build a reproducible pitch-controlled manifest."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.data.nsynth_metadata import create_figures as create_core_audit_figures


SPLITS = ("train", "valid", "test")
REGISTER_LABELS = ("lower", "middle", "upper")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "metadata_path",
        "tables_dir",
        "figures_dir",
        "reports_dir",
        "manifests_dir",
        "selection",
        "balancing",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Missing audit configuration keys: {sorted(missing)}")
    return config


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_audit_tables(metadata: pd.DataFrame, tables_dir: Path) -> list[Path]:
    """Write all split-level tables promised by the metadata-audit protocol."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    overview = (
        metadata.groupby(["split", "instrument_family_str"], observed=True)
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
            source_types=("instrument_source_str", "nunique"),
            pitches=("pitch", "nunique"),
            pitch_min=("pitch", "min"),
            pitch_max=("pitch", "max"),
            velocities=("velocity", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "instrument_family_str"])
    )
    written.append(_write(overview, tables_dir / "family_overview.csv"))

    pitch = (
        metadata.groupby(
            ["split", "instrument_family_str", "pitch"], observed=True
        )
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    pitch["family_note_fraction"] = pitch["notes"] / pitch.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    written.append(_write(pitch, tables_dir / "family_pitch_counts.csv"))

    velocity = (
        metadata.groupby(
            ["split", "instrument_family_str", "velocity"], observed=True
        )
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    velocity["family_note_fraction"] = velocity["notes"] / velocity.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    written.append(_write(velocity, tables_dir / "family_velocity_counts.csv"))

    instruments = (
        metadata.groupby(
            [
                "split",
                "instrument_family_str",
                "instrument_str",
                "instrument_source_str",
            ],
            observed=True,
        )
        .agg(
            notes=("item_id", "size"),
            pitches=("pitch", "nunique"),
            pitch_min=("pitch", "min"),
            pitch_max=("pitch", "max"),
            velocities=("velocity", "nunique"),
        )
        .reset_index()
    )
    instruments["family_note_fraction"] = instruments["notes"] / instruments.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    written.append(
        _write(instruments, tables_dir / "source_instrument_contributions.csv")
    )

    source_types = (
        metadata.groupby(
            ["split", "instrument_family_str", "instrument_source_str"],
            observed=True,
        )
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    source_types["family_note_fraction"] = source_types["notes"] / source_types.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    source_types["family_instrument_fraction"] = source_types[
        "source_instruments"
    ] / source_types.groupby(["split", "instrument_family_str"], observed=True)[
        "source_instruments"
    ].transform("sum")
    written.append(_write(source_types, tables_dir / "source_type_composition.csv"))
    return written


def _register_edges(low: int, width: int) -> list[int]:
    third = width // 3
    return [low, low + third, low + 2 * third, low + width]


def add_register_bin(frame: pd.DataFrame, low: int, high: int) -> pd.DataFrame:
    """Restrict a frame to an interval and attach three equal register bins."""

    width = high - low + 1
    if width % 3:
        raise ValueError("The selected interval must divide into three equal bins")
    result = frame.loc[frame["pitch"].between(low, high)].copy()
    result["pitch_bin"] = pd.cut(
        result["pitch"],
        bins=_register_edges(low, width),
        labels=REGISTER_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")
    return result


def eligible_families(metadata: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Report and flag families with enough unseen evaluation instruments."""

    counts = (
        metadata.groupby(["instrument_family_str", "split"], observed=True)[
            "instrument_str"
        ]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=SPLITS, fill_value=0)
        .reset_index()
        .rename(
            columns={
                "train": "train_instruments",
                "valid": "valid_instruments",
                "test": "test_instruments",
            }
        )
    )
    minimum = int(settings["min_eval_instruments_per_split"])
    counts["eligible"] = (
        counts["valid_instruments"].ge(minimum)
        & counts["test_instruments"].ge(minimum)
    )
    counts["exclusion_reason"] = np.where(
        counts["eligible"],
        "",
        f"fewer than {minimum} source instruments in valid or test",
    )
    return counts.sort_values("instrument_family_str").reset_index(drop=True)


def _cell_support(
    metadata: pd.DataFrame,
    families: tuple[str, ...],
    low: int,
    high: int,
    settings: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    subset = add_register_bin(
        metadata.loc[metadata["instrument_family_str"].isin(families)], low, high
    )
    width = high - low + 1
    bin_width = width // 3
    expected = pd.MultiIndex.from_product(
        [SPLITS, families, REGISTER_LABELS],
        names=["split", "instrument_family_str", "pitch_bin"],
    )
    cells = (
        subset.groupby(
            ["split", "instrument_family_str", "pitch_bin"], observed=True
        )
        .agg(
            notes=("item_id", "size"),
            instruments=("instrument_str", "nunique"),
            pitches=("pitch", "nunique"),
        )
        .reindex(expected, fill_value=0)
    )
    coverage = cells["pitches"] / bin_width

    exact = (
        subset.groupby(
            ["split", "instrument_family_str", "pitch"], observed=True
        )
        .agg(notes=("item_id", "size"), instruments=("instrument_str", "nunique"))
        .reset_index()
    )
    exact["supported"] = exact["notes"].ge(
        int(settings["exact_pitch_min_notes"])
    ) & exact["instruments"].ge(int(settings["exact_pitch_min_instruments"]))

    shared_counts: list[int] = []
    edges = _register_edges(low, width)
    for split in SPLITS:
        for start, stop in zip(edges[:-1], edges[1:]):
            pitch_values = range(start, stop)
            supported = 0
            for pitch in pitch_values:
                rows = exact.loc[
                    exact["split"].eq(split)
                    & exact["pitch"].eq(pitch)
                    & exact["instrument_family_str"].isin(families)
                ]
                if len(rows) == len(families) and rows["supported"].all():
                    supported += 1
            shared_counts.append(supported)

    metrics = {
        "min_cell_notes": float(cells["notes"].min()),
        "min_cell_instruments": float(cells["instruments"].min()),
        "min_pitch_coverage": float(coverage.min()),
        "min_shared_supported_pitches": float(min(shared_counts)),
        "total_notes_in_interval": float(len(subset)),
    }
    feasible = (
        metrics["min_cell_notes"]
        >= int(settings["min_notes_per_family_register_split"])
        and metrics["min_cell_instruments"]
        >= int(settings["min_instruments_per_family_register_split"])
        and metrics["min_pitch_coverage"]
        >= float(settings["min_pitch_coverage_fraction"])
        and metrics["min_shared_supported_pitches"]
        >= int(settings["min_shared_supported_pitches_per_register_split"])
    )
    return feasible, metrics


def select_families_and_interval(
    metadata: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Apply the predeclared support rule and choose a deterministic optimum."""

    eligibility = eligible_families(metadata, settings)
    eligible = eligibility.loc[eligibility["eligible"], "instrument_family_str"].tolist()
    min_families = int(settings["min_families"])
    max_families = min(int(settings["max_families"]), len(eligible))
    if max_families < min_families:
        raise RuntimeError(
            f"Only {len(eligible)} families meet the evaluation-instrument rule; "
            f"at least {min_families} are required"
        )

    # Aggregate once. The interval search below then uses small dictionaries
    # rather than repeatedly grouping the full note-level table.
    exact = (
        metadata.loc[metadata["instrument_family_str"].isin(eligible)]
        .groupby(["split", "instrument_family_str", "pitch"], observed=True)
        .agg(notes=("item_id", "size"), instruments=("instrument_str", "nunique"))
        .reset_index()
    )
    note_lookup = {
        (str(row.split), str(row.instrument_family_str), int(row.pitch)): int(row.notes)
        for row in exact.itertuples(index=False)
    }
    exact_instrument_lookup = {
        (str(row.split), str(row.instrument_family_str), int(row.pitch)): int(
            row.instruments
        )
        for row in exact.itertuples(index=False)
    }
    instrument_sets = {
        (str(split), str(family), int(pitch)): set(values)
        for (split, family, pitch), values in metadata.loc[
            metadata["instrument_family_str"].isin(eligible)
        ].groupby(
            ["split", "instrument_family_str", "pitch"], observed=True
        )["instrument_str"]
        .unique()
        .items()
    }

    pitch_min = int(metadata["pitch"].min())
    pitch_max = int(metadata["pitch"].max())
    candidates: list[dict[str, Any]] = []
    for family_count in range(max_families, min_families - 1, -1):
        for families in itertools.combinations(eligible, family_count):
            for width in range(
                int(settings["min_interval_semitones"]),
                int(settings["max_interval_semitones"]) + 1,
                int(settings["interval_step_semitones"]),
            ):
                if width % 3:
                    continue
                for low in range(pitch_min, pitch_max - width + 2):
                    high = low + width - 1
                    edges = _register_edges(low, width)
                    cell_notes: list[int] = []
                    cell_instruments: list[int] = []
                    pitch_coverages: list[float] = []
                    shared_pitch_counts: list[int] = []
                    for split in SPLITS:
                        for start, stop in zip(edges[:-1], edges[1:]):
                            shared = 0
                            for pitch in range(start, stop):
                                if all(
                                    note_lookup.get((split, family, pitch), 0)
                                    >= int(settings["exact_pitch_min_notes"])
                                    and exact_instrument_lookup.get(
                                        (split, family, pitch), 0
                                    )
                                    >= int(settings["exact_pitch_min_instruments"])
                                    for family in families
                                ):
                                    shared += 1
                            shared_pitch_counts.append(shared)
                            for family in families:
                                keys = [
                                    (split, family, pitch)
                                    for pitch in range(start, stop)
                                ]
                                notes = sum(note_lookup.get(key, 0) for key in keys)
                                instruments: set[str] = set()
                                pitches_present = 0
                                for key in keys:
                                    if note_lookup.get(key, 0) > 0:
                                        pitches_present += 1
                                    instruments.update(instrument_sets.get(key, set()))
                                cell_notes.append(notes)
                                cell_instruments.append(len(instruments))
                                pitch_coverages.append(
                                    pitches_present / (stop - start)
                                )
                    metrics = {
                        "min_cell_notes": float(min(cell_notes)),
                        "min_cell_instruments": float(min(cell_instruments)),
                        "min_pitch_coverage": float(min(pitch_coverages)),
                        "min_shared_supported_pitches": float(
                            min(shared_pitch_counts)
                        ),
                        "total_notes_in_interval": float(sum(cell_notes)),
                    }
                    feasible = (
                        metrics["min_cell_notes"]
                        >= int(settings["min_notes_per_family_register_split"])
                        and metrics["min_cell_instruments"]
                        >= int(
                            settings["min_instruments_per_family_register_split"]
                        )
                        and metrics["min_pitch_coverage"]
                        >= float(settings["min_pitch_coverage_fraction"])
                        and metrics["min_shared_supported_pitches"]
                        >= int(
                            settings[
                                "min_shared_supported_pitches_per_register_split"
                            ]
                        )
                    )
                    if feasible:
                        candidates.append(
                            {
                                "families": "|".join(families),
                                "family_count": family_count,
                                "pitch_low": low,
                                "pitch_high": high,
                                "interval_semitones": width,
                                **metrics,
                            }
                        )
        # The ranking prioritises family count, so smaller combinations cannot
        # beat a feasible result found at this family count.
        if candidates:
            break
    if not candidates:
        raise RuntimeError(
            "No family/interval combination satisfies the configured support rule"
        )

    candidate_frame = pd.DataFrame(candidates).sort_values(
        [
            "family_count",
            "interval_semitones",
            "min_shared_supported_pitches",
            "min_cell_notes",
            "min_pitch_coverage",
            "total_notes_in_interval",
            "pitch_low",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    best = candidate_frame.iloc[0]
    low = int(best["pitch_low"])
    high = int(best["pitch_high"])
    edges = _register_edges(low, high - low + 1)
    selection = {
        "families": str(best["families"]).split("|"),
        "pitch_low": low,
        "pitch_high": high,
        "interval_semitones": int(best["interval_semitones"]),
        "pitch_bins": {
            label: [int(start), int(stop - 1)]
            for label, start, stop in zip(
                REGISTER_LABELS, edges[:-1], edges[1:]
            )
        },
        "selection_metrics": {
            key: float(best[key])
            for key in (
                "min_cell_notes",
                "min_cell_instruments",
                "min_pitch_coverage",
                "min_shared_supported_pitches",
                "total_notes_in_interval",
            )
        },
    }
    selected_families = selection["families"]
    shared_supported_pitches: dict[str, dict[str, list[int]]] = {}
    for split in SPLITS:
        shared_supported_pitches[split] = {}
        for label, start, stop in zip(REGISTER_LABELS, edges[:-1], edges[1:]):
            shared_supported_pitches[split][label] = [
                pitch
                for pitch in range(start, stop)
                if all(
                    note_lookup.get((split, family, pitch), 0)
                    >= int(settings["exact_pitch_min_notes"])
                    and exact_instrument_lookup.get((split, family, pitch), 0)
                    >= int(settings["exact_pitch_min_instruments"])
                    for family in selected_families
                )
            ]
    selection["shared_supported_pitches"] = shared_supported_pitches
    return selection, eligibility, candidate_frame


def selected_support_tables(
    metadata: pd.DataFrame,
    selection: dict[str, Any],
    tables_dir: Path,
) -> list[Path]:
    subset = add_register_bin(
        metadata.loc[
            metadata["instrument_family_str"].isin(selection["families"])
        ],
        selection["pitch_low"],
        selection["pitch_high"],
    )
    pitch_bins = (
        subset.groupby(
            ["split", "instrument_family_str", "pitch_bin"], observed=True
        )
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
            pitches=("pitch", "nunique"),
            velocities=("velocity", "nunique"),
        )
        .reset_index()
    )
    exact = (
        subset.groupby(
            ["split", "instrument_family_str", "pitch_bin", "pitch"],
            observed=True,
        )
        .agg(
            notes=("item_id", "size"),
            source_instruments=("instrument_str", "nunique"),
        )
        .reset_index()
    )
    return [
        _write(pitch_bins, tables_dir / "selected_family_pitch_bin_counts.csv"),
        _write(exact, tables_dir / "selected_family_pitch_counts.csv"),
    ]


def _common_target(
    groups: dict[str, pd.DataFrame],
    maximum: int,
    maximum_share: float,
    minimum_instruments: int,
) -> int:
    if any(group["instrument_str"].nunique() < minimum_instruments for group in groups.values()):
        return 0
    upper = min(maximum, *(len(group) for group in groups.values()))
    for target in range(upper, 1, -1):
        instrument_cap = max(1, math.floor(target * maximum_share))
        if all(
            group.groupby("instrument_str").size().clip(upper=instrument_cap).sum()
            >= target
            for group in groups.values()
        ):
            return target
    return 0


def _sample_with_instrument_cap(
    group: pd.DataFrame,
    target: int,
    maximum_share: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cap = max(1, math.floor(target * maximum_share))
    pieces: list[pd.DataFrame] = []
    for _, instrument_rows in group.groupby("instrument_str", observed=True):
        take = min(cap, len(instrument_rows))
        indices = rng.choice(instrument_rows.index.to_numpy(), size=take, replace=False)
        pieces.append(group.loc[indices])
    pool = pd.concat(pieces)
    chosen = rng.choice(pool.index.to_numpy(), size=target, replace=False)
    return pool.loc[chosen]


def _balanced_velocity_quota(
    groups: dict[str, pd.DataFrame], target: int
) -> dict[int, int] | None:
    """Allocate a common, near-uniform velocity quota for every family."""

    velocities = sorted(
        set.intersection(
            *(set(group["velocity"].astype(int)) for group in groups.values())
        )
    )
    availability = {
        velocity: min(
            int(group["velocity"].eq(velocity).sum()) for group in groups.values()
        )
        for velocity in velocities
    }
    if sum(availability.values()) < target:
        return None
    quota = {velocity: 0 for velocity in velocities}
    for _ in range(target):
        candidates = [
            velocity
            for velocity in velocities
            if quota[velocity] < availability[velocity]
        ]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda value: (quota[value], value))
        quota[chosen] += 1
    return {velocity: count for velocity, count in quota.items() if count}


def _sample_with_velocity_and_instrument_caps(
    group: pd.DataFrame,
    quota: dict[int, int],
    maximum_share: float,
    rng: np.random.Generator,
    attempts: int = 100,
) -> pd.DataFrame | None:
    """Sample one pitch while meeting a velocity quota and instrument cap."""

    target = sum(quota.values())
    instrument_cap = max(1, math.floor(target * maximum_share))
    slots = [velocity for velocity, count in quota.items() for _ in range(count)]
    for _ in range(attempts):
        rng.shuffle(slots)
        chosen: list[int] = []
        instrument_counts: dict[str, int] = {}
        success = True
        for velocity in slots:
            candidates = group.loc[
                group["velocity"].eq(velocity) & ~group.index.isin(chosen)
            ].copy()
            candidates["used"] = candidates["instrument_str"].map(
                instrument_counts
            ).fillna(0)
            candidates = candidates.loc[candidates["used"].lt(instrument_cap)]
            if candidates.empty:
                success = False
                break
            minimum_used = candidates["used"].min()
            candidates = candidates.loc[candidates["used"].eq(minimum_used)]
            index = int(rng.choice(candidates.index.to_numpy()))
            chosen.append(index)
            instrument = str(group.loc[index, "instrument_str"])
            instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
        if success:
            return group.loc[chosen]
    return None


def build_exact_pitch_sensitivity_manifest(
    metadata: pd.DataFrame,
    selection: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match exact-pitch/velocity counts and cap source-instrument contribution."""

    families = selection["families"]
    subset = add_register_bin(
        metadata.loc[metadata["instrument_family_str"].isin(families)],
        selection["pitch_low"],
        selection["pitch_high"],
    )
    supported_rows = []
    for split in SPLITS:
        supported = {
            pitch
            for values in selection["shared_supported_pitches"][split].values()
            for pitch in values
        }
        supported_rows.append(
            subset.loc[subset["split"].eq(split) & subset["pitch"].isin(supported)]
        )
    subset = pd.concat(supported_rows, ignore_index=True)
    seed = int(settings["random_seed"])
    rng = np.random.default_rng(seed)
    maximum = int(settings["max_notes_per_family_exact_pitch_cell"])
    maximum_share = float(settings["max_instrument_share_per_cell"])
    minimum_instruments = int(settings["min_instruments_per_cell"])
    sampled: list[pd.DataFrame] = []
    strata: list[dict[str, Any]] = []

    for split in SPLITS:
        split_rows = subset.loc[subset["split"].eq(split)]
        for pitch in sorted(split_rows["pitch"].unique()):
            cell = split_rows.loc[split_rows["pitch"].eq(pitch)]
            groups = {
                family: cell.loc[cell["instrument_family_str"].eq(family)]
                for family in families
            }
            if any(
                group["instrument_str"].nunique() < minimum_instruments
                for group in groups.values()
            ):
                continue
            upper = min(maximum, *(len(group) for group in groups.values()))
            chosen_groups: dict[str, pd.DataFrame] | None = None
            chosen_quota: dict[int, int] | None = None
            for target in range(upper, 1, -1):
                quota = _balanced_velocity_quota(groups, target)
                if quota is None:
                    continue
                trial: dict[str, pd.DataFrame] = {}
                for family, group in groups.items():
                    result = _sample_with_velocity_and_instrument_caps(
                        group, quota, maximum_share, rng
                    )
                    if result is None:
                        break
                    trial[family] = result
                if len(trial) == len(families):
                    chosen_groups = trial
                    chosen_quota = quota
                    break
            if chosen_groups is None or chosen_quota is None:
                continue
            sampled.extend(chosen_groups.values())
            pitch_bin = str(next(iter(chosen_groups.values()))["pitch_bin"].iloc[0])
            for velocity, count in chosen_quota.items():
                strata.append(
                    {
                        "split": split,
                        "pitch_bin": pitch_bin,
                        "pitch": int(pitch),
                        "velocity": int(velocity),
                        "notes_per_family": int(count),
                        "families": len(families),
                    }
                )

    if not sampled:
        raise RuntimeError("The balancing rule did not produce any samples")
    manifest = pd.concat(sampled, ignore_index=True).sort_values(
        ["split", "instrument_family_str", "pitch", "velocity", "item_id"]
    )
    manifest["selection_seed"] = seed
    stratum_frame = pd.DataFrame(strata)
    return manifest.reset_index(drop=True), stratum_frame


def validate_exact_pitch_manifest(
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Prove exact-pitch/velocity balance and official split separation."""

    families = selection["families"]
    counts = manifest.groupby(
        ["split", "pitch", "velocity", "instrument_family_str"],
        observed=True,
    ).size()
    balance = counts.groupby(level=[0, 1, 2]).nunique()
    if not balance.eq(1).all():
        raise RuntimeError("Manifest is not balanced by exact pitch and velocity")
    if manifest["item_id"].duplicated().any():
        raise RuntimeError("Manifest contains duplicate item identifiers")

    split_instruments = {
        split: set(manifest.loc[manifest["split"].eq(split), "instrument_str"])
        for split in SPLITS
    }
    if split_instruments["train"] & (
        split_instruments["valid"] | split_instruments["test"]
    ):
        raise RuntimeError("Manifest violates train/evaluation instrument separation")
    if any(not split_instruments[split] for split in SPLITS):
        raise RuntimeError("Balanced manifest must contain every official split")

    cap = float(settings["max_instrument_share_per_cell"])
    family_cells = manifest.groupby(
        ["split", "pitch", "instrument_family_str"],
        observed=True,
    ).size()
    instrument_cells = manifest.groupby(
        [
            "split",
            "pitch",
            "instrument_family_str",
            "instrument_str",
        ],
        observed=True,
    ).size()
    shares = instrument_cells / family_cells
    if shares.max() > cap + 1e-12:
        raise RuntimeError("Manifest exceeds the configured instrument-share cap")

    summary: dict[str, Any] = {
        "rows": int(len(manifest)),
        "families": families,
        "pitch_low": int(selection["pitch_low"]),
        "pitch_high": int(selection["pitch_high"]),
        "exact_pitch_velocity_balance": True,
        "maximum_observed_instrument_share_per_cell": float(shares.max()),
        "split_rows": {
            split: int(manifest["split"].eq(split).sum()) for split in SPLITS
        },
        "split_source_instruments": {
            split: int(
                manifest.loc[manifest["split"].eq(split), "instrument_str"].nunique()
            )
            for split in SPLITS
        },
        "register_rows_per_family": {},
    }
    register_counts = manifest.groupby(
        ["split", "pitch_bin", "instrument_family_str"], observed=True
    ).size()
    for split in SPLITS:
        summary["register_rows_per_family"][split] = {
            register: int(
                register_counts.loc[(split, register)].iloc[0]
            )
            for register in REGISTER_LABELS
            if (split, register) in register_counts.index.droplevel(2)
        }
    return summary


def build_balanced_manifest(
    metadata: pd.DataFrame,
    selection: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the primary register/velocity-balanced evaluation manifest."""

    families = selection["families"]
    subset = add_register_bin(
        metadata.loc[metadata["instrument_family_str"].isin(families)],
        selection["pitch_low"],
        selection["pitch_high"],
    )
    seed = int(settings["random_seed"])
    rng = np.random.default_rng(seed)
    maximum = int(settings["max_notes_per_family_register_velocity_cell"])
    maximum_share = float(settings["max_instrument_share_per_cell"])
    minimum_instruments = int(settings["min_instruments_per_cell"])
    sampled: list[pd.DataFrame] = []
    strata: list[dict[str, Any]] = []

    for split in SPLITS:
        split_rows = subset.loc[subset["split"].eq(split)]
        for pitch_bin in REGISTER_LABELS:
            for velocity in sorted(split_rows["velocity"].unique()):
                cell = split_rows.loc[
                    split_rows["pitch_bin"].eq(pitch_bin)
                    & split_rows["velocity"].eq(velocity)
                ]
                groups = {
                    family: cell.loc[cell["instrument_family_str"].eq(family)]
                    for family in families
                }
                target = _common_target(
                    groups, maximum, maximum_share, minimum_instruments
                )
                if target == 0:
                    continue
                for group in groups.values():
                    sampled.append(
                        _sample_with_instrument_cap(
                            group, target, maximum_share, rng
                        )
                    )
                strata.append(
                    {
                        "split": split,
                        "pitch_bin": pitch_bin,
                        "velocity": int(velocity),
                        "notes_per_family": target,
                        "families": len(families),
                    }
                )

    if not sampled:
        raise RuntimeError("The register-balancing rule did not produce samples")
    manifest = pd.concat(sampled, ignore_index=True).sort_values(
        ["split", "instrument_family_str", "pitch", "velocity", "item_id"]
    )
    manifest["selection_seed"] = seed
    return manifest.reset_index(drop=True), pd.DataFrame(strata)


def validate_manifest(
    manifest: pd.DataFrame,
    selection: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Validate the primary register/velocity-balanced manifest."""

    counts = manifest.groupby(
        ["split", "pitch_bin", "velocity", "instrument_family_str"],
        observed=True,
    ).size()
    if not counts.groupby(level=[0, 1, 2]).nunique().eq(1).all():
        raise RuntimeError("Manifest is not balanced by register and velocity")
    if manifest["item_id"].duplicated().any():
        raise RuntimeError("Manifest contains duplicate item identifiers")

    split_instruments = {
        split: set(manifest.loc[manifest["split"].eq(split), "instrument_str"])
        for split in SPLITS
    }
    if split_instruments["train"] & (
        split_instruments["valid"] | split_instruments["test"]
    ):
        raise RuntimeError("Manifest violates train/evaluation instrument separation")
    if any(not split_instruments[split] for split in SPLITS):
        raise RuntimeError("Balanced manifest must contain every official split")

    family_cells = manifest.groupby(
        ["split", "pitch_bin", "velocity", "instrument_family_str"],
        observed=True,
    ).size()
    instrument_cells = manifest.groupby(
        [
            "split",
            "pitch_bin",
            "velocity",
            "instrument_family_str",
            "instrument_str",
        ],
        observed=True,
    ).size()
    shares = instrument_cells / family_cells
    cap = float(settings["max_instrument_share_per_cell"])
    if shares.max() > cap + 1e-12:
        raise RuntimeError("Manifest exceeds the configured instrument-share cap")

    register_counts = manifest.groupby(
        ["split", "pitch_bin", "instrument_family_str"], observed=True
    ).size()
    summary: dict[str, Any] = {
        "rows": int(len(manifest)),
        "families": selection["families"],
        "pitch_low": int(selection["pitch_low"]),
        "pitch_high": int(selection["pitch_high"]),
        "register_velocity_balance": True,
        "maximum_observed_instrument_share_per_cell": float(shares.max()),
        "split_rows": {
            split: int(manifest["split"].eq(split).sum()) for split in SPLITS
        },
        "split_source_instruments": {
            split: int(
                manifest.loc[manifest["split"].eq(split), "instrument_str"].nunique()
            )
            for split in SPLITS
        },
        "register_rows_per_family": {},
    }
    for split in SPLITS:
        summary["register_rows_per_family"][split] = {
            register: int(register_counts.loc[(split, register)].iloc[0])
            for register in REGISTER_LABELS
            if (split, register) in register_counts.index.droplevel(2)
        }
    return summary


def create_additional_figures(
    metadata: pd.DataFrame,
    selection: dict[str, Any],
    manifest: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    written: list[Path] = []

    contribution = (
        metadata.groupby(
            ["split", "instrument_family_str", "instrument_str"], observed=True
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    contribution["family_fraction"] = contribution["notes"] / contribution.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)
    for axis, split in zip(axes, SPLITS):
        sns.stripplot(
            data=contribution.loc[contribution["split"].eq(split)],
            x="instrument_family_str",
            y="family_fraction",
            jitter=0.25,
            alpha=0.65,
            size=4,
            ax=axis,
        )
        axis.set_title(f"{split.capitalize()} split")
        axis.set_ylabel("Fraction of family notes")
        axis.set_xlabel("")
    axes[-1].tick_params(axis="x", rotation=35)
    axes[-1].set_xlabel("Instrument family")
    fig.suptitle("Contribution of individual source instruments", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = figures_dir / "source_instrument_contributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    source = (
        metadata.groupby(
            ["split", "instrument_family_str", "instrument_source_str"],
            observed=True,
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    source["fraction"] = source["notes"] / source.groupby(
        ["split", "instrument_family_str"], observed=True
    )["notes"].transform("sum")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    for axis, split in zip(axes, SPLITS):
        pivot = source.loc[source["split"].eq(split)].pivot(
            index="instrument_family_str",
            columns="instrument_source_str",
            values="fraction",
        ).fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=axis, width=0.85)
        axis.set_title(f"{split.capitalize()} split")
        axis.set_ylabel("Fraction of family notes")
        axis.set_xlabel("")
        axis.legend(title="Source type", bbox_to_anchor=(1.01, 1), loc="upper left")
    axes[-1].set_xlabel("Instrument family")
    fig.suptitle("Source-type composition by family", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.88, 0.98))
    path = figures_dir / "source_type_composition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    balanced = (
        manifest.groupby(
            ["split", "instrument_family_str", "pitch_bin"], observed=True
        )
        .size()
        .rename("notes")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, split in zip(axes, SPLITS):
        sns.barplot(
            data=balanced.loc[balanced["split"].eq(split)],
            x="pitch_bin",
            y="notes",
            hue="instrument_family_str",
            order=REGISTER_LABELS,
            hue_order=selection["families"],
            ax=axis,
        )
        axis.set_title(f"{split.capitalize()} split")
        axis.set_xlabel("Register bin")
        axis.set_ylabel("Selected notes" if axis is axes[0] else "")
        if axis is not axes[-1] and axis.legend_ is not None:
            axis.legend_.remove()
    axes[-1].legend(title="Family", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Pitch-balanced manifest counts", y=1.02)
    fig.tight_layout()
    path = figures_dir / "balanced_manifest_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def write_report(
    path: Path,
    selection: dict[str, Any],
    eligibility: pd.DataFrame,
    manifest_summary: dict[str, Any],
    exact_sensitivity_summary: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    settings = config["selection"]
    ineligible = eligibility.loc[
        ~eligibility["eligible"], "instrument_family_str"
    ].tolist()
    eligible_not_selected = eligibility.loc[
        eligibility["eligible"]
        & ~eligibility["instrument_family_str"].isin(selection["families"]),
        "instrument_family_str",
    ].tolist()
    bins = selection["pitch_bins"]
    lines = [
        "# NSynth Metadata Audit and Controlled-Subset Decision",
        "",
        "## Predeclared selection rule",
        "",
        f"- Require at least {settings['min_eval_instruments_per_split']} distinct source instruments in both validation and test.",
        f"- Consider {settings['min_families']}-{settings['max_families']} eligible families.",
        f"- Search contiguous intervals of {settings['min_interval_semitones']}-{settings['max_interval_semitones']} semitones, in steps of {settings['interval_step_semitones']}, divisible into three equal bins.",
        f"- Require every split x family x register cell to contain at least {settings['min_notes_per_family_register_split']} notes, {settings['min_instruments_per_family_register_split']} instruments, and {float(settings['min_pitch_coverage_fraction']):.0%} of pitches in the bin.",
        f"- Require at least {settings['min_shared_supported_pitches_per_register_split']} shared pitches per split and register; a supported pitch has at least {settings['exact_pitch_min_notes']} notes from {settings['exact_pitch_min_instruments']} instruments in every selected family.",
        "- Rank feasible results by: more families, wider interval, stronger shared-pitch support, larger weakest cell, greater coverage, then total usable notes.",
        "",
        "## Selected design",
        "",
        f"- Families: {', '.join(selection['families'])}",
        f"- Common MIDI interval: {selection['pitch_low']}-{selection['pitch_high']} ({selection['interval_semitones']} semitones)",
        f"- Lower register: {bins['lower'][0]}-{bins['lower'][1]}",
        f"- Middle register: {bins['middle'][0]}-{bins['middle'][1]}",
        f"- Upper register: {bins['upper'][0]}-{bins['upper'][1]}",
        f"- Shared supported pitches by split/register: {selection['shared_supported_pitches']}",
        f"- Ineligible families: {', '.join(ineligible) if ineligible else 'none'}",
        f"- Eligible but not selected by the joint-overlap rule: {', '.join(eligible_not_selected) if eligible_not_selected else 'none'}",
        "",
        "## Primary register-balanced manifest",
        "",
        f"- Rows: {manifest_summary['rows']:,}",
        f"- Rows by split: {manifest_summary['split_rows']}",
        "- Register-bin and velocity counts are identical across selected families within each split.",
        f"- Maximum source-instrument share in a register x velocity cell: {manifest_summary['maximum_observed_instrument_share_per_cell']:.1%}",
        f"- Random seed: {config['balancing']['random_seed']}",
        "- Official train/validation/test labels are preserved, and train instruments remain disjoint from evaluation instruments.",
        "",
        "## Strict exact-pitch sensitivity manifest",
        "",
        f"- Rows: {exact_sensitivity_summary['rows']:,}",
        f"- Rows by split: {exact_sensitivity_summary['split_rows']}",
        "- Uses only pitches jointly supported by every selected family.",
        "- Exact pitch x velocity counts are identical across families, with the instrument cap checked per exact pitch.",
        "- The strict test subset is too small for the primary per-register analysis and is retained only as a sensitivity check.",
        "",
        "## Core comparison protocol",
        "",
        "- Use the same six-family class universe in every condition.",
        "- Train one classifier per representation on official-train data and tune it with validation data only.",
        "- Compare the same fitted classifier on the unbalanced and primary balanced test conditions over the same MIDI interval.",
        "- Treat the full-range test and strict exact-pitch manifest as secondary analyses.",
        "",
        "## Interpretation boundary",
        "",
        "The selection uses metadata only. Test audio, acoustic features, embeddings, predictions, and test scores were not inspected. The saved item identifiers freeze the evaluation subset before classifier development.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_audit(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_config(config_path)
    metadata_path = _resolve(root, config["metadata_path"])
    tables_dir = _resolve(root, config["tables_dir"])
    figures_dir = _resolve(root, config["figures_dir"])
    reports_dir = _resolve(root, config["reports_dir"])
    manifests_dir = _resolve(root, config["manifests_dir"])
    metadata = pd.read_parquet(metadata_path)

    table_paths = build_audit_tables(metadata, tables_dir)
    selection, eligibility, candidates = select_families_and_interval(
        metadata, config["selection"]
    )
    table_paths.append(_write(eligibility, tables_dir / "family_eligibility.csv"))
    table_paths.append(
        _write(candidates, tables_dir / "feasible_family_interval_candidates.csv")
    )
    table_paths.extend(
        selected_support_tables(metadata, selection, tables_dir)
    )

    manifest, strata = build_balanced_manifest(
        metadata, selection, config["balancing"]
    )
    manifest_summary = validate_manifest(
        manifest, selection, config["balancing"]
    )
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "nsynth_pitch_balanced_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    manifest_csv = manifests_dir / "nsynth_pitch_balanced_item_ids.csv"
    manifest[
        [
            "item_id",
            "split",
            "instrument_family_str",
            "instrument_str",
            "pitch",
            "pitch_bin",
            "velocity",
            "audio_relpath",
            "selection_seed",
        ]
    ].to_csv(manifest_csv, index=False)
    table_paths.append(_write(strata, tables_dir / "balanced_sampling_strata.csv"))

    exact_manifest, exact_strata = build_exact_pitch_sensitivity_manifest(
        metadata, selection, config["balancing"]
    )
    exact_summary = validate_exact_pitch_manifest(
        exact_manifest, selection, config["balancing"]
    )
    exact_path = manifests_dir / "nsynth_exact_pitch_sensitivity_manifest.parquet"
    exact_manifest.to_parquet(exact_path, index=False)
    exact_csv = manifests_dir / "nsynth_exact_pitch_sensitivity_item_ids.csv"
    exact_manifest[
        [
            "item_id",
            "split",
            "instrument_family_str",
            "instrument_str",
            "pitch",
            "pitch_bin",
            "velocity",
            "audio_relpath",
            "selection_seed",
        ]
    ].to_csv(exact_csv, index=False)
    table_paths.append(
        _write(exact_strata, tables_dir / "exact_pitch_sensitivity_strata.csv")
    )

    figure_paths = create_core_audit_figures(metadata, figures_dir)
    figure_paths.extend(
        create_additional_figures(metadata, selection, manifest, figures_dir)
    )
    report_path = write_report(
        reports_dir / "nsynth_metadata_audit.md",
        selection,
        eligibility,
        manifest_summary,
        exact_summary,
        config,
    )
    decision = {
        "config": config_path.relative_to(root).as_posix(),
        "selection": selection,
        "manifest_summary": manifest_summary,
        "manifest_parquet": manifest_path.relative_to(root).as_posix(),
        "manifest_item_ids": manifest_csv.relative_to(root).as_posix(),
        "exact_pitch_sensitivity_summary": exact_summary,
        "exact_pitch_sensitivity_parquet": exact_path.relative_to(root).as_posix(),
        "exact_pitch_sensitivity_item_ids": exact_csv.relative_to(root).as_posix(),
        "tables": [path.relative_to(root).as_posix() for path in table_paths],
        "figures": [path.relative_to(root).as_posix() for path in figure_paths],
        "report": report_path.relative_to(root).as_posix(),
    }
    decision_path = manifests_dir / "nsynth_pitch_control_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NSynth metadata and freeze a pitch-controlled subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "nsynth_audit.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_audit(args.config.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
