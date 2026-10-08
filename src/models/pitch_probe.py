"""Train coarse-pitch probes on frozen handcrafted and MERT representations."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.handcrafted_baseline import load_feature_cache


PITCH_ORDER = ("lower", "middle", "upper")
SPLIT_ORDER = ("train", "valid", "test")


def assign_pitch_bins(
    pitches: pd.Series,
    pitch_bins: dict[str, list[int]],
) -> np.ndarray:
    """Map every MIDI pitch to one non-overlapping, inclusive register bin."""

    values = pitches.to_numpy(dtype=np.int64)
    labels = np.full(len(values), "", dtype=object)
    assigned = np.zeros(len(values), dtype=bool)
    previous_high: int | None = None
    for label in PITCH_ORDER:
        if label not in pitch_bins or len(pitch_bins[label]) != 2:
            raise ValueError(f"Missing or invalid pitch-bin boundary for {label}")
        low, high = (int(value) for value in pitch_bins[label])
        if low > high or (previous_high is not None and low != previous_high + 1):
            raise ValueError("Pitch bins must be ordered, contiguous inclusive intervals")
        mask = (values >= low) & (values <= high)
        if (assigned & mask).any():
            raise ValueError("Pitch-bin definitions overlap")
        labels[mask] = label
        assigned |= mask
        previous_high = high
    if not assigned.all():
        outside = sorted(np.unique(values[~assigned]).tolist())
        raise ValueError(f"Pitches outside configured bins: {outside[:10]}")
    return labels.astype(str)


def _load_handcrafted_subset(
    root: Path,
    settings: dict[str, Any],
    families: list[str],
    pitch_low: int,
    pitch_high: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    family_bundle = joblib.load(root / settings["family_model_path"])
    feature_columns = list(family_bundle["feature_columns"])
    frame = load_feature_cache(root / settings["feature_cache_dir"])
    frame = frame.loc[
        frame["instrument_family_str"].isin(families)
        & frame["pitch"].between(pitch_low, pitch_high)
    ].reset_index(drop=True)
    matrix = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Selected handcrafted features contain non-finite values")
    metadata_columns = [
        "split",
        "item_id",
        "instrument_family_str",
        "instrument_str",
        "pitch",
        "velocity",
    ]
    metadata = frame[metadata_columns].copy()
    representation_metadata = {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    return metadata, matrix, representation_metadata


def _load_mert_subset(
    root: Path,
    settings: dict[str, Any],
    families: list[str],
    pitch_low: int,
    pitch_high: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    family_bundle = joblib.load(root / settings["family_model_path"])
    index = pd.read_parquet(root / settings["embedding_index"])
    selected = index.loc[
        index["instrument_family_str"].isin(families)
        & index["pitch"].between(pitch_low, pitch_high)
    ].reset_index(drop=True)
    if selected.duplicated(["split", "item_id"]).any():
        raise RuntimeError("Selected MERT index contains duplicate keys")
    dimensions = selected["embedding_dim"].unique()
    expected_dim = int(family_bundle["embedding_dim"])
    if len(dimensions) != 1 or int(dimensions[0]) != expected_dim:
        raise RuntimeError("MERT index and family model embedding dimensions differ")
    expected_signature = str(
        family_bundle["embedding_metadata"]["cache_signature"]
    )
    if set(selected["cache_signature"].astype(str).unique()) != {
        expected_signature
    }:
        raise RuntimeError("MERT index and family model cache signatures differ")

    matrix = np.empty((len(selected), expected_dim), dtype=np.float32)
    for shard_name, rows in selected.groupby("embedding_shard", sort=False):
        shard = np.load(root / str(shard_name), mmap_mode="r", allow_pickle=False)
        embedding_rows = rows["embedding_row"].to_numpy(dtype=np.int64)
        if embedding_rows.min() < 0 or embedding_rows.max() >= len(shard):
            raise RuntimeError(f"Embedding row outside shard bounds: {shard_name}")
        matrix[rows.index.to_numpy()] = shard[embedding_rows]
    if not np.isfinite(matrix).all():
        raise RuntimeError("Selected MERT embeddings contain non-finite values")

    metadata_columns = [
        "split",
        "item_id",
        "instrument_family_str",
        "instrument_str",
        "pitch",
        "velocity",
    ]
    representation_metadata = {
        "embedding_dim": expected_dim,
        "embedding_metadata": family_bundle["embedding_metadata"],
    }
    return selected[metadata_columns].copy(), matrix, representation_metadata


def balanced_row_indices(
    metadata: pd.DataFrame,
    balanced_manifest: pd.DataFrame,
) -> np.ndarray:
    """Return balanced-test row indices in manifest order with metadata checks."""

    manifest = balanced_manifest.loc[
        balanced_manifest["split"].eq("test"),
        ["split", "item_id", "instrument_family_str", "pitch", "velocity"],
    ].copy()
    available = metadata.reset_index(names="matrix_row")[
        [
            "matrix_row",
            "split",
            "item_id",
            "instrument_family_str",
            "pitch",
            "velocity",
        ]
    ]
    aligned = manifest.merge(
        available,
        on=["split", "item_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", "_cache"),
        indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise RuntimeError("Balanced test contains items absent from representation cache")
    for field in ("instrument_family_str", "pitch", "velocity"):
        if not np.array_equal(
            aligned[f"{field}_manifest"].to_numpy(),
            aligned[f"{field}_cache"].to_numpy(),
        ):
            raise RuntimeError(f"Balanced/cache metadata mismatch for {field}")
    return aligned["matrix_row"].to_numpy(dtype=np.int64)


def pitch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    recalls = recall_score(y_true, y_pred, labels=list(PITCH_ORDER), average=None)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(PITCH_ORDER), average="macro")
        ),
        "per_pitch_bin_recall": {
            label: float(value) for label, value in zip(PITCH_ORDER, recalls)
        },
    }


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    representation: str,
    condition: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(PITCH_ORDER), normalize="true"
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        vmin=0,
        vmax=1,
        xticklabels=PITCH_ORDER,
        yticklabels=PITCH_ORDER,
        ax=axis,
    )
    axis.set_xlabel("Predicted pitch bin")
    axis.set_ylabel("True pitch bin")
    axis.set_title(f"{representation}: {condition.replace('_', ' ')} pitch probe")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_one_pitch_probe(
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    balanced_indices: np.ndarray,
    pitch_bins: dict[str, list[int]],
    classifier_settings: dict[str, Any],
    representation: str,
    representation_metadata: dict[str, Any],
    model_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Fit one train-only-scaled pitch probe and evaluate fixed conditions."""

    if len(metadata) != len(matrix):
        raise RuntimeError("Metadata and representation matrices have different lengths")
    if set(metadata["split"].unique()) != set(SPLIT_ORDER):
        raise RuntimeError("Pitch probe requires train, valid, and test splits")
    metadata = metadata.copy()
    metadata["pitch_bin"] = assign_pitch_bins(metadata["pitch"], pitch_bins)
    targets = metadata["pitch_bin"].to_numpy()
    masks = {
        split: metadata["split"].eq(split).to_numpy() for split in SPLIT_ORDER
    }

    scaler = StandardScaler()
    print(
        f"[{representation}] fitting train-only scaler on "
        f"{int(masks['train'].sum()):,} rows...",
        flush=True,
    )
    scaled = {
        "train": scaler.fit_transform(matrix[masks["train"]]),
        "valid": scaler.transform(matrix[masks["valid"]]),
        "test": scaler.transform(matrix[masks["test"]]),
        "balanced_test": scaler.transform(matrix[balanced_indices]),
    }
    split_targets = {
        "train": targets[masks["train"]],
        "valid": targets[masks["valid"]],
        "test": targets[masks["test"]],
        "balanced_test": targets[balanced_indices],
    }

    candidates: list[dict[str, float]] = []
    classifiers: dict[float, LinearSVC] = {}
    seed = int(classifier_settings["random_seed"])
    for candidate in classifier_settings["c_values"]:
        c_value = float(candidate)
        started = time.perf_counter()
        classifier = LinearSVC(
            C=c_value,
            class_weight=classifier_settings.get("class_weight"),
            max_iter=int(classifier_settings["max_iter"]),
            random_state=seed,
            dual="auto",
        )
        classifier.fit(scaled["train"], split_targets["train"])
        prediction = classifier.predict(scaled["valid"])
        metrics = pitch_metrics(split_targets["valid"], prediction)
        candidates.append(
            {"C": c_value, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        )
        classifiers[c_value] = classifier
        print(
            f"[{representation}] C={c_value:g}: valid macro-F1="
            f"{metrics['macro_f1']:.4f} ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["macro_f1", "accuracy", "C"], ascending=[False, False, True]
    )
    selected_c = float(candidate_frame.iloc[0]["C"])
    classifier = classifiers[selected_c]
    conditions = {
        "valid": (scaled["valid"], split_targets["valid"], metadata.loc[masks["valid"]]),
        "ordinary_test": (scaled["test"], split_targets["test"], metadata.loc[masks["test"]]),
        "balanced_test": (
            scaled["balanced_test"],
            split_targets["balanced_test"],
            metadata.iloc[balanced_indices],
        ),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(
        results_dir / f"{representation}_validation_c_selection.csv", index=False
    )
    condition_metrics: dict[str, Any] = {}
    for condition, (values, y_true, rows) in conditions.items():
        y_pred = classifier.predict(values)
        condition_metrics[condition] = pitch_metrics(y_true, y_pred)
        predictions = rows[
            [
                "item_id",
                "split",
                "instrument_family_str",
                "instrument_str",
                "pitch",
                "pitch_bin",
                "velocity",
            ]
        ].copy()
        predictions["predicted_pitch_bin"] = y_pred
        predictions.to_csv(
            results_dir / f"{representation}_{condition}_predictions.csv", index=False
        )
        _plot_confusion(
            y_true,
            y_pred,
            representation,
            condition,
            figures_dir / f"{representation}_{condition}_confusion_matrix.png",
        )

    balanced_predictions = classifier.predict(scaled["balanced_test"])
    balanced_rows = metadata.iloc[balanced_indices].copy()
    balanced_rows["predicted_pitch_bin"] = balanced_predictions
    family_rows: list[dict[str, Any]] = []
    for family, group in balanced_rows.groupby("instrument_family_str", sort=True):
        metrics = pitch_metrics(
            group["pitch_bin"].to_numpy(),
            group["predicted_pitch_bin"].to_numpy(),
        )
        family_rows.append(
            {
                "representation": representation,
                "instrument_family": family,
                "rows": metrics["rows"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    pd.DataFrame(family_rows).to_csv(
        results_dir / f"{representation}_balanced_pitch_probe_by_family.csv",
        index=False,
    )

    bundle = {
        "scaler": scaler,
        "classifier": classifier,
        "pitch_bins": pitch_bins,
        "pitch_labels": list(PITCH_ORDER),
        "selected_C": selected_c,
        "random_seed": seed,
        "representation": representation,
        "representation_metadata": representation_metadata,
    }
    joblib.dump(bundle, model_path)
    return {
        "representation": representation,
        "target": "coarse_pitch_bin",
        "classifier": "LinearSVC",
        "scaler_fit_split": "train",
        "classifier_fit_split": "train",
        "hyperparameter_selection_split": "valid",
        "selected_C": selected_c,
        "source_split_rows": {
            split: int(masks[split].sum()) for split in SPLIT_ORDER
        },
        "representation_metadata": representation_metadata,
        "metrics": condition_metrics,
    }


def _write_summary_outputs(
    results: dict[str, Any],
    family_metrics: dict[str, Any],
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation in ("handcrafted", "mert"):
        family = family_metrics["systems"][representation]["balanced"]
        probe = results["representations"][representation]["metrics"]
        rows.append(
            {
                "representation": representation,
                "balanced_family_accuracy": family["accuracy"],
                "balanced_family_macro_f1": family["macro_f1"],
                "ordinary_pitch_accuracy": probe["ordinary_test"]["accuracy"],
                "ordinary_pitch_macro_f1": probe["ordinary_test"]["macro_f1"],
                "balanced_pitch_accuracy": probe["balanced_test"]["accuracy"],
                "balanced_pitch_macro_f1": probe["balanced_test"]["macro_f1"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "family_robustness_vs_pitch_probe.csv", index=False)

    pitch_plot = summary.melt(
        id_vars="representation",
        value_vars=["ordinary_pitch_macro_f1", "balanced_pitch_macro_f1"],
        var_name="condition",
        value_name="macro_f1",
    )
    pitch_plot["condition"] = pitch_plot["condition"].map(
        {
            "ordinary_pitch_macro_f1": "Ordinary test",
            "balanced_pitch_macro_f1": "Pitch-balanced test",
        }
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=pitch_plot,
        x="representation",
        y="macro_f1",
        hue="condition",
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Representation")
    axis.set_ylabel("Pitch-bin macro-F1")
    axis.set_title("Coarse-pitch probe performance")
    figure.tight_layout()
    figure.savefig(
        figures_dir / "pitch_probe_ordinary_vs_balanced.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    task_plot = summary.melt(
        id_vars="representation",
        value_vars=["balanced_family_macro_f1", "balanced_pitch_macro_f1"],
        var_name="task",
        value_name="macro_f1",
    )
    task_plot["task"] = task_plot["task"].map(
        {
            "balanced_family_macro_f1": "Instrument family",
            "balanced_pitch_macro_f1": "Pitch bin",
        }
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=task_plot,
        x="representation",
        y="macro_f1",
        hue="task",
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Representation")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Family robustness and pitch decodability on the balanced test")
    figure.tight_layout()
    figure.savefig(
        figures_dir / "family_robustness_vs_pitch_probe.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    return summary


def _write_markdown_report(
    report: dict[str, Any],
    summary: pd.DataFrame,
    results_dir: Path,
) -> None:
    lines = [
        "# Coarse Pitch-Probe Results",
        "",
        "## Protocol",
        "",
        "- Target: lower, middle, or upper register within the frozen MIDI 43-69 interval.",
        "- Representation: the same cached handcrafted features or frozen MERT embeddings used for family recognition.",
        "- StandardScaler and LinearSVC fit: official train split only.",
        "- C selection: official validation macro-F1 only.",
        "- Evaluation: ordinary test and the same 594-note pitch-balanced test used for family robustness.",
        "",
        "## Results beside family robustness",
        "",
        "| Representation | Balanced family macro-F1 | Ordinary pitch macro-F1 | Balanced pitch macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.balanced_family_macro_f1:.4f} | "
            f"{row.ordinary_pitch_macro_f1:.4f} | {row.balanced_pitch_macro_f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "High pitch predictability is not inherently undesirable: a useful music representation may encode both pitch and timbre. The central question is whether family recognition remains robust while pitch is still decodable.",
            "",
            "Per-bin recall, prediction tables, confusion matrices, and per-family pitch-probe results are saved beside this report.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python scripts\\train_pitch_probe.py --config configs\\pitch_probe.yaml",
            "```",
        ]
    )
    (results_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_pitch_probes(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    decision = json.loads(
        (root / config["selection_decision_path"]).read_text(encoding="utf-8")
    )["selection"]
    families = [str(value) for value in decision["families"]]
    pitch_low = int(decision["pitch_low"])
    pitch_high = int(decision["pitch_high"])
    pitch_bins = decision["pitch_bins"]
    balanced_manifest = pd.read_parquet(root / config["balanced_manifest_path"])
    family_metrics = json.loads(
        (root / config["family_robustness_metrics_path"]).read_text(encoding="utf-8")
    )
    model_dir = root / config["model_dir"]
    results_dir = root / config["results_dir"]
    figures_dir = root / config["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    representations: dict[str, Any] = {}
    loaders = {
        "handcrafted": _load_handcrafted_subset,
        "mert": _load_mert_subset,
    }
    for representation, loader in loaders.items():
        print(f"Loading frozen {representation} representation...", flush=True)
        metadata, matrix, representation_metadata = loader(
            root,
            config[representation],
            families,
            pitch_low,
            pitch_high,
        )
        indices = balanced_row_indices(metadata, balanced_manifest)
        representations[representation] = train_one_pitch_probe(
            metadata,
            matrix,
            indices,
            pitch_bins,
            config["classifier"],
            representation,
            representation_metadata,
            model_dir / f"{representation}_pitch_probe.joblib",
            results_dir,
            figures_dir,
        )
        del metadata, matrix, indices
        gc.collect()

    report = {
        "target": "coarse_pitch_bin",
        "pitch_range": [pitch_low, pitch_high],
        "pitch_bins": pitch_bins,
        "families": families,
        "balanced_test_rows": int(balanced_manifest["split"].eq("test").sum()),
        "representations": representations,
    }
    summary = _write_summary_outputs(
        report,
        family_metrics,
        results_dir,
        figures_dir,
    )
    (results_dir / "pitch_probe_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown_report(report, summary, results_dir)
    return report
