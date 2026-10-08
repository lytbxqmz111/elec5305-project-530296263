"""Train and evaluate a simple LinearSVC on frozen MERT embeddings."""

from __future__ import annotations

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


SPLITS = ("train", "valid", "test")


def load_embedding_matrix(
    root: Path,
    index_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load sharded embeddings in exact index order and validate every row."""

    index = pd.read_parquet(index_path).reset_index(drop=True)
    if index.empty:
        raise RuntimeError("MERT embedding index is empty")
    if index.duplicated(["split", "item_id"]).any():
        raise RuntimeError("MERT index contains duplicate split/item_id keys")
    if set(index["split"].unique()) != set(SPLITS):
        raise RuntimeError("MERT index must contain train, valid, and test")
    dimensions = index["embedding_dim"].unique()
    if len(dimensions) != 1:
        raise RuntimeError(f"Mixed embedding dimensions: {dimensions.tolist()}")
    embedding_dim = int(dimensions[0])
    matrix = np.empty((len(index), embedding_dim), dtype=np.float32)

    for shard_name, rows in index.groupby("embedding_shard", sort=False):
        shard_path = root / str(shard_name)
        shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        embedding_rows = rows["embedding_row"].to_numpy(dtype=np.int64)
        if embedding_rows.min() < 0 or embedding_rows.max() >= len(shard):
            raise RuntimeError(f"Embedding row outside shard bounds: {shard_path}")
        values = np.asarray(shard[embedding_rows], dtype=np.float32)
        if values.shape != (len(rows), embedding_dim):
            raise RuntimeError(
                f"Invalid selected shape {values.shape} from {shard_path}"
            )
        matrix[rows.index.to_numpy()] = values

    if not np.isfinite(matrix).all():
        raise RuntimeError("MERT matrix contains NaN or infinite values")
    return index, matrix


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted family")
    axis.set_ylabel("True family")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_mert_ordinary_baseline(
    index: pd.DataFrame,
    embeddings: np.ndarray,
    classifier_settings: dict[str, Any],
    ordinary_pitch_range: tuple[int, int],
    model_dir: Path,
    results_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Select C on validation and perform one ordinary test evaluation."""

    if len(index) != len(embeddings):
        raise RuntimeError("Index and embedding row counts do not match")
    labels = sorted(index["instrument_family_str"].unique().tolist())
    train_mask = index["split"].eq("train").to_numpy()
    valid_mask = index["split"].eq("valid").to_numpy()
    pitch_low, pitch_high = ordinary_pitch_range
    test_mask = (
        index["split"].eq("test")
        & index["pitch"].between(pitch_low, pitch_high)
    ).to_numpy()
    if not train_mask.any() or not valid_mask.any() or not test_mask.any():
        raise RuntimeError("Train, validation, and ordinary-test rows are required")

    targets = index["instrument_family_str"].to_numpy()
    x_train = embeddings[train_mask]
    x_valid = embeddings[valid_mask]
    x_test = embeddings[test_mask]
    y_train = targets[train_mask]
    y_valid = targets[valid_mask]
    y_test = targets[test_mask]

    # The scaler is fitted strictly on official-train embeddings.
    print(
        f"Fitting train-only StandardScaler on {len(x_train):,} x "
        f"{x_train.shape[1]} embeddings...",
        flush=True,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    seed = int(classifier_settings["random_seed"])
    candidates: list[dict[str, float]] = []
    models: dict[float, LinearSVC] = {}
    for value in classifier_settings["c_values"]:
        c_value = float(value)
        started = time.perf_counter()
        print(f"Training LinearSVC with C={c_value:g}...", flush=True)
        classifier = LinearSVC(
            C=c_value,
            class_weight=classifier_settings.get("class_weight"),
            max_iter=int(classifier_settings["max_iter"]),
            random_state=seed,
            dual="auto",
        )
        classifier.fit(x_train, y_train)
        valid_prediction = classifier.predict(x_valid)
        candidates.append(
            {"C": c_value, **_metrics(y_valid, valid_prediction)}
        )
        print(
            f"C={c_value:g} complete in {time.perf_counter() - started:.1f}s; "
            f"validation accuracy={candidates[-1]['accuracy']:.4f}, "
            f"macro-F1={candidates[-1]['macro_f1']:.4f}",
            flush=True,
        )
        models[c_value] = classifier

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["macro_f1", "accuracy", "C"], ascending=[False, False, True]
    )
    selected_c = float(candidate_frame.iloc[0]["C"])
    classifier = models[selected_c]
    print(
        f"Selected C={selected_c:g}; running the single ordinary-test evaluation...",
        flush=True,
    )
    valid_prediction = classifier.predict(x_valid)
    # This is the single ordinary-test prediction after validation selection.
    test_prediction = classifier.predict(x_test)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(results_dir / "validation_c_selection.csv", index=False)

    condition_data = {
        "valid": (valid_mask, y_valid, valid_prediction),
        "ordinary_test": (test_mask, y_test, test_prediction),
    }
    condition_metrics: dict[str, Any] = {}
    for condition, (mask, y_true, y_pred) in condition_data.items():
        recalls = recall_score(y_true, y_pred, labels=labels, average=None)
        condition_metrics[condition] = {
            **_metrics(y_true, y_pred),
            "rows": int(len(y_true)),
            "per_family_recall": {
                label: float(value) for label, value in zip(labels, recalls)
            },
        }
        predictions = index.loc[
            mask,
            ["item_id", "split", "instrument_family_str", "instrument_str", "pitch", "velocity"],
        ].copy()
        predictions["predicted_family"] = y_pred
        predictions.to_csv(results_dir / f"{condition}_predictions.csv", index=False)
        pd.DataFrame(
            {"instrument_family": labels, "recall": recalls}
        ).to_csv(results_dir / f"{condition}_per_family_recall.csv", index=False)
        _plot_confusion(
            y_true,
            y_pred,
            labels,
            f"Frozen MERT + LinearSVC: {condition.replace('_', ' ')}",
            figures_dir / f"{condition}_confusion_matrix.png",
        )

    model_metadata = {
        "model_id": str(index["model_id"].iloc[0]),
        "model_revision": str(index["model_revision"].iloc[0]),
        "hidden_layer": int(index["hidden_layer"].iloc[0]),
        "pooling": str(index["pooling"].iloc[0]),
        "cache_signature": str(index["cache_signature"].iloc[0]),
    }
    bundle = {
        "scaler": scaler,
        "classifier": classifier,
        "families": labels,
        "selected_C": selected_c,
        "random_seed": seed,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_metadata": model_metadata,
    }
    joblib.dump(bundle, model_dir / "mert_linear_svm.joblib")

    report = {
        "representation": "frozen_MERT_v0_public_final_layer_mean_pool",
        "classifier": "LinearSVC",
        "scaler_fit_split": "train",
        "classifier_fit_split": "train",
        "hyperparameter_selection_split": "valid",
        "test_evaluations": 1,
        "selected_C": selected_c,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_metadata": model_metadata,
        "families": labels,
        "source_split_rows": {
            split: int(index["split"].eq(split).sum()) for split in SPLITS
        },
        "ordinary_test_pitch_range": [pitch_low, pitch_high],
        "metrics": condition_metrics,
    }
    (results_dir / "ordinary_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    markdown = [
        "# Frozen MERT + LinearSVC: Ordinary Result",
        "",
        "## Protocol",
        "",
        f"- MERT representation: final hidden layer, time-mean pooled, {embeddings.shape[1]} dimensions.",
        "- Training data: all official-train notes from the six selected families.",
        "- StandardScaler and LinearSVC fit: train only.",
        "- C selection: validation macro-F1 over 0.01, 0.1, and 1.0.",
        f"- Ordinary test: MIDI {pitch_low}-{pitch_high}, without pitch balancing.",
        "- Test evaluated once after validation selection.",
        "",
        "## Results",
        "",
        "| Condition | Rows | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
        f"| Validation | {condition_metrics['valid']['rows']:,} | {condition_metrics['valid']['accuracy']:.4f} | {condition_metrics['valid']['macro_f1']:.4f} |",
        f"| Ordinary test | {condition_metrics['ordinary_test']['rows']:,} | {condition_metrics['ordinary_test']['accuracy']:.4f} | {condition_metrics['ordinary_test']['macro_f1']:.4f} |",
        "",
        f"Selected C: **{selected_c:g}**",
        "",
        "## Ordinary-test recall by family",
        "",
        "| Family | Recall |",
        "|---|---:|",
    ]
    markdown.extend(
        f"| {label} | {condition_metrics['ordinary_test']['per_family_recall'][label]:.4f} |"
        for label in labels
    )
    markdown.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python scripts\\train_mert_baseline.py --config configs\\mert_baseline.yaml",
            "```",
        ]
    )
    (results_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report
