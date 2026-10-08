"""Train and evaluate a train-only-scaled linear SVM baseline."""

from __future__ import annotations

import json
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


def load_feature_cache(cache_dir: Path) -> pd.DataFrame:
    parts = sorted((cache_dir / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature parts found under {cache_dir / 'parts'}")
    frame = pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)
    if frame["item_id"].duplicated().any():
        raise RuntimeError("Feature cache contains duplicate item identifiers")
    return frame


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    split: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted family")
    axis.set_ylabel("True family")
    axis.set_title(f"Handcrafted baseline: {split} confusion matrix")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    features: pd.DataFrame,
    feature_columns: list[str],
    classifier_settings: dict[str, Any],
    model_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    ordinary_test_pitch_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Select C on validation, then perform one untouched test evaluation."""

    if set(features["split"].unique()) != set(SPLITS):
        raise RuntimeError("Feature cache must contain train, valid, and test")
    labels = sorted(features["instrument_family_str"].unique().tolist())
    split_frames = {
        split: features.loc[features["split"].eq(split)].copy() for split in SPLITS
    }
    arrays = {
        split: split_frames[split][feature_columns].to_numpy(dtype=np.float64)
        for split in SPLITS
    }
    targets = {
        split: split_frames[split]["instrument_family_str"].to_numpy()
        for split in SPLITS
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Feature matrix contains non-finite values")

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(arrays["train"])
    scaled_valid = scaler.transform(arrays["valid"])
    scaled_test = scaler.transform(arrays["test"])

    seed = int(classifier_settings["random_seed"])
    candidates: list[dict[str, float]] = []
    models: dict[float, LinearSVC] = {}
    for c_value in classifier_settings["c_values"]:
        c_float = float(c_value)
        model = LinearSVC(
            C=c_float,
            class_weight=classifier_settings.get("class_weight"),
            max_iter=int(classifier_settings["max_iter"]),
            random_state=seed,
            dual="auto",
        )
        model.fit(scaled_train, targets["train"])
        prediction = model.predict(scaled_valid)
        candidates.append(
            {"C": c_float, **metrics_for(targets["valid"], prediction)}
        )
        models[c_float] = model

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["macro_f1", "accuracy", "C"], ascending=[False, False, True]
    )
    best_c = float(candidate_frame.iloc[0]["C"])
    final_model = models[best_c]
    full_predictions = {
        "valid": final_model.predict(scaled_valid),
        "test": final_model.predict(scaled_test),
    }
    evaluation_frames = {
        "valid": split_frames["valid"],
        "test": split_frames["test"],
    }
    evaluation_targets = {
        "valid": targets["valid"],
        "test": targets["test"],
    }
    predictions = dict(full_predictions)
    if ordinary_test_pitch_range is not None:
        pitch_low, pitch_high = ordinary_test_pitch_range
        test_mask = split_frames["test"]["pitch"].between(
            pitch_low, pitch_high
        ).to_numpy()
        evaluation_frames["test"] = split_frames["test"].loc[test_mask].copy()
        evaluation_targets["test"] = targets["test"][test_mask]
        predictions["test"] = full_predictions["test"][test_mask]

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(results_dir / "validation_c_selection.csv", index=False)

    split_metrics: dict[str, Any] = {}
    for split in ("valid", "test"):
        y_true = evaluation_targets[split]
        y_pred = predictions[split]
        recalls = recall_score(y_true, y_pred, labels=labels, average=None)
        split_metrics[split] = {
            **metrics_for(y_true, y_pred),
            "rows": int(len(y_true)),
            "per_family_recall": {
                label: float(value) for label, value in zip(labels, recalls)
            },
        }
        prediction_table = evaluation_frames[split][
            ["item_id", "instrument_family_str", "instrument_str", "pitch", "velocity"]
        ].copy()
        prediction_table["predicted_family"] = y_pred
        prediction_table.to_csv(results_dir / f"{split}_predictions.csv", index=False)
        pd.DataFrame(
            {"instrument_family": labels, "recall": recalls}
        ).to_csv(results_dir / f"{split}_per_family_recall.csv", index=False)
        _plot_confusion(
            y_true,
            y_pred,
            labels,
            split,
            figures_dir / f"{split}_confusion_matrix.png",
        )

    bundle = {
        "scaler": scaler,
        "classifier": final_model,
        "feature_columns": feature_columns,
        "families": labels,
        "selected_C": best_c,
        "random_seed": seed,
    }
    joblib.dump(bundle, model_dir / "handcrafted_linear_svm.joblib")
    report = {
        "representation": "handcrafted_acoustic_features",
        "classifier": "LinearSVC",
        "scaler_fit_split": "train",
        "classifier_fit_split": "train",
        "hyperparameter_selection_split": "valid",
        "test_evaluations": 1,
        "selected_C": best_c,
        "feature_count": len(feature_columns),
        "families": labels,
        "source_split_rows": {
            split: int(len(frame)) for split, frame in split_frames.items()
        },
        "ordinary_test_pitch_range": (
            list(ordinary_test_pitch_range)
            if ordinary_test_pitch_range is not None
            else None
        ),
        "metrics": split_metrics,
    }
    (results_dir / "ordinary_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pitch_description = (
        f"MIDI {ordinary_test_pitch_range[0]}-{ordinary_test_pitch_range[1]}"
        if ordinary_test_pitch_range is not None
        else "the full selected-family test split"
    )
    markdown = [
        "# Handcrafted Baseline: Ordinary Result",
        "",
        "## Protocol",
        "",
        "- Representation: 76 aggregated MFCC/delta, spectral, temporal, and attack features.",
        "- Training data: all official-train notes from the six selected families.",
        "- StandardScaler fit: train only.",
        "- LinearSVC fit: train only.",
        "- C selection: validation macro-F1 over 0.01, 0.1, and 1.0.",
        f"- Ordinary test condition: {pitch_description}, without pitch balancing.",
        "- Test evaluation: one run after C was selected.",
        "",
        "## Results",
        "",
        "| Split | Rows | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
        f"| Validation | {split_metrics['valid']['rows']:,} | {split_metrics['valid']['accuracy']:.4f} | {split_metrics['valid']['macro_f1']:.4f} |",
        f"| Ordinary test | {split_metrics['test']['rows']:,} | {split_metrics['test']['accuracy']:.4f} | {split_metrics['test']['macro_f1']:.4f} |",
        "",
        f"Selected C: **{best_c:g}**",
        "",
        "## Ordinary-test recall by family",
        "",
        "| Family | Recall |",
        "|---|---:|",
    ]
    markdown.extend(
        f"| {label} | {split_metrics['test']['per_family_recall'][label]:.4f} |"
        for label in labels
    )
    markdown.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python scripts\\extract_handcrafted_features.py --config configs\\handcrafted_baseline.yaml --cache-tag ordinary",
            "python scripts\\train_handcrafted_baseline.py --config configs\\handcrafted_baseline.yaml --cache-tag ordinary",
            "```",
            "",
            "Feature parts are cached under `data/cache/handcrafted/ordinary/` and are ignored by Git. Re-running extraction resumes completed parts.",
        ]
    )
    (results_dir / "README.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report
