"""Evaluate already-fitted handcrafted and MERT systems on a frozen manifest."""

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

from src.models.handcrafted_baseline import load_feature_cache


REGISTER_ORDER = ("lower", "middle", "upper")


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split",
        "item_id",
        "instrument_family_str",
        "pitch",
        "pitch_bin",
        "velocity",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Balanced manifest missing columns: {sorted(missing)}")
    test = manifest.loc[manifest["split"].eq("test")].copy()
    if test.empty:
        raise RuntimeError("Balanced manifest has no test rows")
    if test.duplicated(["split", "item_id"]).any():
        raise RuntimeError("Balanced test manifest contains duplicate keys")
    if set(test["pitch_bin"].unique()) != set(REGISTER_ORDER):
        raise RuntimeError("Balanced test must contain lower, middle, and upper bins")
    counts = test.groupby(
        ["pitch_bin", "instrument_family_str"], observed=True
    ).size()
    if counts.groupby(level=0).nunique().ne(1).any():
        raise RuntimeError("Family counts are not balanced within every register")
    return test.sort_values(["pitch_bin", "instrument_family_str", "item_id"]).reset_index(
        drop=True
    )


def _align_handcrafted(
    manifest_test: pd.DataFrame,
    feature_cache_dir: Path,
    bundle: dict[str, Any],
) -> np.ndarray:
    features = load_feature_cache(feature_cache_dir)
    feature_columns = list(bundle["feature_columns"])
    selected = features.loc[
        features["split"].eq("test"),
        ["split", "item_id", "instrument_family_str", "pitch", "velocity"]
        + feature_columns,
    ].copy()
    aligned = manifest_test[
        ["split", "item_id", "instrument_family_str", "pitch", "velocity"]
    ].merge(
        selected,
        on=["split", "item_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", "_cache"),
        indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise RuntimeError("Some balanced items have no handcrafted feature row")
    for field in ("instrument_family_str", "pitch", "velocity"):
        if not np.array_equal(
            aligned[f"{field}_manifest"].to_numpy(),
            aligned[f"{field}_cache"].to_numpy(),
        ):
            raise RuntimeError(f"Handcrafted metadata mismatch for {field}")
    matrix = aligned[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Balanced handcrafted matrix contains non-finite values")
    return matrix


def _align_mert(
    root: Path,
    manifest_test: pd.DataFrame,
    index_path: Path,
    bundle: dict[str, Any],
) -> np.ndarray:
    index = pd.read_parquet(index_path)
    selected = index.loc[
        index["split"].eq("test"),
        [
            "split",
            "item_id",
            "instrument_family_str",
            "pitch",
            "velocity",
            "embedding_shard",
            "embedding_row",
            "embedding_dim",
            "cache_signature",
        ],
    ].copy()
    aligned = manifest_test[
        ["split", "item_id", "instrument_family_str", "pitch", "velocity"]
    ].merge(
        selected,
        on=["split", "item_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", "_cache"),
        indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise RuntimeError("Some balanced items have no MERT embedding row")
    for field in ("instrument_family_str", "pitch", "velocity"):
        if not np.array_equal(
            aligned[f"{field}_manifest"].to_numpy(),
            aligned[f"{field}_cache"].to_numpy(),
        ):
            raise RuntimeError(f"MERT metadata mismatch for {field}")
    signature = str(bundle["embedding_metadata"]["cache_signature"])
    if set(aligned["cache_signature"].unique()) != {signature}:
        raise RuntimeError("MERT model and embedding cache signatures do not match")
    dimensions = aligned["embedding_dim"].unique()
    expected_dim = int(bundle["embedding_dim"])
    if len(dimensions) != 1 or int(dimensions[0]) != expected_dim:
        raise RuntimeError("MERT embedding dimensions do not match fitted model")

    matrix = np.empty((len(aligned), expected_dim), dtype=np.float32)
    for shard_name, rows in aligned.groupby("embedding_shard", sort=False):
        shard = np.load(root / str(shard_name), mmap_mode="r", allow_pickle=False)
        embedding_rows = rows["embedding_row"].to_numpy(dtype=np.int64)
        matrix[rows.index.to_numpy()] = shard[embedding_rows]
    if not np.isfinite(matrix).all():
        raise RuntimeError("Balanced MERT matrix contains non-finite values")
    return matrix


def evaluate_predictions(
    manifest_test: pd.DataFrame,
    predictions: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Calculate overall, per-register, and family-by-register results."""

    y_true = manifest_test["instrument_family_str"].to_numpy()
    if len(predictions) != len(y_true):
        raise RuntimeError("Prediction and manifest lengths do not match")
    overall = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=labels, average="macro")
        ),
        "per_family_recall": {
            label: float(value)
            for label, value in zip(
                labels,
                recall_score(y_true, predictions, labels=labels, average=None),
            )
        },
    }
    register_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for register in REGISTER_ORDER:
        mask = manifest_test["pitch_bin"].eq(register).to_numpy()
        true_register = y_true[mask]
        predicted_register = predictions[mask]
        register_rows.append(
            {
                "pitch_bin": register,
                "rows": int(mask.sum()),
                "accuracy": float(accuracy_score(true_register, predicted_register)),
                "macro_f1": float(
                    f1_score(
                        true_register,
                        predicted_register,
                        labels=labels,
                        average="macro",
                    )
                ),
            }
        )
        recalls = recall_score(
            true_register,
            predicted_register,
            labels=labels,
            average=None,
        )
        supports = pd.Series(true_register).value_counts()
        for label, recall in zip(labels, recalls):
            recall_rows.append(
                {
                    "pitch_bin": register,
                    "instrument_family": label,
                    "support": int(supports.get(label, 0)),
                    "recall": float(recall),
                }
            )
    return overall, pd.DataFrame(register_rows), pd.DataFrame(recall_rows)


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    representation: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted family")
    axis.set_ylabel("True family")
    axis.set_title(f"{representation}: pitch-balanced test")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_family_recall_heatmaps(recalls: pd.DataFrame, path: Path) -> None:
    """Plot family recall across registers for both representations."""

    families = sorted(recalls["instrument_family"].unique().tolist())
    expected_rows = len(families) * len(REGISTER_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for index, representation in enumerate(("handcrafted", "mert")):
        selected = recalls.loc[recalls["representation"].eq(representation)]
        if len(selected) != expected_rows:
            raise RuntimeError(
                f"Incomplete family/register recall grid for {representation}"
            )
        matrix = selected.pivot(
            index="instrument_family", columns="pitch_bin", values="recall"
        ).reindex(index=families, columns=list(REGISTER_ORDER))
        if matrix.isna().any().any():
            raise RuntimeError(
                f"Missing family/register recall values for {representation}"
            )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar=index == 1,
            ax=axes[index],
        )
        axes[index].set_title(
            "Handcrafted" if representation == "handcrafted" else "Frozen MERT"
        )
        axes[index].set_xlabel("Register")
        axes[index].set_ylabel("Instrument family" if index == 0 else "")
        axes[index].tick_params(axis="x", rotation=0)
        axes[index].tick_params(axis="y", rotation=0)
    fig.suptitle("Pitch-balanced family recall by register", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ordinary_metrics(path: Path, representation: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if representation == "handcrafted":
        values = payload["metrics"]["test"]
    else:
        values = payload["metrics"]["ordinary_test"]
    return {
        "accuracy": float(values["accuracy"]),
        "macro_f1": float(values["macro_f1"]),
        "rows": int(values["rows"]),
    }


def run_balanced_evaluation(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both fitted systems without fitting or changing either model."""

    manifest = pd.read_parquet(root / config["manifest_path"])
    manifest_test = _validate_manifest(manifest)
    results_dir = root / config["results_dir"]
    figures_dir = root / config["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    systems: dict[str, dict[str, Any]] = {}
    predictions_by_system: dict[str, np.ndarray] = {}

    handcrafted_bundle = joblib.load(root / config["handcrafted"]["model_path"])
    handcrafted_matrix = _align_handcrafted(
        manifest_test,
        root / config["handcrafted"]["feature_cache_dir"],
        handcrafted_bundle,
    )
    handcrafted_prediction = handcrafted_bundle["classifier"].predict(
        handcrafted_bundle["scaler"].transform(handcrafted_matrix)
    )
    predictions_by_system["handcrafted"] = handcrafted_prediction

    mert_bundle = joblib.load(root / config["mert"]["model_path"])
    mert_matrix = _align_mert(
        root,
        manifest_test,
        root / config["mert"]["embedding_index"],
        mert_bundle,
    )
    mert_prediction = mert_bundle["classifier"].predict(
        mert_bundle["scaler"].transform(mert_matrix)
    )
    predictions_by_system["mert"] = mert_prediction

    comparison_rows: list[dict[str, Any]] = []
    register_tables: list[pd.DataFrame] = []
    recall_tables: list[pd.DataFrame] = []
    for name, bundle in (
        ("handcrafted", handcrafted_bundle),
        ("mert", mert_bundle),
    ):
        labels = list(bundle["families"])
        prediction = predictions_by_system[name]
        overall, register_table, recall_table = evaluate_predictions(
            manifest_test, prediction, labels
        )
        ordinary = _ordinary_metrics(
            root / config[name]["ordinary_metrics_path"], name
        )
        systems[name] = {
            "model_path": config[name]["model_path"],
            "model_reused_without_refit": True,
            "ordinary": ordinary,
            "balanced": overall,
            "delta_balanced_minus_ordinary": {
                "accuracy": overall["accuracy"] - ordinary["accuracy"],
                "macro_f1": overall["macro_f1"] - ordinary["macro_f1"],
            },
        }
        comparison_rows.append(
            {
                "representation": name,
                "ordinary_rows": ordinary["rows"],
                "ordinary_accuracy": ordinary["accuracy"],
                "ordinary_macro_f1": ordinary["macro_f1"],
                "balanced_rows": overall["rows"],
                "balanced_accuracy": overall["accuracy"],
                "balanced_macro_f1": overall["macro_f1"],
                "accuracy_change": overall["accuracy"] - ordinary["accuracy"],
                "macro_f1_change": overall["macro_f1"] - ordinary["macro_f1"],
            }
        )
        register_table.insert(0, "representation", name)
        recall_table.insert(0, "representation", name)
        register_tables.append(register_table)
        recall_tables.append(recall_table)

        prediction_table = manifest_test[
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
        prediction_table["predicted_family"] = prediction
        prediction_table.to_csv(
            results_dir / f"{name}_balanced_predictions.csv", index=False
        )
        _plot_confusion(
            manifest_test["instrument_family_str"].to_numpy(),
            prediction,
            labels,
            "Handcrafted" if name == "handcrafted" else "Frozen MERT",
            figures_dir / f"{name}_balanced_confusion_matrix.png",
        )

    comparison = pd.DataFrame(comparison_rows)
    registers = pd.concat(register_tables, ignore_index=True)
    recalls = pd.concat(recall_tables, ignore_index=True)
    comparison.to_csv(results_dir / "ordinary_vs_balanced.csv", index=False)
    registers.to_csv(results_dir / "balanced_metrics_by_register.csv", index=False)
    recalls.to_csv(results_dir / "balanced_family_recall_by_register.csv", index=False)
    _plot_family_recall_heatmaps(
        recalls,
        figures_dir / "balanced_family_recall_by_register_heatmap.png",
    )

    melted = comparison.melt(
        id_vars="representation",
        value_vars=["ordinary_macro_f1", "balanced_macro_f1"],
        var_name="condition",
        value_name="macro_f1",
    )
    melted["condition"] = melted["condition"].map(
        {
            "ordinary_macro_f1": "Ordinary",
            "balanced_macro_f1": "Pitch-balanced",
        }
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=melted,
        x="representation",
        y="macro_f1",
        hue="condition",
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Representation")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Ordinary versus pitch-balanced instrument-family recognition")
    axis.legend(title="Test condition")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "ordinary_vs_balanced_macro_f1.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=registers,
        x="pitch_bin",
        y="macro_f1",
        hue="representation",
        style="representation",
        markers=True,
        sort=False,
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Register")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Pitch-balanced performance by register")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "balanced_macro_f1_by_register.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    report = {
        "manifest": config["manifest_path"],
        "balanced_test_rows": int(len(manifest_test)),
        "models_refit": False,
        "systems": systems,
    }
    (results_dir / "balanced_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown_report(results_dir / "README.md", comparison, registers, systems)
    return report


def _write_markdown_report(
    path: Path,
    comparison: pd.DataFrame,
    registers: pd.DataFrame,
    systems: dict[str, Any],
) -> None:
    lines = [
        "# Ordinary versus Pitch-Balanced Evaluation",
        "",
        "Both train-only-fitted scaler/classifier bundles were reused without refitting.",
        "",
        "## Overall results",
        "",
        "| Representation | Ordinary macro-F1 | Balanced macro-F1 | Change | Ordinary accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.ordinary_macro_f1:.4f} | "
            f"{row.balanced_macro_f1:.4f} | {row.macro_f1_change:+.4f} | "
            f"{row.ordinary_accuracy:.4f} | {row.balanced_accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Balanced overall performance by register",
            "",
            "| Representation | Register | Rows | Accuracy | Macro-F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for representation in ("handcrafted", "mert"):
        rows = registers.loc[registers["representation"].eq(representation)].set_index(
            "pitch_bin"
        )
        for register in REGISTER_ORDER:
            lines.append(
                f"| {representation} | {register} | "
                f"{int(rows.loc[register, 'rows']):,} | "
                f"{rows.loc[register, 'accuracy']:.4f} | "
                f"{rows.loc[register, 'macro_f1']:.4f} |"
            )
    lines.extend(
        [
            "",
            "![Balanced macro-F1 by register](../../figures/balanced_evaluation/balanced_macro_f1_by_register.png)",
            "",
            "![Family recall by register](../../figures/balanced_evaluation/balanced_family_recall_by_register_heatmap.png)",
            "",
            "## Interpretation guardrail",
            "",
            "A performance decrease after balancing is evidence that the ordinary score benefited from family-specific pitch/register distributions. Stability or improvement suggests stronger robustness under the controlled distribution. The balanced test has 594 notes, so per-register and per-family estimates should be interpreted with their support counts.",
            "",
            "Detailed predictions, the family-by-register recall table, and its heatmap are stored beside this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
