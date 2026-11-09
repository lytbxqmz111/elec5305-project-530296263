"""Controlled cross-register generalisation on the frozen NSynth manifest."""

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
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.handcrafted_baseline import load_feature_cache


REGISTERS = ("lower", "middle", "upper")
SPLITS = ("train", "valid", "test")
KEY = ["split", "item_id"]
META = ["instrument_family_str", "instrument_str", "pitch", "velocity"]


def validate_manifest(manifest: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    """Check the frozen design and return rows in stable order."""
    required = set(KEY + META + ["pitch_bin"])
    if required - set(manifest):
        raise KeyError(f"Manifest missing columns: {sorted(required - set(manifest))}")
    if manifest.duplicated(KEY).any():
        raise ValueError("Duplicate manifest split/item_id keys")
    if set(manifest["split"]) != set(SPLITS):
        raise ValueError("Manifest must contain official train, valid, and test splits")
    selection = decision["selection"]
    families = sorted(selection["families"])
    if set(manifest["instrument_family_str"]) != set(families):
        raise ValueError("Manifest families differ from frozen selection")
    for register, bounds in selection["pitch_bins"].items():
        rows = manifest.loc[manifest["pitch_bin"].eq(register)]
        if rows.empty or not rows["pitch"].between(*bounds).all():
            raise ValueError(f"Pitch-bin mismatch for {register}")
    if not manifest["pitch_bin"].isin(REGISTERS).all():
        raise ValueError("Unknown pitch bin")
    # NSynth train instruments are separate from both evaluation splits;
    # validation and test may share source instruments in the official split.
    instruments = [set(manifest.loc[manifest["split"].eq(split), "instrument_str"]) for split in SPLITS]
    if instruments[0] & (instruments[1] | instruments[2]):
        raise ValueError("Train source instrument overlaps validation or test")
    counts = manifest.groupby(["split", "pitch_bin", "instrument_family_str"], observed=True).size()
    if len(counts) != len(SPLITS) * len(REGISTERS) * len(families) or (counts <= 0).any():
        raise ValueError("Incomplete family/register grid")
    if counts.groupby(level=[0, 1]).nunique().ne(1).any():
        raise ValueError("Families are not balanced within split and register")
    return manifest.sort_values(KEY).reset_index(drop=True)


def align_cache(manifest: pd.DataFrame, cache_metadata: pd.DataFrame) -> np.ndarray:
    """Map manifest rows to cache rows while verifying all shared metadata."""
    if cache_metadata.duplicated(KEY).any():
        raise ValueError("Duplicate cache split/item_id keys")
    available = cache_metadata.reset_index(names="cache_row")[KEY + META + ["cache_row"]]
    aligned = manifest[KEY + META].merge(
        available, on=KEY, how="left", validate="one_to_one",
        suffixes=("_manifest", "_cache"), indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Manifest rows missing from representation cache")
    for field in META:
        if not np.array_equal(aligned[f"{field}_manifest"].to_numpy(), aligned[f"{field}_cache"].to_numpy()):
            raise ValueError(f"Cache metadata mismatch: {field}")
    return aligned["cache_row"].to_numpy(dtype=np.int64)


def load_handcrafted(root: Path, settings: dict[str, Any], manifest: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(root / settings["family_model_path"])
    frame = load_feature_cache(root / settings["feature_cache_dir"])
    rows = align_cache(manifest, frame)
    matrix = frame.iloc[rows][list(bundle["feature_columns"])].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite handcrafted features")
    return matrix


def load_mert(root: Path, settings: dict[str, Any], manifest: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(root / settings["family_model_path"])
    index = pd.read_parquet(root / settings["embedding_index"]).reset_index(drop=True)
    rows = align_cache(manifest, index)
    selected = index.iloc[rows].reset_index(drop=True)
    expected_dim = int(bundle["embedding_dim"])
    expected_signature = str(bundle["embedding_metadata"]["cache_signature"])
    if set(selected["embedding_dim"]) != {expected_dim} or set(selected["cache_signature"].astype(str)) != {expected_signature}:
        raise ValueError("MERT cache dimension or signature mismatch")
    matrix = np.empty((len(selected), expected_dim), dtype=np.float32)
    for shard_name, group in selected.groupby("embedding_shard", sort=False):
        shard = np.load(root / str(shard_name), mmap_mode="r", allow_pickle=False)
        shard_rows = group["embedding_row"].to_numpy(dtype=np.int64)
        if shard_rows.min() < 0 or shard_rows.max() >= len(shard):
            raise ValueError(f"MERT shard row out of bounds: {shard_name}")
        matrix[group.index.to_numpy()] = shard[shard_rows]
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite MERT embeddings")
    return matrix


def score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    recalls = recall_score(y_true, y_pred, labels=labels, average=None)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "per_family_recall": {label: float(value) for label, value in zip(labels, recalls)},
    }


def fit_source(
    manifest: pd.DataFrame, matrix: np.ndarray, source: str,
    settings: dict[str, Any], labels: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Tune on source-register validation; evaluate both test registers once."""
    if source not in ("lower", "upper"):
        raise ValueError("Source register must be lower or upper")
    if len(manifest) != len(matrix):
        raise ValueError("Manifest and matrix lengths differ")
    masks = {split: (manifest["split"].eq(split) & manifest["pitch_bin"].eq(source)).to_numpy() for split in ("train", "valid")}
    target = manifest["instrument_family_str"].to_numpy()
    for split in masks:
        if set(target[masks[split]]) != set(labels):
            raise ValueError(f"{split} source register lacks a family")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(matrix[masks["train"]])
    valid_x = scaler.transform(matrix[masks["valid"]])
    candidates = []
    models = {}
    for value in settings["c_values"]:
        c = float(value)
        model = LinearSVC(C=c, class_weight=settings.get("class_weight"), max_iter=int(settings["max_iter"]), random_state=int(settings["random_seed"]), dual="auto")
        model.fit(train_x, target[masks["train"]])
        metrics = score(target[masks["valid"]], model.predict(valid_x), labels)
        candidates.append({"C": c, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        models[c] = model
    selection = pd.DataFrame(candidates).sort_values(["macro_f1", "accuracy", "C"], ascending=[False, False, True])
    best_c = float(selection.iloc[0]["C"])
    model = models[best_c]
    predictions = []
    metrics = {}
    for target_register in ("lower", "upper"):
        mask = (manifest["split"].eq("test") & manifest["pitch_bin"].eq(target_register)).to_numpy()
        y_pred = model.predict(scaler.transform(matrix[mask]))
        metrics[target_register] = score(target[mask], y_pred, labels)
        table = manifest.loc[mask, KEY + META + ["pitch_bin"]].copy()
        table["source_register"] = source
        table["predicted_family"] = y_pred
        predictions.append(table)
    other = "upper" if source == "lower" else "lower"
    metrics["cross_minus_same_macro_f1"] = metrics[other]["macro_f1"] - metrics[source]["macro_f1"]
    report = {"source_register": source, "train_rows": int(masks["train"].sum()), "valid_rows": int(masks["valid"].sum()), "selected_C": best_c, "validation_selection": candidates, "test": metrics}
    return {"scaler": scaler, "classifier": model, "families": labels, "source_register": source, "selected_C": best_c}, pd.concat(predictions, ignore_index=True), report


def run_cross_register(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    decision = json.loads((root / config["selection_decision_path"]).read_text(encoding="utf-8"))
    manifest = validate_manifest(pd.read_parquet(root / config["manifest_path"]), decision)
    labels = sorted(decision["selection"]["families"])
    results_dir = root / config["results_dir"]
    figures_dir = root / config["figures_dir"]
    model_dir = root / config["model_dir"]
    for directory in (results_dir, figures_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"design": "Frozen balanced manifest; source-register train/valid, official instrument-disjoint test; same and cross targets", "manifest": config["manifest_path"], "systems": {}}
    rows = []
    for name, loader in (("handcrafted", load_handcrafted), ("mert", load_mert)):
        print(f"Loading {name} cache...", flush=True)
        matrix = loader(root, config[name], manifest)
        summary["systems"][name] = {}
        for source in config["source_registers"]:
            print(f"Training {name}: {source} register...", flush=True)
            bundle, predictions, report = fit_source(manifest, matrix, source, config["classifier"], labels)
            joblib.dump(bundle, model_dir / f"{name}_{source}.joblib")
            predictions.to_csv(results_dir / f"{name}_{source}_test_predictions.csv", index=False)
            pd.DataFrame(report["validation_selection"]).to_csv(results_dir / f"{name}_{source}_validation_c.csv", index=False)
            summary["systems"][name][source] = report
            for target_register in ("lower", "upper"):
                metric = report["test"][target_register]
                rows.append({"representation": name, "train_register": source, "test_register": target_register, "condition": "same" if source == target_register else "cross", "rows": metric["rows"], "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"]})
        del matrix
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "cross_register_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, name in zip(axes, ("handcrafted", "mert")):
        subset = table.loc[table["representation"].eq(name)]
        for index, source in enumerate(("lower", "upper")):
            pair = subset.loc[subset["train_register"].eq(source)].set_index("test_register")
            axis.bar([index * 3, index * 3 + 1], [pair.loc[source, "macro_f1"], pair.loc["upper" if source == "lower" else "lower", "macro_f1"]], color=["#3b7f75", "#c48251"])
        axis.set_xticks([0, 1, 3, 4], ["L→L", "L→U", "U→U", "U→L"])
        axis.set_title(name.capitalize())
        axis.set_ylim(0, 1)
        axis.set_ylabel("Test macro-F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "same_vs_cross_macro_f1.png", dpi=180)
    plt.close(fig)
    (results_dir / "cross_register_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
