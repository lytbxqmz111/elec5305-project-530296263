"""Frozen MERT extraction, sharded caching, and one-to-one coverage checks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.io import wavfile


SPLITS = ("train", "valid", "test")
INDEX_COLUMNS = [
    "item_id",
    "split",
    "instrument_family_str",
    "instrument_str",
    "pitch",
    "velocity",
    "audio_relpath",
    "embedding_shard",
    "embedding_row",
    "embedding_dim",
    "model_id",
    "model_revision",
    "hidden_layer",
    "pooling",
    "cache_signature",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "metadata_path",
        "selection_decision_path",
        "embedding_dir",
        "manifests_to_validate",
        "model",
        "inference",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Missing MERT config keys: {sorted(missing)}")
    return config


def cache_signature(config: dict[str, Any]) -> str:
    """Hash every setting that changes the numerical embedding cache."""

    decision_path = project_root() / config["selection_decision_path"]
    decision_hash = (
        hashlib.sha256(decision_path.read_bytes()).hexdigest()
        if decision_path.is_file()
        else str(config["selection_decision_path"])
    )
    relevant = {
        "model": config["model"],
        "output_dtype": config["inference"]["output_dtype"],
        "use_fp16_on_cuda": config["inference"].get("use_fp16_on_cuda", False),
        "selection_decision_sha256": decision_hash,
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def selected_metadata(
    root: Path,
    config: dict[str, Any],
    limit_per_split_family: int | None = None,
) -> pd.DataFrame:
    """Return the full official splits restricted only to selected families."""

    metadata = pd.read_parquet(root / config["metadata_path"])
    decision = json.loads(
        (root / config["selection_decision_path"]).read_text(encoding="utf-8")
    )
    families = decision["selection"]["families"]
    selected = metadata.loc[
        metadata["instrument_family_str"].isin(families)
    ].copy()
    if limit_per_split_family is not None:
        selected = (
            selected.sort_values(["split", "instrument_family_str", "item_id"])
            .groupby(
                ["split", "instrument_family_str"],
                observed=True,
                group_keys=False,
            )
            .head(limit_per_split_family)
        )
    selected = selected.sort_values(["split", "item_id"]).reset_index(drop=True)
    if selected.duplicated(["split", "item_id"]).any():
        raise RuntimeError("Selected metadata has duplicate split/item_id keys")
    if set(selected["split"].unique()) != set(SPLITS):
        raise RuntimeError("Selected metadata must contain all official splits")
    return selected


def read_nsynth_audio(
    path: Path,
    expected_sample_rate: int,
    expected_samples: int,
) -> np.ndarray:
    """Read and validate one fixed-length mono NSynth WAV."""

    sample_rate, audio = wavfile.read(path, mmap=True)
    if int(sample_rate) != expected_sample_rate:
        raise ValueError(
            f"Expected {expected_sample_rate} Hz, found {sample_rate} Hz: {path}"
        )
    values = np.asarray(audio)
    if values.ndim != 1:
        raise ValueError(f"Expected mono audio, found shape {values.shape}: {path}")
    if len(values) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, found {len(values)}: {path}"
        )
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        scale = float(max(abs(info.min), info.max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    return values


def resolve_device(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or set inference.device to cpu."
        )
    return requested


def load_frozen_mert(config: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    """Lazily import ML dependencies and return processor/model/device."""

    try:
        import torch
        from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "MERT dependencies are missing. Install CUDA PyTorch, then run "
            "pip install -r requirements-mert.txt"
        ) from exc

    model_config = config["model"]
    inference = config["inference"]
    device = resolve_device(str(inference["device"]), torch)
    common = {
        "revision": str(model_config["revision"]),
        "trust_remote_code": bool(model_config["trust_remote_code"]),
    }
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_config["model_id"], **common
    )
    loaded_config = AutoConfig.from_pretrained(model_config["model_id"], **common)
    for name, value in model_config.get("compatibility_overrides", {}).items():
        setattr(loaded_config, name, value)
    model = AutoModel.from_pretrained(
        model_config["model_id"],
        config=loaded_config,
        use_safetensors=True,
        **common,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if int(processor.sampling_rate) != int(model_config["sample_rate"]):
        raise RuntimeError(
            f"Processor sample rate {processor.sampling_rate} does not match "
            f"configured {model_config['sample_rate']}"
        )
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return torch, processor, model, device


def infer_batch(
    audio_batch: list[np.ndarray],
    torch_module: Any,
    processor: Any,
    model: Any,
    device: str,
    config: dict[str, Any],
) -> np.ndarray:
    """Run frozen inference and mean-pool the configured hidden layer."""

    model_config = config["model"]
    inference = config["inference"]
    inputs = processor(
        audio_batch,
        sampling_rate=int(model_config["sample_rate"]),
        return_tensors="pt",
        padding=False,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    hidden_layer = int(model_config["hidden_layer"])
    need_hidden_states = hidden_layer != -1
    use_amp = device == "cuda" and bool(inference["use_fp16_on_cuda"])
    amp_context = (
        torch_module.autocast(
            device_type="cuda",
            dtype=torch_module.float16,
        )
        if use_amp
        else nullcontext()
    )
    with torch_module.inference_mode():
        with amp_context:
            outputs = model(
                **inputs,
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )
            hidden = (
                outputs.last_hidden_state
                if hidden_layer == -1
                else outputs.hidden_states[hidden_layer]
            )
            pooled = hidden.mean(dim=1)
    embeddings = pooled.float().cpu().numpy()
    expected_dim = int(model_config["expected_embedding_dim"])
    if embeddings.shape != (len(audio_batch), expected_dim):
        raise RuntimeError(
            f"Expected embedding shape {(len(audio_batch), expected_dim)}, "
            f"found {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("MERT produced non-finite embeddings")
    return embeddings.astype(str(inference["output_dtype"]), copy=False)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_save_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def shard_paths(cache_dir: Path, split: str, shard_index: int) -> tuple[Path, Path]:
    stem = f"{split}_{shard_index:04d}"
    return (
        cache_dir / "shards" / f"{stem}.npy",
        cache_dir / "metadata" / f"{stem}.parquet",
    )


def validate_shard_pair(
    embedding_path: Path,
    metadata_path: Path,
    expected_rows: int,
    expected_dim: int,
    expected_signature: str,
    check_finite: bool = True,
) -> pd.DataFrame:
    """Validate one embedding/metadata pair and return its index rows."""

    if embedding_path.is_file() != metadata_path.is_file():
        raise RuntimeError(
            f"Incomplete shard pair; both files must exist: {embedding_path}, "
            f"{metadata_path}"
        )
    if not embedding_path.is_file():
        raise FileNotFoundError(embedding_path)
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    index = pd.read_parquet(metadata_path)
    if embeddings.shape != (expected_rows, expected_dim):
        raise RuntimeError(
            f"Invalid shape {embeddings.shape} for {embedding_path}; expected "
            f"{(expected_rows, expected_dim)}"
        )
    if len(index) != expected_rows:
        raise RuntimeError(f"Metadata row mismatch for {metadata_path}")
    if index["embedding_row"].tolist() != list(range(expected_rows)):
        raise RuntimeError(f"Non-contiguous embedding_row in {metadata_path}")
    if set(index["cache_signature"].unique()) != {expected_signature}:
        raise RuntimeError(f"Cache signature mismatch in {metadata_path}")
    if check_finite and not np.isfinite(embeddings).all():
        raise RuntimeError(f"Non-finite values in {embedding_path}")
    return index


def build_shard_index(
    rows: pd.DataFrame,
    embedding_path: Path,
    config: dict[str, Any],
    signature: str,
) -> pd.DataFrame:
    model_config = config["model"]
    index = rows[
        [
            "item_id",
            "split",
            "instrument_family_str",
            "instrument_str",
            "pitch",
            "velocity",
            "audio_relpath",
        ]
    ].copy()
    index["embedding_shard"] = embedding_path.relative_to(project_root()).as_posix()
    index["embedding_row"] = np.arange(len(index), dtype=np.int32)
    index["embedding_dim"] = int(model_config["expected_embedding_dim"])
    index["model_id"] = str(model_config["model_id"])
    index["model_revision"] = str(model_config["revision"])
    index["hidden_layer"] = int(model_config["hidden_layer"])
    index["pooling"] = str(model_config["pooling"])
    index["cache_signature"] = signature
    return index[INDEX_COLUMNS]


def validate_manifest_coverage(
    index: pd.DataFrame,
    manifest_paths: Iterable[Path],
) -> dict[str, Any]:
    """Require every manifest split/item key to match exactly one embedding."""

    if index.duplicated(["split", "item_id"]).any():
        raise RuntimeError("Embedding index has duplicate split/item_id keys")
    reports: dict[str, Any] = {}
    keys = index[["split", "item_id"]]
    for path in manifest_paths:
        manifest = pd.read_parquet(path, columns=["split", "item_id"])
        if manifest.duplicated(["split", "item_id"]).any():
            raise RuntimeError(f"Manifest has duplicate keys: {path}")
        joined = manifest.merge(
            keys,
            on=["split", "item_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int(joined["_merge"].ne("both").sum())
        if missing:
            raise RuntimeError(f"{path} has {missing} items without MERT embeddings")
        reports[path.name] = {
            "rows": int(len(manifest)),
            "matched_embeddings": int(len(joined)),
            "missing_embeddings": 0,
            "one_to_one": True,
        }
    return reports


def validate_complete_cache(
    target: pd.DataFrame,
    cache_dir: Path,
    config: dict[str, Any],
    validate_manifests: bool = True,
) -> dict[str, Any]:
    """Validate all shards, exact target coverage, and manifest coverage."""

    signature = cache_signature(config)
    expected_dim = int(config["model"]["expected_embedding_dim"])
    rows_per_shard = int(config["inference"]["rows_per_shard"])
    all_indices: list[pd.DataFrame] = []
    shard_count = 0
    for split in SPLITS:
        split_rows = target.loc[target["split"].eq(split)].reset_index(drop=True)
        for shard_index, start in enumerate(range(0, len(split_rows), rows_per_shard)):
            stop = min(start + rows_per_shard, len(split_rows))
            embedding_path, metadata_path = shard_paths(
                cache_dir, split, shard_index
            )
            all_indices.append(
                validate_shard_pair(
                    embedding_path,
                    metadata_path,
                    stop - start,
                    expected_dim,
                    signature,
                )
            )
            shard_count += 1
    index = pd.concat(all_indices, ignore_index=True)
    target_keys = target[["split", "item_id"]].sort_values(
        ["split", "item_id"]
    ).reset_index(drop=True)
    index_keys = index[["split", "item_id"]].sort_values(
        ["split", "item_id"]
    ).reset_index(drop=True)
    if not target_keys.equals(index_keys):
        raise RuntimeError("Embedding index does not exactly cover target metadata")
    if index.duplicated(["split", "item_id"]).any():
        raise RuntimeError("Embedding index contains duplicate target keys")

    index_path = cache_dir / "index.parquet"
    _atomic_save_parquet(index_path, index)
    manifest_report: dict[str, Any] = {}
    if validate_manifests:
        manifest_paths = [
            project_root() / value for value in config["manifests_to_validate"]
        ]
        manifest_report = validate_manifest_coverage(index, manifest_paths)
    return {
        "rows": int(len(index)),
        "embedding_dim": expected_dim,
        "shards": shard_count,
        "unique_split_item_keys": int(
            index[["split", "item_id"]].drop_duplicates().shape[0]
        ),
        "all_finite": True,
        "exact_target_coverage": True,
        "manifest_coverage": manifest_report,
        "index_path": index_path.relative_to(project_root()).as_posix(),
    }


def extract_embeddings(
    config_path: Path,
    cache_tag: str = "ordinary",
    limit_per_split_family: int | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Extract or resume the complete selected-family MERT cache."""

    root = project_root()
    config = load_config(config_path)
    target = selected_metadata(root, config, limit_per_split_family)
    base_cache_dir = root / config["embedding_dir"]
    cache_dir = base_cache_dir if cache_tag == "ordinary" else base_cache_dir / cache_tag
    (cache_dir / "shards").mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata").mkdir(parents=True, exist_ok=True)
    signature = cache_signature(config)
    expected_dim = int(config["model"]["expected_embedding_dim"])
    rows_per_shard = int(config["inference"]["rows_per_shard"])

    if validate_only:
        validation = validate_complete_cache(
            target,
            cache_dir,
            config,
            validate_manifests=limit_per_split_family is None,
        )
        return {"mode": "validate_only", **validation}

    torch_module, processor, model, device = load_frozen_mert(config)
    batch_size = int(config["inference"]["batch_size"])
    expected_sample_rate = int(config["model"]["sample_rate"])
    expected_samples = int(config["model"]["expected_audio_samples"])
    written_rows = 0
    resumed_rows = 0
    start_time = time.time()

    for split in SPLITS:
        split_rows = target.loc[target["split"].eq(split)].reset_index(drop=True)
        shard_total = (len(split_rows) + rows_per_shard - 1) // rows_per_shard
        for shard_index, start in enumerate(range(0, len(split_rows), rows_per_shard)):
            stop = min(start + rows_per_shard, len(split_rows))
            rows = split_rows.iloc[start:stop].copy()
            embedding_path, metadata_path = shard_paths(
                cache_dir, split, shard_index
            )
            if embedding_path.is_file() or metadata_path.is_file():
                validate_shard_pair(
                    embedding_path,
                    metadata_path,
                    len(rows),
                    expected_dim,
                    signature,
                    check_finite=True,
                )
                resumed_rows += len(rows)
                print(
                    f"{split}: shard {shard_index + 1}/{shard_total} resumed "
                    f"({resumed_rows:,} rows)",
                    flush=True,
                )
                continue

            batch_embeddings: list[np.ndarray] = []
            for batch_start in range(0, len(rows), batch_size):
                batch = rows.iloc[batch_start : batch_start + batch_size]
                audio_batch = [
                    read_nsynth_audio(
                        root / audio_path,
                        expected_sample_rate,
                        expected_samples,
                    )
                    for audio_path in batch["audio_relpath"]
                ]
                batch_embeddings.append(
                    infer_batch(
                        audio_batch,
                        torch_module,
                        processor,
                        model,
                        device,
                        config,
                    )
                )
            embeddings = np.concatenate(batch_embeddings, axis=0)
            index = build_shard_index(rows, embedding_path, config, signature)
            _atomic_save_npy(embedding_path, embeddings)
            _atomic_save_parquet(metadata_path, index)
            validate_shard_pair(
                embedding_path,
                metadata_path,
                len(rows),
                expected_dim,
                signature,
            )
            written_rows += len(rows)
            print(
                f"{split}: shard {shard_index + 1}/{shard_total} written; "
                f"new={written_rows:,}, resumed={resumed_rows:,}",
                flush=True,
            )

    validation = validate_complete_cache(
        target,
        cache_dir,
        config,
        validate_manifests=limit_per_split_family is None,
    )
    report = {
        "model_id": config["model"]["model_id"],
        "model_revision": config["model"]["revision"],
        "hidden_layer": int(config["model"]["hidden_layer"]),
        "pooling": config["model"]["pooling"],
        "device": device,
        "cache_signature": signature,
        "target_rows": int(len(target)),
        "written_this_run": written_rows,
        "resumed_rows": resumed_rows,
        "elapsed_seconds": time.time() - start_time,
        "validation": validation,
    }
    (cache_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
