from pathlib import Path

import numpy as np
import pandas as pd

from src.features.mert import (
    _atomic_save_npy,
    _atomic_save_parquet,
    cache_signature,
    validate_manifest_coverage,
    validate_shard_pair,
)


def _config() -> dict:
    return {
        "selection_decision_path": "data/manifests/decision.json",
        "model": {
            "model_id": "m-a-p/MERT-v0-public",
            "revision": "fixed-revision",
            "sample_rate": 16000,
            "expected_audio_samples": 64000,
            "hidden_layer": -1,
            "expected_embedding_dim": 3,
            "pooling": "mean",
            "trust_remote_code": True,
            "compatibility_overrides": {"conv_pos_batch_norm": False},
        },
        "inference": {"output_dtype": "float32"},
    }


def test_shard_pair_and_manifest_are_one_to_one(tmp_path: Path) -> None:
    config = _config()
    signature = cache_signature(config)
    embedding_path = tmp_path / "test_0000.npy"
    metadata_path = tmp_path / "test_0000.parquet"
    embeddings = np.arange(9, dtype=np.float32).reshape(3, 3)
    index = pd.DataFrame(
        {
            "split": ["test"] * 3,
            "item_id": ["a", "b", "c"],
            "embedding_row": [0, 1, 2],
            "cache_signature": [signature] * 3,
        }
    )
    _atomic_save_npy(embedding_path, embeddings)
    _atomic_save_parquet(metadata_path, index)
    validated = validate_shard_pair(
        embedding_path,
        metadata_path,
        expected_rows=3,
        expected_dim=3,
        expected_signature=signature,
    )
    assert len(validated) == 3

    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {"split": ["test", "test"], "item_id": ["c", "a"]}
    ).to_parquet(manifest_path, index=False)
    report = validate_manifest_coverage(index, [manifest_path])
    assert report["manifest.parquet"]["matched_embeddings"] == 2
    assert report["manifest.parquet"]["one_to_one"] is True


def test_cache_signature_changes_with_hidden_layer() -> None:
    first = _config()
    second = _config()
    second["model"]["hidden_layer"] = 6
    assert cache_signature(first) != cache_signature(second)
