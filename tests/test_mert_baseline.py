from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mert_baseline import load_embedding_matrix, train_mert_ordinary_baseline


def test_sharded_loader_and_train_only_protocol(tmp_path: Path) -> None:
    root = tmp_path
    shard_dir = root / "cache"
    shard_dir.mkdir()
    rng = np.random.default_rng(5305)
    rows = []
    shard_values = []
    for split, count in (("train", 24), ("valid", 12), ("test", 12)):
        values = []
        for index in range(count):
            family = "a" if index % 2 == 0 else "b"
            center = -1.0 if family == "a" else 1.0
            values.append(rng.normal(center, 0.2, size=4))
            rows.append(
                {
                    "item_id": f"{split}_{index}",
                    "split": split,
                    "instrument_family_str": family,
                    "instrument_str": f"{split}_{family}_{index // 4}",
                    "pitch": 60 + index % 2,
                    "velocity": 50,
                    "embedding_shard": f"cache/{split}.npy",
                    "embedding_row": index,
                    "embedding_dim": 4,
                    "model_id": "test-model",
                    "model_revision": "fixed",
                    "hidden_layer": -1,
                    "pooling": "mean",
                    "cache_signature": "abc",
                }
            )
        shard = np.asarray(values, dtype=np.float32)
        np.save(shard_dir / f"{split}.npy", shard)
        shard_values.append(shard)

    index_path = root / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    index, matrix = load_embedding_matrix(root, index_path)
    np.testing.assert_allclose(matrix, np.concatenate(shard_values))
    report = train_mert_ordinary_baseline(
        index,
        matrix,
        {
            "c_values": [0.1],
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_seed": 5305,
        },
        (60, 61),
        root / "models",
        root / "results",
        root / "figures",
    )
    assert report["scaler_fit_split"] == "train"
    assert report["classifier_fit_split"] == "train"
    assert report["test_evaluations"] == 1
    assert report["metrics"]["ordinary_test"]["rows"] == 12
