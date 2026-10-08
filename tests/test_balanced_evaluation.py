import numpy as np
import pandas as pd

from src.models.balanced_evaluation import evaluate_predictions


def test_balanced_metrics_include_every_register_and_family() -> None:
    rows = []
    predictions = []
    for register in ("lower", "middle", "upper"):
        for family in ("a", "b"):
            for index in range(2):
                rows.append(
                    {
                        "item_id": f"{register}_{family}_{index}",
                        "instrument_family_str": family,
                        "pitch_bin": register,
                    }
                )
                predictions.append(family if index == 0 else ("b" if family == "a" else "a"))
    overall, registers, recalls = evaluate_predictions(
        pd.DataFrame(rows), np.asarray(predictions), ["a", "b"]
    )
    assert overall["rows"] == 12
    assert set(registers["pitch_bin"]) == {"lower", "middle", "upper"}
    assert len(recalls) == 6
    assert registers["macro_f1"].between(0, 1).all()
