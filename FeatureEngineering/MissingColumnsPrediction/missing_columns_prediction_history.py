"""Append comparable model results to the missing-column history CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HISTORY_PATH = Path(__file__).with_name(
    "missing_columns_prediction_history.csv"
)

HISTORY_COLUMNS = [
    "model",
    "training_mae",
    "validation_mae",
    "testing_mae",
    "training_rmse",
    "validation_rmse",
    "testing_rmse",
    "training_r2",
    "validation_r2",
    "testing_r2",
]


def record_prediction_history(**metrics) -> Path:
    """Append one model run to the shared prediction-history CSV."""

    unknown_columns = sorted(set(metrics).difference(HISTORY_COLUMNS))
    if unknown_columns:
        raise ValueError(
            f"Unknown prediction-history columns: {unknown_columns}"
        )

    row = pd.DataFrame(
        [{column: metrics.get(column, pd.NA) for column in HISTORY_COLUMNS}]
    )

    file_exists = HISTORY_PATH.is_file()
    row.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )

    print(f"Prediction history recorded in: {HISTORY_PATH}")
    return HISTORY_PATH

