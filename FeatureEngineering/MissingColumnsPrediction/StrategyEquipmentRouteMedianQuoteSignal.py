"""Equipment-route median baseline for quote_signal prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis


TRAINING_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

TARGET_COLUMN = "quote_signal"
PREDICTION_COLUMN = "median_imputed_quote_signal"
ROUTE_COLUMNS = [
    "equipment",
    "pickup",
    "delivery",
]


def evaluate_quote_signal_equipment_route_median(
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Evaluate hierarchical median imputation for quote_signal."""

    data = pd.read_csv(TRAINING_INPUT_PATH)

    required_columns = ROUTE_COLUMNS + [TARGET_COLUMN]
    missing_columns = sorted(
        set(required_columns).difference(data.columns)
    )
    if missing_columns:
        raise ValueError(
            "Required quote_signal baseline columns are missing: "
            f"{missing_columns}"
        )

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    observed_target_data = data.loc[
        data[TARGET_COLUMN].notna()
    ].reset_index(drop=True)

    if len(observed_target_data) <= 8_000:
        raise ValueError(
            "More than 8,000 rows with known quote_signal values "
            "are required."
        )

    testing_indices = observed_target_data.sample(
        n=8_000,
        random_state=42,
    ).index

    testing_partition = observed_target_data.loc[
        testing_indices
    ].reset_index(drop=True)

    training_validation_data = observed_target_data.drop(
        index=testing_indices,
    ).reset_index(drop=True)

    training_partition, validation_partition = train_test_split(
        training_validation_data,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    training_partition = training_partition.reset_index(drop=True)
    validation_partition = validation_partition.reset_index(drop=True)

    # Calculate every median exclusively from the training partition.
    route_medians = (
        training_partition.groupby(
            ROUTE_COLUMNS,
            dropna=False,
        )[TARGET_COLUMN]
        .median()
        .rename("_route_median")
        .reset_index()
    )

    equipment_medians = training_partition.groupby(
        "equipment",
        dropna=False,
    )[TARGET_COLUMN].median()

    overall_median = training_partition[TARGET_COLUMN].median()
    if pd.isna(overall_median):
        raise ValueError(
            "Cannot calculate quote_signal medians because the training "
            "partition has no valid quote_signal values."
        )

    def evaluate_partition(
        partition: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        result = partition.copy()

        route_predictions = (
            result.loc[:, ROUTE_COLUMNS]
            .merge(
                route_medians,
                how="left",
                on=ROUTE_COLUMNS,
                sort=False,
                validate="many_to_one",
            )["_route_median"]
        )

        equipment_predictions = (
            result["equipment"]
            .map(equipment_medians)
            .reset_index(drop=True)
        )

        result[PREDICTION_COLUMN] = (
            route_predictions
            .fillna(equipment_predictions)
            .fillna(overall_median)
            .to_numpy(dtype=float)
        )

        actual_values = result[TARGET_COLUMN].to_numpy(dtype=float)
        predicted_values = result[PREDICTION_COLUMN].to_numpy(dtype=float)

        metrics = {
            "mae": mean_absolute_error(
                actual_values,
                predicted_values,
            ),
            "rmse": mean_squared_error(
                actual_values,
                predicted_values,
            ) ** 0.5,
            "r2": r2_score(
                actual_values,
                predicted_values,
            ),
        }

        return result, metrics

    _, training_metrics = evaluate_partition(
        training_partition
    )
    _, validation_metrics = evaluate_partition(
        validation_partition
    )
    testing_results, testing_metrics = evaluate_partition(
        testing_partition
    )

    validation_mae_gap = (
        validation_metrics["mae"] - training_metrics["mae"]
    )
    test_mae_gap = (
        testing_metrics["mae"] - training_metrics["mae"]
    )
    validation_rmse_gap = (
        validation_metrics["rmse"] - training_metrics["rmse"]
    )
    test_rmse_gap = (
        testing_metrics["rmse"] - training_metrics["rmse"]
    )
    test_r2_gap = training_metrics["r2"] - testing_metrics["r2"]

    print("#############################################")
    print("# Equipment-route median regression performance")
    print(f"Training for {TARGET_COLUMN} prediction")
    print(
        f"Training:   MAE={training_metrics['mae']:.6f}, "
        f"RMSE={training_metrics['rmse']:.6f}, "
        f"R²={training_metrics['r2']:.6f}"
    )
    print(
        f"Validation: MAE={validation_metrics['mae']:.6f}, "
        f"RMSE={validation_metrics['rmse']:.6f}, "
        f"R²={validation_metrics['r2']:.6f}"
    )
    print(
        f"Testing:    MAE={testing_metrics['mae']:.6f}, "
        f"RMSE={testing_metrics['rmse']:.6f}, "
        f"R²={testing_metrics['r2']:.6f}"
    )
    print("# Generalization gaps")
    print(f"Validation MAE gap:  {validation_mae_gap:.6f}")
    print(f"Test MAE gap:        {test_mae_gap:.6f}")
    print(f"Validation RMSE gap: {validation_rmse_gap:.6f}")
    print(f"Test RMSE gap:       {test_rmse_gap:.6f}")
    print(f"Test R² gap:         {test_r2_gap:.6f}")
    print("#############################################")

    vis.plot_regression_performance(
        model_name="Equipment-route median quote_signal baseline",
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        testing_metrics=testing_metrics,
    )

    mcph.record_prediction_history(
        model="EquipmentRouteMedian_quote_signal",
        training_mae=training_metrics["mae"],
        validation_mae=validation_metrics["mae"],
        testing_mae=testing_metrics["mae"],
        training_rmse=training_metrics["rmse"],
        validation_rmse=validation_metrics["rmse"],
        testing_rmse=testing_metrics["rmse"],
        training_r2=training_metrics["r2"],
        validation_r2=validation_metrics["r2"],
        testing_r2=testing_metrics["r2"],
    )

    # return (
    #     testing_results,
    #     training_metrics,
    #     validation_metrics,
    #     testing_metrics,
    # )

