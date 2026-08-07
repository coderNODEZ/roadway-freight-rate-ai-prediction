# """GPU CatBoost strategy for predicting a missing freight-data column."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
import importlib

import FeatureEngineering.feature_engr_constants as fec
fec = importlib.reload(fec)
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis
from pathlib import Path


INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)


def predict_market_index_column_catboost():
    """Train and evaluate a GPU CatBoost market-index regression model."""

    df = pd.read_csv(INPUT_PATH)

    target_column = fec.missing_columns_target
    if isinstance(target_column, list):
        if len(target_column) != 1:
            raise ValueError("Exactly one target column is required.")
        target_column = target_column[0]

    if target_column not in df.columns:
        raise ValueError(f"Target column is missing: {target_column}")

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    observed_target_df = df.loc[
        df[target_column].notna()
    ].reset_index(drop=True)

    if len(observed_target_df) <= 8_000:
        raise ValueError(
            "More than 8,000 rows with known market_index values "
            "are required."
        )

    testing_indices = observed_target_df.sample(
        n=8_000,
        random_state=42,
    ).index

    df_testing = observed_target_df.loc[
        testing_indices
    ].reset_index(drop=True)

    df_training = observed_target_df.drop(
        index=testing_indices,
    ).reset_index(drop=True)

    categorical_columns = list(
        fec.missing_columns_categorical_columns
    )
    continuous_columns = list(
        fec.missing_columns_continuous_columns
    )
    feature_columns = categorical_columns + continuous_columns

    duplicate_features = sorted(
        set(categorical_columns) & set(continuous_columns)
    )
    if duplicate_features:
        raise ValueError(
            "Features cannot be both categorical and continuous: "
            f"{duplicate_features}"
        )

    prohibited_features = {
        target_column,
        "quote_signal",
        "posted_rate",
    }
    leaked_features = sorted(
        prohibited_features & set(feature_columns)
    )
    if leaked_features:
        raise ValueError(
            "Target/leakage columns are configured as features: "
            f"{leaked_features}"
        )

    missing_features = sorted(
        set(feature_columns).difference(df.columns)
    )
    if missing_features:
        raise ValueError(
            f"Configured feature columns are missing: {missing_features}"
        )

    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    gpu_count = get_gpu_device_count()
    if gpu_count < 1:
        raise RuntimeError(
            "CatBoost cannot find an available CUDA GPU."
        )
    print(f"CatBoost CUDA GPU count: {gpu_count}")
    print("CatBoost CUDA device: 0")

    def prepare_features(partition: pd.DataFrame) -> pd.DataFrame:
        features = partition.loc[:, feature_columns].copy()

        for column in categorical_columns:
            features[column] = (
                features[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

        for column in continuous_columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

        return features

    training_partition, validation_partition = train_test_split(
        df_training,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    x_training = prepare_features(training_partition)
    y_training = training_partition[target_column].to_numpy(dtype=float)
    x_validation = prepare_features(validation_partition)
    y_validation = validation_partition[target_column].to_numpy(dtype=float)

    training_pool = Pool(
        data=x_training,
        label=y_training,
        cat_features=categorical_columns,
    )
    validation_pool = Pool(
        data=x_validation,
        label=y_validation,
        cat_features=categorical_columns,
    )

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=8_000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=42,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
        verbose=100,
    )

    model.fit(
        training_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=100,
    )

    def evaluate_partition(
        partition: pd.DataFrame,
    ) -> tuple[dict[str, float], np.ndarray]:
        y_actual = partition[target_column].to_numpy(dtype=float)
        features = prepare_features(partition)
        predictions = np.asarray(model.predict(features)).reshape(-1)

        metrics = {
            "mae": mean_absolute_error(y_actual, predictions),
            "rmse": mean_squared_error(y_actual, predictions) ** 0.5,
            "r2": r2_score(y_actual, predictions),
        }
        return metrics, predictions

    training_metrics, _ = evaluate_partition(training_partition)
    validation_metrics, _ = evaluate_partition(validation_partition)
    testing_metrics, testing_predictions = evaluate_partition(df_testing)

    df_testing["predicted_market_index"] = testing_predictions

    validation_mae_gap = (
        validation_metrics["mae"] - training_metrics["mae"]
    )
    test_mae_gap = testing_metrics["mae"] - training_metrics["mae"]
    validation_rmse_gap = (
        validation_metrics["rmse"] - training_metrics["rmse"]
    )
    test_rmse_gap = testing_metrics["rmse"] - training_metrics["rmse"]
    test_r2_gap = training_metrics["r2"] - testing_metrics["r2"]

    print("#############################################")
    print("# CatBoost regression performance")
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
        model_name="CatBoost",
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        testing_metrics=testing_metrics,
    )      

    mcph.record_prediction_history(
        model="CatBoost",
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

    # model_directory = Path(
    #     "FeatureEngineering/MissingColumnsPrediction/saved_models"
    # )
    # model_directory.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )

    # model_path = model_directory / "market_index_catboost.cbm"

    # model.save_model(
    #     str(model_path),
    #     format="cbm",
    # )

    # print(f"CatBoost model saved to: {model_path}")   
    
    # return df_testing, model    