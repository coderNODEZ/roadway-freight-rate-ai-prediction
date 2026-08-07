"""CUDA LightGBM strategy for predicting a missing freight-data column."""

from __future__ import annotations

import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import FeatureEngineering.feature_engr_constants as fec
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis


fec = importlib.reload(fec)

INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)


def predict_market_index_column_lightgbm():
    """Train and evaluate a CUDA LightGBM market-index regressor."""

    print("CUDA support min-test")

    x = np.random.rand(1_000, 10)
    y = np.random.rand(1_000)

    model = lgb.LGBMRegressor(
        n_estimators=10,
        device_type="cuda",
    )

    model.fit(x, y)

    print("LightGBM CUDA support min-test succeeded.")

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

    training_partition, validation_partition = train_test_split(
        df_training,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    category_levels = {
        column: pd.Index(
            training_partition[column]
            .astype("string")
            .fillna("__MISSING__")
            .unique()
        )
        for column in categorical_columns
    }

    def prepare_features(partition: pd.DataFrame) -> pd.DataFrame:
        features = partition.loc[:, feature_columns].copy()

        for column in categorical_columns:
            values = (
                features[column]
                .astype("string")
                .fillna("__MISSING__")
            )
            features[column] = pd.Categorical(
                values,
                categories=category_levels[column],
            )

        for column in continuous_columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

        return features

    x_training = prepare_features(training_partition)
    y_training = training_partition[target_column].to_numpy(dtype=float)
    x_validation = prepare_features(validation_partition)
    y_validation = validation_partition[target_column].to_numpy(dtype=float)

    print("LightGBM device type: CUDA")
    print("LightGBM CUDA device: 0")

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=8_000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_bin=255,
        random_state=42,
        device_type="cuda",
        gpu_device_id=0,
        verbosity=1,
    )

    try:
        model.fit(
            x_training,
            y_training,
            eval_set=[(x_validation, y_validation)],
            eval_names=["validation"],
            eval_metric="mae",
            categorical_feature=categorical_columns,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    first_metric_only=True,
                    verbose=True,
                ),
                lgb.log_evaluation(period=100),
            ],
        )
    except lgb.basic.LightGBMError as error:
        if "CUDA Tree Learner was not enabled" in str(error):
            raise RuntimeError(
                "The installed LightGBM package was not built with CUDA. "
                "Reinstall it from source with USE_CUDA=ON."
            ) from error
        raise

    def evaluate_partition(
        partition: pd.DataFrame,
    ) -> tuple[dict[str, float], np.ndarray]:
        y_actual = partition[target_column].to_numpy(dtype=float)
        features = prepare_features(partition)
        predictions = np.asarray(
            model.predict(
                features,
                num_iteration=model.best_iteration_,
            )
        ).reshape(-1)

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
    print("# LightGBM CUDA regression performance")
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
        model_name="TabICLv2",
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        testing_metrics=testing_metrics,
    )      

    mcph.record_prediction_history(
        model="LightGBM_CUDA",
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

    # return df_testing, model

