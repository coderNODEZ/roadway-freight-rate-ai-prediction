"""Corrected PySR strategy with CPU search and CUDA equation evaluation."""

from __future__ import annotations

import importlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pysr import PySRRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import FeatureEngineering.feature_engr_constants as fec
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis


fec = importlib.reload(fec)

INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)
OUTPUT_DIRECTORY = Path(
    "FeatureEngineering/MissingColumnsPrediction/pysr_runs"
)


def predict_market_index_column_pysr():
    """Fit PySR and evaluate its selected equation with CUDA."""

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
    configured_feature_columns = (
        categorical_columns + continuous_columns
    )

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
        prohibited_features & set(configured_feature_columns)
    )
    if leaked_features:
        raise ValueError(
            "Target/leakage columns are configured as features: "
            f"{leaked_features}"
        )

    missing_features = sorted(
        set(configured_feature_columns).difference(df.columns)
    )
    if missing_features:
        raise ValueError(
            f"Configured feature columns are missing: {missing_features}"
        )

    if not configured_feature_columns:
        raise ValueError("At least one feature column is required.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to evaluate the selected PySR equation."
        )

    print(
        "PySR search backend: Julia CPU multithreading "
        "(PySR does not provide CUDA symbolic search)."
    )
    print(f"PySR equation-evaluation device: {torch.cuda.get_device_name(0)}")

    training_partition, validation_partition = train_test_split(
        df_training,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    if categorical_columns:
        categorical_encoder.fit(
            training_partition.loc[:, categorical_columns]
            .astype("string")
            .fillna("__MISSING__")
        )

    continuous_medians = (
        training_partition.loc[:, continuous_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0.0)
    )

    raw_feature_names = list(continuous_columns)
    if categorical_columns:
        raw_feature_names.extend(
            categorical_encoder.get_feature_names_out(
                categorical_columns
            ).tolist()
        )

    def valid_variable_names(names: list[str]) -> list[str]:
        cleaned_names = []
        used_names = set()

        for position, name in enumerate(names):
            cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if not cleaned or cleaned[0].isdigit():
                cleaned = f"feature_{position}_{cleaned}"

            candidate = cleaned
            suffix = 2
            while candidate in used_names:
                candidate = f"{cleaned}_{suffix}"
                suffix += 1

            used_names.add(candidate)
            cleaned_names.append(candidate)

        return cleaned_names

    transformed_feature_names = valid_variable_names(raw_feature_names)

    def prepare_features(partition: pd.DataFrame) -> pd.DataFrame:
        arrays = []

        continuous_values = (
            partition.loc[:, continuous_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(continuous_medians)
            .to_numpy(dtype=np.float32, copy=True)
        )
        arrays.append(continuous_values)

        if categorical_columns:
            categorical_values = (
                partition.loc[:, categorical_columns]
                .astype("string")
                .fillna("__MISSING__")
            )
            arrays.append(
                categorical_encoder.transform(categorical_values)
            )

        feature_array = np.concatenate(arrays, axis=1)
        return pd.DataFrame(
            feature_array,
            columns=transformed_feature_names,
        )

    x_training = prepare_features(training_partition)
    y_training = training_partition[target_column].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime(
        "market_index_%Y%m%dT%H%M%SZ"
    )

    model = PySRRegressor(
        niterations=100,
        populations=20,
        population_size=50,
        ncycles_per_iteration=500,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        constraints={
            "/": (-1, 8),
            "square": 8,
        },
        nested_constraints={
            "square": {"square": 1},
        },
        maxsize=25,
        maxdepth=12,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        select_k_features=min(24, x_training.shape[1]),
        batching=True,
        batch_size=2_048,
        parallelism="multithreading",
        precision=32,
        turbo=True,
        warm_start=False,
        random_state=42,
        progress=True,
        verbosity=1,
        input_stream="devnull",
        output_directory=str(OUTPUT_DIRECTORY),
        run_id=run_id,
    )

    print("PySR is searching for symbolic equations.")
    model.fit(x_training, y_training)
    print("PySR symbolic search is complete.")
    print("Selected equation:", model.sympy())

    cuda_device = torch.device("cuda:0")
    torch_equation = model.pytorch().to(cuda_device)
    torch_equation.eval()

    def predict_with_cuda(
        features: pd.DataFrame,
        partition_name: str,
        batch_size: int = 8_000,
    ) -> np.ndarray:
        prediction_batches = []
        feature_array = features.to_numpy(
            dtype=np.float32,
            copy=True,
        )
        total_batches = (
            len(feature_array) + batch_size - 1
        ) // batch_size

        with torch.inference_mode():
            for batch_number, start in enumerate(
                range(0, len(feature_array), batch_size),
                start=1,
            ):
                print(
                    f"PySR CUDA {partition_name} evaluation batch "
                    f"{batch_number}/{total_batches}"
                )
                batch_tensor = torch.from_numpy(
                    feature_array[start:start + batch_size]
                ).to(cuda_device)
                batch_predictions = torch_equation(batch_tensor)
                prediction_batches.append(
                    batch_predictions.reshape(-1).cpu().numpy()
                )

        predictions = np.concatenate(prediction_batches)
        if not np.isfinite(predictions).all():
            raise ValueError(
                f"The selected PySR equation produced non-finite "
                f"{partition_name} predictions."
            )
        return predictions

    def evaluate_partition(
        partition: pd.DataFrame,
        partition_name: str,
    ) -> tuple[dict[str, float], np.ndarray]:
        y_actual = partition[target_column].to_numpy(dtype=float)
        features = prepare_features(partition)
        predictions = predict_with_cuda(features, partition_name)

        metrics = {
            "mae": mean_absolute_error(y_actual, predictions),
            "rmse": mean_squared_error(y_actual, predictions) ** 0.5,
            "r2": r2_score(y_actual, predictions),
        }
        return metrics, predictions

    training_metrics, _ = evaluate_partition(
        training_partition,
        "training",
    )
    validation_metrics, _ = evaluate_partition(
        validation_partition,
        "validation",
    )
    testing_metrics, testing_predictions = evaluate_partition(
        df_testing,
        "testing",
    )

    df_testing["predicted_market_index"] = testing_predictions

    validation_mae_gap = (
        validation_metrics["mae"] - training_metrics["mae"]
    )
    test_mae_gap = testing_metrics["mae"] - training_metrics["mae"]
    validation_rmse_gap = (
        validation_metrics["rmse"] - training_metrics["rmse"]
    )
    test_rmse_gap = (
        testing_metrics["rmse"] - training_metrics["rmse"]
    )
    test_r2_gap = training_metrics["r2"] - testing_metrics["r2"]

    print("#############################################")
    print("# PySR regression performance")
    print(f"Training for {target_column} prediction")    
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
        model="PySR",
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

