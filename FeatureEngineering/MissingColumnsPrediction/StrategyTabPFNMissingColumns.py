"""GPU TabPFN 3 strategy for predicting a missing freight-data column."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import FeatureEngineering.feature_engr_constants as fec
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis
from pathlib import Path
import joblib

from tabpfn.model_loading import save_fitted_tabpfn_model

fec = importlib.reload(fec)

TRAINING_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

PREDICTION_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)


def predict_market_index_column_tabpfn(predict=False, target="market_index"):
    """Fit and evaluate a CUDA TabPFN 3 market-index regressor."""

    if not predict:
        df = pd.read_csv(TRAINING_INPUT_PATH)

        target_column = target
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

        if target_column == "market_index":
            prohibited_features = {
                target_column,
                "quote_signal",
                "posted_rate",
            }
        elif target_column == "quote_signal":
            prohibited_features = {
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

        if not torch.cuda.is_available():
            raise RuntimeError("TabPFN cannot find an available CUDA GPU.")

        print(f"TabPFN CUDA device: {torch.cuda.get_device_name(0)}")

        training_partition, validation_partition = train_test_split(
            df_training,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        category_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )

        category_encoder.fit(
            training_partition.loc[:, categorical_columns]
            .astype("string")
            .fillna("__MISSING__")
        )

        def prepare_features(partition: pd.DataFrame) -> pd.DataFrame:
            features = pd.DataFrame(index=partition.index)

            if categorical_columns:
                categorical_values = (
                    partition.loc[:, categorical_columns]
                    .astype("string")
                    .fillna("__MISSING__")
                )
                encoded_categories = category_encoder.transform(
                    categorical_values
                )
                features[categorical_columns] = encoded_categories

            for column in continuous_columns:
                features[column] = pd.to_numeric(
                    partition[column],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan)

            return features.loc[:, feature_columns].reset_index(drop=True)

        x_training = prepare_features(training_partition)
        y_training = training_partition[target_column].to_numpy(dtype=float)

        categorical_feature_indices = list(
            range(len(categorical_columns))
        )

        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            device="cuda",
            categorical_features_indices=categorical_feature_indices,
            n_estimators=16,
            random_state=42,
            memory_saving_mode="auto",
        )

        print("TabPFN configured device:", model.device)
        print("TabPFN is fitting the training partition.")
        model.fit(x_training, y_training)
        print("TabPFN fit is complete.")

        def predict_in_batches(
            features: pd.DataFrame,
            partition_name: str,
            batch_size: int = 8_000,
        ) -> np.ndarray:
            prediction_batches = []
            total_batches = (
                len(features) + batch_size - 1
            ) // batch_size

            for batch_number, start in enumerate(
                range(0, len(features), batch_size),
                start=1,
            ):
                print(
                    f"TabPFN {partition_name} prediction batch "
                    f"{batch_number}/{total_batches}"
                )
                batch = features.iloc[start:start + batch_size]
                prediction_batches.append(
                    np.asarray(model.predict(batch)).reshape(-1)
                )

            return np.concatenate(prediction_batches)

        def evaluate_partition(
            partition: pd.DataFrame,
            partition_name: str,
        ) -> tuple[dict[str, float], np.ndarray]:
            y_actual = partition[target_column].to_numpy(dtype=float)
            features = prepare_features(partition)
            predictions = predict_in_batches(
                features,
                partition_name,
            )

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
        test_rmse_gap = testing_metrics["rmse"] - training_metrics["rmse"]
        test_r2_gap = training_metrics["r2"] - testing_metrics["r2"]

        print("#############################################")
        print("# TabPFN 3 regression performance")
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
            model="TabPFN_3",
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

        # model_path = (
        #     model_directory / f"TabPFN3_{target_column}.tabpfn_fit"
        # )
        # preprocessing_path = (
        #     model_directory / f"TabPFN3_preprocessing_{target_column}.joblib"
        # )

        # save_fitted_tabpfn_model(
        #     model,
        #     model_path,
        # )

        # joblib.dump(
        #     {
        #         "category_encoder": category_encoder,
        #         "categorical_columns": categorical_columns,
        #         "continuous_columns": continuous_columns,
        #         "feature_columns": feature_columns,
        #         "target_column": target_column,
        #     },
        #     preprocessing_path,
        # )

        # print(f"TabPFN fitted model saved to: {model_path}")
        # print(f"TabPFN preprocessing saved to: {preprocessing_path}")

    elif predict:
        df = pd.read_csv(PREDICTION_INPUT_PATH)       

        # TODO: correct the following so as to load the model for market_index, 
        # as in make use of 
        # TabPFN3_market_index.tabpfn_fit and TabPFN3_preprocessing_market_index.joblib to
        # predict the market_index column
        # and add this new predicted column to the dataframe 
        # and write the new dataframe to 
        # FeatureEngineering/MissingColumnsPrediction/missing_columns_feature_engr_output.csv



