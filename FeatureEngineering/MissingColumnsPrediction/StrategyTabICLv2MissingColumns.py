"""CUDA TabICLv2 strategy for predicting a missing freight-data column."""

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
from tabicl import TabICLRegressor

import FeatureEngineering.feature_engr_constants as fec
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis
from pathlib import Path

fec = importlib.reload(fec)

TRAINING_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

PREDICTION_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

FINAL_PREDICTION_INPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "final_missing_columns_feature_engr_output.csv"
)

MARKET_INDEX_PREDICTION_OUTPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

QUOTE_SIGNAL_PREDICTION_OUTPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

POSTED_RATE_PREDICTION_OUTPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "missing_columns_feature_engr_output.csv"
)

FINAL_POSTED_RATE_PREDICTION_OUTPUT_PATH = (
    "FeatureEngineering/MissingColumnsPrediction/"
    "final_missing_columns_feature_engr_output.csv"
)

POSTED_RATE_PATH = (
        "FeatureEngineering/MissingColumnsPrediction/"
        "feature_engr_posted_rate_data.csv"
)



def predict_market_index_column_TabICLv2(predict_missing_columns=False, target_column="market_index", final_prediction=False) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fit and evaluate a CUDA TabICLv2 market-index regressor."""

    train_model = False

    if not predict_missing_columns:
        train_model = True

    if train_model:
        df = pd.read_csv(TRAINING_INPUT_PATH)

        #target_column = fec.missing_columns_target

        if isinstance(target_column, list):
            if len(target_column) != 1:
                raise ValueError("Exactly one target column is required.")
            target_column = target_column[0]

        if target_column == "posted_rate" and target_column not in df.columns:
            posted_rate_data = pd.read_csv(
                POSTED_RATE_PATH
            )

            if target_column not in posted_rate_data.columns:
                raise ValueError(
                    f"{POSTED_RATE_PATH} does not contain "
                    f"the required {target_column} column."
                )

            if len(posted_rate_data) != len(df):
                raise ValueError(
                    "The feature data and posted-rate target data have "
                    "different row counts: "
                    f"{len(df):,} feature rows versus "
                    f"{len(posted_rate_data):,} target rows."
                )

            df[target_column] = posted_rate_data[
                target_column
            ].to_numpy()

        elif target_column not in df.columns:
            raise ValueError(
                f"Target column is missing: {target_column}"
            )            

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

      

        if target_column == "market_index":
            categorical_columns = list(
                fec.missing_columns_categorical_columns
            )                 
            continuous_columns = list(
                fec.missing_columns_continuous_columns
            )
            prohibited_features = {
                "market_index",
                "quote_signal",
                "posted_rate",
            }            
        elif target_column == "quote_signal":
            categorical_columns = list(
                fec.missing_quote_signal_categorical_columns
            )             
            continuous_columns = list(
                fec.missing_quote_signal_continuous_columns
            )  
            prohibited_features = {
                "quote_signal",
                "posted_rate",
            }            
        elif target_column == "posted_rate":
            categorical_columns = list(
                fec.categorical_columns
            )               
            continuous_columns = list(
                fec.continuous_columns
            )  
            prohibited_features = {
                "posted_rate",
            }                         


        feature_columns = categorical_columns + continuous_columns

        duplicate_features = sorted(
            set(categorical_columns) & set(continuous_columns)
        )
        if duplicate_features:
            raise ValueError(
                "Features cannot be both categorical and continuous: "
                f"{duplicate_features}"
            )


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
            raise RuntimeError("TabICLv2 cannot find an available CUDA GPU.")

        print(f"TabICLv2 CUDA device: {torch.cuda.get_device_name(0)}")

        training_partition, validation_partition = train_test_split(
            df_training,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        def prepare_features(partition: pd.DataFrame) -> pd.DataFrame:
            features = partition.loc[:, feature_columns].copy()

            # TabICLv2 detects and ordinal-encodes string/category/Boolean
            # columns internally, so categorical values stay categorical here.
            for column in categorical_columns:
                features[column] = (
                    features[column]
                    .astype("string")
                    .fillna("__MISSING__")
                )

            for column in continuous_columns:
                features[column] = pd.to_numeric(
                    features[column],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan)

            return features.reset_index(drop=True)

        x_training = prepare_features(training_partition)
        y_training = training_partition[target_column].to_numpy(dtype=float)

        model = TabICLRegressor(
            n_estimators=16,
            batch_size=1,
            kv_cache=False,
            checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
            device="cuda",
            use_amp="auto",
            use_fa3="auto",
            offload_mode="auto",
            random_state=42,
            verbose=True,
        )

        print("TabICLv2 is fitting the training partition.")
        model.fit(x_training, y_training)
        print("TabICLv2 fit is complete.")

        def predict_in_batches(
            features: pd.DataFrame,
            partition_name: str,
            batch_size: int = 8_000,
        ) -> np.ndarray:
            prediction_batches = []
            total_batches = (len(features) + batch_size - 1) // batch_size

            for batch_number, start in enumerate(
                range(0, len(features), batch_size),
                start=1,
            ):
                print(
                    f"TabICLv2 {partition_name} prediction batch "
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
            predictions = predict_in_batches(features, partition_name)

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

        df_testing[f"predicted_{target_column}"] = testing_predictions

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
        print("# TabICLv2 regression performance")
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
            model="TabICLv2",
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

        # if target_column == "posted_rate":
        model_directory = Path(
            "FeatureEngineering/MissingColumnsPrediction/saved_models"
        )
        model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            model_directory / f"TabICLv2_{target_column}.pkl"
        )

        model.save(
            model_path,
            save_model_weights=False,
            save_training_data=True,
            save_kv_cache=False,
        )

        print(f"TabICLv2 model saved to: {model_path}")

    elif predict_missing_columns:

        if not final_prediction:
            df = pd.read_csv(PREDICTION_INPUT_PATH)
        else:
            df = pd.read_csv(FINAL_PREDICTION_INPUT_PATH)

        print(f"##########################{df.shape}")

        # target_column = "market_index"
        if final_prediction and target_column == "predicted_rate":
            model_path = Path(
                "FeatureEngineering/MissingColumnsPrediction/saved_models"
            ) / f"TabICLv2_posted_rate.pkl"
        else:            
            model_path = Path(
                "FeatureEngineering/MissingColumnsPrediction/saved_models"
            ) / f"TabICLv2_{target_column}.pkl"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved TabICLv2 model was not found: {model_path}"
            )

        categorical_columns = list(
            fec.missing_columns_categorical_columns
        )

        if target_column == "market_index":
            continuous_columns = list(
                fec.missing_columns_continuous_columns
            )

            prohibited_features = {
                "market_index",
                "quote_signal",
                "posted_rate",
            }    

            output_path = Path(
                MARKET_INDEX_PREDICTION_OUTPUT_PATH
            )                          

        elif target_column == "quote_signal":
            continuous_columns = list(
                fec.missing_quote_signal_continuous_columns
            )

            prohibited_features = {
                "quote_signal",
                "posted_rate",
            }   

            output_path = Path(
                QUOTE_SIGNAL_PREDICTION_OUTPUT_PATH
            )                           

        elif target_column == "posted_rate" or target_column == "predicted_rate":
            continuous_columns = list(
                fec.continuous_columns
            )  

            prohibited_features = {
                "posted_rate",
                "predicted_rate"
            }            

            if not final_prediction:
                output_path = Path(
                    POSTED_RATE_PREDICTION_OUTPUT_PATH
                )     
            else:
                output_path = Path(
                    FINAL_POSTED_RATE_PREDICTION_OUTPUT_PATH
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
            raise RuntimeError("TabICLv2 cannot find an available CUDA GPU.")

        if target_column not in df.columns:
            df[target_column] = np.nan
        else:
            df[target_column] = pd.to_numeric(
                df[target_column],
                errors="coerce",
            )

        missing_target_mask = df[target_column].isna()
        rows_to_predict = int(missing_target_mask.sum())

        if rows_to_predict:
            features = df.loc[
                missing_target_mask,
                feature_columns,
            ].copy()

            for column in categorical_columns:
                features[column] = (
                    features[column]
                    .astype("string")
                    .fillna("__MISSING__")
                )

            for column in continuous_columns:
                features[column] = pd.to_numeric(
                    features[column],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan)

            features = features.reset_index(drop=True)

            print(f"Loading saved TabICLv2 model from: {model_path}")
            model = TabICLRegressor.load(model_path)

            prediction_batches = []
            batch_size = 8_000
            total_batches = (
                len(features) + batch_size - 1
            ) // batch_size

            for batch_number, start in enumerate(
                range(0, len(features), batch_size),
                start=1,
            ):
                print(
                    f"TabICLv2 {target_column} prediction batch "
                    f"{batch_number}/{total_batches}"
                )
                batch = features.iloc[start:start + batch_size]
                prediction_batches.append(
                    np.asarray(model.predict(batch)).reshape(-1)
                )

            predictions = np.concatenate(prediction_batches)
            df.loc[missing_target_mask, target_column] = predictions
            print(
                f"Predicted {rows_to_predict:,} missing "
                f"{target_column} values."
            )
        else:
            print(f"No missing {target_column} values require prediction.")


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        df.to_csv(output_path, index=False)
        print(
            f"{target_column} prediction output written to: "
            f"{output_path}"
        )

        predicted_column_df = df.loc[:, [target_column]].copy()

        return df, predicted_column_df


