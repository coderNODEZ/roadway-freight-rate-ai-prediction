from pathlib import Path

import pandas as pd


def fill_validation_predicted_values() -> pd.DataFrame:
    """Fill predicted_rate using the saved posted-rate predictions."""

    prediction_path = Path(
        "FeatureEngineering/MissingColumnsPrediction/"
        "final_posted_rate_prediction.csv"
    )
    validation_path = Path(
        "Data-Testing/validation_predictions_template.csv"
    )

    prediction_df = pd.read_csv(prediction_path)
    validation_df = pd.read_csv(validation_path)

    if prediction_df.shape[1] != 1:
        raise ValueError(
            "The posted-rate prediction file must contain exactly one "
            f"column, but contains {prediction_df.shape[1]} columns."
        )

    if len(prediction_df) != len(validation_df):
        raise ValueError(
            "Row-count mismatch: "
            f"{len(prediction_df):,} predictions versus "
            f"{len(validation_df):,} validation rows."
        )

    prediction_values = prediction_df.iloc[:, 0]

    missing_prediction_count = int(prediction_values.isna().sum())
    if missing_prediction_count:
        raise ValueError(
            f"The prediction file contains {missing_prediction_count:,} "
            "missing values."
        )

    validation_df["predicted_rate"] = prediction_values.to_numpy()

    if validation_df["predicted_rate"].isna().any():
        raise ValueError(
            "The completed predicted_rate column contains missing values."
        )

    validation_df.to_csv(validation_path, index=False)

    print(
        f"Filled and saved {len(validation_df):,} predicted_rate values."
    )

    return validation_df

def fill_december_chart_predicted_values() -> pd.DataFrame:
    """Fill predicted_rate using the saved posted-rate predictions."""

    prediction_path = Path(
        "FeatureEngineering/MissingColumnsPrediction/"
        "final_posted_rate_prediction.csv"
    )
    validation_path = Path(
        "Data-Testing/december_chart_inputs.csv"
    )

    prediction_df = pd.read_csv(prediction_path)
    validation_df = pd.read_csv(validation_path)

    if prediction_df.shape[1] != 1:
        raise ValueError(
            "The posted-rate prediction file must contain exactly one "
            f"column, but contains {prediction_df.shape[1]} columns."
        )

    if len(prediction_df) != len(validation_df):
        raise ValueError(
            "Row-count mismatch: "
            f"{len(prediction_df):,} predictions versus "
            f"{len(validation_df):,} validation rows."
        )

    prediction_values = prediction_df.iloc[:, 0]

    missing_prediction_count = int(prediction_values.isna().sum())
    if missing_prediction_count:
        raise ValueError(
            f"The prediction file contains {missing_prediction_count:,} "
            "missing values."
        )

    validation_df["predicted_rate"] = prediction_values.to_numpy()

    if validation_df["predicted_rate"].isna().any():
        raise ValueError(
            "The completed predicted_rate column contains missing values."
        )

    validation_df.to_csv(validation_path, index=False)

    print(
        f"Filled and saved {len(validation_df):,} predicted_rate values."
    )

    return validation_df



from pathlib import Path

import pandas as pd


def write_predicted_market_index(
    predicted_market_index: pd.DataFrame,
) -> None:
    """Insert predicted market_index as the second-to-last CSV column."""

    feature_data_path = Path(
        "FeatureEngineering/MissingColumnsPrediction/"
        "final_missing_columns_feature_engr_output.csv"
    )

    feature_data = pd.read_csv(feature_data_path)

    if "market_index" not in predicted_market_index.columns:
        raise ValueError(
            "The prediction DataFrame does not contain market_index."
        )

    if len(feature_data) != len(predicted_market_index):
        raise ValueError(
            "Row-count mismatch: "
            f"{len(feature_data):,} feature rows versus "
            f"{len(predicted_market_index):,} predictions."
        )

    predicted_values = pd.to_numeric(
        predicted_market_index["market_index"],
        errors="coerce",
    )

    if predicted_values.isna().any():
        raise ValueError(
            "The predicted market_index column contains missing or "
            "nonnumeric values."
        )

    if "market_index" in feature_data.columns:
        feature_data = feature_data.drop(columns=["market_index"])

    second_to_last_position = max(len(feature_data.columns) - 1, 0)

    feature_data.insert(
        second_to_last_position,
        "market_index",
        predicted_values.to_numpy(),
    )

    feature_data.to_csv(feature_data_path, index=False)

    print(
        f"Inserted {len(predicted_values):,} market_index predictions "
        f"and rewrote: {feature_data_path}"
    )

    # return feature_data        


def write_predicted_quote_signal(
    predicted_quote_signal: pd.DataFrame,
) -> None:
    """Insert predicted quote_signal as the final CSV column."""

    feature_data_path = Path(
        "FeatureEngineering/MissingColumnsPrediction/"
        "final_missing_columns_feature_engr_output.csv"
    )

    feature_data = pd.read_csv(feature_data_path)

    if "quote_signal" not in predicted_quote_signal.columns:
        raise ValueError(
            "The prediction DataFrame does not contain quote_signal."
        )

    if len(feature_data) != len(predicted_quote_signal):
        raise ValueError(
            "Row-count mismatch: "
            f"{len(feature_data):,} feature rows versus "
            f"{len(predicted_quote_signal):,} predictions."
        )

    predicted_values = pd.to_numeric(
        predicted_quote_signal["quote_signal"],
        errors="coerce",
    )

    if predicted_values.isna().any():
        raise ValueError(
            "The predicted quote_signal column contains missing or "
            "nonnumeric values."
        )

    # Remove an existing version so the predicted column can be
    # reinserted at the intended final position.
    if "quote_signal" in feature_data.columns:
        feature_data = feature_data.drop(
            columns=["quote_signal"]
        )

    feature_data.insert(
        len(feature_data.columns),
        "quote_signal",
        predicted_values.to_numpy(),
    )

    feature_data.to_csv(
        feature_data_path,
        index=False,
    )

    print(
        f"Inserted {len(predicted_values):,} quote_signal predictions "
        f"and rewrote: {feature_data_path}"
    )

    # return feature_data    