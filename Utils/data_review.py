import pandas as pd
from pathlib import Path
import numpy as np

def basic_head_tail(df: pd.DataFrame) -> None:
    # csv_path = Path("Data/train-test.csv")

    # df = pd.read_csv(csv_path)

    print("Dataframe head")
    display(df.head())


    print("\nDataframe tail")
    display(df.tail())

    # return df


# def basic_columns_data_types(df: pd.DataFrame) -> None: 

#     print("DataFrame Shape:", df.shape)
#     print("\n DataFrame Columns:")
#     print(df.columns.tolist())

#     print("\nData types:")
#     print(df.dtypes)

def check_missing_data(df: pd.DataFrame) -> None:
    """Check every DataFrame column for missing values."""

    missing_report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean().mul(100),
    })

    missing_report["missing_percentage"] = (
        missing_report["missing_percentage"].round(2)
    )

    display(missing_report)

    missing_column_names = missing_report.index[
        missing_report["missing_count"].gt(0)
    ].tolist()

    print(
        f"\nColumns with missing data: "
        f"{len(missing_column_names)} of {df.shape[1]}"
    )

    if missing_column_names:
        print("\nColumn names with missing data:")

        for column in missing_column_names:
            print(f"  - {column}")
    else:
        print("Confirmed: no columns contain missing data.")



def missing_column_data_review(
    df: pd.DataFrame,
    column_name: str,
) -> None:
    """Review missing, zero, negative, and positive values for a column."""

    if column_name not in df.columns:
        print(f"Column is not present: {column_name}")
        return

    numeric_values = pd.to_numeric(
        df[column_name],
        errors="coerce",
    )

    column_audit = pd.Series({
        "missing": df[column_name].isna().sum(),
        "non_numeric": (
            numeric_values.isna() & df[column_name].notna()
        ).sum(),
        "zero": numeric_values.eq(0).sum(),
        "negative": numeric_values.lt(0).sum(),
        "positive": numeric_values.gt(0).sum(),
    })

    display(
        column_audit.to_frame(
            name=f"{column_name} count"
        )
    )

    
def confirm_no_leakage_columns(df: pd.DataFrame) -> None:
    """Report whether potential target-leakage columns are present."""

    potential_leakage_columns = [
        "quote_signal_difference",
        "posted_rate",
        "posted_rate_per_mile",
    ]

    present_leakage_columns = [
        column
        for column in potential_leakage_columns
        if column in df.columns
    ]

    print("Checked potential leakage columns:")
    for column in potential_leakage_columns:
        print(f"  - {column}")

    if present_leakage_columns:
        print(
            f"\nPotential leakage columns present: "
            f"{len(present_leakage_columns)}"
        )

        for column in present_leakage_columns:
            print(f"  - {column}")
    else:
        print(
            "\nConfirmed: no potential leakage columns are present."
        )    


def basic_columns_data_types(df: pd.DataFrame) -> None:
    """Display the DataFrame shape, columns in three columns, and data types."""

    print("DataFrame Shape:", df.shape)

    print("\nDataFrame Columns:")
    column_names = df.columns.tolist()

    if column_names:
        display_width = max(len(column) for column in column_names) + 4

        for start in range(0, len(column_names), 3):
            row = column_names[start:start + 3]
            print(
                "".join(
                    f"{column:<{display_width}}"
                    for column in row
                ).rstrip()
            )

    print("\nData types:")
    print(df.dtypes)

def white_spaces_check(df: pd.DataFrame) -> None: 

    text_columns = ["load_id", "pickup", "delivery", "equipment"]

    print("\nLeading/trailing white space check:\n")
    for column in text_columns:
        original = df[column].astype("string")
        stripped = original.str.strip()

        changed_mask = original.ne(stripped).fillna(False)

        print(f"Column: {column}")
        print(f"Values with leading/trailing whitespace: {changed_mask.sum()}")

        if changed_mask.any():
            display(
                pd.DataFrame({
                    "original_value": original[changed_mask],
                    "stripped_value": stripped[changed_mask],
                })
            )    

    print("\nInternal white space check:\n")
    for column in text_columns:
        values = df[column].astype("string")

        extra_internal_spaces = values.str.contains(r"\s{2,}", regex=True, na=False)

        print(f"Column: {column}")
        print(f"Values with repeated internal whitespace: {extra_internal_spaces.sum()}")        

        # print(
        #     f"{column}: "
        #     f"{extra_internal_spaces.sum()} values with repeated internal whitespace"
        # )

        if extra_internal_spaces.any():
            display(df.loc[extra_internal_spaces, [column]])   




def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
        "unknown",
        "?",
        "-",
        "--",
        ".",
        "..",
    }

    def extended_missing_mask(series: pd.Series) -> pd.Series:
        # Actual pandas missing values
        mask = series.isna()

        # Additional text-based missing representations
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            normalized = (
                series.astype("string")
                .str.strip()
                .str.lower()
            )

            mask = mask | normalized.isin(missing_tokens)

        return mask

    missing_value_df = pd.DataFrame(index=df.columns)

    missing_value_df["standard_missing_count"] = [
        df[column].isna().sum()
        for column in df.columns
    ]

    missing_value_df["extended_missing_count"] = [
        extended_missing_mask(df[column]).sum()
        for column in df.columns
    ]

    missing_value_df["additional_text_missing"] = (
        missing_value_df["extended_missing_count"]
        - missing_value_df["standard_missing_count"]
    )

    missing_value_df["missing_percent"] = (
        missing_value_df["extended_missing_count"]
        / len(df)
        * 100
    ).round(2)

    display(missing_value_df)

    # return missing_value_df


def check_duplicate_load_id(df: pd.DataFrame) -> None:

    duplicate_load_ids = df["load_id"].duplicated(keep=False)
    print("Duplicate load IDs:", duplicate_load_ids.sum())
    if duplicate_load_ids.sum() > 0:
        display(df.loc[duplicate_load_ids].sort_values("load_id"))


def check_duplicate_rows(df: pd.DataFrame) -> None:

    duplicate_rows = df.duplicated(keep=False)
    print("Duplicate complete rows:", duplicate_rows.sum())
    if duplicate_rows.sum() > 0:
        display(df.loc[duplicate_rows])


def check_load_id_format(df: pdDataFrame) -> None:

    valid_load_id = df["load_id"].str.fullmatch(r"TR-\d{6}", na=False)

    invalid_load_ids = df.loc[
        ~valid_load_id,
        ["load_id"]
    ]

    print("Invalid load IDs:", len(invalid_load_ids))
    if len(invalid_load_ids) > 0:
        display(invalid_load_ids)    


