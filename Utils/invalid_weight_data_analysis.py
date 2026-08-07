import pandas as pd
import numpy as np

def missing_weight_data_review(df: pd.DataFrame) -> None:

    weight_audit = pd.Series({
        "missing": df["weight"].isna().sum(),
        "zero": df["weight"].eq(0).sum(),
        "negative": df["weight"].lt(0).sum(),
        "positive": df["weight"].gt(0).sum(),
    })

    display(weight_audit.to_frame("weight count"))    


def invalid_weight_rows(df: pd.DataFrame) -> pd.DataFrame:
        
    invalid_weight_mask = (
        df["weight"].isna()
        | df["weight"].le(0)
    )

    invalid_weight_rows = df.loc[
        invalid_weight_mask,
        [
            "load_id",
            "pickup",
            "delivery",
            "equipment",
            "weight",
            "distance",
            "date",
            "posted_rate",
        ],
    ].copy()

    print("Rows with missing or nonpositive weight:", len(invalid_weight_rows))

    print("\ninvalid_weight_rows.head(20): ")
    display(invalid_weight_rows.head(20))     

    return invalid_weight_rows

     
def invalid_weight_by_equipment(df: pd.DataFrame) -> None:

    weight_issues_by_equipment = (
        df.groupby("equipment", dropna=False)
        .agg(
            missing_weight_count=(
                "weight",
                lambda values: values.isna().sum(),
            ),
            negative_weight_count=(
                "weight",
                lambda values: values.lt(0).sum(),
            ),
        )
    )

    weight_issues_by_equipment["invalid_weight_count"] = (
        weight_issues_by_equipment["missing_weight_count"]
        + weight_issues_by_equipment["negative_weight_count"]
    )

    weight_issues_by_equipment = (
        weight_issues_by_equipment
        .sort_values("invalid_weight_count", ascending=False)
    )

    display(weight_issues_by_equipment)          

def invalid_weight_by_date(invalid_weight_rows: pd.DataFrame) -> None:

    display(
        invalid_weight_rows.groupby("date")
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_frame("invalid_weight_count")
    )       


def invalid_weight_pickup_delivery(invalid_weight_rows: pd.DataFrame) -> None:

    display(
        invalid_weight_rows.groupby(
            ["pickup", "delivery"]
        )
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_frame("invalid_weight_count")
    )


def weight_comparison(df: pd.DataFrame) -> None:

    invalid_weight_mask = (
        df["weight"].isna()
        | df["weight"].le(0)
    )    

    weight_comparison = df.assign(
        weight_status=np.where(
            invalid_weight_mask,
            "Missing or nonpositive",
            "Positive",
        )
    ).groupby("weight_status").agg(
        rows=("load_id", "size"),
        average_distance=("distance", "mean"),
        median_distance=("distance", "median"),
        average_posted_rate=("posted_rate", "mean"),
        median_posted_rate=("posted_rate", "median"),
        average_quote_signal=("quote_signal", "mean"),
    )

    display(weight_comparison)    