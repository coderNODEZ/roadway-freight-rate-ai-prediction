"""Mixed-vintage, real-data truck-insurance risk features.

This module intentionally contains no example, premium-benchmark, or synthetic
values.  It reads six standardized extracts made from the documented public
sources and refuses to run when a required extract or provenance field is
missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIRECTORY = Path(__file__).with_name("insurance_risk_data")

COMPONENT_WEIGHTS = {
    "commercial_auto_loss_ratio": 0.40,
    "truck_crash_rate": 0.20,
    "vehicle_theft_rate": 0.15,
    "congestion_score": 0.10,
    "severe_weather_risk": 0.10,
    "litigation_risk": 0.05,
}

# Expected mixed-vintage source files.  Each CSV must also contain year and
# source_url so the model input remains auditable.
SOURCE_FILES = {
    "commercial_auto_loss_ratio": (
        "state_commercial_auto.csv",
        ["state", "commercial_auto_loss_ratio", "year", "source_url"],
    ),
    "truck_crash_rate": (
        "city_crashes.csv",
        ["city", "state", "truck_crash_rate", "year", "source_url"],
    ),
    "vehicle_theft_rate": (
        "city_vehicle_theft.csv",
        ["city", "state", "vehicle_theft_rate", "year", "source_url"],
    ),
    "congestion_score": (
        "city_congestion.csv",
        ["city", "state", "congestion_score", "year", "source_url"],
    ),
    "severe_weather_risk": (
        "city_weather_risk.csv",
        ["city", "state", "severe_weather_risk", "year", "source_url"],
    ),
    "litigation_risk": (
        "state_litigation_risk.csv",
        ["state", "litigation_risk", "year", "source_url"],
    ),
}


def add_insurance_risk_features2(
    data: pd.DataFrame,
    data_directory: str | Path = DATA_DIRECTORY,
) -> pd.DataFrame:
    """Add four risk-index columns from cached, sourced real observations."""

    required = {
        "pickup",
        "pickup_state",
        "delivery",
        "delivery_state",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Insurance risk V2 is missing columns: {missing}")

    source_tables = _load_source_tables(Path(data_directory))
    city_scores = _build_city_scores(data, source_tables)
    score_lookup = city_scores.set_index(["city_key", "state"])[
        "insurance_risk2"
    ]

    result = data.copy()
    result["pickup_insurance_risk2"] = _map_endpoint(
        result["pickup"], result["pickup_state"], score_lookup
    )
    result["delivery_insurance_risk2"] = _map_endpoint(
        result["delivery"], result["delivery_state"], score_lookup
    )
    result["route_insurance_risk2"] = result[
        ["pickup_insurance_risk2", "delivery_insurance_risk2"]
    ].mean(axis=1)
    result["route_max_insurance_risk2"] = result[
        ["pickup_insurance_risk2", "delivery_insurance_risk2"]
    ].max(axis=1)
    return result


def _load_source_tables(data_directory: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    problems = []

    for component, (filename, required_columns) in SOURCE_FILES.items():
        path = data_directory / filename
        if not path.is_file():
            problems.append(str(path))
            continue

        table = pd.read_csv(path)
        missing = sorted(set(required_columns).difference(table.columns))
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        if table.empty:
            raise ValueError(f"{filename} contains no observations.")
        if table["year"].isna().any() or table["source_url"].isna().any():
            raise ValueError(
                f"{filename} must identify year and source_url for every row."
            )

        table = table.copy()
        table["state"] = table["state"].astype("string").str.upper().str.strip()
        if "city" in table:
            table["city_key"] = _city_key(table["city"])
        table[component] = pd.to_numeric(table[component], errors="coerce")
        if not np.isfinite(table[component]).any():
            raise ValueError(f"{filename} has no finite {component} values.")
        tables[component] = table

    if problems:
        files = "\n  - ".join(problems)
        raise FileNotFoundError(
            "Insurance risk V2 requires real-data source extracts:\n  - " + files
        )
    return tables


def _build_city_scores(
    freight_data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pickup = freight_data[["pickup", "pickup_state"]].rename(
        columns={"pickup": "city", "pickup_state": "state"}
    )
    delivery = freight_data[["delivery", "delivery_state"]].rename(
        columns={"delivery": "city", "delivery_state": "state"}
    )
    cities = pd.concat([pickup, delivery], ignore_index=True).drop_duplicates()
    cities["state"] = cities["state"].astype("string").str.upper().str.strip()
    cities["city_key"] = _city_key(cities["city"])

    for component in COMPONENT_WEIGHTS:
        source = tables[component].copy()
        index_column = f"{component}_index"
        source[index_column] = _robust_zero_to_one(source[component])
        if "city_key" in source:
            values = source[["city_key", "state", index_column]].drop_duplicates(
                ["city_key", "state"], keep="last"
            )
            cities = cities.merge(
                values, on=["city_key", "state"], how="left", validate="one_to_one"
            )
            state_median = source.groupby("state")[index_column].median()
            cities[index_column] = cities[index_column].fillna(
                cities["state"].map(state_median)
            )
        else:
            values = source[["state", index_column]].drop_duplicates(
                "state", keep="last"
            )
            cities = cities.merge(values, on="state", how="left", validate="many_to_one")

        # A real national median is used only when a city/state observation is
        # absent.  No random, fabricated, or premium-benchmark value is used.
        cities[index_column] = cities[index_column].fillna(
            source[index_column].median()
        )

    cities["insurance_risk2"] = sum(
        weight * cities[f"{component}_index"]
        for component, weight in COMPONENT_WEIGHTS.items()
    )
    return cities


def _robust_zero_to_one(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    lower = numeric.quantile(0.05)
    upper = numeric.quantile(0.95)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("A risk component contains no usable numeric values.")
    if upper == lower:
        return pd.Series(0.5, index=values.index, dtype=float)
    return numeric.clip(lower, upper).sub(lower).div(upper - lower)


def _map_endpoint(
    cities: pd.Series,
    states: pd.Series,
    lookup: pd.Series,
) -> pd.Series:
    keys = pd.MultiIndex.from_arrays(
        [_city_key(cities), states.astype("string").str.upper().str.strip()]
    )
    values = lookup.reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Could not map every freight endpoint to a V2 risk index.")
    return pd.Series(values, index=cities.index)


def _city_key(cities: pd.Series) -> pd.Series:
    return (
        cities.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )

