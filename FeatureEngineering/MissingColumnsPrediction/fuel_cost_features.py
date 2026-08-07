"""Real 2025 diesel and crude-oil features for freight-rate modeling."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


FUEL_FEATURE_COLUMNS = [
    "national_diesel_price",
    "pickup_region_diesel_price",
    "delivery_region_diesel_price",
    "route_average_diesel_price",
    "diesel_change_1_week",
    "diesel_change_4_weeks",
    # "national_diesel_change_1_year"
    "estimated_fuel_gallons",
    "estimated_fuel_cost",
    "estimated_fuel_cost_per_mile",
    # "pickup_diesel_minus_national",
    # "delivery_diesel_minus_national",
    "wti_spot_price",
    "wti_change_5d",
    "wti_change_20d",
    "wti_volatility_20d",
    "brent_spot_price",
    "brent_wti_spread",
]

CACHE_DIRECTORY = Path(__file__).with_name("fuel_cost_data")
CACHE_PATH = CACHE_DIRECTORY / "fuel_cost_market_data_2025.csv"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

SERIES_IDS = {
    "national_diesel_price": "GASDESW",
    "East Coast": "GASDESECW",
    "Midwest": "GASDESMWW",
    "Gulf Coast": "GASDESGCW",
    "Rocky Mountain": "GASDESRMW",
    "West Coast": "GASDESWCW",
    "wti_spot_price": "DCOILWTICO",
    "brent_spot_price": "DCOILBRENTEU",
}

STATE_TO_DIESEL_REGION = {
    "CT": "East Coast", "DC": "East Coast", "DE": "East Coast",
    "FL": "East Coast", "GA": "East Coast", "MA": "East Coast",
    "MD": "East Coast", "ME": "East Coast", "NC": "East Coast",
    "NH": "East Coast", "NJ": "East Coast", "NY": "East Coast",
    "PA": "East Coast", "RI": "East Coast", "SC": "East Coast",
    "VA": "East Coast", "VT": "East Coast", "WV": "East Coast",
    "IA": "Midwest", "IL": "Midwest", "IN": "Midwest",
    "KS": "Midwest", "KY": "Midwest", "MI": "Midwest",
    "MN": "Midwest", "MO": "Midwest", "ND": "Midwest",
    "NE": "Midwest", "OH": "Midwest", "OK": "Midwest",
    "SD": "Midwest", "TN": "Midwest", "WI": "Midwest",
    "AL": "Gulf Coast", "AR": "Gulf Coast", "LA": "Gulf Coast",
    "MS": "Gulf Coast", "NM": "Gulf Coast", "TX": "Gulf Coast",
    "CO": "Rocky Mountain", "ID": "Rocky Mountain",
    "MT": "Rocky Mountain", "UT": "Rocky Mountain",
    "WY": "Rocky Mountain",
    "AK": "West Coast", "AZ": "West Coast", "CA": "West Coast",
    "HI": "West Coast", "NV": "West Coast", "OR": "West Coast",
    "WA": "West Coast",
}


def add_fuel_cost_features(
    data: pd.DataFrame,
    *,
    truck_miles_per_gallon: float = 6.5,
    refresh_data: bool = False,
    cache_path: str | Path = CACHE_PATH,
) -> pd.DataFrame:
    """Add exactly the requested 2025 diesel and crude-oil features."""

    required = {"date", "distance", "pickup_state", "delivery_state"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Fuel-cost features are missing columns: {missing}")
    if not np.isfinite(truck_miles_per_gallon) or truck_miles_per_gallon <= 0:
        raise ValueError("truck_miles_per_gallon must be a positive finite value.")

    result = data.copy()
    dates = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    distance = pd.to_numeric(result["distance"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Fuel-cost features require valid dates.")
    if distance.isna().any() or distance.le(0).any():
        raise ValueError("Fuel-cost features require positive numeric distance.")

    market = load_or_download_2025_fuel_data(
        refresh=refresh_data,
        cache_path=cache_path,
    ).set_index("date")
    aligned = market.reindex(dates, method="ffill")
    aligned.index = result.index

    result["national_diesel_price"] = aligned["national_diesel_price"]
    result["pickup_region_diesel_price"] = _regional_prices(
        result["pickup_state"], aligned
    )
    result["delivery_region_diesel_price"] = _regional_prices(
        result["delivery_state"], aligned
    )
    result["route_average_diesel_price"] = result[
        ["pickup_region_diesel_price", "delivery_region_diesel_price"]
    ].mean(axis=1)
    result["diesel_change_1_week"] = aligned["diesel_change_1_week"]
    result["diesel_change_4_weeks"] = aligned["diesel_change_4_weeks"]
    result["estimated_fuel_gallons"] = distance / truck_miles_per_gallon
    result["estimated_fuel_cost"] = (
        result["estimated_fuel_gallons"]
        * result["route_average_diesel_price"]
    )
    result["estimated_fuel_cost_per_mile"] = (
        result["estimated_fuel_cost"] / distance
    )
    result["pickup_diesel_minus_national"] = (
        result["pickup_region_diesel_price"]
        - result["national_diesel_price"]
    )
    result["delivery_diesel_minus_national"] = (
        result["delivery_region_diesel_price"]
        - result["national_diesel_price"]
    )
    for column in [
        "wti_spot_price", "wti_change_5d", "wti_change_20d",
        "wti_volatility_20d", "brent_spot_price", "brent_wti_spread",
    ]:
        result[column] = aligned[column]

    invalid = result[FUEL_FEATURE_COLUMNS].isna().sum()
    invalid = invalid[invalid.gt(0)]
    if not invalid.empty:
        raise ValueError(
            "Fuel-cost data could not cover every requested date: "
            f"{invalid.to_dict()}"
        )
    return result


def load_or_download_2025_fuel_data(
    *,
    refresh: bool = False,
    cache_path: str | Path = CACHE_PATH,
) -> pd.DataFrame:
    """Return cached daily 2025 data or download and build it from FRED/EIA."""

    cache_path = Path(cache_path)
    if cache_path.is_file() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return _validate_market_data(cached)

    series = {
        output_name: _download_fred_series(series_id)
        for output_name, series_id in SERIES_IDS.items()
    }
    calendar = pd.DataFrame(
        {"date": pd.date_range("2024-10-01", "2025-12-31", freq="D")}
    ).set_index("date")

    diesel_names = [
        "national_diesel_price", "East Coast", "Midwest", "Gulf Coast",
        "Rocky Mountain", "West Coast",
    ]
    weekly = pd.concat([series[name] for name in diesel_names], axis=1).sort_index()
    weekly["diesel_change_1_week"] = weekly["national_diesel_price"].diff(1)
    weekly["diesel_change_4_weeks"] = weekly["national_diesel_price"].diff(4)
    calendar = calendar.join(weekly).ffill()

    crude = pd.concat(
        [series["wti_spot_price"], series["brent_spot_price"]], axis=1
    ).sort_index().ffill()
    crude["wti_change_5d"] = crude["wti_spot_price"].diff(5)
    crude["wti_change_20d"] = crude["wti_spot_price"].diff(20)
    crude["wti_volatility_20d"] = (
        crude["wti_spot_price"].pct_change(fill_method=None).rolling(20).std()
    )
    crude["brent_wti_spread"] = (
        crude["brent_spot_price"] - crude["wti_spot_price"]
    )
    calendar = calendar.join(crude).ffill()
    calendar = calendar.loc["2025-01-01":"2025-12-31"].reset_index()
    calendar = _validate_market_data(calendar)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(cache_path, index=False)
    return calendar


def _download_fred_series(series_id: str) -> pd.Series:
    url = FRED_CSV_URL.format(series_id=series_id)
    try:
        with urlopen(url, timeout=120) as response:
            frame = pd.read_csv(response)
    except Exception as error:
        raise RuntimeError(f"Could not download FRED series {series_id}.") from error
    if frame.shape[1] < 2:
        raise ValueError(f"FRED series {series_id} returned an invalid table.")
    date = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    result = pd.Series(values.to_numpy(), index=date, name=_series_output_name(series_id))
    return result.loc[result.index.notna()].dropna().sort_index()


def _series_output_name(series_id: str) -> str:
    for output_name, configured_id in SERIES_IDS.items():
        if series_id == configured_id:
            return output_name
    raise KeyError(series_id)


def _regional_prices(states: pd.Series, aligned: pd.DataFrame) -> pd.Series:
    regions = states.astype("string").str.upper().str.strip().map(
        STATE_TO_DIESEL_REGION
    )
    missing_states = sorted(states.loc[regions.isna()].dropna().unique().tolist())
    if missing_states:
        raise ValueError(f"No diesel-region mapping for states: {missing_states}")
    prices = pd.Series(index=states.index, dtype=float)
    for region in regions.dropna().unique():
        mask = regions.eq(region)
        prices.loc[mask] = aligned.loc[mask, region]
    return prices


def _validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "national_diesel_price", "East Coast", "Midwest",
        "Gulf Coast", "Rocky Mountain", "West Coast",
        "diesel_change_1_week", "diesel_change_4_weeks",
        "wti_spot_price", "wti_change_5d", "wti_change_20d",
        "wti_volatility_20d", "brent_spot_price", "brent_wti_spread",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fuel market cache is missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("Fuel market cache contains invalid or duplicate dates.")
    return result.sort_values("date")



def add_national_diesel_change_1_year(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the EIA national on-highway diesel-price change from one year ago.

    The value is measured in dollars per gallon:

        current weekly price - price 52 weekly observations earlier
    """
    required_columns = {
        "date",
        "national_diesel_price",
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "One-year diesel change is missing columns: "
            f"{missing_columns}"
        )

    result = data.copy()

    dates = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError(
            "One-year diesel change requires valid dates."
        )

    # Download the complete weekly EIA history. The 2024 observations
    # are necessary to calculate the 2025 year-over-year changes.
    weekly_diesel = _download_fred_series(
        SERIES_IDS["national_diesel_price"]
    ).sort_index()

    weekly_market = pd.DataFrame({
        "national_diesel_price": weekly_diesel,
    })

    # EIA publishes this series weekly, so 52 observations represent
    # the corresponding reporting week approximately one year earlier.
    weekly_market["national_diesel_change_1_year"] = (
        weekly_market["national_diesel_price"]
        - weekly_market["national_diesel_price"].shift(52)
    )

    daily_calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                weekly_market.index.min(),
                dates.max(),
                freq="D",
            )
        }
    ).set_index("date")

    daily_calendar = daily_calendar.join(
        weekly_market[
            ["national_diesel_change_1_year"]
        ]
    ).ffill()

    aligned = daily_calendar.reindex(
        dates,
        method="ffill",
    )

    aligned.index = result.index

    result["national_diesel_change_1_year"] = (
        aligned["national_diesel_change_1_year"]
    )

    if result["national_diesel_change_1_year"].isna().any():
        missing_count = (
            result["national_diesel_change_1_year"]
            .isna()
            .sum()
        )

        raise ValueError(
            "EIA diesel history could not provide the one-year "
            f"change for {missing_count} rows."
        )

    return result