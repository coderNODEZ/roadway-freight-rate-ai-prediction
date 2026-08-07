import math
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = (
    PROJECT_ROOT
    / "FeatureEngineering"
    / "MissingColumnsPrediction"
    / "preprocessed_output.csv"
)

FINAL_PREDICTION_INPUT_PATH = (
    PROJECT_ROOT
    / "FeatureEngineering"
    / "MissingColumnsPrediction"
    / "final_preprocessed_output.csv"
)

STATE_TO_CENSUS_REGION = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "DC": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}


@cache
def _get_us_cities_by_name() -> dict[str, list[dict]]:
    """Load GeoNames once and index U.S. cities by normalized city name."""

    try:
        import geonamescache
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "State resolution requires geonamescache. "
            "Install it with: pip install geonamescache"
        ) from error

    geonames = geonamescache.GeonamesCache(min_city_population=500)
    us_cities_by_name = {}

    for candidate in geonames.get_cities().values():
        if candidate["countrycode"] == "US":
            city_key = candidate["name"].strip().casefold()
            us_cities_by_name.setdefault(city_key, []).append(candidate)

    return us_cities_by_name


def remove_posted_rate_if_present(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Remove posted_rate and return it separately when it exists."""

    if "posted_rate" not in data.columns:
        return data, None

    posted_rate_data = data.loc[:, ["posted_rate"]].copy()
    data = data.drop(columns=["posted_rate"])

    return data, posted_rate_data


def remove_leakage_columns_if_present(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove derived leakage columns when they are present."""

    leakage_columns = [
        "quote_signal_difference",
        "posted_rate_per_mile",
    ]

    columns_to_remove = [
        column
        for column in leakage_columns
        if column in data.columns
    ]

    if columns_to_remove:
        data = data.drop(columns=columns_to_remove)

        print(
            "Removed potential leakage columns: "
            f"{columns_to_remove}"
        )
    else:
        print("No derived leakage columns needed removal.")

    return data    

def add_pickup_state(data: pd.DataFrame) -> pd.DataFrame:
    """Add pickup_state and report any rows with a missing state."""

    CITY_NAME_ALIASES = {
        "new york": "new york city",
    }

    print(add_pickup_state.__doc__)

    us_cities_by_name = _get_us_cities_by_name()

    def resolve_state(row: pd.Series):
        city = str(row["pickup"]).strip()
        city_key = city.casefold()

        city_state_overrides = {
            "new york": "NY",
        }

        if city_key in city_state_overrides:
            return city_state_overrides[city_key]

        candidates = us_cities_by_name.get(city_key, [])

        if not candidates:
            return pd.NA

        city = str(row["pickup"]).strip()
        candidates = us_cities_by_name.get(city.casefold(), [])

        if not candidates:
            return pd.NA

        pickup_latitude = math.radians(float(row["pickup_lat"]))
        pickup_longitude = math.radians(float(row["pickup_lon"]))
        state_candidates = {}

        for candidate in candidates:
            candidate_latitude = math.radians(
                float(candidate["latitude"])
            )
            candidate_longitude = math.radians(
                float(candidate["longitude"])
            )

            latitude_difference = (
                candidate_latitude - pickup_latitude
            )
            longitude_difference = (
                candidate_longitude - pickup_longitude
            )

            haversine = (
                math.sin(latitude_difference / 2) ** 2
                + math.cos(pickup_latitude)
                * math.cos(candidate_latitude)
                * math.sin(longitude_difference / 2) ** 2
            )

            distance_miles = (
                3958.8
                * 2
                * math.asin(math.sqrt(haversine))
            )

            state = candidate["admin1code"]
            population = int(
                candidate.get("population") or 0
            )

            if (
                state not in state_candidates
                or distance_miles
                < state_candidates[state]["distance"]
            ):
                state_candidates[state] = {
                    "distance": distance_miles,
                    "population": population,
                }

        nearest_distance = min(
            candidate["distance"]
            for candidate in state_candidates.values()
        )

        close_states = {
            state: candidate
            for state, candidate in state_candidates.items()
            if candidate["distance"] <= nearest_distance + 25
        }

        return max(
            close_states,
            key=lambda state: (
                close_states[state]["population"],
                -close_states[state]["distance"],
            ),
        )

    unique_pickups = data[
        ["pickup", "pickup_lat", "pickup_lon"]
    ].drop_duplicates().copy()

    unique_pickups["pickup_state"] = unique_pickups.apply(
        resolve_state,
        axis=1,
    )

    data = data.merge(
        unique_pickups,
        on=["pickup", "pickup_lat", "pickup_lon"],
        how="left",
        validate="many_to_one",
    )

    missing_state_mask = data["pickup_state"].isna()
    missing_state_count = int(missing_state_mask.sum())

    print(f"Missing pickup_state rows: {missing_state_count:,}")

    if missing_state_count > 0:
        missing_pickups = (
            data.loc[
                missing_state_mask,
                ["pickup", "pickup_lat", "pickup_lon"],
            ]
            .value_counts(dropna=False)
            .rename("missing_row_count")
            .reset_index()
        )

        print(missing_pickups.to_string(index=False))

    return data


def add_delivery_state(data: pd.DataFrame) -> pd.DataFrame:
    """Add delivery_state and report any rows with a missing state."""

    print(add_delivery_state.__doc__)

    CITY_NAME_ALIASES = {
        "new york": "new york city",
    }


    us_cities_by_name = _get_us_cities_by_name()

    def resolve_state(row: pd.Series):
        city = str(row["delivery"]).strip()
        city_key = city.casefold()

        city_state_overrides = {
            "new york": "NY",
        }

        if city_key in city_state_overrides:
            return city_state_overrides[city_key]

        candidates = us_cities_by_name.get(city_key, [])

        if not candidates:
            return pd.NA

        city = str(row["delivery"]).strip()
        candidates = us_cities_by_name.get(city.casefold(), [])

        if not candidates:
            return pd.NA

        delivery_latitude = math.radians(
            float(row["delivery_lat"])
        )
        delivery_longitude = math.radians(
            float(row["delivery_lon"])
        )

        state_candidates = {}

        for candidate in candidates:
            candidate_latitude = math.radians(
                float(candidate["latitude"])
            )
            candidate_longitude = math.radians(
                float(candidate["longitude"])
            )

            latitude_difference = (
                candidate_latitude - delivery_latitude
            )
            longitude_difference = (
                candidate_longitude - delivery_longitude
            )

            haversine = (
                math.sin(latitude_difference / 2) ** 2
                + math.cos(delivery_latitude)
                * math.cos(candidate_latitude)
                * math.sin(longitude_difference / 2) ** 2
            )

            distance_miles = (
                3958.8
                * 2
                * math.asin(math.sqrt(haversine))
            )

            state = candidate["admin1code"]
            population = int(
                candidate.get("population") or 0
            )

            if (
                state not in state_candidates
                or distance_miles
                < state_candidates[state]["distance"]
            ):
                state_candidates[state] = {
                    "distance": distance_miles,
                    "population": population,
                }

        nearest_distance = min(
            candidate["distance"]
            for candidate in state_candidates.values()
        )

        close_states = {
            state: candidate
            for state, candidate in state_candidates.items()
            if candidate["distance"] <= nearest_distance + 25
        }

        if row["pickup_state"] in close_states:
            return row["pickup_state"]

        return max(
            close_states,
            key=lambda state: (
                close_states[state]["population"],
                -close_states[state]["distance"],
            ),
        )

    unique_deliveries = data[
        [
            "delivery",
            "delivery_lat",
            "delivery_lon",
            "pickup_state",
        ]
    ].drop_duplicates().copy()

    unique_deliveries["delivery_state"] = (
        unique_deliveries.apply(
            resolve_state,
            axis=1,
        )
    )

    data = data.merge(
        unique_deliveries,
        on=[
            "delivery",
            "delivery_lat",
            "delivery_lon",
            "pickup_state",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_state_mask = data["delivery_state"].isna()
    missing_state_count = int(missing_state_mask.sum())

    print(
        f"Missing delivery_state rows: "
        f"{missing_state_count:,}"
    )

    if missing_state_count > 0:
        missing_deliveries = (
            data.loc[
                missing_state_mask,
                [
                    "delivery",
                    "delivery_lat",
                    "delivery_lon",
                ],
            ]
            .value_counts(dropna=False)
            .rename("missing_row_count")
            .reset_index()
        )

        print(
            missing_deliveries.to_string(index=False)
        )

    return data


def interstate_trip_column(data: pd.DataFrame) -> pd.DataFrame:
    """Add a Boolean indicating whether pickup and delivery states differ."""

    print(interstate_trip_column.__doc__)

    data = data.copy()
    data["interstate_bool"] = data["delivery_state"].ne(
        data["pickup_state"]
    )
    return data


def census_region(data: pd.DataFrame) -> pd.DataFrame:
    """Add pickup, delivery, and inter-region columns."""

    print(census_region.__doc__)

    data = data.copy()
    data["pickup_region"] = data["pickup_state"].map(
        STATE_TO_CENSUS_REGION
    )
    data["delivery_region"] = data["delivery_state"].map(
        STATE_TO_CENSUS_REGION
    )
    data["inter_region_bool"] = data["pickup_region"].ne(
        data["delivery_region"]
    )
    return data


def timezone(data: pd.DataFrame) -> pd.DataFrame:
    """Add pickup, delivery, and inter-timezone columns offline."""

    print(timezone.__doc__)

    try:
        from timezonefinder import TimezoneFinder
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "timezone requires timezonefinder. "
            "Install it with: pip install timezonefinder"
        ) from error

    data = data.copy()
    finder = TimezoneFinder(in_memory=True)

    pickup_coordinates = data[
        ["pickup_lat", "pickup_lon"]
    ].drop_duplicates().copy()
    pickup_coordinates["pickup_timezone"] = [
        finder.timezone_at(lng=float(longitude), lat=float(latitude))
        for latitude, longitude in pickup_coordinates.itertuples(
            index=False,
            name=None,
        )
    ]
    data = data.merge(
        pickup_coordinates,
        on=["pickup_lat", "pickup_lon"],
        how="left",
        validate="many_to_one",
    )

    delivery_coordinates = data[
        ["delivery_lat", "delivery_lon"]
    ].drop_duplicates().copy()
    delivery_coordinates["delivery_timezone"] = [
        finder.timezone_at(lng=float(longitude), lat=float(latitude))
        for latitude, longitude in delivery_coordinates.itertuples(
            index=False,
            name=None,
        )
    ]
    data = data.merge(
        delivery_coordinates,
        on=["delivery_lat", "delivery_lon"],
        how="left",
        validate="many_to_one",
    )

    data["inter_timezone_bool"] = data["pickup_timezone"].ne(
        data["delivery_timezone"]
    )
    return data


def day_trips_count(data: pd.DataFrame) -> pd.DataFrame:
    """Add the running trip count for each date in load_id order."""

    print(day_trips_count.__doc__)

    data = data.copy()
    trip_order = data[["date", "load_id"]].copy()
    trip_order["date"] = pd.to_datetime(
        trip_order["date"],
        errors="coerce",
    ).dt.normalize()
    trip_order["original_order"] = range(len(trip_order))
    trip_order = trip_order.sort_values(
        ["date", "load_id", "original_order"],
        kind="stable",
    )
    trip_order["day_trips_count"] = (
        trip_order.groupby("date", dropna=False).cumcount() + 1
    )

    data["day_trips_count"] = (
        trip_order.sort_values("original_order")["day_trips_count"]
        .to_numpy()
    )
    return data


def month_trips_count(data: pd.DataFrame) -> pd.DataFrame:
    """Add the running trip count for each month in load_id order."""

    print(month_trips_count.__doc__)

    data = data.copy()
    trip_order = data[["date", "load_id"]].copy()
    trip_order["month"] = pd.to_datetime(
        trip_order["date"],
        errors="coerce",
    ).dt.to_period("M")
    trip_order["original_order"] = range(len(trip_order))
    trip_order = trip_order.sort_values(
        ["month", "load_id", "original_order"],
        kind="stable",
    )
    trip_order["month_trips_count"] = (
        trip_order.groupby("month", dropna=False).cumcount() + 1
    )

    data["month_trips_count"] = (
        trip_order.sort_values("original_order")["month_trips_count"]
        .to_numpy()
    )
    return data


def week_day(data: pd.DataFrame) -> pd.DataFrame:
    """Add weekday text and cyclical weekday features."""

    print(week_day.__doc__)

    data = data.copy()
    date = pd.to_datetime(data["date"], errors="coerce")
    day_of_week = date.dt.dayofweek

    data["day"] = date.dt.day_name()
    data["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    data["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    return data


def holidays(data: pd.DataFrame) -> pd.DataFrame:
    """Add a Boolean indicating whether each date is a U.S. holiday."""

    print(holidays.__doc__)
    import holidays as holidays_library

    data = data.copy()
    date = pd.to_datetime(data["date"], errors="coerce")
    holiday_calendar = holidays_library.US(
        years=date.dropna().dt.year.unique().tolist()
    )
    data["holiday_boolean"] = date.dt.date.map(
        lambda value: (
            value in holiday_calendar if pd.notna(value) else False
        )
    )
    return data


# def snowfall(data: pd.DataFrame) -> pd.DataFrame:
#     """Add historical daily snowfall for pickup and delivery locations."""

#     print(snowfall.__doc__)

#     #import requests

#     data = data.copy()
#     date = pd.to_datetime(data["date"], errors="coerce").dt.date

#     coordinates = pd.concat(
#         [
#             data[["pickup_lat", "pickup_lon"]].rename(
#                 columns={"pickup_lat": "latitude", "pickup_lon": "longitude"}
#             ),
#             data[["delivery_lat", "delivery_lon"]].rename(
#                 columns={
#                     "delivery_lat": "latitude",
#                     "delivery_lon": "longitude",
#                 }
#             ),
#         ],
#         ignore_index=True,
#     ).drop_duplicates()

#     response = requests.get(
#         "https://archive-api.open-meteo.com/v1/archive",
#         params={
#             "latitude": ",".join(coordinates["latitude"].astype(str)),
#             "longitude": ",".join(coordinates["longitude"].astype(str)),
#             "start_date": date.min().isoformat(),
#             "end_date": date.max().isoformat(),
#             "daily": "snowfall_sum",
#             "timezone": "auto",
#         },
#         timeout=60,
#     )
#     response.raise_for_status()

#     weather_results = response.json()
#     if isinstance(weather_results, dict):
#         weather_results = [weather_results]

#     snowfall_lookup = {}
#     for coordinate, weather in zip(
#         coordinates.itertuples(index=False, name=None),
#         weather_results,
#         strict=True,
#     ):
#         for weather_date, snowfall_sum in zip(
#             weather["daily"]["time"],
#             weather["daily"]["snowfall_sum"],
#             strict=True,
#         ):
#             snowfall_lookup[(*coordinate, weather_date)] = snowfall_sum

#     date_text = date.astype("string")
#     data["pickup_historical_snowfall"] = [
#         snowfall_lookup.get((latitude, longitude, weather_date), np.nan)
#         for latitude, longitude, weather_date in zip(
#             data["pickup_lat"],
#             data["pickup_lon"],
#             date_text,
#             strict=True,
#         )
#     ]
#     data["delivery_historical_snowfall"] = [
#         snowfall_lookup.get((latitude, longitude, weather_date), np.nan)
#         for latitude, longitude, weather_date in zip(
#             data["delivery_lat"],
#             data["delivery_lon"],
#             date_text,
#             strict=True,
#         )
#     ]
#     return data

def snowfall(data: pd.DataFrame) -> pd.DataFrame:
    """Add daily snowfall amounts and snowfall indicators for each location."""

    print(snowfall.__doc__)

    data = data.copy()
    date = pd.to_datetime(data["date"], errors="coerce").dt.date

    coordinates = pd.concat(
        [
            data[["pickup_lat", "pickup_lon"]].rename(
                columns={
                    "pickup_lat": "latitude",
                    "pickup_lon": "longitude",
                }
            ),
            data[["delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery_lat": "latitude",
                    "delivery_lon": "longitude",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": ",".join(
                coordinates["latitude"].astype(str)
            ),
            "longitude": ",".join(
                coordinates["longitude"].astype(str)
            ),
            "start_date": date.min().isoformat(),
            "end_date": date.max().isoformat(),
            "daily": "snowfall_sum",
            "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()

    weather_results = response.json()

    if isinstance(weather_results, dict):
        weather_results = [weather_results]

    snowfall_lookup = {}

    for coordinate, weather in zip(
        coordinates.itertuples(index=False, name=None),
        weather_results,
        strict=True,
    ):
        for weather_date, snowfall_sum in zip(
            weather["daily"]["time"],
            weather["daily"]["snowfall_sum"],
            strict=True,
        ):
            snowfall_lookup[
                (*coordinate, weather_date)
            ] = snowfall_sum

    date_text = date.astype("string")

    data["pickup_historical_snowfall"] = [
        snowfall_lookup.get(
            (latitude, longitude, weather_date),
            np.nan,
        )
        for latitude, longitude, weather_date in zip(
            data["pickup_lat"],
            data["pickup_lon"],
            date_text,
            strict=True,
        )
    ]

    data["delivery_historical_snowfall"] = [
        snowfall_lookup.get(
            (latitude, longitude, weather_date),
            np.nan,
        )
        for latitude, longitude, weather_date in zip(
            data["delivery_lat"],
            data["delivery_lon"],
            date_text,
            strict=True,
        )
    ]

    data["pickup_snowfall_bool"] = (
        data["pickup_historical_snowfall"].gt(0)
    )

    data["delivery_snowfall_bool"] = (
        data["delivery_historical_snowfall"].gt(0)
    )

    return data

# def freezing_days(data: pd.DataFrame) -> pd.DataFrame:
#     """Add pickup and delivery freezing-day Boolean columns."""

#     print(freezing_days.__doc__)

#     #import requests

#     data = data.copy()
#     date = pd.to_datetime(data["date"], errors="coerce").dt.date

#     coordinates = pd.concat(
#         [
#             data[["pickup_lat", "pickup_lon"]].rename(
#                 columns={"pickup_lat": "latitude", "pickup_lon": "longitude"}
#             ),
#             data[["delivery_lat", "delivery_lon"]].rename(
#                 columns={
#                     "delivery_lat": "latitude",
#                     "delivery_lon": "longitude",
#                 }
#             ),
#         ],
#         ignore_index=True,
#     ).drop_duplicates()

#     response = requests.get(
#         "https://archive-api.open-meteo.com/v1/archive",
#         params={
#             "latitude": ",".join(coordinates["latitude"].astype(str)),
#             "longitude": ",".join(coordinates["longitude"].astype(str)),
#             "start_date": date.min().isoformat(),
#             "end_date": date.max().isoformat(),
#             "daily": "temperature_2m_min",
#             "timezone": "auto",
#         },
#         timeout=60,
#     )
#     response.raise_for_status()

#     weather_results = response.json()
#     if isinstance(weather_results, dict):
#         weather_results = [weather_results]

#     freezing_lookup = {}
#     for coordinate, weather in zip(
#         coordinates.itertuples(index=False, name=None),
#         weather_results,
#         strict=True,
#     ):
#         for weather_date, minimum_temperature in zip(
#             weather["daily"]["time"],
#             weather["daily"]["temperature_2m_min"],
#             strict=True,
#         ):
#             freezing_lookup[(*coordinate, weather_date)] = (
#                 minimum_temperature is not None
#                 and minimum_temperature <= 0
#             )

#     date_text = date.astype("string")
#     data["pickup_freezing_days"] = [
#         freezing_lookup.get((latitude, longitude, weather_date), False)
#         for latitude, longitude, weather_date in zip(
#             data["pickup_lat"],
#             data["pickup_lon"],
#             date_text,
#             strict=True,
#         )
#     ]
#     data["delivery_freezing_days"] = [
#         freezing_lookup.get((latitude, longitude, weather_date), False)
#         for latitude, longitude, weather_date in zip(
#             data["delivery_lat"],
#             data["delivery_lon"],
#             date_text,
#             strict=True,
#         )
#     ]
#     return data

def freezing_days(data: pd.DataFrame) -> pd.DataFrame:
    """Add pickup and delivery freezing-day Boolean columns."""

    print(freezing_days.__doc__)

    data = data.copy()
    date = pd.to_datetime(data["date"], errors="coerce").dt.date

    coordinates = pd.concat(
        [
            data[["pickup_lat", "pickup_lon"]].rename(
                columns={
                    "pickup_lat": "latitude",
                    "pickup_lon": "longitude",
                }
            ),
            data[["delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery_lat": "latitude",
                    "delivery_lon": "longitude",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": ",".join(
                coordinates["latitude"].astype(str)
            ),
            "longitude": ",".join(
                coordinates["longitude"].astype(str)
            ),
            "start_date": date.min().isoformat(),
            "end_date": date.max().isoformat(),
            "daily": "temperature_2m_min",
            "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()

    weather_results = response.json()

    if isinstance(weather_results, dict):
        weather_results = [weather_results]

    freezing_lookup = {}

    for coordinate, weather in zip(
        coordinates.itertuples(index=False, name=None),
        weather_results,
        strict=True,
    ):
        for weather_date, minimum_temperature in zip(
            weather["daily"]["time"],
            weather["daily"]["temperature_2m_min"],
            strict=True,
        ):
            freezing_lookup[
                (*coordinate, weather_date)
            ] = (
                minimum_temperature is not None
                and minimum_temperature <= 0
            )

    date_text = date.astype("string")

    data["pickup_freezing_days"] = [
        freezing_lookup.get(
            (latitude, longitude, weather_date),
            False,
        )
        for latitude, longitude, weather_date in zip(
            data["pickup_lat"],
            data["pickup_lon"],
            date_text,
            strict=True,
        )
    ]

    data["delivery_freezing_days"] = [
        freezing_lookup.get(
            (latitude, longitude, weather_date),
            False,
        )
        for latitude, longitude, weather_date in zip(
            data["delivery_lat"],
            data["delivery_lon"],
            date_text,
            strict=True,
        )
    ]

    data["pickup_freezing_day_bool"] = (
        data["pickup_freezing_days"].astype(bool)
    )

    data["delivery_freezing_day_bool"] = (
        data["delivery_freezing_days"].astype(bool)
    )

    return data

def elevation(data: pd.DataFrame) -> pd.DataFrame:
    """Add pickup and delivery terrain altitude in meters."""

    print(elevation.__doc__)

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    data = data.copy()
    coordinates = pd.concat(
        [
            data[["pickup_lat", "pickup_lon"]].rename(
                columns={"pickup_lat": "latitude", "pickup_lon": "longitude"}
            ),
            data[["delivery_lat", "delivery_lon"]].rename(
                columns={
                    "delivery_lat": "latitude",
                    "delivery_lon": "longitude",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    altitude_lookup = {}
    batch_size = 50

    for start in range(0, len(coordinates), batch_size):
        batch = coordinates.iloc[start:start + batch_size]
        response = session.get(
            "https://api.open-meteo.com/v1/elevation",
            params={
                "latitude": ",".join(batch["latitude"].astype(str)),
                "longitude": ",".join(batch["longitude"].astype(str)),
            },
            timeout=(15, 120),
        )
        response.raise_for_status()

        elevations = response.json().get("elevation", [])
        if len(elevations) != len(batch):
            raise ValueError(
                "Open-Meteo returned an unexpected number of elevations."
            )

        altitude_lookup.update(
            {
                coordinate: altitude
                for coordinate, altitude in zip(
                    batch.itertuples(index=False, name=None),
                    elevations,
                    strict=True,
                )
            }
        )

    session.close()

    data["pickup_altitude"] = [
        altitude_lookup.get((latitude, longitude), np.nan)
        for latitude, longitude in zip(
            data["pickup_lat"],
            data["pickup_lon"],
            strict=True,
        )
    ]
    data["delivery_altitude"] = [
        altitude_lookup.get((latitude, longitude), np.nan)
        for latitude, longitude in zip(
            data["delivery_lat"],
            data["delivery_lon"],
            strict=True,
        )
    ]
    return data


def insurance_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Add insurance-risk features through the external feature module."""

    print(insurance_risk.__doc__)

    from FeatureEngineering.MissingColumnsPrediction import (
        insurance_risk_features,
    )

    return insurance_risk_features.add_insurance_risk_features(data)


def insurance_risk2(data: pd.DataFrame) -> pd.DataFrame:
    """Add mixed-vintage real-data risk features through the V2 module."""

    print(insurance_risk2.__doc__)

    from FeatureEngineering.MissingColumnsPrediction import (
        insurance_risk_features2,
    )

    return insurance_risk_features2.add_insurance_risk_features2(data)


def move_market_columns_last(data: pd.DataFrame) -> pd.DataFrame:
    """Move market_index and quote_signal to the end of the DataFrame."""

    print(move_market_columns_last.__doc__)

    columns_to_move = [
        column
        for column in ["market_index", "quote_signal", "posted_rate"]
        if column in data.columns
    ]
    remaining_columns = [
        column
        for column in data.columns
        if column not in columns_to_move
    ]
    return data.loc[:, remaining_columns + columns_to_move]


def fuel_cost_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add real 2025 diesel and crude-oil features through an external module."""

    print(fuel_cost_features.__doc__)

    from FeatureEngineering.MissingColumnsPrediction import (
        fuel_cost_features as fuel_cost_features_module,
    )

    return fuel_cost_features_module.add_fuel_cost_features(data)


def add_month_column(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    output_column: str = "month",
) -> pd.DataFrame:
    """Extract the calendar month number from the date column."""

    print(add_month_column.__doc__)

    if date_column not in data.columns:
        raise ValueError(
            f"Month feature requires the {date_column!r} column."
        )

    result = data.copy()

    dates = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    invalid_date_count = dates.isna().sum()

    if invalid_date_count:
        raise ValueError(
            f"Could not extract the month because {invalid_date_count} "
            f"values in {date_column!r} are invalid or missing."
        )

    result[output_column] = dates.dt.month.astype("int8")

    return result    


def add_season_column(
    data: pd.DataFrame,
    *,
    month_column: str = "month",
    output_column: str = "season",
) -> pd.DataFrame:
    """
    Categorize each calendar month as winter, spring, summer, or autumn.

    Winter: December, January, February
    Spring: March, April, May
    Summer: June, July, August
    Autumn: September, October, November
    """

    print(add_season_column.__doc__)    
    if month_column not in data.columns:
        raise ValueError(
            f"Season feature requires the {month_column!r} column."
        )

    result = data.copy()

    months = pd.to_numeric(
        result[month_column],
        errors="coerce",
    )

    invalid_months = months.isna() | ~months.between(1, 12)

    if invalid_months.any():
        raise ValueError(
            "Could not assign seasons because "
            f"{invalid_months.sum()} values in {month_column!r} "
            "are missing or outside the range 1–12."
        )

    season_mapping = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }

    result[output_column] = (
        months.astype("int8")
        .map(season_mapping)
        .astype("category")
    )

    return result    


def add_dist_times_quote_signal(
    data: pd.DataFrame,
    *,
    distance_column: str = "distance",
    quote_signal_column: str = "quote_signal",
    output_column: str = "dist_times_quote_signal",
) -> pd.DataFrame:
    """
    Add distance multiplied by quote_signal.

    If quote_signal is absent or missing, dist_times_quote_signal is
    created with NaN values for the affected rows. It can be recalculated
    after predicted quote_signal values are inserted.
    """
    if distance_column not in data.columns:
        raise ValueError(
            f"dist_times_quote_signal requires {distance_column!r}."
        )

    result = data.copy()

    distance = pd.to_numeric(
        result[distance_column],
        errors="coerce",
    )

    # Final-prediction input may not yet contain quote_signal.
    if quote_signal_column not in result.columns:
        result[output_column] = np.nan
        return result

    quote_signal = pd.to_numeric(
        result[quote_signal_column],
        errors="coerce",
    )

    # Multiplication naturally produces NaN wherever distance or
    # quote_signal is unavailable.
    result[output_column] = distance * quote_signal

    return result  

# def national_diesel_change_1_year(data: pd.DataFrame) -> pd.DataFrame:
#     """Add the current EIA national on-highway diesel price minus the corresponding weekly price one year earlier."""

#     print(national_diesel_change_1_year.__doc__)

#     from FeatureEngineering.MissingColumnsPrediction import (
#         fuel_cost_features as fuel_cost_features_module,
#     )

#     return fuel_cost_features_module.add_national_diesel_change_1_year(data)   


def missing_columns_feature_engr_controller(final_prediction=False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the preprocessed input and run feature-engineering functions."""

    print(missing_columns_feature_engr_controller.__doc__)

    if not final_prediction:
        data = pd.read_csv(INPUT_PATH)
    else:
        data = pd.read_csv(FINAL_PREDICTION_INPUT_PATH)

    data, posted_rate_data = remove_posted_rate_if_present(data)
    data = remove_leakage_columns_if_present(data)    

    data = add_pickup_state(data)
    data = add_delivery_state(data)
    data = interstate_trip_column(data)

    data = census_region(data)
    data = timezone(data)

    data = day_trips_count(data)
    data = month_trips_count(data)

    data = add_month_column(data)
    data = add_season_column(data)

    data = week_day(data)
    data = holidays(data)

    data = snowfall(data)
    data = freezing_days(data)

    data = elevation(data)
    data = insurance_risk(data)

    # data = insurance_risk2(data)
    data = fuel_cost_features(data)

    data = add_dist_times_quote_signal(data)
    # data = add_national_diesel_change_1_year(data)
    data = move_market_columns_last(data)

    print("missing columns feature engr is complete")
    return data, posted_rate_data