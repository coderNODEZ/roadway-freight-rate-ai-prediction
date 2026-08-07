import pandas as pd
import time

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from geopy.distance import geodesic


def basic_coordinates_validity(df: pd.DataFrame) -> None:

    coordinate_checks = {
        "Pickup latitude present": df["pickup_lat"].notna(),
        "Pickup longitude present": df["pickup_lon"].notna(),
        "Pickup latitude within -90 to 90": df["pickup_lat"].between(-90, 90),
        "Pickup longitude within -180 to 180": df["pickup_lon"].between(-180, 180),

        "Delivery latitude present": df["delivery_lat"].notna(),
        "Delivery longitude present": df["delivery_lon"].notna(),
        "Delivery latitude within -90 to 90": df["delivery_lat"].between(-90, 90),
        "Delivery longitude within -180 to 180": df["delivery_lon"].between(-180, 180),
    }

    print("\033[1mBasic coordinate validation\033[0m")
    print("-" * 60)

    for description, valid_mask in coordinate_checks.items():
        invalid_count = (~valid_mask).sum()

        print(
            f"{description:<45} "
            f"invalid: {invalid_count:,}"
        )

    df["pickup_coordinates_in_range"] = (
        coordinate_checks["Pickup latitude present"]
        & coordinate_checks["Pickup longitude present"]
        & coordinate_checks["Pickup latitude within -90 to 90"]
        & coordinate_checks["Pickup longitude within -180 to 180"]
    )

    df["delivery_coordinates_in_range"] = (
        coordinate_checks["Delivery latitude present"]
        & coordinate_checks["Delivery longitude present"]
        & coordinate_checks["Delivery latitude within -90 to 90"]
        & coordinate_checks["Delivery longitude within -180 to 180"]
    )

    print()
    print("\033[1mCombined coordinate-pair results\033[0m")
    print("-" * 60)

    print(
        f"{'Invalid pickup coordinate pairs:':<45} "
        f"{(~df['pickup_coordinates_in_range']).sum():,}"
    )

    print(
        f"{'Invalid delivery coordinate pairs:':<45} "
        f"{(~df['delivery_coordinates_in_range']).sum():,}"
    )


def unique_city_review(df: pd.DataFrame) -> list[str]:
    print("\033[1mUnique city review\033[0m")
    print("-" * 60)

    print("Combining pickup and delivery city columns...")

    unique_cities = pd.concat(
        [
            df["pickup"],
            df["delivery"],
        ],
        ignore_index=True,
    )

    print(f"Combined city entries:                 {len(unique_cities):,}")

    missing_count = unique_cities.isna().sum()
    print(f"Missing city entries removed:          {missing_count:,}")

    unique_cities = (
        unique_cities
        .dropna()
        .astype("string")
        .str.strip()
    )

    blank_count = unique_cities.eq("").sum()
    print(f"Blank city entries removed:            {blank_count:,}")

    unique_cities = sorted(
        city
        for city in unique_cities.unique()
        if city
    )

    print(f"Unique cities ready for geocoding:     {len(unique_cities):,}")

    # print("\n\033[1mUnique cities\033[0m")
    # print("-" * 60)

    for index, city in enumerate(unique_cities, start=1):
        print(f"{index:>2}. {city}")

    return unique_cities     



def geocode_unique_cities(
    unique_cities: list[str],
) -> pd.DataFrame:
    print("\033[1mForward-geocoding unique cities\033[0m")
    print("-" * 60)

    geolocator = Nominatim(
        user_agent="freight-rate-coordinate-validation"
    )

    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        max_retries=2,
        error_wait_seconds=5,
        swallow_exceptions=True,
    )

    city_reference_records = []

    print(f"Cities queued for geocoding:           {len(unique_cities):,}")
    print("Country restriction:                   United States")
    print("Minimum delay between requests:        1 second")
    print()

    start_time = time.perf_counter()

    for index, city in enumerate(unique_cities, start=1):
        query = f"{city}, USA"

        location = geocode(
            query,
            exactly_one=True,
            addressdetails=True,
            country_codes="us",
        )

        if location is None:
            city_reference_records.append(
                {
                    "city": city,
                    "city_reference_lat": pd.NA,
                    "city_reference_lon": pd.NA,
                    "city_geocoded_address": pd.NA,
                    "city_geocode_success": False,
                }
            )

            print(
                f"{index:>3}/{len(unique_cities):<3} "
                f"{city:<25} Not found"
            )

            continue

        city_reference_records.append(
            {
                "city": city,
                "city_reference_lat": location.latitude,
                "city_reference_lon": location.longitude,
                "city_geocoded_address": location.address,
                "city_geocode_success": True,
            }
        )

        print(
            f"{index:>3}/{len(unique_cities):<3} "
            f"{city:<25} "
            f"{location.latitude:>10.5f}, "
            f"{location.longitude:>11.5f}"
        )

    geocoding_runtime = time.perf_counter() - start_time

    city_reference_df = pd.DataFrame(city_reference_records)

    successful_lookups = int(
        city_reference_df["city_geocode_success"].sum()
    )

    failed_lookups = (
        len(city_reference_df) - successful_lookups
    )

    print()
    print("\033[1mGeocoding summary\033[0m")
    print("-" * 60)
    print(f"Geocoding runtime:                     {geocoding_runtime:.2f} seconds")
    print(f"Cities processed:                      {len(city_reference_df):,}")
    print(f"Successful city lookups:               {successful_lookups:,}")
    print(f"Failed city lookups:                   {failed_lookups:,}")

    return city_reference_df    


def map_city_references(
    df: pd.DataFrame,
    city_reference_df: pd.DataFrame,
) -> None:
    print("\033[1mMapping city-reference data to freight records\033[0m")
    print("-" * 60)

    city_lookup = city_reference_df.set_index("city")

    for location_type in ["pickup", "delivery"]:
        city_column = location_type

        df[f"{location_type}_city_reference_lat"] = (
            df[city_column].map(
                city_lookup["city_reference_lat"]
            )
        )

        df[f"{location_type}_city_reference_lon"] = (
            df[city_column].map(
                city_lookup["city_reference_lon"]
            )
        )

        df[f"{location_type}_city_geocoded_address"] = (
            df[city_column].map(
                city_lookup["city_geocoded_address"]
            )
        )

        df[f"{location_type}_city_geocode_success"] = (
            df[city_column]
            .map(city_lookup["city_geocode_success"])
            .fillna(False)
            .astype(bool)
        )

        successful_count = int(
            df[f"{location_type}_city_geocode_success"].sum()
        )

        failed_count = len(df) - successful_count

        print(
            f"{location_type.title():<12} "
            f"successful mappings: {successful_count:,}"
        )

        print(
            f"{location_type.title():<12} "
            f"failed mappings:     {failed_count:,}"
        )

    print()
    print("City-reference columns added successfully.")    


def distance_to_city_center_miles(
    actual_lat,
    actual_lon,
    reference_lat,
    reference_lon,
) -> pd.DataFrame:
    values = [
        actual_lat,
        actual_lon,
        reference_lat,
        reference_lon,
    ]

    if any(pd.isna(value) for value in values):
        return pd.NA

    if not (-90 <= actual_lat <= 90):
        return pd.NA

    if not (-180 <= actual_lon <= 180):
        return pd.NA

    if not (-90 <= reference_lat <= 90):
        return pd.NA

    if not (-180 <= reference_lon <= 180):
        return pd.NA

    return geodesic(
        (actual_lat, actual_lon),
        (reference_lat, reference_lon),
    ).miles    


import time

import pandas as pd


def validate_city_coordinate_distances(
    df: pd.DataFrame,
    city_tolerance_miles: float = 50,
) -> None:
    print("\033[1mValidating coordinates against city centers\033[0m")
    print("-" * 60)
    print(f"City-distance tolerance:               {city_tolerance_miles:g} miles")
    print(f"Rows to process:                       {len(df):,}")
    print()

    start_time = time.perf_counter()

    df["pickup_city_distance_miles"] = df.apply(
        lambda row: distance_to_city_center_miles(
            row["pickup_lat"],
            row["pickup_lon"],
            row["pickup_city_reference_lat"],
            row["pickup_city_reference_lon"],
        ),
        axis=1,
    )

    pickup_distance_runtime = time.perf_counter() - start_time

    print(
        f"Pickup distance-calculation runtime:   "
        f"{pickup_distance_runtime:.2f} seconds"
    )

    start_time = time.perf_counter()

    df["delivery_city_distance_miles"] = df.apply(
        lambda row: distance_to_city_center_miles(
            row["delivery_lat"],
            row["delivery_lon"],
            row["delivery_city_reference_lat"],
            row["delivery_city_reference_lon"],
        ),
        axis=1,
    )

    delivery_distance_runtime = time.perf_counter() - start_time

    print(
        f"Delivery distance-calculation runtime: "
        f"{delivery_distance_runtime:.2f} seconds"
    )

    df["pickup_coordinate_matches_city"] = (
        df["pickup_coordinates_in_range"]
        & df["pickup_city_geocode_success"]
        & df["pickup_city_distance_miles"].le(city_tolerance_miles)
    )

    df["delivery_coordinate_matches_city"] = (
        df["delivery_coordinates_in_range"]
        & df["delivery_city_geocode_success"]
        & df["delivery_city_distance_miles"].le(city_tolerance_miles)
    )

    df["all_coordinates_match_cities"] = (
        df["pickup_coordinate_matches_city"]
        & df["delivery_coordinate_matches_city"]
    )

    def coordinate_validation_status(
        coordinates_in_range,
        geocode_success,
        distance_miles,
        tolerance_miles,
    ):
        if not coordinates_in_range:
            return "Invalid or missing coordinate"

        if not geocode_success:
            return "Named city could not be geocoded"

        if pd.isna(distance_miles):
            return "Distance could not be calculated"

        if distance_miles <= tolerance_miles:
            return "Within tolerance"

        return "Outside tolerance"

    df["pickup_coordinate_status"] = df.apply(
        lambda row: coordinate_validation_status(
            row["pickup_coordinates_in_range"],
            row["pickup_city_geocode_success"],
            row["pickup_city_distance_miles"],
            city_tolerance_miles,
        ),
        axis=1,
    )

    df["delivery_coordinate_status"] = df.apply(
        lambda row: coordinate_validation_status(
            row["delivery_coordinates_in_range"],
            row["delivery_city_geocode_success"],
            row["delivery_city_distance_miles"],
            city_tolerance_miles,
        ),
        axis=1,
    )

    pickup_matches = int(
        df["pickup_coordinate_matches_city"].sum()
    )

    delivery_matches = int(
        df["delivery_coordinate_matches_city"].sum()
    )

    complete_matches = int(
        df["all_coordinates_match_cities"].sum()
    )

    print()
    print("\033[1mCoordinate-validation summary\033[0m")
    print("-" * 60)
    print(
        f"Pickup coordinates within tolerance:   "
        f"{pickup_matches:,} / {len(df):,}"
    )
    print(
        f"Delivery coordinates within tolerance: "
        f"{delivery_matches:,} / {len(df):,}"
    )
    print(
        f"Both coordinates within tolerance:     "
        f"{complete_matches:,} / {len(df):,}"
    )
    print(
        f"Total distance-calculation runtime:     "
        f"{pickup_distance_runtime + delivery_distance_runtime:.2f} seconds"
    )

    coordinate_validation_columns = [
        "load_id",

        "pickup",
        "pickup_lat",
        "pickup_lon",
        "pickup_city_reference_lat",
        "pickup_city_reference_lon",
        "pickup_city_distance_miles",
        "pickup_coordinate_matches_city",
        "pickup_coordinate_status",

        "delivery",
        "delivery_lat",
        "delivery_lon",
        "delivery_city_reference_lat",
        "delivery_city_reference_lon",
        "delivery_city_distance_miles",
        "delivery_coordinate_matches_city",
        "delivery_coordinate_status",

        "all_coordinates_match_cities",
    ]

    print()
    print("\033[1mFirst 20 coordinate-validation results\033[0m")
    print("-" * 60)

    display(
        df[coordinate_validation_columns].head(20)
    )   
    
    coordinate_mismatches = df.loc[
        ~df["all_coordinates_match_cities"],
        coordinate_validation_columns,
    ].copy()

    print("Rows with one or more coordinate problems:", len(coordinate_mismatches))

    coordinate_summary = pd.DataFrame(
        {
            "check": [
                "Pickup coordinates in valid numeric range",
                "Delivery coordinates in valid numeric range",
                "Pickup city geocoded successfully",
                "Delivery city geocoded successfully",
                f"Pickup within {city_tolerance_miles} miles",
                f"Delivery within {city_tolerance_miles} miles",
                "Both pickup and delivery passed",
            ],
            "passed": [
                df["pickup_coordinates_in_range"].sum(),
                df["delivery_coordinates_in_range"].sum(),
                df["pickup_city_geocode_success"].sum(),
                df["delivery_city_geocode_success"].sum(),
                df["pickup_coordinate_matches_city"].sum(),
                df["delivery_coordinate_matches_city"].sum(),
                df["all_coordinates_match_cities"].sum(),
            ],
        }
    )

    coordinate_summary["failed"] = (
        len(df) - coordinate_summary["passed"]
    )

    coordinate_summary["pass_percent"] = (
        coordinate_summary["passed"] / len(df) * 100
    ).round(2)

    display(coordinate_summary)

    return coordinate_validation_columns 


def review_city_coordinate_mapping(
    df: pd.DataFrame,
    location_type: str = "pickup",
) -> pd.DataFrame:
    city_column = location_type
    latitude_column = f"{location_type}_lat"
    longitude_column = f"{location_type}_lon"

    coordinate_variants = (
        df[
            [
                city_column,
                latitude_column,
                longitude_column,
            ]
        ]
        .drop_duplicates()
        .groupby(city_column)
        .size()
        .sort_values(ascending=False)
        .to_frame("unique_coordinate_pairs")
    )

    print(
        f"\033[1m{location_type.title()} city-to-coordinate "
        f"mapping consistency\033[0m"
    )
    print("-" * 60)
    print(f"Cities reviewed:                       {len(coordinate_variants):,}")
    print(
        f"Cities with one coordinate pair:       "
        f"{coordinate_variants['unique_coordinate_pairs'].eq(1).sum():,}"
    )
    print(
        f"Cities with multiple coordinate pairs: "
        f"{coordinate_variants['unique_coordinate_pairs'].gt(1).sum():,}"
    )

    display(coordinate_variants)

    return coordinate_variants    


def count_coordinate_pairs_by_city(data: pd.DataFrame) -> pd.DataFrame:
    """List the number of unique coordinate pairs found for each city."""

    pickup_coordinates = data[
        ["pickup", "pickup_lat", "pickup_lon"]
    ].rename(
        columns={
            "pickup": "city",
            "pickup_lat": "latitude",
            "pickup_lon": "longitude",
        }
    )

    delivery_coordinates = data[
        ["delivery", "delivery_lat", "delivery_lon"]
    ].rename(
        columns={
            "delivery": "city",
            "delivery_lat": "latitude",
            "delivery_lon": "longitude",
        }
    )

    coordinate_pairs = pd.concat(
        [pickup_coordinates, delivery_coordinates],
        ignore_index=True,
    ).dropna().drop_duplicates()

    return (
        coordinate_pairs.groupby("city")
        .size()
        .reset_index(name="coordinate_pair_count")
        .sort_values(
            ["coordinate_pair_count", "city"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )    