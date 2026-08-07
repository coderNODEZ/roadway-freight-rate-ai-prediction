# Preprocessing/preprocessing.py

import json
from pathlib import Path

import pandas as pd


def add_absolute_weight_column(data: pd.DataFrame) -> pd.DataFrame:
    """Add weight_abs while leaving the original weight column unchanged."""

    data = data.copy()
    data["weight_abs"] = pd.to_numeric(
        data["weight"],
        errors="coerce",
    ).abs()
    return data


def add_weight_status_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add Boolean columns describing the original weight value."""

    data = data.copy()
    weight = pd.to_numeric(data["weight"], errors="coerce")
    data["weight_was_zero"] = weight.eq(0)
    data["weight_was_missing"] = weight.isna()
    data["weight_was_negative"] = weight.lt(0)
    return data


def add_equipment_median_imputed_weight(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing weight_abs values with the equipment median."""

    data = data.copy()
    equipment_median = data.groupby("equipment")["weight_abs"].transform(
        "median"
    )
    data["weight_imputed_equipment"] = data["weight_abs"].fillna(
        equipment_median
    )
    return data


# def add_equipment_route_median_imputed_weight(
#     data: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Fill missing weight_abs using the equipment and route median."""

#     data = data.copy()
#     route_median = data.groupby(
#         ["equipment", "pickup", "delivery"]
#     )["weight_abs"].transform("median")
#     data["weight_imputed_equipment_route"] = data["weight_abs"].fillna(
#         route_median
#     )
#     return data

def add_equipment_route_median_imputed_weight(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Impute weight using route, equipment, and overall median fallbacks."""

    data = data.copy()

    equipment_route_median = data.groupby(
        ["equipment", "pickup", "delivery"]
    )["weight_abs"].transform("median")

    equipment_median = data.groupby(
        "equipment"
    )["weight_abs"].transform("median")

    overall_median = data["weight_abs"].median()

    data["weight_imputed_equipment_route"] = (
        data["weight_abs"]
        .fillna(equipment_route_median)
        .fillna(equipment_median)
        .fillna(overall_median)
    )

    return data    


def add_missing_coordinate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add absent pickup and delivery coordinate columns from the city map."""

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]
    missing_columns = [
        column for column in coordinate_columns if column not in data.columns
    ]

    if not missing_columns:
        return data

    data = data.copy()

    coordinates_path = Path(__file__).with_name("city_coordinates.json")
    with coordinates_path.open("r", encoding="utf-8") as coordinates_file:
        city_coordinates = json.load(coordinates_file)

    latitude_by_city = {
        city: coordinates["latitude"]
        for city, coordinates in city_coordinates.items()
    }
    longitude_by_city = {
        city: coordinates["longitude"]
        for city, coordinates in city_coordinates.items()
    }

    if "pickup_lat" in missing_columns:
        data["pickup_lat"] = data["pickup"].map(latitude_by_city)
    if "pickup_lon" in missing_columns:
        data["pickup_lon"] = data["pickup"].map(longitude_by_city)
    if "delivery_lat" in missing_columns:
        data["delivery_lat"] = data["delivery"].map(latitude_by_city)
    if "delivery_lon" in missing_columns:
        data["delivery_lon"] = data["delivery"].map(longitude_by_city)

    return data


def add_missing_load_id_column(data: pd.DataFrame) -> pd.DataFrame:
    """Add sequential load IDs when the load_id column is absent."""

    if "load_id" in data.columns:
        return data

    data = data.copy()
    data["load_id"] = [
        f"TR-{load_number:06d}"
        for load_number in range(60000, 60000 + len(data))
    ]
    return data


def remove_original_weight_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove the original weight column when it is present."""

    if "weight" in data.columns:
        data = data.drop(columns=["weight"])
    if "weight_abs":        
        data = data.drop(columns=["weight_abs"])

    return data


# def preprocess_freight_data_controller(
#     data: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Controller for the freight-data preprocessing functions."""

#     data = add_missing_load_id_column(data)
#     data = add_missing_coordinate_columns(data)

#     data = add_weight_status_columns(data)
#     data = add_equipment_median_imputed_weight(data)
#     data = add_equipment_route_median_imputed_weight(data)
#     data = add_absolute_weight_column(data)
#     data = remove_leakage_columns(data)

#     data = remove_original_weight_column(data)

#     return data    


def remove_leakage_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Remove quote_signal and posted_rate_per_mile when present."""

    return data.drop(
        columns=[
            "quote_signal_difference"
            "posted_rate_per_mile",
        ],
        errors="ignore",
    )    


def preprocess_freight_data_controller(data: pd.DataFrame) -> pd.DataFrame:
    """Controller for the freight-data preprocessing functions."""

    data = add_missing_load_id_column(data)
    data = add_missing_coordinate_columns(data)
    
    data = add_weight_status_columns(data)
    data = add_absolute_weight_column(data) 
    data = add_equipment_median_imputed_weight(data)
    
    data = add_equipment_route_median_imputed_weight(data)
      
    data = remove_original_weight_columns(data)
    data = remove_leakage_columns(data)
    return data

