import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "Data" / "train-test.csv"
OUTPUT_PATH = PROJECT_ROOT / "Preprocessing" / "city_coordinates.json"


def extract_city_coordinates(data: pd.DataFrame) -> dict:
    """Extract one latitude/longitude pair for every city."""

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

    coordinates = (
        pd.concat(
            [pickup_coordinates, delivery_coordinates],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
    )

    coordinate_counts = coordinates.groupby("city").size()
    inconsistent_cities = coordinate_counts[coordinate_counts.gt(1)]

    if not inconsistent_cities.empty:
        raise ValueError(
            "Multiple coordinate pairs found for cities: "
            f"{inconsistent_cities.index.tolist()}"
        )

    coordinates = coordinates.sort_values("city")

    return {
        row.city: {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in coordinates.itertuples(index=False)
    }


def create_city_coordinates_json() -> None:
    """Read the training CSV and write the city-coordinate JSON file."""

    data = pd.read_csv(INPUT_PATH)
    city_coordinates = extract_city_coordinates(data)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(
            city_coordinates,
            output_file,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved {len(city_coordinates):,} cities to {OUTPUT_PATH}")


if __name__ == "__main__":
    create_city_coordinates_json()

