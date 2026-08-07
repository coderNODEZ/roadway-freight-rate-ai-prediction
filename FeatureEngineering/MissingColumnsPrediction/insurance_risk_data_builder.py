"""Build the six sourced CSV inputs required by insurance risk V2.

The builder accepts official source extracts as local files or direct download
URLs.  It standardizes their differing column names, calculates rates and
component scores when raw counts are supplied, and records publication year
and source URL in every output row.  It never generates example observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_MANIFEST = MODULE_DIRECTORY / "insurance_risk_sources.json"
DEFAULT_OUTPUT_DIRECTORY = MODULE_DIRECTORY / "insurance_risk_data"

OUTPUT_SCHEMAS = {
    "state_commercial_auto": [
        "state", "commercial_auto_loss_ratio", "year", "source_url"
    ],
    "city_crashes": ["city", "state", "truck_crash_rate", "year", "source_url"],
    "city_vehicle_theft": [
        "city", "state", "vehicle_theft_rate", "year", "source_url"
    ],
    "city_congestion": [
        "city", "state", "congestion_score", "year", "source_url"
    ],
    "city_weather_risk": [
        "city", "state", "severe_weather_risk", "year", "source_url"
    ],
    "state_litigation_risk": [
        "state", "litigation_risk", "year", "source_url"
    ],
}


def build_insurance_risk_data(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    """Download, standardize, validate, and write all six V2 source files."""

    manifest_path = Path(manifest_path)
    output_directory = Path(output_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    builders = {
        "state_commercial_auto": _standardize_commercial_auto,
        "city_crashes": _standardize_crashes,
        "city_vehicle_theft": _standardize_vehicle_theft,
        "city_congestion": _standardize_congestion,
        "city_weather_risk": _standardize_weather,
        "state_litigation_risk": _standardize_litigation,
    }
    written = {}

    for source_name, builder in builders.items():
        if source_name not in manifest:
            raise ValueError(f"Manifest is missing source: {source_name}")
        configuration = manifest[source_name]
        raw_path = _obtain_source(source_name, configuration, raw_directory)
        raw = _read_table(raw_path, configuration)
        standardized = builder(raw)
        standardized["year"] = _required_year(configuration, source_name)
        standardized["source_url"] = _required_source_url(
            configuration, source_name
        )
        standardized = _validate_output(source_name, standardized)
        output_path = output_directory / f"{source_name}.csv"
        standardized.to_csv(output_path, index=False)
        written[source_name] = output_path

    return written


def _obtain_source(name: str, config: dict, raw_directory: Path) -> Path:
    local_path = config.get("local_path")
    if local_path:
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} file does not exist: {path}")
        return path

    download_url = config.get("download_url")
    if not download_url:
        raise ValueError(f"{name} requires local_path or download_url.")
    suffix = Path(urlparse(download_url).path).suffix or ".csv"
    path = raw_directory / f"{name}{suffix}"
    if path.is_file() and path.stat().st_size:
        return path
    try:
        import requests
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "URL downloads require requests. Install it with: pip install requests"
        ) from error
    response = requests.get(download_url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _read_table(path: Path, config: dict) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(
            path,
            sheet_name=config.get("sheet_name", 0),
            skiprows=config.get("skiprows"),
        )
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, skiprows=config.get("skiprows"))
    raise ValueError(f"Unsupported source format: {path}")


def _standardize_commercial_auto(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "state": ["state", "state_abbreviation", "st"],
        "commercial_auto_loss_ratio": [
            "commercial_auto_loss_ratio", "loss_ratio", "total_industry_loss_ratio"
        ],
    })
    return frame[["state", "commercial_auto_loss_ratio"]]


def _standardize_crashes(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "city": ["city", "city_name", "place"],
        "state": ["state", "state_abbreviation", "st"],
        "truck_crash_rate": ["truck_crash_rate", "crash_rate"],
        "truck_crashes": ["truck_crashes", "truck_fatal_crashes", "crashes"],
        "exposure": ["exposure", "population", "truck_vmt"],
    }, required={"city", "state"})
    if "truck_crash_rate" not in frame:
        _require(frame, {"truck_crashes", "exposure"}, "city_crashes")
        frame["truck_crash_rate"] = _rate(frame, "truck_crashes", "exposure")
    return frame[["city", "state", "truck_crash_rate"]]


def _standardize_vehicle_theft(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "city": ["city", "city_name", "place"],
        "state": ["state", "state_abbreviation", "st"],
        "vehicle_theft_rate": ["vehicle_theft_rate", "motor_vehicle_theft_rate"],
        "vehicle_thefts": ["vehicle_thefts", "motor_vehicle_thefts", "thefts"],
        "population": ["population", "population_covered"],
        "reporting_coverage": ["reporting_coverage", "coverage"],
    }, required={"city", "state"})
    if "reporting_coverage" in frame:
        coverage = pd.to_numeric(frame["reporting_coverage"], errors="coerce")
        frame = frame.loc[coverage.ge(0.90)].copy()
    if "vehicle_theft_rate" not in frame:
        _require(frame, {"vehicle_thefts", "population"}, "city_vehicle_theft")
        frame["vehicle_theft_rate"] = _rate(frame, "vehicle_thefts", "population")
    return frame[["city", "state", "vehicle_theft_rate"]]


def _standardize_congestion(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "city": ["city", "urban_area", "urban area", "area"],
        "state": ["state", "state_abbreviation", "st"],
        "congestion_score": ["congestion_score", "congestion_index"],
        "travel_time_index": ["travel_time_index", "travel time index"],
        "planning_time_index": ["planning_time_index", "planning time index"],
        "congested_hours": ["congested_hours", "hours_of_delay", "annual_delay_hours"],
    }, required={"city", "state"})
    if "congestion_score" not in frame:
        metrics = [
            column for column in
            ["travel_time_index", "planning_time_index", "congested_hours"]
            if column in frame
        ]
        if not metrics:
            raise ValueError("city_congestion contains no supported congestion metric.")
        indexed = pd.concat(
            [_percentile_index(frame[column]) for column in metrics], axis=1
        )
        frame["congestion_score"] = indexed.mean(axis=1)
    return frame[["city", "state", "congestion_score"]]


def _standardize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "city": ["city", "city_name", "place"],
        "state": ["state", "state_abbreviation", "st"],
        "severe_weather_risk": ["severe_weather_risk", "weather_risk"],
        "winter_weather_risk": ["winter_weather_risk", "winter_storm_risk"],
        "flood_risk": ["flood_risk"],
        "hurricane_risk": ["hurricane_risk"],
        "tornado_risk": ["tornado_risk"],
        "hail_risk": ["hail_risk"],
        "strong_wind_risk": ["strong_wind_risk", "wind_risk"],
        "wildfire_risk": ["wildfire_risk"],
    }, required={"city", "state"})
    if "severe_weather_risk" not in frame:
        weights = {
            "winter_weather_risk": 0.30, "flood_risk": 0.20,
            "hurricane_risk": 0.15, "tornado_risk": 0.15,
            "hail_risk": 0.05, "strong_wind_risk": 0.10,
            "wildfire_risk": 0.05,
        }
        available = {key: value for key, value in weights.items() if key in frame}
        if not available:
            raise ValueError("city_weather_risk contains no supported hazard metric.")
        total = sum(available.values())
        frame["severe_weather_risk"] = sum(
            (weight / total) * _percentile_index(frame[column])
            for column, weight in available.items()
        )
    return frame[["city", "state", "severe_weather_risk"]]


def _standardize_litigation(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _rename(raw, {
        "state": ["state", "state_abbreviation", "st"],
        "litigation_risk": ["litigation_risk", "litigation_risk_score", "civil_case_rate"],
    })
    return frame[["state", "litigation_risk"]]


def _rename(
    frame: pd.DataFrame,
    aliases: dict[str, list[str]],
    required: set[str] | None = None,
) -> pd.DataFrame:
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    rename = {}
    for target, options in aliases.items():
        for option in options:
            if option.casefold() in normalized:
                rename[normalized[option.casefold()]] = target
                break
    result = frame.rename(columns=rename).copy()
    _require(result, required or {next(iter(aliases)), list(aliases)[1]}, "source")
    return result


def _require(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing supported columns: {missing}")


def _rate(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    count = pd.to_numeric(frame[numerator], errors="coerce")
    exposure = pd.to_numeric(frame[denominator], errors="coerce")
    return count.div(exposure.where(exposure.gt(0))).mul(100_000)


def _percentile_index(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(pct=True).mul(100)


def _required_year(config: dict, name: str) -> int:
    try:
        return int(config["year"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} requires a numeric publication year.") from error


def _required_source_url(config: dict, name: str) -> str:
    value = str(config.get("source_url", "")).strip()
    if not value.startswith(("https://", "http://")):
        raise ValueError(f"{name} requires an http(s) source_url.")
    return value


def _validate_output(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = OUTPUT_SCHEMAS[name]
    _require(frame, set(columns), name)
    result = frame.loc[:, columns].copy()
    result["state"] = result["state"].astype("string").str.upper().str.strip()
    value_column = columns[-3]
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["state", value_column]
    )
    keys = ["state"] if "city" not in result else ["city", "state"]
    result = result.drop_duplicates(keys, keep="last")
    if result.empty:
        raise ValueError(f"{name} produced no usable real observations.")
    return result

