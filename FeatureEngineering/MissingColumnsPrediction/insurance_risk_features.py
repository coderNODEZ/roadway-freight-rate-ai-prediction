import pandas as pd


# Published state commercial-truck premium benchmarks are used only as a
# relative risk proxy. They are not predicted premiums or loss ratios.
STATE_COMMERCIAL_TRUCK_PREMIUM_BENCHMARK = {
    "AK": 6915,
    "AL": 10284,
    "AR": 10973,
    "AZ": 6102,
    "CA": 11834,
    "CO": 7294,
    "CT": 16946,
    "DE": 17351,
    "FL": 12872,
    "GA": 15200,
    "IA": 5615,
    "ID": 6887,
    "IL": 7704,
    "IN": 8430,
    "KS": 6645,
    "KY": 11555,
    "LA": 19736,
    "MA": 5447,
    "MD": 11112,
    "ME": 9535,
    "MI": 8910,
    "MN": 9669,
    "MO": 7646,
    "MS": 3552,
    "MT": 6501,
    "NC": 7450,
    "ND": 6456,
    "NE": 6259,
    "NH": 6817,
    "NJ": 20763,
    "NM": 7298,
    "NV": 10681,
    "NY": 16949,
    "OH": 7094,
    "OK": 9376,
    "OR": 8484,
    "PA": 7536,
    "RI": 14046,
    "SC": 9390,
    "SD": 6689,
    "TN": 9592,
    "TX": 10533,
    "UT": 9121,
    "VA": 9957,
    "VT": 6937,
    "WA": 8484,
    "WI": 6714,
    "WV": 11687,
    "WY": 4927,
}


def add_insurance_risk_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add normalized endpoint and route truck-insurance risk proxies."""

    data = data.copy()
    minimum = min(STATE_COMMERCIAL_TRUCK_PREMIUM_BENCHMARK.values())
    maximum = max(STATE_COMMERCIAL_TRUCK_PREMIUM_BENCHMARK.values())
    state_risk_index = {
        state: (premium - minimum) / (maximum - minimum)
        for state, premium in STATE_COMMERCIAL_TRUCK_PREMIUM_BENCHMARK.items()
    }
    default_risk = pd.Series(state_risk_index.values()).median()

    data["pickup_insurance_risk"] = (
        data["pickup_state"].map(state_risk_index).fillna(default_risk)
    )
    data["delivery_insurance_risk"] = (
        data["delivery_state"].map(state_risk_index).fillna(default_risk)
    )

    endpoint_mean = (
        data["pickup_insurance_risk"]
        + data["delivery_insurance_risk"]
    ) / 2

    if "route_state_insurance_risk_mean" in data.columns:
        route_state_mean = pd.to_numeric(
            data["route_state_insurance_risk_mean"],
            errors="coerce",
        ).fillna(endpoint_mean)
        data["route_insurance_risk"] = (
            0.25 * data["pickup_insurance_risk"]
            + 0.25 * data["delivery_insurance_risk"]
            + 0.50 * route_state_mean
        )
    else:
        data["route_insurance_risk"] = endpoint_mean

    data["route_max_insurance_risk"] = data[
        ["pickup_insurance_risk", "delivery_insurance_risk"]
    ].max(axis=1)
    return data

