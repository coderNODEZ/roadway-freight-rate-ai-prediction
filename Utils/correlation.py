# from __future__ import annotations

# import warnings

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from matplotlib.patches import Patch
# from scipy.stats import chi2_contingency, pearsonr, spearmanr


# def _is_date_like(series: pd.Series) -> bool:
#     """Return True when a column is already datetime or is named like a date."""
#     return (
#         pd.api.types.is_datetime64_any_dtype(series)
#         or "date" in str(series.name).lower()
#     )


# def _date_to_ordinal(series: pd.Series) -> pd.Series:
#     """Convert dates to elapsed days from the earliest valid date."""
#     dates = pd.to_datetime(series, errors="coerce")

#     if not dates.notna().any():
#         return pd.Series(np.nan, index=series.index, dtype=float)

#     return (dates - dates.min()).dt.total_seconds() / 86_400


# def _numeric_series(series: pd.Series) -> pd.Series:
#     """Convert numeric or date-like data to numeric values."""
#     if _is_date_like(series):
#         return _date_to_ordinal(series)

#     return pd.to_numeric(series, errors="coerce")


# def _correlation_ratio(
#     categories: pd.Series,
#     numeric_values: pd.Series,
# ) -> float:
#     """
#     Calculate correlation ratio eta between categorical and numeric data.

#     Values range from 0 to 1.
#     """
#     data = pd.DataFrame({
#         "category": categories,
#         "value": pd.to_numeric(numeric_values, errors="coerce"),
#     }).dropna()

#     if (
#         data.empty
#         or data["category"].nunique() < 2
#         or data["value"].nunique() < 2
#     ):
#         return np.nan

#     overall_mean = data["value"].mean()

#     between_group_variation = sum(
#         len(group) * (group["value"].mean() - overall_mean) ** 2
#         for _, group in data.groupby("category", observed=True)
#     )

#     total_variation = (
#         (data["value"] - overall_mean) ** 2
#     ).sum()

#     if total_variation == 0:
#         return np.nan

#     return float(
#         np.sqrt(between_group_variation / total_variation)
#     )


# def _corrected_cramers_v(
#     first: pd.Series,
#     second: pd.Series,
# ) -> tuple[float, float]:
#     """
#     Calculate bias-corrected Cramer's V and its chi-square p-value.

#     Values range from 0 to 1.
#     """
#     data = pd.DataFrame({
#         "first": first,
#         "second": second,
#     }).dropna()

#     contingency_table = pd.crosstab(
#         data["first"],
#         data["second"],
#     )

#     rows, columns = contingency_table.shape
#     observations = contingency_table.to_numpy().sum()

#     if rows < 2 or columns < 2 or observations <= 1:
#         return np.nan, np.nan

#     try:
#         chi_squared, p_value, _, _ = chi2_contingency(
#             contingency_table,
#             correction=False,
#         )
#     except ValueError:
#         return np.nan, np.nan

#     phi_squared = chi_squared / observations

#     corrected_phi_squared = max(
#         0,
#         phi_squared
#         - ((columns - 1) * (rows - 1)) / (observations - 1),
#     )

#     corrected_rows = rows - ((rows - 1) ** 2) / (observations - 1)
#     corrected_columns = (
#         columns
#         - ((columns - 1) ** 2) / (observations - 1)
#     )

#     denominator = min(
#         corrected_columns - 1,
#         corrected_rows - 1,
#     )

#     if denominator <= 0:
#         return np.nan, p_value

#     cramers_v = np.sqrt(corrected_phi_squared / denominator)

#     return float(cramers_v), float(p_value)


# def _column_kind(series: pd.Series) -> str:
#     """Classify a column as numeric or categorical."""
#     if pd.api.types.is_numeric_dtype(series):
#         return "numeric"

#     if _is_date_like(series):
#         converted = pd.to_datetime(series, errors="coerce")

#         if converted.notna().mean() >= 0.80:
#             return "numeric"

#     return "categorical"


# def analyze_column_associations(
#     df: pd.DataFrame,
#     target_column: str,
#     excluded_columns: list[str] | None = None,
#     graph: bool = True,
#     top_n: int | None = 20,
#     figure_size: tuple[float, float] = (8, 5.6),
# ) -> pd.DataFrame:
#     """
#     Compare one target column with the other DataFrame columns.

#     The function uses:
#       - Pearson and Spearman: numeric vs numeric
#       - Correlation ratio eta: numeric vs categorical
#       - Corrected Cramer's V: categorical vs categorical

#     Parameters
#     ----------
#     df:
#         Input DataFrame.

#     target_column:
#         Column to compare against the other columns.

#     excluded_columns:
#         Optional columns to exclude from the analysis.

#     graph:
#         Display a lightweight ranked association chart.

#     top_n:
#         Maximum number of associations shown in the graph.
#         Use None to show all associations.

#     figure_size:
#         Matplotlib graph size.

#     Returns
#     -------
#     pd.DataFrame
#         Ranked association results.
#     """
#     if target_column not in df.columns:
#         raise ValueError(
#             f"Target column {target_column!r} was not found."
#         )

#     exclusions = set(excluded_columns or [])
#     exclusions.add(target_column)

#     target = df[target_column]
#     target_kind = _column_kind(target)

#     results = []

#     for feature_column in df.columns:
#         if feature_column in exclusions:
#             continue

#         feature = df[feature_column]
#         feature_kind = _column_kind(feature)

#         result = {
#             "target": target_column,
#             "target_type": target_kind,
#             "column": feature_column,
#             "column_type": feature_kind,
#             "method": None,
#             "observations": 0,
#             "association": np.nan,
#             "signed_association": np.nan,
#             "pearson_r": np.nan,
#             "pearson_p_value": np.nan,
#             "spearman_r": np.nan,
#             "spearman_p_value": np.nan,
#             "p_value": np.nan,
#         }

#         # Numeric target versus numeric feature
#         if target_kind == "numeric" and feature_kind == "numeric":
#             paired = pd.DataFrame({
#                 "target": _numeric_series(target),
#                 "feature": _numeric_series(feature),
#             }).dropna()

#             result["observations"] = len(paired)

#             if (
#                 len(paired) >= 3
#                 and paired["target"].nunique() >= 2
#                 and paired["feature"].nunique() >= 2
#             ):
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")

#                     pearson_r, pearson_p = pearsonr(
#                         paired["feature"],
#                         paired["target"],
#                     )

#                     spearman_r, spearman_p = spearmanr(
#                         paired["feature"],
#                         paired["target"],
#                     )

#                 result.update({
#                     "method": "Spearman",
#                     "association": abs(spearman_r),
#                     "signed_association": spearman_r,
#                     "pearson_r": pearson_r,
#                     "pearson_p_value": pearson_p,
#                     "spearman_r": spearman_r,
#                     "spearman_p_value": spearman_p,
#                     "p_value": spearman_p,
#                 })

#         # Numeric target versus categorical feature
#         elif target_kind == "numeric":
#             paired = pd.DataFrame({
#                 "category": feature,
#                 "numeric": _numeric_series(target),
#             }).dropna()

#             eta = _correlation_ratio(
#                 paired["category"],
#                 paired["numeric"],
#             )

#             result.update({
#                 "method": "Correlation ratio eta",
#                 "observations": len(paired),
#                 "association": eta,
#                 "signed_association": eta,
#             })

#         # Categorical target versus numeric feature
#         elif feature_kind == "numeric":
#             paired = pd.DataFrame({
#                 "category": target,
#                 "numeric": _numeric_series(feature),
#             }).dropna()

#             eta = _correlation_ratio(
#                 paired["category"],
#                 paired["numeric"],
#             )

#             result.update({
#                 "method": "Correlation ratio eta",
#                 "observations": len(paired),
#                 "association": eta,
#                 "signed_association": eta,
#             })

#         # Categorical target versus categorical feature
#         else:
#             paired = pd.DataFrame({
#                 "target": target,
#                 "feature": feature,
#             }).dropna()

#             cramers_v, p_value = _corrected_cramers_v(
#                 paired["target"],
#                 paired["feature"],
#             )

#             result.update({
#                 "method": "Cramer's V",
#                 "observations": len(paired),
#                 "association": cramers_v,
#                 "signed_association": cramers_v,
#                 "p_value": p_value,
#             })

#         results.append(result)

#     results_df = pd.DataFrame(results)

#     results_df = results_df.sort_values(
#         "association",
#         ascending=False,
#         na_position="last",
#     ).reset_index(drop=True)

#     if graph and not results_df.empty:
#         graph_data = results_df.dropna(
#             subset=["association"]
#         ).copy()

#         if top_n is not None:
#             graph_data = graph_data.head(top_n)

#         graph_data = graph_data.sort_values(
#             "association",
#             ascending=True,
#         )

#         colors = [
#             (
#                 "#C0504D"
#                 if row["method"] == "Spearman"
#                 and row["signed_association"] < 0
#                 else "#5B9BD5"
#             )
#             for _, row in graph_data.iterrows()
#         ]

#         _, axis = plt.subplots(figsize=figure_size)

#         axis.barh(
#             graph_data["column"],
#             graph_data["association"],
#             color=colors,
#             edgecolor="white",
#         )

#         axis.set_title(
#             f"Association with {target_column}",
#             fontsize=14,
#             fontweight="bold",
#         )
#         axis.set_xlabel(
#             "Association strength (absolute value)"
#         )
#         axis.set_ylabel("")
#         axis.set_xlim(0, 1)
#         axis.grid(
#             axis="x",
#             alpha=0.25,
#             linestyle="--",
#         )

#         axis.legend(
#             handles=[
#                 Patch(
#                     facecolor="#5B9BD5",
#                     label="Positive or directionless association",
#                 ),
#                 Patch(
#                     facecolor="#C0504D",
#                     label="Negative Spearman correlation",
#                 ),
#             ],
#             loc="lower right",
#             fontsize=8,
#             frameon=True,
#         )

#         for position, value in enumerate(
#             graph_data["association"]
#         ):
#             axis.text(
#                 min(value + 0.01, 0.97),
#                 position,
#                 f"{value:.3f}",
#                 va="center",
#                 fontsize=9,
#             )

#         plt.tight_layout()
#         plt.show()

#     return results_df


from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, pearsonr, spearmanr


def _is_date_like(series: pd.Series) -> bool:
    """Return True when a column is already datetime or is named like a date."""
    return (
        pd.api.types.is_datetime64_any_dtype(series)
        or "date" in str(series.name).lower()
    )


def _date_to_ordinal(series: pd.Series) -> pd.Series:
    """Convert dates to elapsed days from the earliest valid date."""
    dates = pd.to_datetime(series, errors="coerce")

    if not dates.notna().any():
        return pd.Series(np.nan, index=series.index, dtype=float)

    return (dates - dates.min()).dt.total_seconds() / 86_400


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert numeric or date-like data to numeric values."""
    if _is_date_like(series):
        return _date_to_ordinal(series)

    return pd.to_numeric(series, errors="coerce")


def _correlation_ratio(
    categories: pd.Series,
    numeric_values: pd.Series,
) -> float:
    """
    Calculate correlation ratio eta between categorical and numeric data.

    Values range from 0 to 1.
    """
    data = pd.DataFrame({
        "category": categories,
        "value": pd.to_numeric(numeric_values, errors="coerce"),
    }).dropna()

    if (
        data.empty
        or data["category"].nunique() < 2
        or data["value"].nunique() < 2
    ):
        return np.nan

    overall_mean = data["value"].mean()

    between_group_variation = sum(
        len(group) * (group["value"].mean() - overall_mean) ** 2
        for _, group in data.groupby("category", observed=True)
    )

    total_variation = (
        (data["value"] - overall_mean) ** 2
    ).sum()

    if total_variation == 0:
        return np.nan

    return float(
        np.sqrt(between_group_variation / total_variation)
    )


def _corrected_cramers_v(
    first: pd.Series,
    second: pd.Series,
) -> tuple[float, float]:
    """
    Calculate bias-corrected Cramer's V and its chi-square p-value.

    Values range from 0 to 1.
    """
    data = pd.DataFrame({
        "first": first,
        "second": second,
    }).dropna()

    contingency_table = pd.crosstab(
        data["first"],
        data["second"],
    )

    rows, columns = contingency_table.shape
    observations = contingency_table.to_numpy().sum()

    if rows < 2 or columns < 2 or observations <= 1:
        return np.nan, np.nan

    try:
        chi_squared, p_value, _, _ = chi2_contingency(
            contingency_table,
            correction=False,
        )
    except ValueError:
        return np.nan, np.nan

    phi_squared = chi_squared / observations

    corrected_phi_squared = max(
        0,
        phi_squared
        - ((columns - 1) * (rows - 1)) / (observations - 1),
    )

    corrected_rows = rows - ((rows - 1) ** 2) / (observations - 1)
    corrected_columns = (
        columns
        - ((columns - 1) ** 2) / (observations - 1)
    )

    denominator = min(
        corrected_columns - 1,
        corrected_rows - 1,
    )

    if denominator <= 0:
        return np.nan, p_value

    cramers_v = np.sqrt(corrected_phi_squared / denominator)

    return float(cramers_v), float(p_value)


def _column_kind(series: pd.Series) -> str:
    """Classify a column as numeric or categorical."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if _is_date_like(series):
        converted = pd.to_datetime(series, errors="coerce")

        if converted.notna().mean() >= 0.80:
            return "numeric"

    return "categorical"


def analyze_column_associations(
    df: pd.DataFrame,
    target_column: str,
    excluded_columns: list[str] | None = None,
    graph: bool = True,
    top_n: int | None = None,
    figure_size: tuple[float, float] = (8, 5.6),
    three_column_graph: bool | None = None,
) -> pd.DataFrame:
    """
    Compare one target column with the other DataFrame columns.

    The function uses:
      - Pearson and Spearman: numeric vs numeric
      - Correlation ratio eta: numeric vs categorical
      - Corrected Cramer's V: categorical vs categorical

    Parameters
    ----------
    df:
        Input DataFrame.

    target_column:
        Column to compare against the other columns.

    excluded_columns:
        Optional columns to exclude from the analysis.

    graph:
        Display a lightweight ranked association chart.

    top_n:
        Maximum number of associations shown in the graph.
        Use None to show all associations.

    figure_size:
        Matplotlib graph size for the standard single chart.

    three_column_graph:
        When True, split the associations across three compact charts.
        When False, always use one chart. When None, automatically use
        three charts when more than 20 associations are displayed.

    Returns
    -------
    pd.DataFrame
        Ranked association results.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found."
        )

    exclusions = set(excluded_columns or [])
    exclusions.add(target_column)

    target = df[target_column]
    target_kind = _column_kind(target)

    results = []

    for feature_column in df.columns:
        if feature_column in exclusions:
            continue

        feature = df[feature_column]
        feature_kind = _column_kind(feature)

        result = {
            "target": target_column,
            "target_type": target_kind,
            "column": feature_column,
            "column_type": feature_kind,
            "method": None,
            "observations": 0,
            "association": np.nan,
            "signed_association": np.nan,
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
            "spearman_r": np.nan,
            "spearman_p_value": np.nan,
            "p_value": np.nan,
        }

        # Numeric target versus numeric feature
        if target_kind == "numeric" and feature_kind == "numeric":
            paired = pd.DataFrame({
                "target": _numeric_series(target),
                "feature": _numeric_series(feature),
            }).dropna()

            result["observations"] = len(paired)

            if (
                len(paired) >= 3
                and paired["target"].nunique() >= 2
                and paired["feature"].nunique() >= 2
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    pearson_r, pearson_p = pearsonr(
                        paired["feature"],
                        paired["target"],
                    )

                    spearman_r, spearman_p = spearmanr(
                        paired["feature"],
                        paired["target"],
                    )

                result.update({
                    "method": "Spearman",
                    "association": abs(spearman_r),
                    "signed_association": spearman_r,
                    "pearson_r": pearson_r,
                    "pearson_p_value": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p_value": spearman_p,
                    "p_value": spearman_p,
                })

        # Numeric target versus categorical feature
        elif target_kind == "numeric":
            paired = pd.DataFrame({
                "category": feature,
                "numeric": _numeric_series(target),
            }).dropna()

            eta = _correlation_ratio(
                paired["category"],
                paired["numeric"],
            )

            result.update({
                "method": "Correlation ratio eta",
                "observations": len(paired),
                "association": eta,
                "signed_association": eta,
            })

        # Categorical target versus numeric feature
        elif feature_kind == "numeric":
            paired = pd.DataFrame({
                "category": target,
                "numeric": _numeric_series(feature),
            }).dropna()

            eta = _correlation_ratio(
                paired["category"],
                paired["numeric"],
            )

            result.update({
                "method": "Correlation ratio eta",
                "observations": len(paired),
                "association": eta,
                "signed_association": eta,
            })

        # Categorical target versus categorical feature
        else:
            paired = pd.DataFrame({
                "target": target,
                "feature": feature,
            }).dropna()

            cramers_v, p_value = _corrected_cramers_v(
                paired["target"],
                paired["feature"],
            )

            result.update({
                "method": "Cramer's V",
                "observations": len(paired),
                "association": cramers_v,
                "signed_association": cramers_v,
                "p_value": p_value,
            })

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "association",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if graph and not results_df.empty:
        graph_data = results_df.dropna(
            subset=["association"]
        ).copy()

        if top_n is not None:
            graph_data = graph_data.head(top_n)

        use_three_columns = (
            len(graph_data) > 20
            if three_column_graph is None
            else three_column_graph
        )

        legend_handles = [
            Patch(
                facecolor="#5B9BD5",
                label="Positive or directionless association",
            ),
            Patch(
                facecolor="#C0504D",
                label="Negative Spearman correlation",
            ),
        ]

        if use_three_columns:
            # Keep the strongest associations in the first chart while
            # dividing the ranked results into three similarly sized groups.
            group_indices = np.array_split(
                np.arange(len(graph_data)),
                3,
            )
            chart_groups = [
                graph_data.iloc[index_group]
                for index_group in group_indices
            ]
            rows_per_chart = max(len(group) for group in chart_groups)
            compact_height = max(4.2, rows_per_chart * 0.30)

            figure, axes = plt.subplots(
                1,
                3,
                figsize=(12, compact_height),
                sharex=True,
            )

            for chart_number, (axis, chart_data) in enumerate(
                zip(axes, chart_groups),
                start=1,
            ):
                chart_data = chart_data.sort_values(
                    "association",
                    ascending=True,
                )

                colors = [
                    (
                        "#C0504D"
                        if row["method"] == "Spearman"
                        and row["signed_association"] < 0
                        else "#5B9BD5"
                    )
                    for _, row in chart_data.iterrows()
                ]

                axis.barh(
                    chart_data["column"],
                    chart_data["association"],
                    color=colors,
                    edgecolor="white",
                )
                axis.set_title(
                    f"{target_column}\nGroup {chart_number}",
                    fontsize=9,
                    fontweight="bold",
                )
                axis.set_xlim(0, 1)
                axis.tick_params(axis="both", labelsize=7)
                axis.grid(axis="x", alpha=0.25, linestyle="--")

                for position, value in enumerate(
                    chart_data["association"]
                ):
                    axis.text(
                        min(value + 0.01, 0.96),
                        position,
                        f"{value:.2f}",
                        va="center",
                        fontsize=6,
                    )

            figure.suptitle(
                f"Association with {target_column}",
                fontsize=12,
                fontweight="bold",
            )
            figure.supxlabel(
                "Association strength (absolute value)",
                fontsize=9,
            )
            figure.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=2,
                fontsize=7,
                frameon=True,
                bbox_to_anchor=(0.5, -0.02),
            )
            plt.tight_layout(rect=(0, 0.06, 1, 0.94))

        else:
            graph_data = graph_data.sort_values(
                "association",
                ascending=True,
            )

            colors = [
                (
                    "#C0504D"
                    if row["method"] == "Spearman"
                    and row["signed_association"] < 0
                    else "#5B9BD5"
                )
                for _, row in graph_data.iterrows()
            ]

            _, axis = plt.subplots(figsize=figure_size)
            axis.barh(
                graph_data["column"],
                graph_data["association"],
                color=colors,
                edgecolor="white",
            )
            axis.set_title(
                f"Association with {target_column}",
                fontsize=14,
                fontweight="bold",
            )
            axis.set_xlabel("Association strength (absolute value)")
            axis.set_ylabel("")
            axis.set_xlim(0, 1)
            axis.grid(axis="x", alpha=0.25, linestyle="--")
            axis.legend(
                handles=legend_handles,
                loc="lower right",
                fontsize=8,
                frameon=True,
            )

            for position, value in enumerate(
                graph_data["association"]
            ):
                axis.text(
                    min(value + 0.01, 0.97),
                    position,
                    f"{value:.3f}",
                    va="center",
                    fontsize=9,
                )

            plt.tight_layout()

        plt.show()

    return results_df

