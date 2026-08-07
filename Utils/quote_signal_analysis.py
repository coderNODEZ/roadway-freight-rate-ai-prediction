# df["posted_rate_per_mile"] = (
#     df["posted_rate"] / df["distance"]
# )

# display(
#     df[
#         [
#             "load_id",
#             "distance",
#             "posted_rate",
#             "posted_rate_per_mile",
#         ]
#     ].head()
# )

# df["quote_signal_difference"] = (
#     df["posted_rate_per_mile"] - df["quote_signal"]
# )

# display(
#     df[
#         [
#             "load_id",
#             "quote_signal",
#             "posted_rate_per_mile",
#             "quote_signal_difference",
#         ]
#     ].head(20)
# )

def quote_signal_basic_check(
    df: pd.DataFrame,
) -> None:
    print("\033[1mPosted rate per mile calculation\033[0m")
    print("-" * 60)

    df["posted_rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    display(
        df[
            [
                "load_id",
                "distance",
                "posted_rate",
                "posted_rate_per_mile",
            ]
        ].head()
    )

    print()
    print("\033[1mQuote signal comparison\033[0m")
    print("-" * 60)

    df["quote_signal_difference"] = (
        df["posted_rate_per_mile"]
        - df["quote_signal"]
    )

    display(
        df[
            [
                "load_id",
                "quote_signal",
                "posted_rate_per_mile",
                "quote_signal_difference",
            ]
        ].head(20)
    )


def posted_rate_per_mile_quote_signal_difference(df: pd.DataFrame) -> None:

    df["posted_rate_per_mile"] = (
        df["posted_rate"] / df["distance"]
    )

    df["quote_signal_difference"] = (
        df["posted_rate_per_mile"] - df["quote_signal"]
    )

    display(
        df[
            [
                "quote_signal",
                "posted_rate_per_mile",
                "quote_signal_difference",
            ]
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )    

def quote_signal_posted_rate_per_mile_correllation(df: pd.DataFrame) -> None:

    correlation = df[
        [
            "quote_signal",
            "posted_rate_per_mile",
        ]
    ].corr()

    display(correlation)       


def quote_signal_outlier_correlation_check(
    df: pd.DataFrame,
) -> None:
    columns = [
        "quote_signal",
        "posted_rate_per_mile",
    ]

    review_df = df[columns].dropna().copy()

    full_pearson = review_df.corr(
        method="pearson"
    ).iloc[0, 1]

    full_spearman = review_df.corr(
        method="spearman"
    ).iloc[0, 1]

    # lower_bounds = review_df.quantile(0.01)
    # upper_bounds = review_df.quantile(0.99)

    lower_bounds = review_df.quantile(0.1)
    upper_bounds = review_df.quantile(0.92)    

    trimmed_df = review_df[
        review_df["quote_signal"].between(
            lower_bounds["quote_signal"],
            upper_bounds["quote_signal"],
        )
        & review_df["posted_rate_per_mile"].between(
            lower_bounds["posted_rate_per_mile"],
            upper_bounds["posted_rate_per_mile"],
        )
    ]

    trimmed_pearson = trimmed_df.corr(
        method="pearson"
    ).iloc[0, 1]

    trimmed_spearman = trimmed_df.corr(
        method="spearman"
    ).iloc[0, 1]

    print("\033[1mQuote-signal outlier correlation check\033[0m")
    print("-" * 60)
    print(f"Full-data rows:                        {len(review_df):,}")
    print(f"Full-data Pearson correlation:         {full_pearson:.6f}")
    print(f"Full-data Spearman correlation:        {full_spearman:.6f}")
    print()
    print(f"Trimmed rows:                          {len(trimmed_df):,}")
    print(f"Trimmed Pearson correlation:           {trimmed_pearson:.6f}")
    print(f"Trimmed Spearman correlation:          {trimmed_spearman:.6f}")