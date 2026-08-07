from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_regression_performance(
    model_name: str,
    training_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    testing_metrics: dict[str, float],
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, list[Axes]]:
    """
    Plot training, validation, and testing regression performance.

    MAE and RMSE gaps are calculated as evaluation error minus training
    error. R² gaps are calculated as training R² minus evaluation R².
    Positive gaps therefore consistently indicate worse generalization.
    """

    required_metrics = {"mae", "rmse", "r2"}

    for partition_name, metrics in {
        "training": training_metrics,
        "validation": validation_metrics,
        "testing": testing_metrics,
    }.items():
        missing_metrics = required_metrics.difference(metrics)

        if missing_metrics:
            raise ValueError(
                f"{partition_name} metrics are missing: "
                f"{sorted(missing_metrics)}"
            )

    performance = pd.DataFrame(
        {
            "Training": training_metrics,
            "Validation": validation_metrics,
            "Testing": testing_metrics,
        }
    )

    figure, axes_array = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        constrained_layout=True,
    )
    axes = list(axes_array)

    metric_settings = [
        ("mae", "MAE", "Mean absolute error", False),
        ("rmse", "RMSE", "Root mean squared error", False),
        ("r2", "R²", "Coefficient of determination", True),
    ]

    partition_colors = [
        "#4F81BD",
        "#9BBB59",
        "#F79646",
    ]

    partition_names = [
        "Training",
        "Validation",
        "Testing",
    ]

    for axis, (
        metric_key,
        metric_label,
        y_axis_label,
        higher_is_better,
    ) in zip(axes, metric_settings):
        values = performance.loc[
            metric_key,
            partition_names,
        ].astype(float)

        bars = axis.bar(
            partition_names,
            values,
            color=partition_colors,
            width=0.68,
        )

        axis.set_title(metric_label, fontsize=14, fontweight="bold")
        axis.set_ylabel(y_axis_label)
        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.35,
        )
        axis.set_axisbelow(True)

        axis.bar_label(
            bars,
            labels=[f"{value:.4f}" for value in values],
            padding=3,
            fontsize=9,
        )

        training_value = values["Training"]

        for position, partition_name in enumerate(
            ["Validation", "Testing"],
            start=1,
        ):
            partition_value = values[partition_name]

            if higher_is_better:
                generalization_gap = (
                    training_value - partition_value
                )
            else:
                generalization_gap = (
                    partition_value - training_value
                )

            axis.annotate(
                f"Gap: {generalization_gap:+.4f}",
                xy=(position, partition_value),
                xytext=(0, 22),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

        minimum_value = min(values.min(), 0)
        maximum_value = max(values.max(), 0)
        value_range = maximum_value - minimum_value

        if value_range == 0:
            value_range = 1

        axis.set_ylim(
            minimum_value - value_range * 0.10,
            maximum_value + value_range * 0.28,
        )

    figure.suptitle(
        f"{model_name} regression performance",
        fontsize=17,
        fontweight="bold",
    )

    figure.text(
        0.5,
        -0.03,
        "Positive generalization gaps indicate worse performance "
        "relative to training.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return figure, axes

import matplotlib.pyplot as plt


def plot_r2_scores(
    targets: list[str],
    r2_scores: list[float],
    title: str = "Model Testing R² by Prediction Target",
) -> None:
    """Plot R² scores for multiple prediction targets."""

    if len(targets) != len(r2_scores):
        raise ValueError(
            "targets and r2_scores must have the same length."
        )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        targets,
        r2_scores,
    )

    plt.title(title)
    plt.xlabel("Prediction Target")
    plt.ylabel("Testing R²")
    plt.ylim(0, 1.05)

    for bar, value in zip(bars, r2_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()    