from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_vs_actual(
    predictions: pd.DataFrame,
    timestamp_col: str,
    output_path: Path,
    title: str,
    max_points: int = 336,
) -> None:
    plot_df = predictions.tail(max_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df[timestamp_col], plot_df["actual"], label="Actual", linewidth=2)
    ax.plot(plot_df[timestamp_col], plot_df["prediction"], label="Forecast", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction_interval(
    predictions: pd.DataFrame,
    timestamp_col: str,
    output_path: Path,
    title: str,
    max_points: int = 336,
) -> None:
    required_columns = {"prediction_lower", "prediction_upper"}
    if not required_columns.issubset(predictions.columns):
        return

    plot_df = predictions.tail(max_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df[timestamp_col], plot_df["actual"], label="Actual", linewidth=1.8, color="black")
    ax.plot(plot_df[timestamp_col], plot_df["prediction"], label="Forecast", linewidth=1.8, color="tab:blue")
    ax.fill_between(
        plot_df[timestamp_col],
        plot_df["prediction_lower"],
        plot_df["prediction_upper"],
        color="tab:blue",
        alpha=0.2,
        label="Prediction interval",
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

