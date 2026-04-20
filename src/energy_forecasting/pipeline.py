from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, prepare_output_dir, save_json
from .data import load_dataset
from .features import build_feature_frame
from .metrics import compute_regression_metrics, interval_coverage, mean_interval_width
from .plotting import plot_forecast_vs_actual, plot_prediction_interval
from .splits import time_series_split


def run_training_pipeline(config_path: Path) -> dict:
    from .models import PersistenceForecaster, XGBoostIntervalForecaster

    config_path = Path(config_path)
    project_root = config_path.resolve().parent
    config = load_config(config_path)
    output_dir = prepare_output_dir(config, base_dir=project_root)

    raw_df = load_dataset(config, base_dir=project_root)
    feature_frame, feature_columns = build_feature_frame(raw_df, config)
    split_frames = time_series_split(feature_frame, config)

    target_col = config["data"]["target_col"]
    timestamp_col = config["data"]["timestamp_col"]
    uncertainty_config = config["uncertainty"]

    model = XGBoostIntervalForecaster(
        model_params=config["model"]["params"],
        uncertainty_enabled=uncertainty_config.get("enabled", False),
        quantiles=tuple(uncertainty_config.get("quantiles", [0.1, 0.9])),
        fallback_to_residual_interval=uncertainty_config.get("fallback_to_residual_interval", True),
    )

    model.fit(
        split_frames["train"][feature_columns],
        split_frames["train"][target_col],
        X_val=split_frames["val"][feature_columns],
        y_val=split_frames["val"][target_col],
    )

    metrics_by_model: dict[str, dict[str, dict]] = {"xgboost": {}}
    plot_last_n = config["results"].get("plot_last_n", 336)

    for split_name in ["val", "test"]:
        predictions = _build_predictions_frame(
            frame=split_frames[split_name],
            feature_columns=feature_columns,
            target_col=target_col,
            timestamp_col=timestamp_col,
            model=model,
        )
        split_metrics = compute_regression_metrics(
            predictions["actual"].to_numpy(),
            predictions["prediction"].to_numpy(),
        )
        if {"prediction_lower", "prediction_upper"}.issubset(predictions.columns):
            split_metrics["interval_coverage"] = interval_coverage(
                predictions["actual"].to_numpy(),
                predictions["prediction_lower"].to_numpy(),
                predictions["prediction_upper"].to_numpy(),
            )
            split_metrics["mean_interval_width"] = mean_interval_width(
                predictions["prediction_lower"].to_numpy(),
                predictions["prediction_upper"].to_numpy(),
            )

        predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
        plot_forecast_vs_actual(
            predictions=predictions,
            timestamp_col=timestamp_col,
            output_path=output_dir / f"{split_name}_forecast_vs_actual.png",
            title=f"{split_name.title()} Forecast vs Actual",
            max_points=plot_last_n,
        )
        plot_prediction_interval(
            predictions=predictions,
            timestamp_col=timestamp_col,
            output_path=output_dir / f"{split_name}_prediction_interval.png",
            title=f"{split_name.title()} Prediction Interval",
            max_points=plot_last_n,
        )
        metrics_by_model["xgboost"][split_name] = split_metrics

    for lag in config.get("baselines", {}).get("persistence_lags", []):
        baseline = PersistenceForecaster(lag=lag, target_col=target_col)
        metrics_by_model[baseline.name] = {}

        for split_name in ["val", "test"]:
            baseline_predictions = baseline.predict_frame(
                frame=split_frames[split_name],
                timestamp_col=timestamp_col,
            )
            baseline_metrics = compute_regression_metrics(
                baseline_predictions["actual"].to_numpy(),
                baseline_predictions["prediction"].to_numpy(),
            )
            baseline_predictions.to_csv(
                output_dir / f"{split_name}_{baseline.name}_predictions.csv",
                index=False,
            )
            plot_forecast_vs_actual(
                predictions=baseline_predictions,
                timestamp_col=timestamp_col,
                output_path=output_dir / f"{split_name}_{baseline.name}_forecast_vs_actual.png",
                title=f"{split_name.title()} {baseline.name} Forecast vs Actual",
                max_points=plot_last_n,
            )
            metrics_by_model[baseline.name][split_name] = baseline_metrics

    save_json(config, output_dir / "resolved_config.json")
    save_json(feature_columns, output_dir / "feature_columns.json")
    save_json(
        {
            "n_rows_raw": len(raw_df),
            "n_rows_model": len(feature_frame),
            "n_train": len(split_frames["train"]),
            "n_val": len(split_frames["val"]),
            "n_test": len(split_frames["test"]),
            "interval_method": model.interval_method_,
        },
        output_dir / "run_summary.json",
    )
    save_json(metrics_by_model, output_dir / "metrics.json")
    joblib.dump(model, output_dir / "model.joblib")

    return {
        "output_dir": str(output_dir),
        "metrics": metrics_by_model,
        "interval_method": model.interval_method_,
    }


def evaluate_saved_predictions(
    predictions_path: Path,
    output_dir: Path | None = None,
    timestamp_col: str = "timestamp",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    lower_col: str = "prediction_lower",
    upper_col: str = "prediction_upper",
) -> dict:
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    predictions[timestamp_col] = pd.to_datetime(predictions[timestamp_col], errors="coerce")
    if predictions[timestamp_col].isna().all():
        raise ValueError(f"Could not parse timestamps from column '{timestamp_col}'.")

    output_dir = predictions_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_frame = predictions.rename(
        columns={
            actual_col: "actual",
            prediction_col: "prediction",
            lower_col: "prediction_lower",
            upper_col: "prediction_upper",
        }
    )

    metrics = compute_regression_metrics(
        eval_frame["actual"].to_numpy(),
        eval_frame["prediction"].to_numpy(),
    )
    if {"prediction_lower", "prediction_upper"}.issubset(eval_frame.columns):
        metrics["interval_coverage"] = interval_coverage(
            eval_frame["actual"].to_numpy(),
            eval_frame["prediction_lower"].to_numpy(),
            eval_frame["prediction_upper"].to_numpy(),
        )
        metrics["mean_interval_width"] = mean_interval_width(
            eval_frame["prediction_lower"].to_numpy(),
            eval_frame["prediction_upper"].to_numpy(),
        )

    save_json(metrics, output_dir / "evaluation_metrics.json")
    plot_forecast_vs_actual(
        predictions=eval_frame,
        timestamp_col=timestamp_col,
        output_path=output_dir / "evaluation_forecast_vs_actual.png",
        title="Forecast vs Actual",
    )
    plot_prediction_interval(
        predictions=eval_frame,
        timestamp_col=timestamp_col,
        output_path=output_dir / "evaluation_prediction_interval.png",
        title="Prediction Interval",
    )
    return metrics


def _build_predictions_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    timestamp_col: str,
    model: XGBoostIntervalForecaster,
) -> pd.DataFrame:
    outputs = model.predict(frame[feature_columns])
    predictions = pd.DataFrame(
        {
            timestamp_col: frame[timestamp_col].to_numpy(),
            "actual": frame[target_col].to_numpy(),
            "prediction": outputs["prediction"],
        }
    )

    if "prediction_lower" in outputs and "prediction_upper" in outputs:
        predictions["prediction_lower"] = outputs["prediction_lower"]
        predictions["prediction_upper"] = outputs["prediction_upper"]

    return predictions
