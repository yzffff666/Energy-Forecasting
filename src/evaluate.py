from pathlib import Path
import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved forecast predictions.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="CSV file with actual, prediction, and optional interval columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics and plots. Defaults to the prediction file directory.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name in the predictions file.",
    )
    parser.add_argument(
        "--actual-col",
        type=str,
        default="actual",
        help="Actual target column name in the predictions file.",
    )
    parser.add_argument(
        "--prediction-col",
        type=str,
        default="prediction",
        help="Point prediction column name in the predictions file.",
    )
    parser.add_argument(
        "--lower-col",
        type=str,
        default="prediction_lower",
        help="Lower interval column name in the predictions file.",
    )
    parser.add_argument(
        "--upper-col",
        type=str,
        default="prediction_upper",
        help="Upper interval column name in the predictions file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from energy_forecasting.pipeline import evaluate_saved_predictions

    metrics = evaluate_saved_predictions(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        timestamp_col=args.timestamp_col,
        actual_col=args.actual_col,
        prediction_col=args.prediction_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
