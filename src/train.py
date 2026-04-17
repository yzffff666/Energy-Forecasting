from pathlib import Path
import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the energy forecasting baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the experiment config JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from energy_forecasting.pipeline import run_training_pipeline

    summary = run_training_pipeline(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
