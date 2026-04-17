from __future__ import annotations

from pathlib import Path
import json


def load_config(config_path: Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    _validate_config(config)
    return config


def prepare_output_dir(config: dict, base_dir: Path | None = None) -> Path:
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    output_root = base_dir / config["results"]["output_dir"]
    output_dir = output_root / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(payload: dict | list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _validate_config(config: dict) -> None:
    required_top_level = [
        "experiment_name",
        "random_seed",
        "data",
        "features",
        "split",
        "model",
        "uncertainty",
        "results",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    split = config["split"]
    total_frac = split["train_frac"] + split["val_frac"] + split["test_frac"]
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")

    quantiles = config["uncertainty"].get("quantiles", [])
    if config["uncertainty"].get("enabled") and len(quantiles) != 2:
        raise ValueError("Uncertainty quantiles must contain exactly two values.")

