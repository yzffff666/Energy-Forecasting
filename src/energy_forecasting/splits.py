from __future__ import annotations

import pandas as pd


def time_series_split(feature_frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    split_config = config["split"]
    n_rows = len(feature_frame)
    if n_rows < 10:
        raise ValueError("Feature frame is too small after preprocessing. Add more data or fewer lags.")

    train_end = int(n_rows * split_config["train_frac"])
    val_end = train_end + int(n_rows * split_config["val_frac"])

    train = feature_frame.iloc[:train_end].copy()
    val = feature_frame.iloc[train_end:val_end].copy()
    test = feature_frame.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("One of the time splits is empty. Adjust split fractions or provide more data.")

    return {
        "train": train,
        "val": val,
        "test": test,
    }

