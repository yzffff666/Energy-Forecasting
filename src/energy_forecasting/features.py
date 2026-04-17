from __future__ import annotations

import pandas as pd


def build_feature_frame(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    data_config = config["data"]
    feature_config = config["features"]
    timestamp_col = data_config["timestamp_col"]
    target_col = data_config["target_col"]
    exogenous_cols = data_config.get("feature_cols", [])

    feature_frame = df[[timestamp_col, target_col, *exogenous_cols]].copy()

    for lag in feature_config.get("lags", []):
        feature_frame[f"{target_col}_lag_{lag}"] = feature_frame[target_col].shift(lag)

    shifted_target = feature_frame[target_col].shift(1) 
    for window in feature_config.get("rolling_windows", []):
        feature_frame[f"{target_col}_roll_mean_{window}"] = shifted_target.rolling(window).mean()
        feature_frame[f"{target_col}_roll_std_{window}"] = shifted_target.rolling(window).std()

    if feature_config.get("add_calendar_features", True):
        feature_frame["hour"] = feature_frame[timestamp_col].dt.hour
        feature_frame["day_of_week"] = feature_frame[timestamp_col].dt.dayofweek
        feature_frame["day_of_month"] = feature_frame[timestamp_col].dt.day
        feature_frame["month"] = feature_frame[timestamp_col].dt.month
        feature_frame["is_weekend"] = feature_frame["day_of_week"].isin([5, 6]).astype(int)

    feature_frame = feature_frame.dropna().reset_index(drop=True)
    feature_columns = [
        column
        for column in feature_frame.columns
        if column not in {timestamp_col, target_col}
    ]
    return feature_frame, feature_columns

