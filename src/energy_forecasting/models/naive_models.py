from __future__ import annotations

import pandas as pd


class PersistenceForecaster:
    def __init__(self, lag: int, target_col: str) -> None:
        if lag <= 0:
            raise ValueError("Persistence lag must be a positive integer.")
        self.lag = lag
        self.target_col = target_col
        self.feature_name = f"{target_col}_lag_{lag}"

    @property
    def name(self) -> str:
        return f"persistence_lag_{self.lag}"

    def predict_frame(self, frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        if self.feature_name not in frame.columns:
            raise ValueError(
                f"Missing feature '{self.feature_name}' for {self.name}. "
                "Add this lag to config['features']['lags']."
            )

        return pd.DataFrame(
            {
                timestamp_col: frame[timestamp_col].to_numpy(),
                "actual": frame[self.target_col].to_numpy(),
                "prediction": frame[self.feature_name].to_numpy(),
            }
        )
