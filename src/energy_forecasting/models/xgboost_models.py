from __future__ import annotations

import warnings

import numpy as np
from xgboost import XGBRegressor


class XGBoostIntervalForecaster:
    def __init__(
        self,
        model_params: dict,
        uncertainty_enabled: bool = True,
        quantiles: tuple[float, float] = (0.1, 0.9),
        fallback_to_residual_interval: bool = True,
    ) -> None:
        self.model_params = dict(model_params)
        self.uncertainty_enabled = uncertainty_enabled
        self.quantiles = quantiles
        self.fallback_to_residual_interval = fallback_to_residual_interval

        self.point_model: XGBRegressor | None = None
        self.lower_model: XGBRegressor | None = None
        self.upper_model: XGBRegressor | None = None
        self.residual_quantiles_: tuple[float, float] | None = None
        self.interval_method_: str | None = None

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ) -> "XGBoostIntervalForecaster":
        self.point_model = XGBRegressor(**self.model_params)
        self.point_model.fit(X_train, y_train)

        if self.uncertainty_enabled:
            self._fit_intervals(X_train, y_train, X_val, y_val)

        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        if self.point_model is None:
            raise ValueError("Model must be fitted before prediction.")

        prediction = self.point_model.predict(X)
        outputs = {"prediction": prediction}

        if not self.uncertainty_enabled:
            return outputs

        if self.interval_method_ == "quantile_xgboost":
            lower = self.lower_model.predict(X)
            upper = self.upper_model.predict(X)
        elif self.interval_method_ == "residual_interval":
            if self.residual_quantiles_ is None:
                raise ValueError("Residual interval statistics are missing.")
            lower = prediction + self.residual_quantiles_[0]
            upper = prediction + self.residual_quantiles_[1]
        else:
            return outputs

        outputs["prediction_lower"] = np.minimum(lower, upper)
        outputs["prediction_upper"] = np.maximum(lower, upper)
        return outputs

    def _fit_intervals(self, X_train, y_train, X_val=None, y_val=None) -> None:
        lower_alpha, upper_alpha = self.quantiles
        interval_X = X_val if X_val is not None else X_train
        interval_y = y_val if y_val is not None else y_train

        try:
            lower_params = dict(self.model_params)
            lower_params["objective"] = "reg:quantileerror"
            lower_params["quantile_alpha"] = lower_alpha

            upper_params = dict(self.model_params)
            upper_params["objective"] = "reg:quantileerror"
            upper_params["quantile_alpha"] = upper_alpha

            self.lower_model = XGBRegressor(**lower_params)
            self.upper_model = XGBRegressor(**upper_params)
            self.lower_model.fit(X_train, y_train)
            self.upper_model.fit(X_train, y_train)
            self.interval_method_ = "quantile_xgboost"
        except Exception as exc:
            if not self.fallback_to_residual_interval:
                raise

            warnings.warn(
                "Quantile XGBoost training failed, falling back to residual intervals. "
                f"Original error: {exc}",
                RuntimeWarning,
            )
            point_predictions = self.point_model.predict(interval_X)
            residuals = interval_y - point_predictions
            residual_quantiles = np.quantile(residuals, [lower_alpha, upper_alpha])
            self.residual_quantiles_ = (float(residual_quantiles[0]), float(residual_quantiles[1]))
            self.interval_method_ = "residual_interval"

