# Energy Forecasting Prototype

A small research-style project for short-term electricity load forecasting with prediction intervals.

I built this as a baseline-first project rather than a large forecasting system. The current version starts from ERCOT hourly load data, builds lag and rolling features, compares XGBoost with simple persistence rules, and checks whether the prediction intervals are reliable enough to be useful.

**Current result:** on the 2025 ERCOT test split, XGBoost reaches **0.99% MAPE**, improving over a one-hour persistence baseline (**2.08% MAPE**) while also producing quantile-based prediction intervals.

## Problem

The goal is to predict short-term electricity demand from historical load patterns and calendar signals, while also reporting a forecast range instead of only a point prediction.

For this version, I focused on three questions:

- Can a tabular ML baseline outperform naive persistence forecasting on hourly system load?
- Can prediction intervals be added without making the model pipeline too complicated?
- Are the forecast intervals actually well calibrated, or are they systematically too narrow?

## Research Relevance

Short-term load forecasts are useful for operational planning tasks such as unit commitment, reserve scheduling, demand response, and congestion management. Even a simple hourly forecast is more useful when it is reproducible, easy to audit, and evaluated with a time-respecting split.

The prediction interval part matters because grid operators often need more than one expected value. A forecast range gives a rough sense of downside and upside risk, which is relevant when deciding how much reserve capacity to hold or how aggressively to schedule flexible demand. In this project, the point forecast works well against naive baselines, but the interval coverage is lower than expected, so calibration becomes a natural next step.

I kept the first version load-only on purpose. Before adding weather, renewable generation, or more complex neural models, I wanted a baseline that makes the data assumptions, leakage controls, and evaluation outputs easy to inspect.

## Data

The default run uses the ERCOT 2025 hourly native load archive:

- input file: `data/Native_Load_2025.zip`
- Excel member: `Native_Load_2025.xlsx`
- timestamp column: `Hour Ending`
- target column: `ERCOT`

The loader supports CSV, Excel, and ZIP archives containing a single CSV/XLSX file. ERCOT's hour-ending timestamps are normalized automatically, including `24:00` and DST/ST suffixes.

Optional external features can be added through `config.json`, but the current default run uses historical load and calendar features only.

For this first version, I intentionally left weather covariates out so that I could verify the load-only baseline and the no-leakage feature pipeline first. The repository reads directly from the ZIP archive, so the manually extracted folder is not needed for training.

## Method

I convert the time series into a supervised learning table by building one row per timestamp with features derived only from past observations.

Feature groups:

- lagged load values, such as `t-1`, `t-24`, and `t-168`
- rolling mean and rolling standard deviation computed after a one-step shift to avoid leakage
- calendar features including hour of day, day of week, month, and weekend indicator

Models:

- `XGBoost`: main point-forecast baseline
- `XGBoost quantile regression`: lower and upper prediction bounds
- `Persistence baselines`: naive forecasts using `t-1` and `t-24`

I kept the modeling layer small so that a future LSTM baseline can be added without rewriting the data loading, split logic, or metric code.

## Experimental Setup

I use a deterministic chronological train/validation/test split:

- train: 70%
- validation: 15%
- test: 15%

This avoids random shuffling across time and reduces leakage risk. For ERCOT 2025, the processed data contains:

- raw rows: 8760
- modeling rows after lag/rolling feature construction: 8592
- train / validation / test rows: 6014 / 1288 / 1290

All key settings live in `config.json`. Each run saves the resolved config, metrics, predictions, plots, feature names, and trained model under `results/<experiment_name>/` so the experiment can be checked later.

## Results

Test-set results on ERCOT 2025 hourly system load:

| Model | MAE | RMSE | MAPE | Interval Coverage | Mean Interval Width |
|---|---:|---:|---:|---:|---:|
| Persistence (`t-1`) | 1051.47 | 1288.17 | 2.08% | - | - |
| Persistence (`t-24`) | 2404.80 | 3210.91 | 4.73% | - | - |
| XGBoost | 501.66 | 653.79 | 0.99% | 69.22% | 1566.72 MW |

What I found:

- XGBoost is clearly better than both persistence rules for point forecasting on this split.
- The 0.1 / 0.9 quantile band should act like an 80% interval, but test coverage is only 69.22% with an average width of 1566.72 MW. That means the interval is too confident right now, so calibration is a good next step.

### Forecast Visualization

Point forecast on the test split:

![XGBoost test forecast vs actual](results/xgboost_baseline/test_forecast_vs_actual.png)

Prediction interval on the test split:

![XGBoost test prediction interval](results/xgboost_baseline/test_prediction_interval.png)

Generated artifacts include:

- `results/xgboost_baseline/metrics.json`
- `results/xgboost_baseline/test_predictions.csv`
- `results/xgboost_baseline/test_forecast_vs_actual.png`
- `results/xgboost_baseline/test_prediction_interval.png`
- persistence baseline prediction files and plots for `t-1` and `t-24`

## Project Structure

```text
energy-forecasting-prototype/
  data/
  notebooks/
  results/
  src/
    train.py
    evaluate.py
    energy_forecasting/
      config.py
      data.py
      features.py
      splits.py
      metrics.py
      plotting.py
      pipeline.py
      models/
        naive_models.py
        xgboost_models.py
  config.json
  requirements.txt
  README.md
```

## Workflow

```mermaid
flowchart LR
  A["ERCOT hourly load data"] --> B["Timestamp parsing and cleaning"]
  B --> C["Lag, rolling, and calendar features"]
  C --> D["Chronological train / val / test split"]
  D --> E["Persistence baselines"]
  D --> F["XGBoost + quantile intervals"]
  E --> G["Metrics and plots"]
  F --> G["Metrics and plots"]
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
python src/train.py --config config.json
```

Re-evaluate a saved prediction file:

```bash
python src/evaluate.py --predictions-path results/xgboost_baseline/test_predictions.csv
```

If you use Windows and the `python` launcher is not available, use `py -3` instead.

## Metrics

Point forecast accuracy:

- MAE
- RMSE
- MAPE

Uncertainty quality:

- empirical interval coverage, when prediction bounds are available
- mean interval width, to show how wide the prediction bands are

## Limitations And Next Steps

Right now this is still a simple baseline study. The default setup is one-step-ahead forecasting, not a full day-ahead or multi-horizon forecast. I also use one fixed chronological split, so rolling-origin backtesting would be a better next evaluation step.

The prediction intervals are useful to inspect, but they are not calibrated yet. The current 80% nominal interval only covers 69.22% of the test points, so I would next try conformal calibration or residual scaling.

I also left out weather variables on purpose for the first version. A natural extension would be to add temperature and humidity, compare load-only vs load-plus-weather features, and later add an LSTM baseline under `src/energy_forecasting/models/`.
