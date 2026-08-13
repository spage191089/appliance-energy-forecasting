"""Benchmark forecasting methods for appliance energy use."""

import numpy as np
import pandas as pd


def mean_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Forecast every future point using the training mean."""
    return np.repeat(train.mean(), horizon)


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    if len(train) == 0:
        raise ValueError("Training series is empty.")
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    """Repeat the most recent complete seasonal cycle."""
    if season_length <= 0:
        raise ValueError("season_length must be positive.")
    if len(train) < season_length:
        raise ValueError("Not enough observations for seasonal naive.")

    last_cycle = train.iloc[-season_length:].to_numpy()
    return np.resize(last_cycle, horizon)


def drift_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Forecast using a linear drift from the first to last observation."""
    if len(train) < 2:
        raise ValueError("At least two observations are required.")

    first = float(train.iloc[0])
    last = float(train.iloc[-1])
    slope = (last - first) / (len(train) - 1)

    steps = np.arange(1, horizon + 1)
    return last + steps * slope


def make_benchmark_forecasts(
    train: pd.Series,
    horizon: int,
    daily_season: int = 24,
    weekly_season: int = 168,
) -> dict[str, np.ndarray]:
    """Create all benchmark forecasts used in the project."""
    return {
        "mean": mean_forecast(train, horizon),
        "naive": naive_forecast(train, horizon),
        "seasonal_naive_daily": seasonal_naive_forecast(
            train, horizon, daily_season
        ),
        "seasonal_naive_weekly": seasonal_naive_forecast(
            train, horizon, weekly_season
        ),
        "drift": drift_forecast(train, horizon),
    }


def rolling_origin_benchmarks(
    series: pd.Series,
    horizon: int = 24,
    n_windows: int = 14,
) -> pd.DataFrame:
    """Generate rolling-origin benchmark forecasts for the final windows."""
    series = series.dropna().sort_index()

    if len(series) <= horizon * n_windows:
        raise ValueError("Series is too short for the requested evaluation.")

    rows = []

    first_origin = len(series) - horizon * n_windows

    for window in range(n_windows):
        train_end = first_origin + window * horizon
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + horizon]

        forecasts = make_benchmark_forecasts(
            train,
            horizon=len(test),
        )

        for step, timestamp in enumerate(test.index):
            row = {
                "window": window + 1,
                "timestamp": timestamp,
                "actual": test.iloc[step],
            }

            for name, forecast in forecasts.items():
                row[name] = forecast[step]

            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_forecast(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE and bias."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError("Actual and forecast arrays must have the same shape.")

    errors = forecast - actual

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Bias": float(np.mean(errors)),
    }


def mase_scale(
    train: pd.Series,
    season_length: int = 24,
) -> float:
    """Calculate the in-sample seasonal-naive MAE used to scale MASE."""
    if len(train) <= season_length:
        raise ValueError("Not enough observations to calculate MASE scale.")

    errors = (
        train.iloc[season_length:].to_numpy()
        - train.iloc[:-season_length].to_numpy()
    )

    return float(np.mean(np.abs(errors)))


def add_mase(
    metrics: dict[str, float],
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
    scale: float,
) -> dict[str, float]:
    """Add MASE to an existing metric dictionary."""
    if scale <= 0:
        raise ValueError("MASE scale must be positive.")

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    metrics = dict(metrics)
    metrics["MASE"] = float(np.mean(np.abs(actual - forecast)) / scale)

    return metrics


def evaluate_all(
    actual: pd.Series,
    forecasts: dict[str, np.ndarray],
    scale: float,
) -> pd.DataFrame:
    """Evaluate all supplied forecasts with MAE, RMSE, MASE and bias."""
    rows = []

    for model, forecast in forecasts.items():
        metrics = evaluate_forecast(actual, forecast)
        metrics = add_mase(metrics, actual, forecast, scale)
        metrics["model"] = model
        rows.append(metrics)

    return pd.DataFrame(rows)[
        ["model", "MAE", "RMSE", "MASE", "Bias"]
    ]
