"""Leakage-safe feature engineering for appliance-energy forecasting."""

import numpy as np
import pandas as pd


TARGET = "Appliances"
HORIZON = 24

# Feature specification used in Notebook 05.
LAGS = [1, 2, 3, 6, 12, 24, 25, 48, 168]
ROLLING_WINDOWS = [3, 6, 24, 168]

INDOOR_TEMP_COLS = [f"T{i}" for i in range(1, 10)]
INDOOR_HUM_COLS = [f"RH_{i}" for i in range(1, 10)]
WEATHER_COLS = [
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create deterministic calendar features known in advance."""
    out = pd.DataFrame(index=index)

    out["hour"] = index.hour
    out["dayofweek"] = index.dayofweek
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)

    return out


def make_features(
    frame: pd.DataFrame,
    min_lag: int = HORIZON,
    covariates: str = "none",
    target: str = TARGET,
) -> pd.DataFrame:
    """Build a leakage-safe supervised feature table.

    Parameters
    ----------
    frame:
        Hourly dataframe indexed by timestamps.
    min_lag:
        Minimum lookback allowed. Target-derived features only use observations
        at least this many hours before the prediction timestamp.
    covariates:
        "none"     : calendar and target-derived features only.
        "lagged"   : sensor/weather values from min_lag hours ago.
        "realised" : sensor/weather values at the target timestamp. This is a
                     conditional analysis and is not operationally available.
    target:
        Target column name.
    """
    if target not in frame.columns:
        raise KeyError(f"Missing target column: {target}")

    if min_lag < 1:
        raise ValueError("min_lag must be at least 1.")

    if covariates not in {"none", "lagged", "realised"}:
        raise ValueError("covariates must be 'none', 'lagged', or 'realised'.")

    series = frame[target]
    out = make_calendar_features(frame.index)

    # Target lags: only lags available at the forecast origin.
    for lag in LAGS:
        if lag >= min_lag:
            out[f"lag_{lag}"] = series.shift(lag)

    # Rolling statistics: shift first so the window ends at the forecast origin.
    for window in ROLLING_WINDOWS:
        shifted = series.shift(min_lag)
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()

    # Mean of the same hour over the previous seven days.
    same_hour = pd.concat(
        [series.shift(24 * k) for k in range(1, 8)],
        axis=1,
    )
    out["same_hour_mean_7d"] = same_hour.mean(axis=1)

    covariate_cols = [c for c in frame.columns if c != target]

    if covariates == "lagged":
        for col in covariate_cols:
            out[f"{col}_lag{min_lag}"] = frame[col].shift(min_lag)

    elif covariates == "realised":
        for col in covariate_cols:
            out[col] = frame[col]

    return out


def assert_no_future_information(
    frame: pd.DataFrame,
    min_lag: int,
    timestamp,
    target: str = TARGET,
) -> pd.Timestamp:
    """Check empirically that features do not react to future target values."""
    timestamp = pd.Timestamp(timestamp)
    origin = timestamp - pd.Timedelta(hours=min_lag)

    corrupted = frame.copy()
    corrupted.loc[corrupted.index > origin, target] += 10_000.0

    original_row = make_features(frame, min_lag=min_lag).loc[timestamp]
    corrupted_row = make_features(corrupted, min_lag=min_lag).loc[timestamp]

    differing = original_row.index[
        ~np.isclose(
            original_row.values.astype(float),
            corrupted_row.values.astype(float),
            equal_nan=True,
        )
    ]

    assert len(differing) == 0, (
        f"Features using future values: {list(differing)}"
    )

    return origin
