"""Data loading and preparation utilities for the appliance-energy project."""

from pathlib import Path

import pandas as pd


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by looking for the repository README."""
    path = Path(start or Path.cwd()).resolve()

    for candidate in [path, *path.parents]:
        if (candidate / "README.md").exists():
            return candidate

    return path


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw appliance-energy dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return df


def prepare_index(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Convert the date column to a datetime index and sort chronologically."""
    if date_column not in df.columns:
        raise KeyError(f"Missing date column: {date_column}")

    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column])
    out = out.sort_values(date_column)
    out = out.set_index(date_column)

    if out.index.has_duplicates:
        raise ValueError("Duplicate timestamps found.")

    return out


def to_hourly(df: pd.DataFrame, target: str = "Appliances") -> pd.DataFrame:
    """Aggregate the 10-minute data to hourly means."""
    if target not in df.columns:
        raise KeyError(f"Missing target column: {target}")

    hourly = df.resample("h").mean(numeric_only=True)

    # Remove incomplete hours.
    counts = df[target].resample("h").count()
    hourly = hourly.loc[counts[counts == 6].index]

    return hourly


def clean_data(
    path: str | Path,
    date_column: str = "date",
    target: str = "Appliances",
) -> pd.DataFrame:
    """Load, index, clean and resample the raw dataset to hourly data."""
    df = load_raw_data(path)
    df = prepare_index(df, date_column=date_column)

    # Remove the random-noise columns used in the original dataset.
    df = df.drop(columns=["rv1", "rv2"], errors="ignore")

    return to_hourly(df, target=target)
