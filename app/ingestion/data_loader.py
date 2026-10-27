"""
Data loader for UCI-style hourly energy demand.
Generates synthetic data locally if the real CSV is not present,
so the project is fully self-contained out of the box.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

RAW_DATA_PATH = DATA_DIR / "raw/energy.csv"
PROCESSED_PATH = Path("data/processed/energy_clean.parquet")


def download_or_load() -> pd.DataFrame:
    """Return raw energy dataframe, generating if not cached."""
    if RAW_DATA_PATH.exists():
        logger.info("Loading cached raw CSV.")
        df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    else:
        logger.info("Generating synthetic UCI-style energy demand (2 years hourly).")
        df = _generate_synthetic()
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Saved to {RAW_DATA_PATH}")
    return df


def _generate_synthetic(n_days: int = 730) -> pd.DataFrame:
    """
    Realistic synthetic hourly electricity demand with:
    - Daily sinusoidal pattern (peak evening)
    - Weekday/weekend effect
    - Winter-peak seasonal effect
    - Gaussian noise
    """
    rng = np.random.default_rng(42)
    periods = n_days * 24
    dates = pd.date_range("2022-01-01", periods=periods, freq="h")

    hour = dates.hour.values
    dow = dates.day_of_week.values
    month = dates.month.values

    daily = 20 * np.sin(2 * np.pi * (hour - 6) / 24) + 20
    weekly = np.where(dow < 5, 5.0, -5.0)
    seasonal = 15 * np.cos(2 * np.pi * (month - 1) / 12)
    noise = rng.normal(0, 3, periods)

    demand = np.clip(100 + daily + weekly + seasonal + noise, 30, 200)
    return pd.DataFrame({"date": dates, "demand_kwh": demand.round(2)})


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Assert required columns exist, coerce types, drop nulls."""
    required = {"date", "demand_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[["date", "demand_kwh"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["demand_kwh"] = pd.to_numeric(df["demand_kwh"], errors="raise").astype(float)
    if df.empty or df.isna().any().any():
        raise ValueError("Data must be nonempty with no missing timestamps or demand")
    if not np.isfinite(df["demand_kwh"]).all() or (df["demand_kwh"] < 0).any():
        raise ValueError("Demand must be finite and nonnegative")
    df = df.sort_values("date").reset_index(drop=True)
    if df.date.duplicated().any() or not df.date.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Timestamps must be unique, contiguous hourly observations")
    return df
