"""
Data loader for UCI-style hourly energy demand.
Generates synthetic data locally if the real CSV is not present,
so the project is fully self-contained out of the box.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/energy.csv")
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
    dow = dates.dayofweek.values
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
    df["date"] = pd.to_datetime(df["date"])
    df["demand_kwh"] = df["demand_kwh"].astype(float)
    before = len(df)
    df = df.dropna(subset=["demand_kwh"]).reset_index(drop=True)
    logger.info(
        f"Schema OK — {len(df)} rows kept ({before - len(df)} dropped), "
        f"{df['date'].min()} → {df['date'].max()}"
    )
    return df
