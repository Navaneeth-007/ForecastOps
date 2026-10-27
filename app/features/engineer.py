"""
Feature engineering for hourly energy demand forecasting.
Produces features used by both LightGBM and Prophet models.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# Lag windows (hours)
LAG_HOURS: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]  # 168 = 1 week
# Rolling windows (hours)
ROLLING_WINDOWS: List[int] = [6, 24, 168]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a validated raw dataframe (date, demand_kwh),
    return a feature-rich dataframe ready for ML training.
    """
    df = df.sort_values("date").copy()
    df = df.set_index("date")

    # --- Calendar features ---
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek          # 0=Mon, 6=Sun
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = df.index.quarter

    # Cyclical encoding (avoids discontinuity at midnight/Sunday)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # --- Lag features ---
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df["demand_kwh"].shift(lag)

    # --- Rolling statistics ---
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}h"] = (
            df["demand_kwh"].shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}h"] = (
            df["demand_kwh"].shift(1).rolling(window).std()
        )

    # Drop rows where lags/rolling aren't available
    df = df.dropna().reset_index()
    logger.info(
        f"Feature engineering complete: {len(df)} rows, "
        f"{len(df.columns)} columns"
    )
    return df


def train_test_split(
    df: pd.DataFrame,
    test_weeks: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — last `test_weeks` weeks become the hold-out set.
    Never shuffles to preserve temporal integrity.
    """
    cutoff = df["date"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    logger.info(
        f"Train: {len(train)} rows up to {cutoff.date()}  |  "
        f"Test: {len(test)} rows"
    )
    return train, test


def save_features(df: pd.DataFrame, name: str = "features") -> Path:
    """Persist feature dataframe as parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved features → {path}")
    return path


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of model input columns (everything except date + target)."""
    exclude = {"date", "demand_kwh"}
    return [c for c in df.columns if c not in exclude]
