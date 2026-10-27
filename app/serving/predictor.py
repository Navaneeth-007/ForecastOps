"""
Predictor service — wraps both Prophet and LightGBM under a unified interface.
Accepts a horizon (hours) and returns a forecast dataframe.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.features.engineer import build_features, feature_columns
from app.serving.model_loader import get_model_info, load_production_model

logger = logging.getLogger(__name__)


def forecast(
    horizon_hours: int = 24,
    history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate a forecast for the next `horizon_hours` hours.

    Args:
        horizon_hours: Number of hours ahead to forecast.
        history: Optional historical dataframe (date, demand_kwh).
                 If None, uses the stored processed data.

    Returns:
        DataFrame with columns: date, yhat, yhat_lower, yhat_upper
    """
    model_info = get_model_info()
    model_type = model_info.get("model_type", "unknown")
    model = load_production_model()

    if history is None:
        from pathlib import Path
        path = Path("data/raw/energy.csv")
        if path.exists():
            history = pd.read_csv(path, parse_dates=["date"])
        else:
            raise FileNotFoundError("No history available. Run ingestion first.")

    last_ts = history["date"].max()
    future_dates = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    if model_type == "prophet":
        future_df = pd.DataFrame({"ds": future_dates})
        preds = model.predict(None, future_df)
        return pd.DataFrame({
            "date": preds["ds"],
            "yhat": preds["yhat"].round(2),
            "yhat_lower": preds["yhat_lower"].round(2),
            "yhat_upper": preds["yhat_upper"].round(2),
        })

    elif model_type == "lightgbm":
        # Build features for future timestamps by extending history
        # with naive carry-forward for the future window
        extended = _build_future_features(history, future_dates)
        feat_cols = feature_columns(extended)
        x_future = extended.tail(horizon_hours)[feat_cols]
        y_pred = model.predict(x_future)
        # LightGBM has no native PI — use ±1.96*MAE as a proxy
        mae = model_info.get("mae", 5.0)
        return pd.DataFrame({
            "date": future_dates,
            "yhat": np.round(y_pred, 2),
            "yhat_lower": np.round(y_pred - 1.96 * mae, 2),
            "yhat_upper": np.round(y_pred + 1.96 * mae, 2),
        })

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_future_features(history: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extend history with synthetic future rows (demand=NaN) so that
    feature_engineer can compute lag/rolling features.
    Lags are filled by rolling the latest known values forward.
    """
    future_stub = pd.DataFrame({
        "date": future_dates,
        "demand_kwh": np.nan,
    })
    combined = pd.concat([history, future_stub], ignore_index=True)
    # Forward-fill demand so lag features can be computed
    combined["demand_kwh"] = combined["demand_kwh"].fillna(
        method=None
    )
    # Simple carry-forward: repeat last known value
    last_known = history["demand_kwh"].iloc[-1]
    combined.loc[combined["demand_kwh"].isna(), "demand_kwh"] = last_known
    return build_features(combined)
