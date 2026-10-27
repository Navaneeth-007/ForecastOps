"""Forecast from one consistent registry snapshot and validated history."""

import numpy as np
import pandas as pd

from app.ingestion.data_loader import RAW_DATA_PATH, validate_schema
from app.serving.model_loader import load_snapshot
from app.serving.recursive import recursive_predict


def forecast(horizon_hours: int = 24, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if not 1 <= horizon_hours <= 168:
        raise ValueError("Forecast horizon must be between 1 and 168 hours")
    model, info = load_snapshot()
    if history is None:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError("No history available. Run training first.")
        history = pd.read_csv(RAW_DATA_PATH)
    history = validate_schema(history)
    dates = pd.date_range(
        history.date.max() + pd.Timedelta(hours=1), periods=horizon_hours, freq="h"
    )
    if info["model_type"] == "prophet":
        result = model.predict(pd.DataFrame({"ds": dates})).rename(columns={"ds": "date"})
        result = result[["date", "yhat", "yhat_lower", "yhat_upper"]]
    elif info["model_type"] == "lightgbm":
        values = recursive_predict(model, history, dates)
        width = 1.96 * float(info["mae"])
        result = pd.DataFrame(
            {
                "date": dates,
                "yhat": values,
                "yhat_lower": np.maximum(0, values - width),
                "yhat_upper": values + width,
            }
        )
    else:
        raise RuntimeError("Unsupported production model")
    result.attrs["model"] = info
    result[["yhat", "yhat_lower", "yhat_upper"]] = result[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].round(2)
    return result
