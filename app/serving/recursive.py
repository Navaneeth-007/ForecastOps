"""Shared recursive inference: future targets are never available as features."""

import numpy as np
import pandas as pd

from app.features.engineer import build_features, feature_columns


def recursive_predict(model, history: pd.DataFrame, dates) -> np.ndarray:
    window = history[["date", "demand_kwh"]].tail(169).copy()
    predictions = []
    for date in dates:
        # Placeholder is excluded from every lag and rolling statistic by shift(1).
        row = pd.DataFrame({"date": [date], "demand_kwh": [0.0]})
        extended = pd.concat([window, row], ignore_index=True)
        features = build_features(extended).tail(1)
        if features.empty:
            raise ValueError("At least 168 hours of history are required")
        value = max(0.0, float(model.predict(features[feature_columns(features)])[0]))
        if not np.isfinite(value):
            raise ValueError("Model produced a non-finite prediction")
        predictions.append(value)
        extended.loc[extended.index[-1], "demand_kwh"] = value
        window = extended.tail(169)
    return np.asarray(predictions)
