"""
Evaluation metrics for forecasting models.
Produces a comparison table: Prophet vs LightGBM across all key metrics.
"""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — handles near-zero actuals better than MAPE."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-6)) * 100)


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Fraction of actuals inside the prediction interval."""
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(inside.mean())


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, float]:
    result = {
        "model": model_name,
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape_%": round(mape(y_true, y_pred), 4),
        "smape_%": round(smape(y_true, y_pred), 4),
    }
    if y_lower is not None and y_upper is not None:
        result["pi_coverage_%"] = round(coverage(y_true, y_lower, y_upper) * 100, 2)
    return result


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Format a list of evaluate_model dicts into a clean comparison table."""
    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("rmse")
