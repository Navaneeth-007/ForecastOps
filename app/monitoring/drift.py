"""
Data drift detection using Population Stability Index (PSI).

PSI thresholds (industry standard):
  < 0.10  → no significant shift
  0.10–0.25 → moderate shift — monitor closely
  > 0.25  → significant drift — consider retraining
"""
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_BINS = 10
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    Compute the Population Stability Index between two distributions.
    Both arrays should be 1-D numeric.
    """
    # Compute bin edges from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates
    if len(breakpoints) < 2:
        return 0.0

    def _bin_pct(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)  # avoid log(0)

    pct_expected = _bin_pct(expected)
    pct_actual = _bin_pct(actual)
    index = np.sum((pct_actual - pct_expected) * np.log(pct_actual / pct_expected))
    return round(float(index), 4)


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = "demand_kwh",
) -> Dict[str, Any]:
    """
    Compare feature distributions between a reference window and
    the current incoming window.

    Returns a dict with per-feature PSI scores and overall alert flag.
    """
    report: Dict[str, Any] = {"features": {}, "alert": False, "retraining_recommended": False}

    shared_cols = [c for c in reference.columns if c in current.columns and c != "date"]

    for col in shared_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(ref_vals) < 20 or len(cur_vals) < 20:
            continue
        score = psi(ref_vals, cur_vals)
        level = (
            "ok" if score < PSI_WARN_THRESHOLD
            else "warn" if score < PSI_ALERT_THRESHOLD
            else "alert"
        )
        report["features"][col] = {"psi": score, "level": level}
        if level == "alert":
            report["alert"] = True

    # Target drift (demand itself)
    target_psi = psi(
        reference[target_col].dropna().values,
        current[target_col].dropna().values,
    )
    report["target_psi"] = target_psi
    report["target_level"] = (
        "ok" if target_psi < PSI_WARN_THRESHOLD
        else "warn" if target_psi < PSI_ALERT_THRESHOLD
        else "alert"
    )

    report["retraining_recommended"] = (
        report["alert"] or report["target_level"] == "alert"
    )

    n_alerts = sum(1 for v in report["features"].values() if v["level"] == "alert")
    logger.info(
        f"Drift check — target PSI={target_psi:.4f} ({report['target_level']}), "
        f"{n_alerts} feature alerts"
    )
    return report


def compute_prediction_error_drift(
    baseline_errors: np.ndarray,
    current_errors: np.ndarray,
) -> Dict[str, Any]:
    """
    Track whether prediction errors are growing (performance drift).
    baseline_errors and current_errors are arrays of |y_true - y_pred|.
    """
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    current_mae = float(np.mean(np.abs(current_errors)))
    pct_change = (current_mae - baseline_mae) / max(baseline_mae, 1e-6) * 100

    return {
        "baseline_mae": round(baseline_mae, 4),
        "current_mae": round(current_mae, 4),
        "pct_change": round(pct_change, 2),
        "degraded": pct_change > 20,  # >20% MAE increase triggers alert
    }
