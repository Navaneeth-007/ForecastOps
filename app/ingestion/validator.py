"""
Lightweight data validation — checks completeness, outliers, and gaps.
Logs warnings rather than hard-failing so the pipeline stays robust.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_checks(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a suite of data quality checks.
    Returns a dict of results (pass/fail + details) suitable for MLflow logging.
    """
    results: dict[str, Any] = {}

    # 1. Null rate
    null_rate = df["demand_kwh"].isna().mean()
    results["null_rate"] = round(null_rate, 4)
    if null_rate > 0.01:
        logger.warning(f"High null rate: {null_rate:.1%}")

    # 2. Outliers (IQR method)
    q1, q3 = df["demand_kwh"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df["demand_kwh"] < q1 - 3 * iqr) | (df["demand_kwh"] > q3 + 3 * iqr)).sum()
    results["outlier_count"] = int(outliers)
    if outliers > len(df) * 0.005:
        logger.warning(f"Outlier count elevated: {outliers}")

    # 3. Time-series gaps (missing hourly steps)
    df_sorted = df.sort_values("date")
    diffs = df_sorted["date"].diff().dropna()
    expected_freq = pd.Timedelta("1h")
    gaps = (diffs > expected_freq * 1.5).sum()
    results["gap_count"] = int(gaps)
    if gaps > 0:
        logger.warning(f"Time-series gaps detected: {gaps}")

    # 4. Basic stats
    results["row_count"] = len(df)
    results["mean_demand"] = round(df["demand_kwh"].mean(), 2)
    results["std_demand"] = round(df["demand_kwh"].std(), 2)

    # 5. Overall pass/fail
    results["validation_passed"] = bool(
        null_rate <= 0.01 and gaps == 0 and outliers <= len(df) * 0.01
    )
    logger.info(f"Validation complete: {results}")
    return results
