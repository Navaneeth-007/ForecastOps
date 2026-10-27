"""Tests for data validation."""

import numpy as np
import pandas as pd

from app.ingestion.validator import run_checks


def _make_df(n=200, gap=False, outlier=False):
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    demand = np.random.default_rng(0).normal(100, 10, n)
    if outlier:
        demand[50] = 9999.0
    df = pd.DataFrame({"date": dates, "demand_kwh": demand})
    if gap:
        df = df.drop(df.index[100])
    return df


def test_run_checks_clean():
    df = _make_df()
    report = run_checks(df)
    assert report["null_rate"] == 0.0
    assert report["validation_passed"] is True


def test_run_checks_gap_detected():
    df = _make_df(gap=True)
    report = run_checks(df)
    assert report["gap_count"] > 0


def test_run_checks_outlier_detected():
    df = _make_df(outlier=True)
    report = run_checks(df)
    assert report["outlier_count"] > 0


def test_run_checks_returns_stats():
    df = _make_df()
    report = run_checks(df)
    assert "mean_demand" in report
    assert "std_demand" in report
    assert report["row_count"] == 200
