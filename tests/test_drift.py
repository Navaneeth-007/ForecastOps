"""Tests for drift detection (PSI)."""
import numpy as np
import pandas as pd
import pytest

from app.monitoring.drift import psi, check_drift, compute_prediction_error_drift


def test_psi_identical_distributions():
    arr = np.random.default_rng(0).normal(0, 1, 1000)
    score = psi(arr, arr)
    assert score < 0.01, "PSI of identical distributions should be near 0"


def test_psi_clearly_different_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)   # completely different mean
    score = psi(ref, cur)
    assert score > 0.25, f"Expected high PSI for shifted distribution, got {score}"


def test_check_drift_no_alert():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    ref = pd.DataFrame({"date": dates[:100], "demand_kwh": rng.normal(100, 10, 100)})
    cur = pd.DataFrame({"date": dates[100:], "demand_kwh": rng.normal(101, 10, 100)})
    report = check_drift(ref, cur)
    # Slight shift — should not trigger alert
    assert isinstance(report["alert"], bool)
    assert "target_psi" in report


def test_check_drift_with_alert():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    ref = pd.DataFrame({"date": dates[:100], "demand_kwh": rng.normal(100, 5, 100)})
    cur = pd.DataFrame({"date": dates[100:], "demand_kwh": rng.normal(200, 5, 100)})
    report = check_drift(ref, cur)
    assert report["retraining_recommended"] is True


def test_prediction_error_drift_degraded():
    baseline = np.abs(np.random.default_rng(0).normal(0, 1, 500))
    current = np.abs(np.random.default_rng(0).normal(0, 5, 500))  # much larger errors
    result = compute_prediction_error_drift(baseline, current)
    assert result["degraded"] is True
    assert result["pct_change"] > 20


def test_prediction_error_drift_stable():
    errors = np.abs(np.random.default_rng(0).normal(0, 1, 500))
    result = compute_prediction_error_drift(errors, errors)
    assert result["degraded"] is False
