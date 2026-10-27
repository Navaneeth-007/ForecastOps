"""Tests for feature engineering pipeline."""
import numpy as np
import pandas as pd
import pytest

from app.ingestion.data_loader import _generate_synthetic, validate_schema
from app.features.engineer import build_features, train_test_split, feature_columns


@pytest.fixture
def raw_df():
    df = _generate_synthetic(n_days=60)
    return validate_schema(df)


def test_synthetic_generation():
    df = _generate_synthetic(n_days=7)
    assert len(df) == 7 * 24
    assert "date" in df.columns
    assert "demand_kwh" in df.columns
    assert df["demand_kwh"].isna().sum() == 0


def test_validate_schema_passes(raw_df):
    assert "date" in raw_df.columns
    assert "demand_kwh" in raw_df.columns
    assert raw_df["demand_kwh"].dtype == float


def test_validate_schema_fails_on_missing_col():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10, freq="h")})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_build_features_shape(raw_df):
    features = build_features(raw_df)
    assert len(features) > 0
    assert "demand_kwh" in features.columns
    assert "lag_24h" in features.columns
    assert "hour_sin" in features.columns
    assert "rolling_mean_24h" in features.columns
    # All lags available — some rows should be dropped
    assert len(features) < len(raw_df)


def test_build_features_no_nulls(raw_df):
    features = build_features(raw_df)
    assert features.isnull().sum().sum() == 0


def test_train_test_split(raw_df):
    features = build_features(raw_df)
    train, test = train_test_split(features, test_weeks=2)
    assert len(train) > 0
    assert len(test) > 0
    assert train["date"].max() < test["date"].min()


def test_feature_columns_excludes_target(raw_df):
    features = build_features(raw_df)
    cols = feature_columns(features)
    assert "demand_kwh" not in cols
    assert "date" not in cols
    assert "lag_24h" in cols
