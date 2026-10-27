"""
Evaluation harness: runs Prophet and LightGBM on the held-out test set
and prints a formatted comparison table.

Usage:
    python eval/run_eval.py
"""
import sys
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 1. Load data + features
    from app.ingestion.data_loader import download_or_load, validate_schema
    from app.features.engineer import build_features, train_test_split, feature_columns
    from eval.metrics import evaluate_model, comparison_table

    raw = validate_schema(download_or_load())
    features = build_features(raw)
    train, test = train_test_split(features, test_weeks=4)

    results = []

    # 2. Prophet
    try:
        from prophet import Prophet
        train_p = train[["date", "demand_kwh"]].rename(columns={"date": "ds", "demand_kwh": "y"})
        test_p = test[["date", "demand_kwh"]].rename(columns={"date": "ds", "demand_kwh": "y"})
        m_prophet = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, seasonality_mode="additive"
        )
        m_prophet.fit(train_p)
        fc = m_prophet.predict(test_p[["ds"]])
        results.append(evaluate_model(
            y_true=test_p["y"].values,
            y_pred=fc["yhat"].values,
            y_lower=fc["yhat_lower"].values,
            y_upper=fc["yhat_upper"].values,
            model_name="Prophet",
        ))
        logger.info("Prophet evaluation done.")
    except Exception as e:
        logger.error(f"Prophet eval failed: {e}")

    # 3. LightGBM
    try:
        import lightgbm as lgb
        feat_cols = feature_columns(train)
        m_lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1)
        m_lgbm.fit(train[feat_cols], train["demand_kwh"])
        y_pred = m_lgbm.predict(test[feat_cols])
        mae_val = np.mean(np.abs(test["demand_kwh"].values - y_pred))
        results.append(evaluate_model(
            y_true=test["demand_kwh"].values,
            y_pred=y_pred,
            y_lower=y_pred - 1.96 * mae_val,
            y_upper=y_pred + 1.96 * mae_val,
            model_name="LightGBM",
        ))
        logger.info("LightGBM evaluation done.")
    except Exception as e:
        logger.error(f"LightGBM eval failed: {e}")

    # 4. Print comparison
    if results:
        print("\n" + "=" * 60)
        print("  FORECASTOPS — Model Comparison (hold-out test set)")
        print("=" * 60)
        table = comparison_table(results)
        print(table.to_string())
        print("=" * 60)
        winner = table.index[0]
        print(f"\n  Winner by RMSE: {winner}")
        print("  (Paste this table into README.md)\n")
    else:
        print("No results — did both models fail?")
        sys.exit(1)


if __name__ == "__main__":
    main()
