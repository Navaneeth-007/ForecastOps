"""
LightGBM trainer with MLflow tracking.
Uses lag + rolling + calendar features for gradient-boosted forecasting.
"""
import logging
from typing import Dict, Any, List

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.features.engineer import feature_columns

logger = logging.getLogger(__name__)

LGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}


def train_and_log(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment_name: str = "ForecastOps",
) -> Dict[str, Any]:
    """
    Train LightGBM with early stopping, log to MLflow, return run metadata.
    Expects train/test to have engineered features + 'demand_kwh' target.
    """
    mlflow.set_experiment(experiment_name)
    feat_cols: List[str] = feature_columns(train)

    X_train = train[feat_cols]
    y_train = train["demand_kwh"]
    X_test = test[feat_cols]
    y_test = test["demand_kwh"]

    with mlflow.start_run(run_name="lgbm_challenger") as run:
        mlflow.log_params(LGBM_PARAMS)
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("framework", "lightgbm")
        mlflow.set_tag("n_features", len(feat_cols))

        logger.info(f"Training LightGBM on {len(X_train)} rows, {len(feat_cols)} features…")
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / np.maximum(y_test.values, 1e-6))) * 100

        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "best_iteration": model.best_iteration_,
        })
        logger.info(f"LGBM — MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

        # Feature importance (top 20)
        importance = pd.Series(
            model.feature_importances_, index=feat_cols
        ).nlargest(20)
        mlflow.log_dict(importance.to_dict(), "feature_importance_top20.json")

        mlflow.lightgbm.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    return {
        "model_name": "lightgbm",
        "run_id": run_id,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "feature_importance": importance.to_dict(),
    }
