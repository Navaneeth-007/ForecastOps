"""
Prophet model trainer with MLflow tracking.
Prophet is the classical baseline — interpretable, requires no feature engineering.
"""

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

PROPHET_PARAMS: dict[str, Any] = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": True,
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "interval_width": 0.95,
}


class ProphetWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper so Prophet can be logged as a pyfunc model."""

    def __init__(self, model: Prophet):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """model_input must have a 'ds' column."""
        forecast = self.model.predict(model_input)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def train_and_log(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment_name: str = "ForecastOps",
) -> dict[str, Any]:
    """
    Train Prophet, log params + metrics to MLflow, return run metadata.
    Expects train/test to have 'date' and 'demand_kwh' columns.
    """
    mlflow.set_experiment(experiment_name)

    # Prophet expects ds / y column names
    train_prophet = train[["date", "demand_kwh"]].rename(columns={"date": "ds", "demand_kwh": "y"})
    test_prophet = test[["date", "demand_kwh"]].rename(columns={"date": "ds", "demand_kwh": "y"})

    with mlflow.start_run(run_name="prophet_baseline") as run:
        mlflow.log_params(PROPHET_PARAMS)
        mlflow.set_tag("model_type", "prophet")
        mlflow.set_tag("framework", "prophet")

        logger.info("Fitting Prophet model…")
        m = Prophet(**PROPHET_PARAMS)
        m.fit(train_prophet)

        # Predict on test set
        forecast = m.predict(test_prophet[["ds"]])
        y_pred = forecast["yhat"].values
        y_true = test_prophet["y"].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        logger.info(f"Prophet — MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

        # Log model as pyfunc
        wrapped = ProphetWrapper(m)
        logged_model = mlflow.pyfunc.log_model(
            name="model",
            input_example=test_prophet[["ds"]].head(3),
            python_model=wrapped,
        )

        run_id = run.info.run_id

    return {
        "model_name": "prophet",
        "run_id": run_id,
        "model_uri": logged_model.model_uri,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
