"""
Model loader that always serves the current Production model from MLflow.
Caches in memory — call reload() to hot-swap after a new promotion.
"""
import logging
from typing import Optional, Any

import mlflow.pyfunc
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from app.training.registry import REGISTERED_MODEL_NAME, get_production_model_uri

logger = logging.getLogger(__name__)

_cached_model: Optional[Any] = None
_cached_model_type: Optional[str] = None
_cached_version: Optional[str] = None


def load_production_model() -> Any:
    """Load (or return cached) current Production model."""
    global _cached_model, _cached_model_type, _cached_version

    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production model found in MLflow registry. Run training first.")

    latest = versions[0]
    if latest.version == _cached_version:
        logger.debug(f"Returning cached model v{_cached_version}")
        return _cached_model

    logger.info(f"Loading Production model v{latest.version} (run {latest.run_id})…")
    run = client.get_run(latest.run_id)
    model_type = run.data.tags.get("model_type", "unknown")

    uri = get_production_model_uri()
    if model_type == "lightgbm":
        model = mlflow.lightgbm.load_model(uri)
    else:
        model = mlflow.pyfunc.load_model(uri)

    _cached_model = model
    _cached_model_type = model_type
    _cached_version = latest.version
    logger.info(f"Loaded {model_type} v{latest.version}")
    return model


def get_model_info() -> dict:
    """Return metadata about the currently loaded Production model."""
    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return {"status": "no_model"}
    v = versions[0]
    run = client.get_run(v.run_id)
    return {
        "version": v.version,
        "model_type": run.data.tags.get("model_type"),
        "rmse": run.data.metrics.get("rmse"),
        "mae": run.data.metrics.get("mae"),
        "mape": run.data.metrics.get("mape"),
        "run_id": v.run_id,
    }


def reload():
    """Force a reload on next call (e.g. after new promotion)."""
    global _cached_version
    _cached_version = None
    logger.info("Model cache cleared — will reload on next request.")
