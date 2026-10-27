"""Thread-safe model snapshots loaded by immutable version, never mutable URI."""

from threading import RLock

import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from app.config import TRACKING_URI
from app.training.registry import REGISTERED_MODEL_NAME

_lock = RLock()
_cache = None


def _version():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except Exception as exc:
        raise RuntimeError("No champion model available. Run training first.") from exc
    run = client.get_run(version.run_id)
    info = {
        "version": version.version,
        "model_type": run.data.tags.get("model_type"),
        "rmse": run.data.metrics.get("rmse"),
        "mae": run.data.metrics.get("mae"),
        "mape": run.data.metrics.get("mape"),
        "run_id": version.run_id,
    }
    return info


def load_snapshot():
    global _cache
    with _lock:
        info = _version()
        if _cache is None or _cache[1]["version"] != info["version"]:
            uri = f"models:/{REGISTERED_MODEL_NAME}/{info['version']}"
            loader = mlflow.lightgbm if info["model_type"] == "lightgbm" else mlflow.pyfunc
            _cache = (loader.load_model(uri), info)
        return _cache


def load_production_model():
    return load_snapshot()[0]


def get_model_info():
    return _version()


def reload():
    global _cache
    with _lock:
        _cache = None
