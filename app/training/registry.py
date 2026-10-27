"""
Model registry utilities.
Compares Prophet vs LightGBM by RMSE and promotes the winner to
MLflow Model Registry as "Production", archiving the loser.
"""

import logging
import math
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from app.config import TRACKING_URI

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "forecastops_energy_demand"


def compare_and_promote(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Given a list of run results (each with model_name, run_id, rmse, mae, mape),
    select the best by RMSE and register it as Production in MLflow.
    Returns the winner's metadata dict.
    """
    if not results:
        raise ValueError("No model results provided for comparison.")

    if any(not math.isfinite(r["rmse"]) for r in results):
        raise ValueError("Model scores must be finite")
    mlflow.set_tracking_uri(TRACKING_URI)
    # Rank by RMSE ascending
    ranked = sorted(results, key=lambda r: r["rmse"])
    winner = ranked[0]
    loser = ranked[1] if len(ranked) > 1 else None

    logger.info("=== Model Comparison ===")
    for r in ranked:
        logger.info(
            f"  {r['model_name']:12s} | RMSE={r['rmse']:.4f} "
            f"| MAE={r['mae']:.4f} | MAPE={r['mape']:.2f}%"
        )
    logger.info(f"Winner: {winner['model_name']} (RMSE={winner['rmse']:.4f})")

    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Register winner
    model_uri = winner.get("model_uri", f"runs:/{winner['run_id']}/model")
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info(f"Registered model version: {mv.version}")

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)

    # Tag the winning run
    client.set_tag(winner["run_id"], "promoted_to_production", "true")
    if loser:
        client.set_tag(loser["run_id"], "outcome", "lost_comparison")

    winner["model_version"] = mv.version
    winner["registry_name"] = REGISTERED_MODEL_NAME
    return winner


def get_production_model_uri() -> str:
    """Return the MLflow URI for the current Production model."""
    return f"models:/{REGISTERED_MODEL_NAME}@champion"


def list_model_versions() -> list[dict[str, Any]]:
    """Return all registered versions with their stages and metrics."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return []
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion").version
    result = []
    for v in versions:
        run = client.get_run(v.run_id)
        result.append(
            {
                "version": v.version,
                "stage": "Production" if v.version == champion else "Archived",
                "model_name": run.data.tags.get("model_type", "unknown"),
                "rmse": run.data.metrics.get("rmse"),
                "mae": run.data.metrics.get("mae"),
                "mape": run.data.metrics.get("mape"),
                "run_id": v.run_id,
                "created_at": v.creation_timestamp,
            }
        )
    return sorted(result, key=lambda x: -int(x["version"]))
