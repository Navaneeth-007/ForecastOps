"""
End-to-end training pipeline.
Orchestrates: load → validate → featurize → train both models → compare → promote.
Called by scripts/train.py and the scheduler.
"""
import logging
from pathlib import Path
from typing import Dict, Any

from app.ingestion.data_loader import download_or_load, validate_schema
from app.ingestion.validator import run_checks
from app.features.engineer import build_features, train_test_split, save_features
from app.training import prophet_trainer, lgbm_trainer
from app.training.registry import compare_and_promote

logger = logging.getLogger(__name__)


def run_pipeline(experiment_name: str = "ForecastOps") -> Dict[str, Any]:
    """
    Full training pipeline. Returns a summary dict with winner metadata.
    """
    logger.info("=" * 60)
    logger.info("ForecastOps Training Pipeline — START")
    logger.info("=" * 60)

    # 1. Ingest
    raw = download_or_load()
    raw = validate_schema(raw)
    validation_report = run_checks(raw)
    logger.info(f"Validation: {validation_report}")

    # 2. Feature engineering
    features = build_features(raw)
    save_features(features)
    train, test = train_test_split(features, test_weeks=4)

    # 3. Train both models
    prophet_result = prophet_trainer.train_and_log(
        train=train[["date", "demand_kwh"]],
        test=test[["date", "demand_kwh"]],
        experiment_name=experiment_name,
    )

    lgbm_result = lgbm_trainer.train_and_log(
        train=train,
        test=test,
        experiment_name=experiment_name,
    )

    # 4. Compare + promote winner
    winner = compare_and_promote([prophet_result, lgbm_result])

    summary = {
        "winner": winner,
        "all_results": [prophet_result, lgbm_result],
        "validation": validation_report,
    }
    logger.info(f"Pipeline complete. Winner: {winner['model_name']} (RMSE={winner['rmse']:.4f})")
    return summary
