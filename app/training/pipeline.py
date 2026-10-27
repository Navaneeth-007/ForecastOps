"""
End-to-end training pipeline.
Orchestrates: load → validate → featurize → train both models → compare → promote.
Called by scripts/train.py and the scheduler.
"""

import hashlib
import logging
from typing import Any

import numpy as np

from app.features.engineer import build_features, save_features, train_test_split
from app.ingestion.data_loader import download_or_load, validate_schema
from app.ingestion.validator import run_checks
from app.training import lgbm_trainer, prophet_trainer
from app.training.registry import compare_and_promote
from eval.metrics import evaluate_model

logger = logging.getLogger(__name__)


def run_pipeline(experiment_name: str = "ForecastOps") -> dict[str, Any]:
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
    if not validation_report["validation_passed"]:
        raise ValueError("Data quality checks failed")

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
        "dataset": {
            "rows": len(raw),
            "start": raw.date.min().isoformat(),
            "end": raw.date.max().isoformat(),
            "sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        },
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": test.date.min().isoformat(),
            "test_end": test.date.max().isoformat(),
        },
        "baseline": evaluate_model(
            test.demand_kwh.to_numpy(),
            np.resize(train.demand_kwh.tail(168).to_numpy(), len(test)),
            model_name="seasonal_naive_weekly",
        ),
    }
    import json

    from app.config import DATA_DIR

    report_path = DATA_DIR / "reports"
    report_path.mkdir(parents=True, exist_ok=True)
    (report_path / "latest.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    logger.info(f"Pipeline complete. Winner: {winner['model_name']} (RMSE={winner['rmse']:.4f})")
    return summary
