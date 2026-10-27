"""
Retraining scheduler using APScheduler.
Runs weekly retraining + drift checks, promotes new model if it beats Production.
Can also be triggered manually via /retrain API endpoint.
"""
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.training.pipeline import run_pipeline
from app.serving.model_loader import reload as reload_model

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_last_run: Optional[datetime] = None
_last_result: Optional[dict] = None


def _retrain_job():
    """Scheduled job: retrain both models, compare, promote winner."""
    global _last_run, _last_result
    logger.info("Scheduled retraining triggered.")
    try:
        result = run_pipeline()
        reload_model()  # Hot-swap serving model
        _last_run = datetime.utcnow()
        _last_result = result
        logger.info(f"Scheduled retraining complete. Winner: {result['winner']['model_name']}")
    except Exception as e:
        logger.error(f"Scheduled retraining failed: {e}", exc_info=True)


def start(weekly_day: str = "sun", weekly_hour: int = 2):
    """
    Start the background scheduler.
    Defaults to every Sunday at 02:00 UTC.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.warning("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _retrain_job,
        trigger=CronTrigger(day_of_week=weekly_day, hour=weekly_hour, minute=0),
        id="weekly_retrain",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(f"Scheduler started — retraining every {weekly_day} at {weekly_hour:02d}:00 UTC")


def stop():
    """Gracefully stop the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


def trigger_now() -> dict:
    """Manually trigger a retraining run (used by /retrain endpoint)."""
    logger.info("Manual retrain triggered.")
    _retrain_job()
    return {"status": "complete", "result": _last_result}


def status() -> dict:
    return {
        "running": _scheduler.running if _scheduler else False,
        "last_run": _last_run.isoformat() if _last_run else None,
        "last_winner": _last_result["winner"]["model_name"] if _last_result else None,
        "last_rmse": _last_result["winner"]["rmse"] if _last_result else None,
    }
