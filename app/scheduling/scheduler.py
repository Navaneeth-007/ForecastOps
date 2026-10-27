"""Single-process retraining coordinator with visible failures and UTC schedule."""

import logging
from datetime import UTC, datetime
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import SCHEDULER_ENABLED
from app.serving.model_loader import reload as reload_model

logger = logging.getLogger(__name__)
_scheduler = None
_lock = Lock()
_last_run = None
_last_result = None
_error = None


def trigger_now():
    global _last_run, _last_result, _error
    if not _lock.acquire(blocking=False):
        raise BlockingIOError("Retraining is already running")
    try:
        from app.training.pipeline import run_pipeline

        _last_run = datetime.now(UTC).isoformat()
        _error = None
        result = run_pipeline()
        reload_model()
        _last_result = result
        return {"status": "complete", "result": result}
    except Exception:
        _error = "Retraining failed; check server logs"
        logger.exception("Retraining failed")
        raise
    finally:
        _lock.release()


def start(weekly_day="sun", weekly_hour=2):
    global _scheduler
    if not SCHEDULER_ENABLED or (_scheduler and _scheduler.running):
        return
    import os

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        trigger_now,
        CronTrigger(
            day_of_week=os.getenv("RETRAIN_DAY", weekly_day),
            hour=int(os.getenv("RETRAIN_HOUR", weekly_hour)),
            minute=0,
            timezone="UTC",
        ),
        id="weekly_retrain",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()


def stop():
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)


def status():
    return {
        "running": bool(_scheduler and _scheduler.running),
        "training": _lock.locked(),
        "last_run": _last_run,
        "error": _error,
        "last_winner": _last_result["winner"]["model_name"] if _last_result else None,
    }
