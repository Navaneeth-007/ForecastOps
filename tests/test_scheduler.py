import pytest

from app.scheduling import scheduler


def test_failure_propagates_and_lock_is_released(monkeypatch):
    from app.training import pipeline

    def fail():
        raise ValueError("training failure")

    monkeypatch.setattr(pipeline, "run_pipeline", fail)
    with pytest.raises(ValueError, match="training failure"):
        scheduler.trigger_now()
    assert scheduler.status()["error"] is not None
    assert scheduler.status()["training"] is False


def test_duplicate_job_rejected():
    scheduler._lock.acquire()
    try:
        with pytest.raises(BlockingIOError):
            scheduler.trigger_now()
    finally:
        scheduler._lock.release()
