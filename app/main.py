"""
ForecastOps — FastAPI application.

Endpoints:
  GET  /health               — liveness + model status
  GET  /forecast?hours=24    — generate forecast from Production model
  POST /retrain              — trigger manual retraining
  GET  /model/info           — current Production model metadata
  GET  /model/versions       — all registered model versions
  GET  /drift                — run drift check on latest data window
  GET  /scheduler/status     — retraining scheduler status
"""

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_KEY
from app.ingestion.data_loader import RAW_DATA_PATH
from app.scheduling import scheduler
from app.serving.model_loader import get_model_info
from app.training.registry import list_model_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background retraining scheduler on startup
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(
    title="ForecastOps",
    description="End-to-end MLOps pipeline for energy demand forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    try:
        info = get_model_info()
        return {"status": "ok", "model": info}
    except Exception:
        return {"status": "ok", "model_ready": False}


# ─── Forecast ─────────────────────────────────────────────────────────────────


@app.get("/ready")
def ready():
    try:
        from app.serving.model_loader import load_snapshot

        _, info = load_snapshot()
        if not RAW_DATA_PATH.exists():
            raise RuntimeError("Missing history")
        return {"status": "ready", "model": info}
    except Exception:
        raise HTTPException(status_code=503, detail="Model or history unavailable; run training")


@app.get("/forecast")
def forecast(hours: int = Query(default=24, ge=1, le=168, description="Hours ahead to forecast")):
    """Generate a forecast for the next N hours from the Production model."""
    try:
        from app.serving.predictor import forecast as predict

        df = predict(horizon_hours=hours)
        return {
            "horizon_hours": hours,
            "model": df.attrs["model"],
            "forecast": df.to_dict(orient="records"),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Service unavailable; check server logs")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Service unavailable; check server logs")
    except Exception:
        logger.exception("Forecast error")
        raise HTTPException(status_code=500, detail="Service unavailable; check server logs")


# ─── Model info ───────────────────────────────────────────────────────────────


@app.get("/model/info")
def model_info():
    """Return metadata for the current Production model."""
    try:
        return get_model_info()
    except Exception:
        raise HTTPException(status_code=503, detail="Service unavailable; check server logs")


@app.get("/model/versions")
def model_versions():
    """Return all registered model versions."""
    try:
        return {"versions": list_model_versions()}
    except Exception:
        raise HTTPException(status_code=503, detail="Service unavailable; check server logs")


# ─── Drift ────────────────────────────────────────────────────────────────────


@app.get("/drift")
def drift_check():
    """
    Run a drift check comparing the training reference window
    against the most recent 4-week window of incoming data.
    """
    try:
        import pandas as pd

        from app.monitoring.drift import check_drift

        path = RAW_DATA_PATH
        if not path.exists():
            raise HTTPException(status_code=503, detail="No data available yet.")

        df = pd.read_csv(path, parse_dates=["date"])
        cutoff = df["date"].max() - pd.Timedelta(weeks=8)
        mid = df["date"].max() - pd.Timedelta(weeks=4)
        reference = df[(df["date"] > cutoff) & (df["date"] <= mid)]
        current = df[df["date"] > mid]

        report = check_drift(reference, current)
        return report
    except HTTPException:
        raise
    except Exception:
        logger.exception("Drift check error")
        raise HTTPException(status_code=500, detail="Service unavailable; check server logs")


# ─── Retraining ───────────────────────────────────────────────────────────────


@app.post("/retrain")
def retrain(x_api_key: str = Header(default="")):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Set FORECASTOPS_API_KEY to enable retraining")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    """Trigger an immediate retraining run."""
    try:
        result = scheduler.trigger_now()
        return result
    except BlockingIOError:
        raise HTTPException(status_code=409, detail="Retraining is already running")
    except Exception:
        logger.exception("Retrain error")
        raise HTTPException(status_code=500, detail="Service unavailable; check server logs")


@app.get("/scheduler/status")
def scheduler_status():
    return scheduler.status()
