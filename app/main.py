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
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        info = get_model_info()
        return {"status": "ok", "model": info}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ─── Forecast ─────────────────────────────────────────────────────────────────

@app.get("/forecast")
def forecast(hours: int = Query(default=24, ge=1, le=168, description="Hours ahead to forecast")):
    """Generate a forecast for the next N hours from the Production model."""
    try:
        from app.serving.predictor import forecast as predict
        df = predict(horizon_hours=hours)
        return {
            "horizon_hours": hours,
            "model": get_model_info(),
            "forecast": df.to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Forecast error")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Model info ───────────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    """Return metadata for the current Production model."""
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model/versions")
def model_versions():
    """Return all registered model versions."""
    try:
        return {"versions": list_model_versions()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─── Drift ────────────────────────────────────────────────────────────────────

@app.get("/drift")
def drift_check():
    """
    Run a drift check comparing the training reference window
    against the most recent 4-week window of incoming data.
    """
    try:
        import pandas as pd
        from pathlib import Path
        from app.monitoring.drift import check_drift

        path = Path("data/raw/energy.csv")
        if not path.exists():
            raise HTTPException(status_code=503, detail="No data available yet.")

        df = pd.read_csv(path, parse_dates=["date"])
        cutoff = df["date"].max() - pd.Timedelta(weeks=8)
        mid = df["date"].max() - pd.Timedelta(weeks=4)
        reference = df[df["date"] <= mid]
        current = df[df["date"] > mid]

        report = check_drift(reference, current)
        return report
    except Exception as e:
        logger.exception("Drift check error")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Retraining ───────────────────────────────────────────────────────────────

@app.post("/retrain")
def retrain():
    """Trigger an immediate retraining run."""
    try:
        result = scheduler.trigger_now()
        return result
    except Exception as e:
        logger.exception("Retrain error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scheduler/status")
def scheduler_status():
    return scheduler.status()
