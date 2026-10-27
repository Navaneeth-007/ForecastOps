"""Central configuration shared by CLI, serving and training."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_TRACKING_URI", TRACKING_URI)
API_KEY = os.getenv("FORECASTOPS_API_KEY", "")
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
