# ForecastOps ⚡

> **End-to-end MLOps pipeline for energy demand forecasting.**  
> Prophet vs LightGBM · MLflow experiment tracking · automated model promotion · drift detection · scheduled retraining.

---

## Why this project

Most ML demos stop at a trained model. This project builds the *system around* the model:

- Two competing models trained, tracked, and **compared automatically**
- The winner is **promoted to a production registry** — no manual step
- A FastAPI endpoint **always serves the current Production model**
- **Drift detection** flags when the incoming data distribution shifts
- A **background scheduler** retrains weekly and hot-swaps the model without downtime

This mirrors the MLOps work I did at GE Aerospace on scheduled forecasting pipelines — here made fully public and reproducible.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Data Layer │ →  │  Feature Store   │ →  │  Training Pipeline │
│             │    │                  │    │                    │
│ UCI Energy  │    │  Lag features    │    │  Prophet (baseline)│
│ (synthetic  │    │  Rolling stats   │    │  LightGBM (ML)     │
│  fallback)  │    │  Calendar enc.   │    │  MLflow tracking   │
└─────────────┘    └──────────────────┘    └────────┬───────────┘
                                                    │
                                         ┌──────────▼───────────┐
                                         │   MLflow Registry    │
                                         │   Auto-promotion     │
                                         │   (winner by RMSE)   │
                                         └──────────┬───────────┘
                                                    │
                        ┌───────────────┬───────────▼───────────┐
                        │   Monitoring  │    FastAPI Serving     │
                        │               │                       │
                        │  PSI drift    │  GET /forecast        │
                        │  detection    │  POST /retrain        │
                        │  per feature  │  GET /drift           │
                        └───────────────┴───────────────────────┘
                                                    │
                                         ┌──────────▼───────────┐
                                         │  Streamlit Dashboard │
                                         │  Forecast chart      │
                                         │  Model registry view │
                                         │  Drift heatmap       │
                                         └──────────────────────┘
```

---

## Model Comparison Results

> **Paste your eval output here after running `python eval/run_eval.py`**

```
============================================================
  FORECASTOPS — Model Comparison (hold-out test set)
============================================================
              mae    rmse   mape_%  smape_%  pi_coverage_%
model                                                      
LightGBM    X.XX   X.XX    X.XX    X.XX     XX.X
Prophet     X.XX   X.XX    X.XX    X.XX     XX.X
============================================================
Winner by RMSE: [LightGBM / Prophet]
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/forecastops.git
cd forecastops

# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # defaults work out of the box

# 3. Train both models + auto-promote winner
python scripts/train.py

# 4. Launch MLflow UI (separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 5. Launch API
uvicorn app.main:app --reload

# 6. Launch dashboard
streamlit run frontend/streamlit_app.py

# 7. Run tests
pytest tests/ -v --cov=app

# 8. Run eval + paste results into README
python eval/run_eval.py
```

**No API keys required** — the project ships with a synthetic energy dataset.

---

## Key Design Decisions

### Why PSI for drift detection?

Population Stability Index is the industry standard for monitoring distribution shift in production ML systems (used in banking credit models). It's interpretable (thresholds: <0.10 = ok, 0.10–0.25 = warn, >0.25 = alert) and doesn't require rerunning the model.

### Why Prophet vs LightGBM?

These represent the two dominant paradigms for time-series forecasting:

| | Prophet | LightGBM |
|---|---|---|
| **Type** | Decomposition (trend + seasonality) | Gradient boosted trees |
| **Strengths** | Interpretable, handles missing data | Captures complex lag patterns |
| **Requires** | date column only | Feature engineering (lags, rolling) |
| **Prediction interval** | Native | Proxy (±1.96 × MAE) |

The pipeline auto-selects the winner by RMSE on the held-out test set.

### Why MLflow Model Registry over pickle files?

- Versioned, auditable promotion history
- Stage transitions (Staging → Production → Archived)
- The serving layer always loads from `models:/forecastops_energy_demand/Production` — no path changes on retrain

---

## Project Structure

```
forecastops/
├── app/
│   ├── ingestion/         # data loading + schema validation
│   ├── features/          # lag, rolling, calendar feature engineering
│   ├── training/          # Prophet trainer, LightGBM trainer, registry
│   ├── serving/           # model loader (from registry), predictor
│   ├── monitoring/        # PSI drift detection
│   ├── scheduling/        # APScheduler weekly retrain
│   └── main.py            # FastAPI application
├── eval/
│   ├── metrics.py         # MAE, RMSE, MAPE, SMAPE, PI coverage
│   └── run_eval.py        # comparison harness
├── frontend/
│   └── streamlit_app.py   # 4-tab dashboard
├── scripts/
│   └── train.py           # CLI training entrypoint
├── tests/                 # pytest suite (features, drift, validation)
├── .github/workflows/ci.yml
├── pyproject.toml
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + current model info |
| `GET` | `/forecast?hours=24` | Generate N-hour forecast |
| `GET` | `/model/info` | Current Production model metadata |
| `GET` | `/model/versions` | All registered versions |
| `GET` | `/drift` | PSI drift report |
| `POST` | `/retrain` | Trigger immediate retraining |
| `GET` | `/scheduler/status` | Weekly retrain scheduler status |

---

## Resume Tie-In

*"At GE Aerospace I built forecasting systems for operational planning. ForecastOps is the public version of that architecture — two competing models (classical vs ML), automated comparison and promotion via MLflow, PSI-based drift detection, and a scheduled retraining loop. The goal was to build the production infrastructure, not just the model."*
