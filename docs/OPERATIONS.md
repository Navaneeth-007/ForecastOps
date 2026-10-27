# Operations

## Supported deployment

The included Compose deployment runs one API process, one dashboard and a one-shot training service with one persistent volume. Ports bind to localhost. Training and serving share an MLflow SQLite database and immutable model artifacts in `/state`. Do not delete the volume unless intentionally resetting all experiments and data.

For an internet-facing deployment, put the services behind a TLS reverse proxy with authentication and request limits. Keep MLflow and the API administration path private. Configure `FORECASTOPS_API_KEY` before enabling dashboard retraining. The key must be sent as `X-API-Key`; it is never returned by the API.

The in-process training lock coordinates scheduled and manual runs only within one process. Run exactly one API worker when the scheduler is enabled, and do not run a second CLI training job concurrently. For multiple replicas, move retraining to an external queue with a distributed lock and use a hosted MLflow server backed by PostgreSQL and shared object storage.

## Data contract

Supply `data/raw/energy.csv` (or `$DATA_DIR/raw/energy.csv`) before training to use your own data. Columns are `date` and `demand_kwh`. Timestamps must be unique, contiguous and hourly; demand must be finite and nonnegative. Offset-aware timestamps are converted to UTC, and naive timestamps are interpreted as UTC. Invalid data fails training rather than silently filling gaps. Use at least 60 days; yearly patterns need much longer histories.

The default is a deterministic, generated two-year dataset starting in 2022. There is no UCI download. Forecast timestamps start after the final observation, not the current clock. Repeated retraining on unchanged data is a demonstration of the lifecycle, not new information or guaranteed improvement.

## Health and incident response

- `/health` indicates that the API is alive; `/ready` returns 503 until the model can be loaded and historical data exists.
- `/scheduler/status` exposes whether training is active and whether the last run failed. Details remain in server logs.
- A failed training run before promotion leaves the existing champion untouched. Verify `/ready` after promotion.
- `/drift` compares the last four weeks with the preceding four weeks of the local CSV. Seasonal changes can trigger PSI; this is descriptive monitoring, not automatic proof that the model needs replacement.
- Manual retraining is synchronous. Allow a long proxy timeout or use CLI training for long jobs. A request timeout does not cancel an already running job; check status before retrying.

## Backup and rollback

Stop writers before copying `/state` (or local `mlflow.db`, model artifacts and `data/`). Back up the database and artifacts together and test restoring them. SQLite is intended for this single-node demo; configure a hosted tracking server for larger deployments.

To roll back, use an authenticated MLflow client against the configured tracking URI:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("forecastops_energy_demand", "champion", "1")
```

Choose a verified version from `/model/versions`. The next forecast resolves the alias and loads that immutable version. Do not load untrusted MLflow artifacts: serialized models execute Python code.

## Evaluation limits

Candidates are scored on the same chronological four-week holdout. LightGBM uses a separate inner validation segment for early stopping, refits on training only, then recursively predicts the entire holdout without observed future targets. Prophet forecasts the same window. Scores select between candidates in the current run; there is no incumbent non-regression guarantee across changing datasets. Neither model's score should be presented as a real-world benchmark on the synthetic dataset.

LightGBM bands use ±1.96 × holdout MAE and are explicitly heuristic, with no 95% coverage guarantee or horizon calibration. Prophet bands are model-based estimates. Validate coverage and performance with rolling-origin backtests on actual operational data before relying on the forecasts.
