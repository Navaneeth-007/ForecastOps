# Validation record

Validated locally on 27 October 2026 with Python 3.11 on macOS ARM64, using the pinned dependency environment.

- **34 tests passed**, including real Prophet and LightGBM training, MLflow registration and promotion, and API inference for both model flavors.
- Application statement coverage: **89.98%** (CI minimum: 75%).
- Ruff lint and formatting checks passed.
- Dependency consistency and Git whitespace checks passed.
- Docker Compose configuration validation passed.
- The default two-year training command completed successfully. Benchmark values and dataset checksum are in `benchmark.json`.
- The Streamlit dashboard loaded against the running API and generated a 24-hour forecast with model metadata, demand metrics and uncertainty chart.

On the follow-up run, GitHub Actions successfully built both the standard and Render demo images on Linux. The demo passed its health check and served a 168-hour forecast under a 512 MB container memory limit. The Linux lint, formatting and full test job passed as well.

Verification run: https://github.com/Navaneeth-007/ForecastOps/actions/runs/40486700477

The hosted-demo supervisor also passed a local startup check: dashboard health returned 200, and the internal API returned real 24-hour predictions. Local Docker export was blocked by exhausted disk space and Docker Desktop subsequently failing to restart; the successful Linux CI run verifies the container path independently.

These checks validate the portfolio application. They do not constitute load testing, real-world forecasting validation or a high-availability deployment certification. Dependency deprecation warnings from upstream libraries remain visible during testing.

## Hosted verification

Render deployment `dep-dbg7lobrjlhs73cspifg` reached `live` on 27 October 2026. The public dashboard at https://forecastops.onrender.com loaded its built-in LightGBM champion, generated a 24-hour forecast through the internal API, and returned `ok` from `/_stcore/health`. The public UI has no retraining secret input or start-training control.

The maximum 168-hour forecast was also generated successfully through the live browser UI. A screenshot is saved in [live-demo.jpg](images/live-demo.jpg). Free-plan CPU limits make longer horizons slower than the local demo.
