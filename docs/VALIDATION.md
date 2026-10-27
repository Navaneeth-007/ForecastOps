# Validation record

Validated locally on 27 October 2026 with Python 3.11 on macOS ARM64, using the pinned dependency environment.

- **34 tests passed**, including real Prophet and LightGBM training, MLflow registration and promotion, and API inference for both model flavors.
- Application statement coverage: **89.98%** (CI minimum: 75%).
- Ruff lint and formatting checks passed.
- Dependency consistency and Git whitespace checks passed.
- Docker Compose configuration validation passed.
- The default two-year training command completed successfully. Benchmark values and dataset checksum are in `benchmark.json`.
- The Streamlit dashboard loaded against the running API and generated a 24-hour forecast with model metadata, demand metrics and uncertainty chart.

The Docker daemon was not running on the validation machine, so a local image build and Compose startup were not executed. The repository includes a Linux container build job and full test job in GitHub Actions; their remote results are not yet verified.

These checks validate the portfolio application. They do not constitute load testing, real-world forecasting validation or a high-availability deployment certification. Dependency deprecation warnings from upstream libraries remain visible during testing.
