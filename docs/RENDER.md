# Free Render demo

The public portfolio demo uses one free Docker web service. `Dockerfile.render` trains the seeded two-year dataset during the build and bundles the SQLite registry, model artifacts and CSV into the image at `/app`. `scripts/serve_demo.py` starts an internal API on loopback, waits for model readiness, then starts the public Streamlit dashboard on `$PORT`. If either process fails, both stop so Render can restart the instance.

Public retraining and scheduled training are disabled by the launcher. Run the full Compose stack locally to demonstrate those operations. No LLM, hosted inference API or paid disk is required. The dashboard uses real trained forecasting models.

## Service settings

- Repository: `https://github.com/Navaneeth-007/ForecastOps`
- Runtime: Docker
- Dockerfile: `./Dockerfile.render`
- Context: `.`
- Plan: **free**
- Region: Singapore
- Start command: Dockerfile default
- Environment: no API keys required
- Recommended HTTP health path: `/_stcore/health`

The service is created directly through Render; these settings document its configuration. There is no Blueprint managing or overwriting it.

## Local equivalent

```sh
docker build -f Dockerfile.render -t forecastops-demo .
docker run --rm --memory=512m --cpus=0.1 -p 127.0.0.1:10000:10000 forecastops-demo
```

Open `http://localhost:10000`. The API's port is not published. Forecast artifacts are regenerated on each image build; writable runtime changes are disposable. Free instances can sleep when idle, so the first visitor may experience a cold start. This is a portfolio demo, not a highly available production service.
