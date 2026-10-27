# Portfolio demo walkthrough

1. Start the project using the README. Explain that the default input is synthetic, reproducible hourly data; bring a real CSV to demonstrate external ingestion.
2. Open the dashboard. Generate a 24-hour forecast and explain the champion version, RMSE and demand curve. Export a CSV, then choose a seven-day horizon to demonstrate recursive serving.
3. Load model history. Explain that both candidates are tracked, but only the winner is registered by the normal pipeline. The `champion` alias is the serving pointer; artifacts remain versioned for rollback.
4. Run a drift check. Explain adjacent-window PSI and why seasonal changes are not proof of degraded predictive performance.
5. Open `/docs`. Show request validation with an invalid forecast horizon and explain the distinction between process liveness and model readiness.
6. Run the test suite. Highlight the integration test: it trains real Prophet and LightGBM models, promotes each, and verifies API forecasts from each artifact.
7. Show `data/reports/latest.json` and the checked-in benchmark. Explain the chronological split, separate early-stopping window, recursive evaluation, and weekly seasonal baseline.

## Engineering choices to discuss

- Future lag features use past predictions rather than unseen actual observations.
- Registry alias resolution and immutable version loading prevent mismatched model metadata during promotion.
- Administrative retraining uses a secret and a single-process lock; a multi-replica system needs an external worker and distributed coordination.
- Prediction bands are labeled honestly; empirical coverage calibration would be needed before operational use.
- Docker keeps data and artifacts on a persistent volume and runs as an unprivileged user.

Avoid presenting synthetic metrics as evidence of real-world generalization or claiming that this demo recreates work at a particular employer. Discuss only experience and results you can substantiate.
