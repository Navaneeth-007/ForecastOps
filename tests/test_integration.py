"""Real MLflow round trip for both model flavors; isolated from local artifacts."""

import mlflow
import pytest
from fastapi.testclient import TestClient

from app.ingestion.data_loader import _generate_synthetic


@pytest.mark.integration
def test_training_registry_forecast_roundtrip(monkeypatch, tmp_path):
    from app import config, main
    from app.features import engineer
    from app.ingestion import data_loader
    from app.serving import model_loader, predictor
    from app.training import pipeline, registry

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(registry, "TRACKING_URI", uri)
    monkeypatch.setattr(model_loader, "TRACKING_URI", uri)
    path = tmp_path / "energy.csv"
    _generate_synthetic(60).to_csv(path, index=False)
    monkeypatch.setattr(data_loader, "RAW_DATA_PATH", path)
    monkeypatch.setattr(predictor, "RAW_DATA_PATH", path)
    monkeypatch.setattr(main, "RAW_DATA_PATH", path)
    monkeypatch.setattr(engineer, "PROCESSED_DIR", tmp_path / "processed")
    model_loader.reload()
    summary = pipeline.run_pipeline("Integration")
    assert summary["winner"]["rmse"] > 0
    # Exercise BOTH deployed flavors, regardless of which wins the comparison.
    for result in summary["all_results"]:
        registry.compare_and_promote([result])
        model_loader.reload()
        with TestClient(main.app) as client:
            assert client.get("/ready").status_code == 200
            assert client.get("/model/info").status_code == 200
            versions = client.get("/model/versions")
            assert versions.status_code == 200
            assert any(v["stage"] == "Production" for v in versions.json()["versions"])
            assert client.get("/drift").status_code == 200
            response = client.get("/forecast?hours=24")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert len(payload["forecast"]) == 24
            assert payload["model"]["model_type"] == result["model_name"]
            assert all(
                row["yhat_lower"] <= row["yhat"] <= row["yhat_upper"] for row in payload["forecast"]
            )
    model_loader.reload()
