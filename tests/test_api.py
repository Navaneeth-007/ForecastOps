from fastapi.testclient import TestClient

from app import main


def test_liveness_survives_missing_model(monkeypatch):
    monkeypatch.setattr(main, "get_model_info", lambda: (_ for _ in ()).throw(RuntimeError()))
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["model_ready"] is False


def test_horizon_validation():
    with TestClient(main.app) as client:
        for hours in (0, 169, "bad"):
            assert client.get("/forecast", params={"hours": hours}).status_code == 422


def test_retraining_requires_configured_secret(monkeypatch):
    with TestClient(main.app) as client:
        monkeypatch.setattr(main, "API_KEY", "")
        assert client.post("/retrain").status_code == 503
        monkeypatch.setattr(main, "API_KEY", "secret")
        assert client.post("/retrain").status_code == 401
        monkeypatch.setattr(main.scheduler, "trigger_now", lambda: {"status": "complete"})
        assert client.post("/retrain", headers={"X-API-Key": "secret"}).status_code == 200


def test_drift_missing_data_is_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RAW_DATA_PATH", tmp_path / "missing.csv")
    with TestClient(main.app) as client:
        assert client.get("/drift").status_code == 503
