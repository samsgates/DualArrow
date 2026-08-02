import os

from fastapi.testclient import TestClient


def test_health(project, monkeypatch):
    monkeypatch.setenv("DUALARROW_CONFIG", str(project / "dualarrow.yaml"))
    from dualarrow.api.app import create_app

    with TestClient(create_app(project / "dualarrow.yaml")) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_api_ingest_and_query(project, monkeypatch):
    monkeypatch.setenv("DUALARROW_CONFIG", str(project / "dualarrow.yaml"))
    from dualarrow.api.app import create_app

    with TestClient(create_app(project / "dualarrow.yaml")) as client:
        ingest = client.post("/v1/ingest")
        assert ingest.status_code == 200
        response = client.post(
            "/v1/query",
            json={"query": "Which bearing is approved for AX-100?"},
        )
        assert response.status_code == 200
        assert "BR-900" in response.json()["answer"]
