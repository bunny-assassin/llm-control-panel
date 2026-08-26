from fastapi.testclient import TestClient

from backend.main import app


def test_index_and_model_list():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Local LLM Control Panel" in page.text
        models = client.get("/api/models")
        assert models.status_code == 200
        body = models.json()
        ids = {row["id"] for row in body["models"]}
        assert "qwen3.6-35b-moe" in ids
        assert "qwen3.6-27b-dense" in ids
        health = client.get("/api/health")
        assert health.json()["status"] == "ok"
        status = client.get("/api/status")
        assert "metrics" in status.json()
        assert "status" in status.json()
