from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["gemini_configured"], bool)


def test_health_does_not_expose_api_key():
    response = client.get("/health")
    assert "google_api_key" not in response.json()


def test_settings_without_key_is_not_configured():
    settings = Settings(_env_file=None, google_api_key=None)
    assert settings.gemini_configured is False


def test_settings_secret_is_masked():
    settings = Settings(_env_file=None, google_api_key="dummy-value")
    assert settings.gemini_configured is True
    assert "dummy-value" not in repr(settings)
