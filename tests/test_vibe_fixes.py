import pytest
from fastapi.testclient import TestClient
import os
import multiprocessing
import time
from unittest import mock

os.environ["ALLOW_REGISTRATION"] = "false"
os.environ["REDIS_HOST"] = "localhost"

from herald.api.main import app, get_redis_client
from herald.predict_with_fallback import PhishingPredictorV3

client = TestClient(app)

def test_dependency_injection():
    def override_get_redis():
        return "mock_redis"
    
    app.dependency_overrides[get_redis_client] = override_get_redis
    assert get_redis_client in app.dependency_overrides
    app.dependency_overrides = {}

def test_registration_disabled():
    os.environ["ALLOW_REGISTRATION"] = "false"
    response = client.post("/api/auth/register", json={"username": "testuser", "password": "password123"})
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()

@mock.patch("herald.api.main.get_db")
def test_registration_enabled(mock_get_db):
    os.environ["ALLOW_REGISTRATION"] = "true"
    # Using mock DB to avoid sqlite errors during testing
    try:
        response = client.post("/api/auth/register", json={"username": "testuser", "password": "password123"})
        assert response.status_code != 403
    finally:
        os.environ["ALLOW_REGISTRATION"] = "false"

@mock.patch("multiprocessing.context.SpawnProcess.is_alive")
@mock.patch("multiprocessing.context.SpawnProcess.join")
@mock.patch("multiprocessing.context.SpawnProcess.start")
@mock.patch("multiprocessing.context.SpawnProcess.terminate")
def test_visual_analyzer_timeout(mock_terminate, mock_start, mock_join, mock_is_alive):
    with mock.patch("joblib.load"), mock.patch("yaml.safe_load"), mock.patch("os.path.exists", return_value=True), mock.patch("builtins.open", mock.mock_open(read_data="{}")):
        predictor = PhishingPredictorV3(model_path="dummy", config_path="dummy")
            
    mock_is_alive.return_value = True
    
    result = predictor.analyze_visual_fallback("example.com", "test", 0.5)
    assert result["cv_ocr_confirmed"] == False
    assert result["final_confidence"] == 0.5
