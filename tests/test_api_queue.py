import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock Redis before importing app to avoid connection errors during test collection
with patch('redis.Redis') as mock_redis:
    from herald.api.main import app

client = TestClient(app)

def test_api_scan_enqueues_job():
    """Test that /api/scan correctly enqueues a job into Redis."""
    # Create a mock queue
    mock_queue = MagicMock()
    mock_queue.depth.return_value = {"ready": 0, "processing": 0, "delayed": 0, "dlq": 0}
    mock_queue.enqueue.return_value = "job-1234"
    
    # Mock the get_domain_queue to return our mock queue
    with patch('herald.api.main.get_domain_queue', return_value=mock_queue):
        # We need to mock dependency get_current_user to bypass authentication
        from herald.api.main import get_current_user
        from herald.db.models import User
        
        app.dependency_overrides[get_current_user] = lambda: User(username="test_analyst")
        
        response = client.post("/api/scan", json={"domain": "example.com"})
        
        # Reset override
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["job_id"] == "job-1234"
        
        # Verify the queue was called correctly
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args[0][0]
        assert call_args["domain"] == "example.com"
        assert call_args["source"] == "api_manual"
        assert "trace_id" in call_args
