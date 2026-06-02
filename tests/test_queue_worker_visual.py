import pytest
from unittest.mock import MagicMock, patch

from herald.detection.models import DetectionResult
from herald.monitoring import queue_worker

def test_process_domain_enqueues_visual_when_required():
    # Setup mocks
    mock_engine = MagicMock()
    mock_visual_queue = MagicMock()
    mock_session = MagicMock()
    mock_emitter = MagicMock()
    
    # Configure the mocked DetectionEngine to return a result that REQUIRES visual analysis
    mock_detection_res = DetectionResult(
        domain="suspicious-example.com",
        verdict="Suspected",
        confidence=0.85,
        scorer="ml",
        model_version="ml-v7",
        features={
            "visual_analysis_required": True,
            "ml_confidence_adjusted": 0.85,
            "target_cse": "Generic"
        }
    )
    mock_engine.score.return_value = mock_detection_res
    mock_visual_queue.depth.return_value = {"ready": 0, "active": 0}
    
    # Ensure domain is NOT whitelisted
    mock_session.query.return_value.filter.return_value.first.return_value = None
    
    job_data = {
        "domain": "suspicious-example.com",
        "target_cse": "Generic",
        "source": "api"
    }

    # Patch all the globals/dependencies in queue_worker
    with patch("herald.monitoring.queue_worker.engine", mock_engine), \
         patch("herald.monitoring.queue_worker.visual_queue", mock_visual_queue), \
         patch("herald.monitoring.queue_worker.emitter", mock_emitter), \
         patch("herald.monitoring.queue_worker.SessionLocal", return_value=mock_session), \
         patch("herald.monitoring.queue_worker.init_worker"), \
         patch("herald.monitoring.queue_worker.validate_url_safe"):
             
        # Act
        queue_worker.process_domain(job_data)
        
    # Assert
    # The engine should have been called
    mock_engine.score.assert_called_once_with("suspicious-example.com")
    
    # The visual queue SHOULD receive an enqueue call
    mock_visual_queue.enqueue.assert_called_once()
    
    # Check the exact payload sent to the visual queue
    enqueued_payload = mock_visual_queue.enqueue.call_args[0][0]
    assert enqueued_payload["domain"] == "suspicious-example.com"
    assert enqueued_payload["initial_confidence"] == 0.85
    assert enqueued_payload["parent_analysis_type"] == "ml-v7"
