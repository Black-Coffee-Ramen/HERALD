import pytest
from unittest.mock import patch, MagicMock

from herald.detection.models import DetectionResult, Verdict, display_verdict, normalize_verdict
from herald.detection.heuristic_scorer import HeuristicScorer
from herald.monitoring.queue_worker import process_domain

def test_legacy_verdict_normalization():
    assert normalize_verdict("Clean") == Verdict.CLEAN
    assert normalize_verdict("Likely Clean") == Verdict.CLEAN
    assert normalize_verdict("Suspected") == Verdict.SUSPICIOUS
    assert normalize_verdict("Phishing") == Verdict.PHISHING
    assert display_verdict(Verdict.SUSPICIOUS) == "Suspected"

def test_heuristic_scorer_contract():
    scorer = HeuristicScorer()
    res = scorer.score("example.com")
    
    assert isinstance(res, DetectionResult)
    assert isinstance(res.verdict, Verdict)
    assert isinstance(res.confidence, float)
    assert isinstance(res.features, dict)
    assert isinstance(res.explanations, list)
    assert res.scorer == "heuristic"
    assert hasattr(res, "model_version")

@patch("herald.monitoring.queue_worker.validate_url_safe")
@patch("herald.monitoring.queue_worker.enqueue_visual_analysis")
@patch("herald.monitoring.queue_worker.engine")
def test_queue_worker_visual_escalation(mock_engine, mock_enqueue, mock_validate):
    """Test that the worker safely escalates borderline ML results to the visual queue."""
    mock_res = DetectionResult(
        domain="borderline.com",
        verdict=Verdict.SUSPICIOUS,
        confidence=0.55,
        scorer="ml",
        model_version="mock-v1",
        features={
            "visual_analysis_required": True, 
            "ml_confidence": 0.55, 
            "ml_confidence_adjusted": 0.55
        }
    )
    mock_engine.score.return_value = mock_res
    
    job_data = {
        "domain": "borderline.com",
        "target_cse": "test_cse",
        "source": "test",
        "trace_id": "test-trace"
    }
    
    with patch("herald.monitoring.queue_worker.SessionLocal") as mock_session:
        # Prevent the whitelist check from returning a truthy mock
        mock_db = mock_session.return_value
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with patch("herald.monitoring.queue_worker.emitter"):
            with patch("herald.monitoring.queue_worker.redis_client"):
                process_domain(job_data)
                
    mock_enqueue.assert_called_once()
    args, kwargs = mock_enqueue.call_args
    
    assert args[0] == "borderline.com"
    assert args[1]["visual_analysis_required"] is True
    assert args[1]["target_cse"] == "test_cse"
