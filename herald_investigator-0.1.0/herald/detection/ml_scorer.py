from functools import cached_property
from herald.detection.interfaces import Scorer
from herald.detection.models import DetectionResult, normalize_verdict

class MLScorer(Scorer):
    @cached_property
    def predictor(self):
        from herald.predict_with_fallback import PhishingPredictorV3
        return PhishingPredictorV3()

    def score(self, domain: str) -> DetectionResult:
        res = self.predictor.predict(domain, include_visual=False)
        confidence = float(res.get("ml_confidence_adjusted", res.get("ml_confidence", 0.0)))
        verdict = normalize_verdict(res.get("status"))
            
        return DetectionResult(
            domain=domain,
            verdict=verdict,
            confidence=confidence,
            scorer="ml",
            model_version=res.get("analysis_type", "ml-v7"),
            risk_factors=[],
            features={
                "ml_confidence": res.get("ml_confidence", confidence),
                "ml_confidence_adjusted": confidence,
                "visual_analysis_required": bool(res.get("visual_analysis_required", False)),
                "target_cse": res.get("target_cse"),
                "content_features": res.get("content_features", {}),
            },
            explanations=[
                f"ML predictor returned {res.get('status', 'Unknown')} via {res.get('analysis_type', 'ml-v7')}."
            ],
            threshold=self.predictor.threshold,
            feature_count=len(self.predictor.feature_names),
            fallback_triggered=bool(res.get("content_features")),
            visual_required=bool(res.get("visual_analysis_required", False))
        )
