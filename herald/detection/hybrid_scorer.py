from herald.detection.interfaces import Scorer
from herald.detection.models import DetectionResult
from herald.detection.ml_scorer import MLScorer
from herald.detection.heuristic_scorer import HeuristicScorer

class HybridScorer(Scorer):
    def __init__(self):
        self.ml_scorer = MLScorer()
        self.heuristic_scorer = HeuristicScorer()

    def score(self, domain: str) -> DetectionResult:
        res_ml = self.ml_scorer.score(domain)
        res_heur = self.heuristic_scorer.score(domain)
        
        return DetectionResult(
            domain=domain,
            verdict=res_ml.verdict,
            confidence=res_ml.confidence,
            scorer="hybrid",
            model_version=f"{res_ml.model_version}+{res_heur.model_version}",
            risk_factors=res_heur.risk_factors,
            features={**res_heur.features, **res_ml.features},
            explanations=[*res_ml.explanations, *res_heur.explanations],
            threshold=res_ml.threshold,
            feature_count=res_ml.feature_count,
            fallback_triggered=res_ml.fallback_triggered,
            visual_required=res_ml.visual_required
        )
