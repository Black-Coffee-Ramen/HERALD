from typing import Literal
from herald.detection.interfaces import Scorer
from herald.detection.models import DetectionResult
from herald.detection.heuristic_scorer import HeuristicScorer
from herald.detection.ml_scorer import MLScorer
from herald.detection.hybrid_scorer import HybridScorer

ScorerType = Literal["heuristic", "ml", "hybrid"]

class DetectionEngine:
    def __init__(self, scorer_type: ScorerType = "heuristic"):
        self.scorer_type = scorer_type
        if scorer_type == "ml":
            self.scorer: Scorer = MLScorer()
        elif scorer_type == "hybrid":
            self.scorer: Scorer = HybridScorer()
        else:
            self.scorer: Scorer = HeuristicScorer()

    def score(self, domain: str) -> DetectionResult:
        return self.scorer.score(domain)

    def preload(self) -> None:
        """Explicitly trigger lazy-loading of models (useful before UI spinners)."""
        if hasattr(self.scorer.__class__, "predictor"):
            _ = getattr(self.scorer, "predictor")
