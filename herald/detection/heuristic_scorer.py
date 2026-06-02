from herald.detection.interfaces import Scorer
from herald.detection.models import DetectionResult, RiskFactor, Verdict
from herald.investigation.scoring import analyze_lexical

class HeuristicScorer(Scorer):
    def score(self, domain: str) -> DetectionResult:
        res = analyze_lexical(domain)
        score = res.get("score", 0.0)
        
        verdict = Verdict.CLEAN
        if score >= 0.7:
            verdict = Verdict.PHISHING
        elif score >= 0.35:
            verdict = Verdict.SUSPICIOUS

        risk_factors = [
            RiskFactor(
                name=rf["name"],
                severity=rf["severity"],
                detail=rf["detail"],
                score_impact=rf.get("score_impact", 0.0)
            )
            for rf in res.get("risk_factors", [])
        ]

        return DetectionResult(
            domain=domain,
            verdict=verdict,
            confidence=score,
            scorer="heuristic",
            model_version="lexical-v1",
            risk_factors=risk_factors,
            features=res.get("features", {}),
            explanations=[rf["detail"] for rf in res.get("risk_factors", [])]
        )
