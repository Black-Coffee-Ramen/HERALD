from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class RiskFactor:
    name: str
    severity: str
    detail: str
    score_impact: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "detail": self.detail,
            "score_impact": self.score_impact,
        }

@dataclass
class DetectionResult:
    domain: str
    verdict: str
    confidence: float
    scorer: str
    model_version: str
    risk_factors: List[RiskFactor] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "scorer": self.scorer,
            "model_version": self.model_version,
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "features": self.features,
            "explanations": self.explanations,
        }
