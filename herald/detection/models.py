from dataclasses import dataclass, field
from typing import Any, List, Dict
from enum import Enum

class Verdict(Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    PHISHING = "PHISHING"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"

VERDICT_DISPLAY = {
    Verdict.CLEAN: "Likely Clean",
    Verdict.SUSPICIOUS: "Suspected",
    Verdict.PHISHING: "Phishing",
    Verdict.PENDING: "Pending",
    Verdict.REJECTED: "Rejected",
    Verdict.DEGRADED: "Degraded",
    Verdict.UNKNOWN: "Unknown",
}

LEGACY_VERDICT_MAP = {
    "CLEAN": Verdict.CLEAN,
    "LIKELY CLEAN": Verdict.CLEAN,
    "SUSPECTED": Verdict.SUSPICIOUS,
    "SUSPICIOUS": Verdict.SUSPICIOUS,
    "PHISHING": Verdict.PHISHING,
    "PENDING": Verdict.PENDING,
    "REJECTED": Verdict.REJECTED,
    "DEGRADED": Verdict.DEGRADED,
    "UNKNOWN": Verdict.UNKNOWN,
}

def normalize_verdict(verdict: Verdict | str | None) -> Verdict:
    """Normalize legacy display/status strings into the internal Verdict enum."""
    if isinstance(verdict, Verdict):
        return verdict
    if not verdict:
        return Verdict.UNKNOWN

    return LEGACY_VERDICT_MAP.get(str(verdict).strip().upper(), Verdict.UNKNOWN)

def display_verdict(verdict: Verdict | str) -> str:
    """Safely map a Verdict enum (or string) to its legacy human-readable display label."""
    normalized = normalize_verdict(verdict)
    if normalized == Verdict.UNKNOWN and isinstance(verdict, str):
        return verdict
    return VERDICT_DISPLAY.get(normalized, "Unknown")

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
    verdict: Verdict
    confidence: float
    scorer: str
    model_version: str
    risk_factors: List[RiskFactor] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    
    # CISO-grade metadata
    threshold: float = 0.5
    feature_count: int = 0
    fallback_triggered: bool = False
    visual_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "verdict": display_verdict(self.verdict),
            "confidence": self.confidence,
            "scorer": self.scorer,
            "model_version": self.model_version,
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "features": self.features,
            "explanations": self.explanations,
            "metadata": {
                "threshold": self.threshold,
                "feature_count": self.feature_count,
                "fallback_triggered": self.fallback_triggered,
                "visual_required": self.visual_required,
            }
        }
