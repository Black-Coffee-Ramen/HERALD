"""Structured result type returned by :class:`~herald.detector.PhishingDetector`."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class RiskLevel(str, Enum):
    """Human-readable risk classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Map a continuous risk *score* in [0, 1] to a :class:`RiskLevel`.

        Thresholds:
          * < 0.25  → LOW
          * < 0.50  → MEDIUM
          * < 0.75  → HIGH
          * ≥ 0.75  → CRITICAL
        """
        if score < 0.25:
            return cls.LOW
        if score < 0.50:
            return cls.MEDIUM
        if score < 0.75:
            return cls.HIGH
        return cls.CRITICAL


@dataclass
class DetectionResult:
    """Full output of a phishing detection analysis.

    Attributes:
        is_phishing: ``True`` when the sample exceeds the configured
            risk threshold.
        risk_score: Continuous score in [0, 1] indicating phishing
            likelihood.  Higher values indicate greater risk.
        risk_level: Categorical risk classification derived from
            *risk_score*.
        confidence: Model confidence in the prediction (0–1).
        indicators: Human-readable explanations of the strongest signals
            that contributed to the score.
        url_score: Contribution from URL-based features (0–1).
        text_score: Contribution from text-based features (0–1).
        header_score: Contribution from email-header features (0–1).
        feature_scores: Raw per-feature score map for auditing and
            integration with SIEM platforms.
    """

    is_phishing: bool
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    indicators: List[str] = field(default_factory=list)
    url_score: float = 0.0
    text_score: float = 0.0
    header_score: float = 0.0
    feature_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary suitable for JSON encoding."""
        return {
            "is_phishing": self.is_phishing,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "confidence": round(self.confidence, 4),
            "indicators": self.indicators,
            "url_score": round(self.url_score, 4),
            "text_score": round(self.text_score, 4),
            "header_score": round(self.header_score, 4),
            "feature_scores": {k: round(v, 4) for k, v in self.feature_scores.items()},
        }

    def summary(self) -> str:
        """Return a short human-readable summary of the result."""
        verdict = "PHISHING" if self.is_phishing else "LEGITIMATE"
        lines = [
            f"Verdict   : {verdict}",
            f"Risk Level: {self.risk_level.value}",
            f"Risk Score: {self.risk_score:.4f}",
            f"Confidence: {self.confidence:.4f}",
        ]
        if self.indicators:
            lines.append("Indicators:")
            for ind in self.indicators:
                lines.append(f"  • {ind}")
        return "\n".join(lines)
