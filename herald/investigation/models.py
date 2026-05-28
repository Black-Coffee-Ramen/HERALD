from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageResult:
    name: str
    status: str
    duration_ms: int
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class InvestigationResult:
    trace_id: str
    input: str
    url: str
    domain: str
    started_at: str
    completed_at: str
    elapsed_ms: int
    verdict: str
    phishing_score: float
    evidence_dir: str
    lexical: dict[str, Any]
    dns: dict[str, Any]
    tls: dict[str, Any]
    visual: dict[str, Any]
    summary: dict[str, Any]
    risk_factors: list[dict[str, Any]]
    stages: list[StageResult]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "input": self.input,
            "url": self.url,
            "domain": self.domain,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_ms": self.elapsed_ms,
            "verdict": self.verdict,
            "phishing_score": self.phishing_score,
            "evidence_dir": self.evidence_dir,
            "lexical": self.lexical,
            "dns": self.dns,
            "tls": self.tls,
            "visual": self.visual,
            "summary": self.summary,
            "risk_factors": self.risk_factors,
            "stages": [stage.to_dict() for stage in self.stages],
            "errors": self.errors,
        }
