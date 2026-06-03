from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from herald.core.security import validate_url_safe
from herald.investigation.intelligence import collect_dns_intelligence, collect_tls_intelligence
from herald.investigation.models import InvestigationResult, StageResult, utc_now_iso
from herald.investigation.persistence import create_evidence_dir, save_investigation
from herald.investigation.scoring import build_summary, combine_scores
from herald.detection.engine import DetectionEngine, ScorerType
from herald.investigation.targets import normalize_target


class InvestigationPipeline:
    def __init__(self, evidence_root: str = "evidence", scorer_type: ScorerType = "heuristic"):
        self.evidence_root = evidence_root
        self.engine = DetectionEngine(scorer_type)

    def investigate(self, target: str, *, include_visual: bool = True, allow_private: bool = False) -> InvestigationResult:
        started = time.monotonic()
        started_at = utc_now_iso()
        trace_id = f"trc-{uuid.uuid4().hex[:10]}"
        stages: list[StageResult] = []
        errors: list[str] = []

        url, domain = normalize_target(target)
        evidence_dir = create_evidence_dir(trace_id, domain, self.evidence_root)
        dns: dict[str, Any] = {}
        tls: dict[str, Any] = {}

        with self._stage(stages, "SSRF validation") as stage:
            private_overrides: list[dict[str, str | None]] = []
            validate_url_safe(url, allow_private=allow_private, trace_id=trace_id, private_overrides=private_overrides)
            stage["details"] = {
                "allow_private": allow_private,
                "private_overrides": private_overrides,
            }

        with self._stage(stages, "Detection Engine scoring") as stage:
            detection_res = self.engine.score(domain)
            lexical = {
                "score": detection_res.confidence,
                "suspicious_keywords": detection_res.features.get("suspicious_keywords", []),
                "risk_factors": [rf.to_dict() for rf in detection_res.risk_factors],
                "features": detection_res.features,
                "explanations": detection_res.explanations,
            }
            from herald.detection.models import display_verdict
            stage["details"] = {
                "scorer": detection_res.scorer,
                "confidence": detection_res.confidence,
                "verdict": display_verdict(detection_res.verdict),
            }

        with self._degraded_stage(stages, errors, "DNS and WHOIS intelligence") as stage:
            dns = collect_dns_intelligence(domain)
            stage["details"] = {
                "registrar": dns.get("registrar"),
                "domain_age_days": dns.get("domain_age_days"),
                "a_records": dns.get("a_records", [])[:3],
            }

        with self._degraded_stage(stages, errors, "TLS intelligence") as stage:
            tls = collect_tls_intelligence(domain)
            stage["details"] = {
                "has_tls": tls.get("has_tls"),
                "issuer": tls.get("issuer"),
                "suspicious": tls.get("suspicious"),
            }

        visual: dict[str, Any] = {
            "success": False,
            "screenshot_path": None,
            "ocr_text": "",
            "ocr_findings": {},
            "skipped": not include_visual,
        }
        if include_visual:
            with self._degraded_stage(stages, errors, "Screenshot and OCR") as stage:
                visual = asyncio.run(self._run_visual(url, os.fspath(evidence_dir)))
                stage["details"] = {
                    "success": visual.get("success"),
                    "screenshot_path": visual.get("screenshot_path"),
                    "ocr_suspicious": (visual.get("ocr_findings") or {}).get("is_suspicious"),
                }
                if not visual.get("success"):
                    raise RuntimeError(visual.get("error", "Screenshot failed to capture"))

        heuristic_verdict, heuristic_score, risk_factors = combine_scores(lexical, dns, tls, visual)
        
        if self.engine.scorer_type in ("ml", "hybrid"):
            # Trust the DetectionEngine's sophisticated verdict, but append the
            # heuristic risk factors for UI explainability.
            from herald.detection.models import display_verdict
            verdict = display_verdict(detection_res.verdict)
            phishing_score = detection_res.confidence
            
            # Safety net: If the pipeline caught severe OCR phishing indicators that 
            # the ML model bypassed (e.g., ISP block pages), override the ML verdict.
            if visual.get("ocr_findings", {}).get("is_suspicious"):
                verdict = "Phishing"
                phishing_score = max(phishing_score, heuristic_score, 0.85)
                
            # If the ML model flagged it, ensure we show a Risk Factor for it 
            # so the user isn't told "No strong indicators observed".
            if verdict == "Phishing" and phishing_score >= 0.7:
                if not any(rf.get("name") == "ML High Confidence" for rf in risk_factors):
                    ml_reasons = []
                    feats = lexical.get("features", {})
                    if feats.get("domain_length", 0) > 22:
                        ml_reasons.append("unusually long length")
                    if feats.get("num_hyphens", 0) > 0:
                        ml_reasons.append("hyphenation")
                    if feats.get("subdomain_count", 0) > 0:
                        ml_reasons.append("subdomain structure")
                    if feats.get("digit_ratio", 0) > 0.1:
                        ml_reasons.append("high digit ratio")
                    
                    reason_str = "The ML ensemble flagged anomalous structural patterns"
                    if ml_reasons:
                        reason_str += f" ({', '.join(ml_reasons[:3])})"
                    
                    risk_factors.insert(0, {
                        "name": "ML High Confidence",
                        "severity": "high",
                        "detail": f"{reason_str}, resulting in a confidence score of {phishing_score:.4f}.",
                        "score_impact": phishing_score
                    })
        else:
            verdict = heuristic_verdict
            phishing_score = heuristic_score

        elapsed_ms = int((time.monotonic() - started) * 1000)
        result_payload = {
            "trace_id": trace_id,
            "input": target,
            "url": url,
            "domain": domain,
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "elapsed_ms": elapsed_ms,
            "verdict": verdict,
            "phishing_score": phishing_score,
            "evidence_dir": os.fspath(evidence_dir),
            "lexical": lexical,
            "dns": dns,
            "tls": tls,
            "visual": visual,
            "risk_factors": risk_factors,
            "stages": [stage.to_dict() for stage in stages],
            "errors": errors,
        }
        summary = build_summary(result_payload)

        result = InvestigationResult(
            trace_id=trace_id,
            input=target,
            url=url,
            domain=domain,
            started_at=started_at,
            completed_at=result_payload["completed_at"],
            elapsed_ms=elapsed_ms,
            verdict=verdict,
            phishing_score=phishing_score,
            evidence_dir=os.fspath(evidence_dir),
            lexical=lexical,
            dns=dns,
            tls=tls,
            visual=visual,
            summary=summary,
            risk_factors=risk_factors,
            stages=stages,
            errors=errors,
        )
        paths = save_investigation(result.to_dict(), result.evidence_dir)
        result.visual.setdefault("report_paths", paths)
        return result

    async def _run_visual(self, domain: str, evidence_dir: str) -> dict[str, Any]:
        from herald.core.playwright_analyzer import PlaywrightVisualAnalyzer

        analyzer = PlaywrightVisualAnalyzer(evidence_dir=evidence_dir)
        return await analyzer.run_analysis(domain)

    @contextmanager
    def _stage(self, stages: list[StageResult], name: str) -> Iterator[dict[str, Any]]:
        started = time.monotonic()
        context: dict[str, Any] = {"details": {}}
        try:
            yield context
            stages.append(StageResult(name=name, status="ok", duration_ms=int((time.monotonic() - started) * 1000), details=context["details"]))
        except Exception as exc:
            stages.append(StageResult(name=name, status="failed", duration_ms=int((time.monotonic() - started) * 1000), details=context["details"], error=str(exc)))
            raise

    @contextmanager
    def _degraded_stage(self, stages: list[StageResult], errors: list[str], name: str) -> Iterator[dict[str, Any]]:
        started = time.monotonic()
        context: dict[str, Any] = {"details": {}}
        try:
            yield context
            stages.append(StageResult(name=name, status="ok", duration_ms=int((time.monotonic() - started) * 1000), details=context["details"]))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            stages.append(StageResult(name=name, status="degraded", duration_ms=int((time.monotonic() - started) * 1000), details=context["details"], error=str(exc)))
