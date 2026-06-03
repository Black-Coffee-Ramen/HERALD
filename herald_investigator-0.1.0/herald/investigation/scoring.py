from __future__ import annotations

from typing import Any

import pandas as pd

from herald.features.lexical_features import extract_url_features


import logging

logger = logging.getLogger(__name__)

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "secure",
    "account",
    "verify",
    "update",
    "support",
    "banking",
    "paypal",
    "auth",
    "security",
    "confirm",
]


def analyze_lexical(domain: str) -> dict[str, Any]:
    """
    Analyzes the lexical features of a domain using a rule-based heuristic scoring engine.
    NOTE: This is a fallback/CLI engine and is distinct from the primary PhishingPredictorV3
    ML model (`ensemble_v7.joblib`) used by the background workers.
    """
    logger.warning("Using heuristic scoring engine (scoring.py) instead of primary ML model for domain: %s", domain)
    df = pd.DataFrame([{"domain": domain}])
    features = extract_url_features(df, domain_col="domain").iloc[0].to_dict()
    found_keywords = [keyword for keyword in SUSPICIOUS_KEYWORDS if features.get(f"has_{keyword}", 0) == 1]

    score = 0.0
    score += min(float(features.get("domain_length", 0)) / 80, 0.15)
    score += min(float(features.get("num_hyphens", 0)) * 0.08, 0.2)
    score += min(float(features.get("digit_ratio", 0)) * 0.25, 0.15)
    score += 0.2 if features.get("is_malicious_gtld", 0) else 0.0
    score += 0.2 if features.get("brand_keyword_position", 0) else 0.0
    score += min(len(found_keywords) * 0.08, 0.25)
    score += 0.1 if features.get("is_punycode", 0) else 0.0
    risk_factors = _lexical_risk_factors(features, found_keywords)

    return {
        "score": round(min(score, 1.0), 4),
        "suspicious_keywords": found_keywords,
        "risk_factors": risk_factors,
        "features": {
            key: _json_safe(value)
            for key, value in features.items()
            if key != "domain"
        },
    }


def combine_scores(lexical: dict[str, Any], dns: dict[str, Any], tls: dict[str, Any], visual: dict[str, Any]) -> tuple[str, float, list[dict[str, Any]]]:
    score = float(lexical.get("score", 0.0))
    risk_factors = list(lexical.get("risk_factors") or [])

    age = dns.get("domain_age_days")
    if isinstance(age, int | float) and age >= 0:
        if age < 30:
            score += 0.18
            risk_factors.append(_factor("New domain", "high", f"WHOIS creation age is {int(age)} days.", 0.18))
        elif age < 90:
            score += 0.08
            risk_factors.append(_factor("Young domain", "medium", f"WHOIS creation age is {int(age)} days.", 0.08))

    if dns.get("errors") and not dns.get("a_records"):
        risk_factors.append(_factor("DNS unresolved", "medium", "No A records were resolved during investigation.", 0.0))

    if tls.get("suspicious"):
        score += 0.12
        risk_factors.append(_factor("TLS anomaly", "medium", "Certificate metadata does not cleanly match expected domain signals.", 0.12))
    elif not tls.get("has_tls"):
        risk_factors.append(_factor("No TLS certificate", "low", "No TLS certificate could be inspected on port 443.", 0.0))

    ocr_findings = visual.get("ocr_findings") or {}
    if ocr_findings.get("is_suspicious"):
        impact = min(float(ocr_findings.get("ocr_risk_score", 0)) / 100 * 0.35, 0.35)
        score += impact
        phrases = ", ".join(ocr_findings.get("phrases_found") or [])
        risk_factors.append(_factor("Suspicious OCR text", "high", f"Screenshot text matched phishing phrases: {phrases}.", impact))
    elif visual.get("success"):
        risk_factors.append(_factor("Screenshot captured", "info", "Visual evidence was captured and no suspicious OCR phrases were found.", 0.0))
    elif visual.get("skipped"):
        risk_factors.append(_factor("Visual analysis skipped", "info", "Screenshot and OCR were intentionally skipped for this command.", 0.0))
    else:
        risk_factors.append(_factor("Visual analysis degraded", "low", visual.get("error") or "Screenshot/OCR did not complete.", 0.0))

    score = round(min(score, 1.0), 4)
    if score >= 0.7:
        return "Phishing", score, risk_factors
    if score >= 0.35:
        return "Suspected", score, risk_factors
    return "Likely Clean", score, risk_factors


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    risk_factors = result.get("risk_factors") or []
    weighted_factors = [factor for factor in risk_factors if factor.get("severity") != "info"]
    top_reasons = [
        factor["detail"]
        for factor in sorted(weighted_factors, key=lambda item: item.get("score_impact", 0), reverse=True)[:3]
    ]
    if not top_reasons:
        top_reasons = ["No strong phishing indicators were observed."]

    visual = result.get("visual") or {}
    return {
        "headline": f"{result['domain']} is {result['verdict']} with score {result['phishing_score']}.",
        "why_flagged": top_reasons,
        "artifact_count": sum(1 for key in ["screenshot_path"] if visual.get(key)) + 2,
        "degraded_stages": [stage["name"] for stage in result.get("stages", []) if stage.get("status") == "degraded"],
    }


def _lexical_risk_factors(features: dict[str, Any], found_keywords: list[str]) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    if found_keywords:
        factors.append(_factor("Suspicious keywords", "medium", f"Domain contains phishing-oriented terms: {', '.join(found_keywords)}.", min(len(found_keywords) * 0.08, 0.25)))
    if float(features.get("num_hyphens", 0)) >= 2:
        factors.append(_factor("Hyphenated domain", "medium", "Domain uses multiple hyphens, a common phishing URL pattern.", min(float(features.get("num_hyphens", 0)) * 0.08, 0.2)))
    if features.get("is_malicious_gtld", 0):
        factors.append(_factor("Risky TLD", "medium", "Domain uses a TLD commonly seen in abuse datasets.", 0.2))
    if features.get("brand_keyword_position", 0):
        factors.append(_factor("Brand keyword", "medium", "Domain contains a monitored brand or sector keyword.", 0.2))
    if float(features.get("digit_ratio", 0)) > 0.15:
        factors.append(_factor("Digit-heavy domain", "low", "Domain contains an elevated ratio of digits.", min(float(features.get("digit_ratio", 0)) * 0.25, 0.15)))
    if features.get("is_punycode", 0):
        factors.append(_factor("Punycode domain", "high", "Domain uses IDN/Punycode encoding, which can support visual impersonation.", 0.1))
    return factors


def _factor(name: str, severity: str, detail: str, score_impact: float) -> dict[str, Any]:
    return {
        "name": name,
        "severity": severity,
        "detail": detail,
        "score_impact": round(score_impact, 4),
    }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
