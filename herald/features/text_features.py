"""Text/content-based feature extraction for phishing detection.

Analyses the plain-text or HTML body of a message for linguistic patterns
commonly associated with phishing attacks against critical infrastructure.
"""

from __future__ import annotations

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

_URGENCY_PHRASES: List[str] = [
    "immediate action",
    "urgent",
    "immediately",
    "as soon as possible",
    "within 24 hours",
    "within 48 hours",
    "act now",
    "respond immediately",
    "failure to respond",
    "your account will be",
    "account suspended",
    "account locked",
    "account disabled",
    "account will be closed",
    "limited time",
    "expires today",
    "expires soon",
]

_CREDENTIAL_REQUESTS: List[str] = [
    "enter your password",
    "confirm your password",
    "verify your password",
    "update your password",
    "enter your username",
    "enter your credentials",
    "your social security",
    "social security number",
    "ssn",
    "credit card",
    "card number",
    "cvv",
    "billing information",
    "bank account",
    "routing number",
    "date of birth",
    "mother's maiden name",
]

_THREAT_PHRASES: List[str] = [
    "will be terminated",
    "legal action",
    "law enforcement",
    "will be prosecuted",
    "account will be suspended",
    "report you to",
    "criminal charges",
    "security breach",
    "unauthorized access",
    "suspicious activity",
    "fraud alert",
    "identity theft",
]

_OCI_KEYWORDS: List[str] = [
    # Operational technology / critical infrastructure lures
    "scada",
    "ics",
    "plc",
    "hmi",
    "dcs",
    "control system",
    "industrial control",
    "operational technology",
    "power grid",
    "smart grid",
    "substation",
    "pipeline",
    "water treatment",
    "nuclear",
    "critical infrastructure",
    "vpn access",
    "remote access",
    "firewall",
    "network access",
]

_DECEPTIVE_PHRASES: List[str] = [
    "click here",
    "click the link",
    "follow this link",
    "download the attachment",
    "open the attachment",
    "enable macros",
    "enable editing",
    "enable content",
    "you have been selected",
    "you have won",
    "congratulations",
    "dear customer",
    "dear user",
    "dear member",
    "valued customer",
]

# HTML-specific risk indicators
_HTML_RISK_PATTERNS: List[re.Pattern] = [
    re.compile(r"<\s*form[^>]+action\s*=", re.IGNORECASE),  # forms that post data
    re.compile(r"<\s*input[^>]+type\s*=\s*['\"]?password", re.IGNORECASE),
    re.compile(r"<\s*iframe", re.IGNORECASE),
    re.compile(r"<\s*script[^>]*src\s*=", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
]


class TextFeatureExtractor:
    """Extract phishing-indicative features from message text or HTML."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, float]:
        """Return a feature dictionary for *text*.

        Args:
            text: Plain-text or HTML message body.

        Returns:
            Dictionary mapping feature names (str) to normalised scores
            in [0, 1].
        """
        text_lower = text.lower()
        features: Dict[str, float] = {
            "text_urgency": self._keyword_score(text_lower, _URGENCY_PHRASES, 3),
            "text_credential_request": self._keyword_score(
                text_lower, _CREDENTIAL_REQUESTS, 2
            ),
            "text_threat_language": self._keyword_score(
                text_lower, _THREAT_PHRASES, 2
            ),
            "text_oci_keywords": self._keyword_score(text_lower, _OCI_KEYWORDS, 2),
            "text_deceptive_phrases": self._keyword_score(
                text_lower, _DECEPTIVE_PHRASES, 3
            ),
            "text_html_risks": self._html_risk_score(text),
            "text_url_count": self._url_density(text),
            "text_external_link_ratio": self._external_link_ratio(text),
            "text_obfuscation": self._obfuscation_score(text),
            "text_all_caps_ratio": self._all_caps_ratio(text),
        }
        return features

    def score(self, text: str) -> float:
        """Return a single risk score in [0, 1] summarising all features."""
        features = self.extract(text)
        weights = {
            "text_urgency": 1.2,
            "text_credential_request": 1.5,
            "text_threat_language": 1.2,
            "text_oci_keywords": 1.3,
            "text_deceptive_phrases": 1.0,
            "text_html_risks": 1.1,
            "text_url_count": 0.8,
            "text_external_link_ratio": 1.0,
            "text_obfuscation": 0.9,
            "text_all_caps_ratio": 0.6,
        }
        total_weight = sum(weights.values())
        weighted_sum = sum(features[k] * w for k, w in weights.items())
        return min(weighted_sum / total_weight, 1.0)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str, keywords: List[str], max_count: int) -> float:
        """Normalised score based on keyword hit count."""
        count = sum(1 for kw in keywords if kw in text)
        return min(count / max_count, 1.0)

    @staticmethod
    def _html_risk_score(text: str) -> float:
        """Score based on presence of risky HTML constructs."""
        hits = sum(1 for pat in _HTML_RISK_PATTERNS if pat.search(text))
        return min(hits / len(_HTML_RISK_PATTERNS), 1.0)

    @staticmethod
    def _url_density(text: str) -> float:
        """Score based on the number of URLs embedded in the text."""
        url_pattern = re.compile(
            r"https?://[^\s\"'>]+|www\.[^\s\"'>]+", re.IGNORECASE
        )
        urls = url_pattern.findall(text)
        count = len(urls)
        if count >= 10:
            return 1.0
        if count >= 5:
            return 0.75
        if count >= 2:
            return 0.4
        return 0.0

    @staticmethod
    def _external_link_ratio(text: str) -> float:
        """Ratio of links pointing to domains *other* than the sender's domain.

        Without a known sender domain this is approximated by the fraction of
        distinct link domains relative to the total link count.
        """
        url_pattern = re.compile(
            r"https?://([^/\s\"'>]+)", re.IGNORECASE
        )
        matches = url_pattern.findall(text)
        if not matches:
            return 0.0
        unique_domains = len(set(m.lower() for m in matches))
        total = len(matches)
        # Many different domains → higher risk
        if unique_domains > 5:
            return 1.0
        if unique_domains > 2:
            return 0.5
        return unique_domains / max(total, 1)

    @staticmethod
    def _obfuscation_score(text: str) -> float:
        """Detect character-substitution obfuscation (e.g. ``p4ypal``)."""
        # Look for words mixing letters and digits
        mixed = re.findall(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b|\b[a-zA-Z]*\d[a-zA-Z]+\d\b", text)
        if len(mixed) >= 4:
            return 1.0
        if len(mixed) >= 2:
            return 0.5
        return 0.0

    @staticmethod
    def _all_caps_ratio(text: str) -> float:
        """Fraction of words that are ALL-CAPS (urgency indicator)."""
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
        if not words:
            return 0.0
        caps = sum(1 for w in words if w.isupper())
        return min(caps / len(words) / 0.3, 1.0)  # normalise: >30 % → 1.0
