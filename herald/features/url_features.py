"""URL-based feature extraction for phishing detection.

Each public method returns a normalised score in [0, 1] where
``1.0`` represents the highest phishing risk.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from typing import Dict, List

from herald.config import Config


# ---------------------------------------------------------------------------
# Known-legitimate domains used for brand-impersonation detection.
# These are high-value targets commonly spoofed in critical-infrastructure
# phishing campaigns.
# ---------------------------------------------------------------------------
_LEGITIMATE_DOMAINS: List[str] = [
    "google.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "facebook.com",
    "linkedin.com",
    "twitter.com",
    "github.com",
    "dropbox.com",
    "office365.com",
    "outlook.com",
    "live.com",
    "hotmail.com",
    "yahoo.com",
    "gmail.com",
    # Critical-infrastructure / government portals
    "gov.uk",
    "gov.au",
    ".gov",
    "irs.gov",
    "fbi.gov",
    "cisa.gov",
    "energy.gov",
    "epa.gov",
    "dhs.gov",
    "cia.gov",
]

_SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "security",
    "update",
    "account",
    "password",
    "credential",
    "confirm",
    "banking",
    "wallet",
    "payment",
    "invoice",
    "billing",
]

# Regex that matches an IPv4 or IPv6 address as the URL host
_IP_HOST_RE = re.compile(
    r"^(\[.*\]|"  # IPv6 in brackets
    r"(?:\d{1,3}\.){3}\d{1,3})$"  # IPv4
)


class URLFeatureExtractor:
    """Extract phishing-indicative features from a URL string.

    Args:
        config: :class:`~herald.config.Config` instance with
            ``suspicious_tlds`` and ``url_shorteners`` lists.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, url: str) -> Dict[str, float]:
        """Return a feature dictionary for *url*.

        All values are normalised to [0, 1].  Keys map directly to the
        feature names used by :class:`~herald.model.PhishingModel`.
        """
        parsed = self._safe_parse(url)
        features: Dict[str, float] = {
            "url_uses_ip": self._uses_ip(parsed),
            "url_is_long": self._is_long(url),
            "url_has_at_sign": self._has_at_sign(url),
            "url_double_slash_redirect": self._double_slash_redirect(url),
            "url_has_dash_in_domain": self._has_dash_in_domain(parsed),
            "url_subdomain_count": self._subdomain_count(parsed),
            "url_is_http": self._is_http(parsed),
            "url_suspicious_tld": self._suspicious_tld(parsed),
            "url_is_shortener": self._is_shortener(parsed),
            "url_suspicious_keywords": self._suspicious_keywords(url),
            "url_brand_impersonation": self._brand_impersonation(url, parsed),
            "url_path_depth": self._path_depth(parsed),
            "url_query_complexity": self._query_complexity(parsed),
            "url_encoded_chars": self._encoded_chars(url),
        }
        return features

    def score(self, url: str) -> float:
        """Return a single risk score in [0, 1] summarising all features."""
        features = self.extract(url)
        # Weighted average – weights reflect empirical importance
        weights = {
            "url_uses_ip": 1.5,
            "url_is_long": 0.8,
            "url_has_at_sign": 1.2,
            "url_double_slash_redirect": 1.0,
            "url_has_dash_in_domain": 0.6,
            "url_subdomain_count": 0.9,
            "url_is_http": 0.7,
            "url_suspicious_tld": 1.3,
            "url_is_shortener": 1.1,
            "url_suspicious_keywords": 0.8,
            "url_brand_impersonation": 1.5,
            "url_path_depth": 0.5,
            "url_query_complexity": 0.6,
            "url_encoded_chars": 0.9,
        }
        total_weight = sum(weights.values())
        weighted_sum = sum(features[k] * w for k, w in weights.items())
        return min(weighted_sum / total_weight, 1.0)

    # ------------------------------------------------------------------
    # Individual feature detectors
    # ------------------------------------------------------------------

    def _safe_parse(self, url: str) -> urllib.parse.ParseResult:
        """Parse *url*, adding a scheme if missing."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
            url = "http://" + url
        return urllib.parse.urlparse(url)

    def _uses_ip(self, parsed: urllib.parse.ParseResult) -> float:
        """Return 1.0 if the host is an IP address rather than a domain name."""
        host = parsed.hostname or ""
        try:
            ipaddress.ip_address(host)
            return 1.0
        except ValueError:
            pass
        return 1.0 if bool(_IP_HOST_RE.match(host)) else 0.0

    def _is_long(self, url: str) -> float:
        """Normalised URL length risk (long URLs are more suspicious)."""
        length = len(url)
        if length > 150:
            return 1.0
        if length > 100:
            return 0.75
        if length > 75:
            return 0.5
        return 0.0

    def _has_at_sign(self, url: str) -> float:
        """Return 1.0 if the URL contains an ``@`` sign (credential lure)."""
        return 1.0 if "@" in url else 0.0

    def _double_slash_redirect(self, url: str) -> float:
        """Detect ``//`` redirect patterns in the URL path."""
        # Ignore the leading scheme ``://``
        without_scheme = re.sub(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", "", url)
        return 1.0 if "//" in without_scheme else 0.0

    def _has_dash_in_domain(self, parsed: urllib.parse.ParseResult) -> float:
        """Return 1.0 if the domain contains a dash (common in brand spoofing)."""
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        domain_part = parts[-2] if len(parts) >= 2 else hostname
        return 1.0 if "-" in domain_part else 0.0

    def _subdomain_count(self, parsed: urllib.parse.ParseResult) -> float:
        """Score based on excessive subdomain nesting."""
        hostname = parsed.hostname or ""
        parts = [p for p in hostname.split(".") if p]
        # Subtract 2 for the base domain + TLD
        subdomains = max(0, len(parts) - 2)
        if subdomains >= 3:
            return 1.0
        if subdomains == 2:
            return 0.75
        if subdomains == 1:
            return 0.5
        return 0.0

    def _is_http(self, parsed: urllib.parse.ParseResult) -> float:
        """Return 1.0 for plain HTTP (no TLS)."""
        return 1.0 if parsed.scheme == "http" else 0.0

    def _suspicious_tld(self, parsed: urllib.parse.ParseResult) -> float:
        """Return 1.0 if the TLD is in the configured suspicious-TLD list."""
        hostname = (parsed.hostname or "").lower()
        for tld in self._config.suspicious_tlds:
            if hostname.endswith(tld):
                return 1.0
        return 0.0

    def _is_shortener(self, parsed: urllib.parse.ParseResult) -> float:
        """Return 1.0 if the host is a known URL shortener."""
        hostname = (parsed.hostname or "").lower()
        for shortener in self._config.url_shorteners:
            if hostname == shortener or hostname.endswith("." + shortener):
                return 1.0
        return 0.0

    def _suspicious_keywords(self, url: str) -> float:
        """Score based on count of security-lure keywords in the URL."""
        url_lower = url.lower()
        count = sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in url_lower)
        return min(count / 3.0, 1.0)

    def _brand_impersonation(
        self, url: str, parsed: urllib.parse.ParseResult
    ) -> float:
        """Detect brand names used in the URL *outside* their official domain.

        E.g. ``paypal-secure-login.xyz`` contains "paypal" but is not
        hosted on ``paypal.com``.
        """
        hostname = (parsed.hostname or "").lower()
        url_lower = url.lower()

        for legit in _LEGITIMATE_DOMAINS:
            # Extract the core brand name (drop TLD / country code)
            brand = legit.split(".")[0]
            if len(brand) < 4:
                continue
            # Brand name appears somewhere in the URL …
            if brand in url_lower:
                # … but the host is NOT the official domain
                if not (hostname == legit or hostname.endswith("." + legit)):
                    return 1.0
        return 0.0

    def _path_depth(self, parsed: urllib.parse.ParseResult) -> float:
        """Score based on abnormally deep URL path (obfuscation indicator)."""
        depth = len([p for p in parsed.path.split("/") if p])
        if depth >= 6:
            return 1.0
        if depth >= 4:
            return 0.5
        return 0.0

    def _query_complexity(self, parsed: urllib.parse.ParseResult) -> float:
        """Score based on number of query parameters (evasion indicator)."""
        params = urllib.parse.parse_qs(parsed.query)
        count = len(params)
        if count >= 6:
            return 1.0
        if count >= 3:
            return 0.5
        return 0.0

    def _encoded_chars(self, url: str) -> float:
        """Detect excessive percent-encoding (obfuscation)."""
        encoded = re.findall(r"%[0-9a-fA-F]{2}", url)
        ratio = len(encoded) / max(len(url), 1)
        return min(ratio * 10, 1.0)
