"""
HERALD configuration management.

Configuration values can be overridden via environment variables or by
supplying a custom :class:`Config` instance to :class:`~herald.detector.PhishingDetector`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Runtime configuration for the HERALD detector.

    All numeric thresholds were chosen based on empirically observed
    phishing patterns in critical-infrastructure environments.

    Attributes:
        phishing_threshold: Minimum combined risk score (0–1) that
            classifies a sample as phishing.  Lower values increase
            recall; higher values increase precision.
        url_weight: Relative weight of URL-based features in the final
            score.
        text_weight: Relative weight of text/content-based features.
        header_weight: Relative weight of email-header–based features.
        suspicious_tlds: Top-level domains that are disproportionately
            used by phishing campaigns.
        url_shorteners: Well-known URL-shortening services often abused
            to hide malicious destinations.
        log_level: Python logging level string (``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    """

    phishing_threshold: float = 0.5
    url_weight: float = 0.4
    text_weight: float = 0.35
    header_weight: float = 0.25
    suspicious_tlds: List[str] = field(
        default_factory=lambda: [
            ".xyz",
            ".tk",
            ".ml",
            ".ga",
            ".cf",
            ".gq",
            ".pw",
            ".top",
            ".click",
            ".link",
            ".work",
            ".date",
            ".download",
            ".racing",
            ".win",
        ]
    )
    url_shorteners: List[str] = field(
        default_factory=lambda: [
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "ow.ly",
            "t.co",
            "buff.ly",
            "short.link",
            "rebrand.ly",
            "cutt.ly",
            "is.gd",
            "tiny.cc",
            "lnkd.in",
        ]
    )
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        """Create a :class:`Config` by reading environment variables.

        Recognised environment variables
        (all optional, fall back to defaults):

        * ``HERALD_THRESHOLD`` – float, phishing score threshold.
        * ``HERALD_LOG_LEVEL`` – string, Python log level.
        * ``HERALD_URL_WEIGHT`` – float, URL feature weight.
        * ``HERALD_TEXT_WEIGHT`` – float, text feature weight.
        * ``HERALD_HEADER_WEIGHT`` – float, email header feature weight.
        """
        cfg = cls()
        if val := os.environ.get("HERALD_THRESHOLD"):
            cfg.phishing_threshold = float(val)
        if val := os.environ.get("HERALD_LOG_LEVEL"):
            cfg.log_level = val.upper()
        if val := os.environ.get("HERALD_URL_WEIGHT"):
            cfg.url_weight = float(val)
        if val := os.environ.get("HERALD_TEXT_WEIGHT"):
            cfg.text_weight = float(val)
        if val := os.environ.get("HERALD_HEADER_WEIGHT"):
            cfg.header_weight = float(val)
        return cfg
