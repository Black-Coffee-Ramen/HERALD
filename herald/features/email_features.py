"""Email-header–based feature extraction for phishing detection.

Parses raw email messages or header dictionaries and extracts features
that indicate spoofing, impersonation, or misconfiguration.
"""

from __future__ import annotations

import email
import email.policy
import re
from typing import Dict, List, Optional, Union


# Domains that should enforce strict DMARC / SPF alignment.
# Spoofed display names using these brands are high-risk.
_HIGH_VALUE_SENDER_DOMAINS: List[str] = [
    "microsoft.com",
    "google.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "linkedin.com",
    "irs.gov",
    "gov.uk",
    ".gov",
]


class EmailFeatureExtractor:
    """Extract phishing-indicative features from email headers.

    Usage::

        extractor = EmailFeatureExtractor()
        # From a raw RFC 2822 message string:
        features = extractor.extract_from_raw(raw_message)
        # From a header dictionary:
        features = extractor.extract_from_headers({"From": "...", "Subject": "..."})
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_raw(self, raw_message: str) -> Dict[str, float]:
        """Parse *raw_message* and return a feature dictionary.

        Args:
            raw_message: A complete RFC 2822 email message as a string.

        Returns:
            Dictionary mapping feature names to normalised scores in [0, 1].
        """
        msg = email.message_from_string(raw_message, policy=email.policy.default)
        headers: Dict[str, str] = {k.lower(): str(v) for k, v in msg.items()}
        return self._extract(headers)

    def extract_from_headers(
        self, headers: Dict[str, str]
    ) -> Dict[str, float]:
        """Return a feature dictionary from a pre-parsed *headers* dict.

        Args:
            headers: Mapping of header name → value (case-insensitive).

        Returns:
            Dictionary mapping feature names to normalised scores in [0, 1].
        """
        normalised = {k.lower(): v for k, v in headers.items()}
        return self._extract(normalised)

    def score_from_raw(self, raw_message: str) -> float:
        """Return a single risk score in [0, 1] for a raw email message."""
        return self._score(self.extract_from_raw(raw_message))

    def score_from_headers(self, headers: Dict[str, str]) -> float:
        """Return a single risk score in [0, 1] from a header dictionary."""
        return self._score(self.extract_from_headers(headers))

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _extract(self, headers: Dict[str, str]) -> Dict[str, float]:
        """Core extraction logic operating on lower-cased *headers*."""
        from_header = headers.get("from", "")
        reply_to = headers.get("reply-to", "")
        return_path = headers.get("return-path", "")
        subject = headers.get("subject", "")
        received = headers.get("received", "")
        x_mailer = headers.get("x-mailer", "")
        content_type = headers.get("content-type", "")
        message_id = headers.get("message-id", "")

        features: Dict[str, float] = {
            "header_from_display_mismatch": self._from_display_mismatch(from_header),
            "header_reply_to_differs": self._reply_to_differs(from_header, reply_to),
            "header_return_path_mismatch": self._return_path_mismatch(
                from_header, return_path
            ),
            "header_subject_urgency": self._subject_urgency(subject),
            "header_subject_credential_lure": self._subject_credential_lure(subject),
            "header_subject_oci_lure": self._subject_oci_lure(subject),
            "header_spoofed_high_value_domain": self._spoofed_high_value_domain(
                from_header
            ),
            "header_suspicious_received": self._suspicious_received(received),
            "header_missing_message_id": self._missing_message_id(message_id),
            "header_html_only": self._html_only(content_type),
        }
        return features

    def _score(self, features: Dict[str, float]) -> float:
        weights = {
            "header_from_display_mismatch": 1.5,
            "header_reply_to_differs": 1.2,
            "header_return_path_mismatch": 1.0,
            "header_subject_urgency": 1.0,
            "header_subject_credential_lure": 1.3,
            "header_subject_oci_lure": 1.2,
            "header_spoofed_high_value_domain": 1.5,
            "header_suspicious_received": 0.8,
            "header_missing_message_id": 0.7,
            "header_html_only": 0.5,
        }
        total_weight = sum(weights.values())
        weighted_sum = sum(features.get(k, 0.0) * w for k, w in weights.items())
        return min(weighted_sum / total_weight, 1.0)

    # ------------------------------------------------------------------
    # Individual feature detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_email_address(header_value: str) -> tuple[str, str]:
        """Return ``(display_name, address)`` from a From-style header value."""
        match = re.match(r"^(.*?)\s*<([^>]+)>", header_value.strip())
        if match:
            return match.group(1).strip(' "\''), match.group(2).strip().lower()
        return "", header_value.strip().lower()

    @staticmethod
    def _extract_domain(email_address: str) -> str:
        """Extract the domain portion of an email address."""
        if "@" in email_address:
            return email_address.split("@")[-1].lower()
        return ""

    def _from_display_mismatch(self, from_header: str) -> float:
        """Detect when the display name contains a domain different from the sender domain."""
        display_name, address = self._parse_email_address(from_header)
        sender_domain = self._extract_domain(address)
        if not sender_domain or not display_name:
            return 0.0
        # Look for a domain-like pattern in the display name
        display_domain_match = re.search(
            r"@([a-z0-9\-]+\.[a-z]{2,})", display_name.lower()
        )
        if display_domain_match:
            display_domain = display_domain_match.group(1)
            if display_domain != sender_domain:
                return 1.0
        # Display name contains a known brand but address is on a different domain
        for brand_domain in _HIGH_VALUE_SENDER_DOMAINS:
            brand = brand_domain.split(".")[0]
            if len(brand) < 4:
                continue
            if brand in display_name.lower() and not sender_domain.endswith(
                brand_domain
            ):
                return 0.8
        return 0.0

    def _reply_to_differs(self, from_header: str, reply_to: str) -> float:
        """Return 1.0 when Reply-To domain differs from From domain."""
        if not reply_to:
            return 0.0
        _, from_address = self._parse_email_address(from_header)
        _, reply_address = self._parse_email_address(reply_to)
        from_domain = self._extract_domain(from_address)
        reply_domain = self._extract_domain(reply_address)
        if not from_domain or not reply_domain:
            return 0.0
        return 1.0 if from_domain != reply_domain else 0.0

    def _return_path_mismatch(self, from_header: str, return_path: str) -> float:
        """Return 1.0 when Return-Path domain differs from From domain."""
        if not return_path:
            return 0.0
        _, from_address = self._parse_email_address(from_header)
        clean_rp = return_path.strip("<>")
        from_domain = self._extract_domain(from_address)
        rp_domain = self._extract_domain(clean_rp)
        if not from_domain or not rp_domain:
            return 0.0
        return 1.0 if from_domain != rp_domain else 0.0

    @staticmethod
    def _subject_urgency(subject: str) -> float:
        """Detect urgency indicators in the email subject."""
        urgency_words = [
            "urgent",
            "immediate",
            "action required",
            "alert",
            "warning",
            "critical",
            "important",
            "expires",
            "suspended",
            "locked",
            "verify now",
            "respond now",
        ]
        subject_lower = subject.lower()
        hits = sum(1 for w in urgency_words if w in subject_lower)
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _subject_credential_lure(subject: str) -> float:
        """Detect credential-harvesting lures in the subject."""
        lures = [
            "password",
            "login",
            "sign in",
            "signin",
            "account",
            "verify",
            "confirm",
            "update your",
            "security",
            "access",
        ]
        subject_lower = subject.lower()
        hits = sum(1 for l in lures if l in subject_lower)
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _subject_oci_lure(subject: str) -> float:
        """Detect critical-infrastructure–specific lures in the subject."""
        oci_terms = [
            "scada",
            "ics",
            "plc",
            "vpn",
            "network access",
            "remote access",
            "firewall",
            "control system",
            "industrial",
            "operational",
            "power",
            "grid",
            "pipeline",
        ]
        subject_lower = subject.lower()
        hits = sum(1 for t in oci_terms if t in subject_lower)
        return min(hits / 2.0, 1.0)

    def _spoofed_high_value_domain(self, from_header: str) -> float:
        """Return 1.0 when the sender impersonates a high-value domain."""
        _, address = self._parse_email_address(from_header)
        sender_domain = self._extract_domain(address)
        if not sender_domain:
            return 0.0
        for hvd in _HIGH_VALUE_SENDER_DOMAINS:
            brand = hvd.split(".")[0]
            if len(brand) < 4:
                continue
            # Brand name appears in domain but is not the official domain
            if brand in sender_domain and not (
                sender_domain == hvd or sender_domain.endswith("." + hvd)
            ):
                return 1.0
        return 0.0

    @staticmethod
    def _suspicious_received(received: str) -> float:
        """Detect suspicious routing in the Received chain."""
        if not received:
            return 0.0
        # Received header containing an IP with no reverse DNS
        ip_only_re = re.compile(
            r"from\s+\[?(\d{1,3}(?:\.\d{1,3}){3})\]?\s+\(", re.IGNORECASE
        )
        if ip_only_re.search(received):
            return 0.8
        return 0.0

    @staticmethod
    def _missing_message_id(message_id: str) -> float:
        """Return 1.0 if the Message-ID header is absent."""
        return 1.0 if not message_id.strip() else 0.0

    @staticmethod
    def _html_only(content_type: str) -> float:
        """Return 1.0 if the email is HTML-only (no plain-text alternative)."""
        ct_lower = content_type.lower()
        if "text/html" in ct_lower and "multipart/alternative" not in ct_lower:
            return 1.0
        return 0.0
