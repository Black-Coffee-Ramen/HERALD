"""Tests for PhishingDetector (orchestrator)."""

from __future__ import annotations

import pytest

from herald.config import Config
from herald.detector import PhishingDetector
from herald.result import DetectionResult, RiskLevel


@pytest.fixture()
def detector() -> PhishingDetector:
    return PhishingDetector()


@pytest.fixture()
def strict_detector() -> PhishingDetector:
    """Detector with a lower threshold to ease phishing detection in tests."""
    config = Config(phishing_threshold=0.2)
    return PhishingDetector(config=config)


# ===========================================================================
# URL analysis
# ===========================================================================


class TestAnalyzeURL:
    def test_returns_detection_result(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://example.com/")
        assert isinstance(result, DetectionResult)

    def test_result_fields_populated(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://example.com/page")
        assert result.risk_level in list(RiskLevel)
        assert 0.0 <= result.risk_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_phishing_url_flagged(self, strict_detector: PhishingDetector) -> None:
        result = strict_detector.analyze_url(
            "http://paypa1-secure.xyz/login?verify=account&update=credentials"
        )
        assert result.is_phishing is True
        assert result.risk_score >= 0.2

    def test_legitimate_url_low_risk(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://www.example.com/products/list")
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_ip_address_url_has_indicators(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("http://192.168.0.1/admin")
        assert any("IP address" in ind for ind in result.indicators)

    def test_url_shortener_has_indicator(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://bit.ly/3AbCdEf")
        assert any("shortening" in ind.lower() for ind in result.indicators)

    def test_to_dict_serialisable(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://example.com/")
        d = result.to_dict()
        import json
        json_str = json.dumps(d)  # must not raise
        assert "is_phishing" in json_str

    def test_summary_contains_verdict(self, detector: PhishingDetector) -> None:
        result = detector.analyze_url("https://example.com/")
        summary = result.summary()
        assert "Verdict" in summary
        assert "Risk" in summary


# ===========================================================================
# Text analysis
# ===========================================================================


class TestAnalyzeText:
    def test_returns_detection_result(self, detector: PhishingDetector) -> None:
        result = detector.analyze_text("Hello, please review the attached report.")
        assert isinstance(result, DetectionResult)

    def test_phishing_text_flagged(self, strict_detector: PhishingDetector) -> None:
        phish_body = (
            "URGENT: Your account will be suspended within 24 hours. "
            "Please enter your password and credit card number immediately. "
            "Click here to verify: http://secure-update.xyz/login "
            "Failure to respond will result in legal action."
        )
        result = strict_detector.analyze_text(phish_body)
        assert result.is_phishing is True

    def test_benign_text_low_risk(self, detector: PhishingDetector) -> None:
        text = "Hi, the project meeting is scheduled for Thursday at 10am. See you there."
        result = detector.analyze_text(text)
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_oci_text_flagged(self, strict_detector: PhishingDetector) -> None:
        oci_body = (
            "Your SCADA system VPN access credentials have expired. "
            "Please enter your password to update access to the industrial control system. "
            "Immediate action required."
        )
        result = strict_detector.analyze_text(oci_body)
        assert result.is_phishing is True
        assert any("critical" in ind.lower() or "OT" in ind or "ICS" in ind or "OCI" in ind or "infrastructure" in ind.lower()
                   for ind in result.indicators)

    def test_credential_request_indicator(self, detector: PhishingDetector) -> None:
        text = "Please enter your password and confirm your credit card number."
        result = detector.analyze_text(text)
        assert any("credential" in ind.lower() or "personal data" in ind.lower()
                   for ind in result.indicators)


# ===========================================================================
# Header analysis
# ===========================================================================


class TestAnalyzeHeaders:
    def test_returns_detection_result(self, detector: PhishingDetector) -> None:
        result = detector.analyze_headers({"From": "test@example.com"})
        assert isinstance(result, DetectionResult)

    def test_spoofed_sender_flagged(self, strict_detector: PhishingDetector) -> None:
        headers = {
            "From": "PayPal Security <security@paypal-account-verify.xyz>",
            "Subject": "Urgent: verify your account",
            "Reply-To": "steal@attacker.com",
        }
        result = strict_detector.analyze_headers(headers)
        assert result.is_phishing is True

    def test_reply_to_mismatch_indicator(self, detector: PhishingDetector) -> None:
        headers = {
            "From": "noreply@legit.com",
            "Reply-To": "harvest@totally-different.net",
            "Subject": "Hello",
        }
        result = detector.analyze_headers(headers)
        assert any("Reply-To" in ind for ind in result.indicators)

    def test_legitimate_headers_low_risk(self, detector: PhishingDetector) -> None:
        headers = {
            "From": "newsletter@company.com",
            "Subject": "Monthly update",
            "Message-ID": "<abc123@company.com>",
        }
        result = detector.analyze_headers(headers)
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)


# ===========================================================================
# Full email analysis
# ===========================================================================


class TestAnalyzeEmail:
    _PHISHING_RAW = (
        "From: PayPal Security <security@paypal-verify.xyz>\r\n"
        "To: victim@example.com\r\n"
        "Subject: URGENT: Your account will be suspended\r\n"
        "Reply-To: phish@attacker.com\r\n"
        "Content-Type: text/html\r\n"
        "\r\n"
        "<html><body>"
        "<p>Dear customer, your account will be suspended within 24 hours.</p>"
        "<p>Please enter your password and credit card number immediately.</p>"
        '<form action="http://paypal-verify.xyz/steal" method="post">'
        '<input type="password" name="pwd">'
        "</form>"
        '<p><a href="http://paypal-verify.xyz/login">Click here to verify</a></p>'
        "</body></html>"
    )

    def test_full_email_analysis_returns_result(
        self, detector: PhishingDetector
    ) -> None:
        result = detector.analyze_email(raw_message=self._PHISHING_RAW)
        assert isinstance(result, DetectionResult)

    def test_phishing_email_detected(
        self, strict_detector: PhishingDetector
    ) -> None:
        result = strict_detector.analyze_email(raw_message=self._PHISHING_RAW)
        assert result.is_phishing is True

    def test_analyze_email_with_parts(self, strict_detector: PhishingDetector) -> None:
        headers = {
            "From": "PayPal <noreply@paypal-fake.xyz>",
            "Subject": "Verify your account immediately",
        }
        body = (
            "URGENT: Enter your password and confirm your account. "
            "Legal action will follow if you do not respond."
        )
        urls = ["http://paypal-fake.xyz/login?verify=1"]
        result = strict_detector.analyze_email(
            headers=headers, body=body, urls=urls
        )
        assert result.is_phishing is True
        assert result.url_score > 0
        assert result.text_score > 0
        assert result.header_score > 0

    def test_combined_scores_populated(self, detector: PhishingDetector) -> None:
        result = detector.analyze_email(
            headers={"From": "test@example.com", "Subject": "Hello"},
            body="This is a normal email.",
            urls=["https://example.com/"],
        )
        assert result.url_score >= 0.0
        assert result.text_score >= 0.0
        assert result.header_score >= 0.0

    def test_no_duplicate_indicators(self, detector: PhishingDetector) -> None:
        result = detector.analyze_email(raw_message=self._PHISHING_RAW)
        assert len(result.indicators) == len(set(result.indicators))


# ===========================================================================
# Config / threshold tests
# ===========================================================================


class TestConfigAndThreshold:
    def test_high_threshold_allows_borderline(self) -> None:
        config = Config(phishing_threshold=0.99)
        detector = PhishingDetector(config=config)
        # Even a clearly phishing URL should not cross a 99% threshold
        result = detector.analyze_url("http://paypa1-fake.xyz/")
        assert result.is_phishing is False  # threshold too high

    def test_zero_threshold_flags_everything(self) -> None:
        config = Config(phishing_threshold=0.0)
        detector = PhishingDetector(config=config)
        result = detector.analyze_url("https://example.com/")
        assert result.is_phishing is True  # every score ≥ 0.0

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERALD_THRESHOLD", "0.3")
        monkeypatch.setenv("HERALD_LOG_LEVEL", "DEBUG")
        config = Config.from_env()
        assert config.phishing_threshold == pytest.approx(0.3)
        assert config.log_level == "DEBUG"


# ===========================================================================
# RiskLevel tests
# ===========================================================================


class TestRiskLevel:
    def test_from_score_low(self) -> None:
        assert RiskLevel.from_score(0.1) == RiskLevel.LOW

    def test_from_score_medium(self) -> None:
        assert RiskLevel.from_score(0.3) == RiskLevel.MEDIUM

    def test_from_score_high(self) -> None:
        assert RiskLevel.from_score(0.6) == RiskLevel.HIGH

    def test_from_score_critical(self) -> None:
        assert RiskLevel.from_score(0.9) == RiskLevel.CRITICAL

    def test_boundary_values(self) -> None:
        assert RiskLevel.from_score(0.0) == RiskLevel.LOW
        assert RiskLevel.from_score(0.25) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.50) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.75) == RiskLevel.CRITICAL
        assert RiskLevel.from_score(1.0) == RiskLevel.CRITICAL
