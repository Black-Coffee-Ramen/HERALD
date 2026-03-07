"""Tests for URL, text, and email feature extractors."""

from __future__ import annotations

import pytest

from herald.config import Config
from herald.features.email_features import EmailFeatureExtractor
from herald.features.text_features import TextFeatureExtractor
from herald.features.url_features import URLFeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def url_extractor(config: Config) -> URLFeatureExtractor:
    return URLFeatureExtractor(config)


@pytest.fixture()
def text_extractor() -> TextFeatureExtractor:
    return TextFeatureExtractor()


@pytest.fixture()
def header_extractor() -> EmailFeatureExtractor:
    return EmailFeatureExtractor()


# ===========================================================================
# URL feature tests
# ===========================================================================


class TestURLFeatureExtractor:
    def test_returns_all_feature_keys(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://example.com/page")
        expected_keys = {
            "url_uses_ip",
            "url_is_long",
            "url_has_at_sign",
            "url_double_slash_redirect",
            "url_has_dash_in_domain",
            "url_subdomain_count",
            "url_is_http",
            "url_suspicious_tld",
            "url_is_shortener",
            "url_suspicious_keywords",
            "url_brand_impersonation",
            "url_path_depth",
            "url_query_complexity",
            "url_encoded_chars",
        }
        assert expected_keys == set(features.keys())

    def test_all_values_in_range(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://192.168.1.1/login?user=admin")
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_ip_host_detected(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://192.168.0.1/")
        assert features["url_uses_ip"] == 1.0

    def test_domain_host_not_ip(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://example.com/")
        assert features["url_uses_ip"] == 0.0

    def test_at_sign_detected(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://example.com@malicious.xyz/")
        assert features["url_has_at_sign"] == 1.0

    def test_no_at_sign(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://example.com/path")
        assert features["url_has_at_sign"] == 0.0

    def test_http_flagged(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://example.com/")
        assert features["url_is_http"] == 1.0

    def test_https_not_flagged(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://example.com/")
        assert features["url_is_http"] == 0.0

    def test_suspicious_tld_detected(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://malicious.xyz/login")
        assert features["url_suspicious_tld"] == 1.0

    def test_legitimate_tld_ok(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://example.com/")
        assert features["url_suspicious_tld"] == 0.0

    def test_shortener_detected(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("https://bit.ly/3xYzAbc")
        assert features["url_is_shortener"] == 1.0

    def test_brand_impersonation(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://paypal-secure.malicious.xyz/login")
        assert features["url_brand_impersonation"] == 1.0

    def test_no_brand_impersonation_on_legit_domain(
        self, url_extractor: URLFeatureExtractor
    ) -> None:
        features = url_extractor.extract("https://www.paypal.com/us/signin")
        assert features["url_brand_impersonation"] == 0.0

    def test_double_slash_redirect(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://example.com//redirect?url=evil.com")
        assert features["url_double_slash_redirect"] == 1.0

    def test_score_phishing_url_higher_than_legit(
        self, url_extractor: URLFeatureExtractor
    ) -> None:
        phish = url_extractor.score("http://paypa1-secure.xyz/login?verify=your+account")
        legit = url_extractor.score("https://www.example.com/")
        assert phish > legit

    def test_score_in_range(self, url_extractor: URLFeatureExtractor) -> None:
        for url in [
            "https://example.com",
            "http://192.168.1.1/",
            "http://paypal-fake.xyz/login",
            "https://bit.ly/abc",
        ]:
            score = url_extractor.score(url)
            assert 0.0 <= score <= 1.0, f"score={score} out of range for {url}"

    def test_no_scheme_handled(self, url_extractor: URLFeatureExtractor) -> None:
        """URLs without a scheme should not raise an exception."""
        features = url_extractor.extract("example.com/page")
        assert isinstance(features, dict)

    def test_excessive_subdomains(self, url_extractor: URLFeatureExtractor) -> None:
        features = url_extractor.extract("http://a.b.c.d.example.com/")
        assert features["url_subdomain_count"] >= 0.75

    def test_encoded_chars_obfuscation(self, url_extractor: URLFeatureExtractor) -> None:
        encoded_url = "http://example.com/" + "%20" * 20
        features = url_extractor.extract(encoded_url)
        assert features["url_encoded_chars"] > 0.0


# ===========================================================================
# Text feature tests
# ===========================================================================


class TestTextFeatureExtractor:
    def test_returns_all_feature_keys(self, text_extractor: TextFeatureExtractor) -> None:
        features = text_extractor.extract("Hello world")
        expected_keys = {
            "text_urgency",
            "text_credential_request",
            "text_threat_language",
            "text_oci_keywords",
            "text_deceptive_phrases",
            "text_html_risks",
            "text_url_count",
            "text_external_link_ratio",
            "text_obfuscation",
            "text_all_caps_ratio",
        }
        assert expected_keys == set(features.keys())

    def test_all_values_in_range(self, text_extractor: TextFeatureExtractor) -> None:
        features = text_extractor.extract("Click here to verify your password immediately!")
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_urgency_detected(self, text_extractor: TextFeatureExtractor) -> None:
        text = "URGENT: Your account will be suspended. Immediate action required."
        features = text_extractor.extract(text)
        assert features["text_urgency"] > 0.0

    def test_no_urgency_in_benign(self, text_extractor: TextFeatureExtractor) -> None:
        text = "Thank you for your recent purchase."
        features = text_extractor.extract(text)
        assert features["text_urgency"] == 0.0

    def test_credential_request(self, text_extractor: TextFeatureExtractor) -> None:
        text = "Please enter your password and credit card number to continue."
        features = text_extractor.extract(text)
        assert features["text_credential_request"] > 0.0

    def test_threat_language(self, text_extractor: TextFeatureExtractor) -> None:
        text = "Legal action will be taken if you do not respond. Account will be suspended."
        features = text_extractor.extract(text)
        assert features["text_threat_language"] > 0.0

    def test_oci_keywords(self, text_extractor: TextFeatureExtractor) -> None:
        text = "Access to SCADA systems requires VPN access and updated credentials."
        features = text_extractor.extract(text)
        assert features["text_oci_keywords"] > 0.0

    def test_html_risks_form(self, text_extractor: TextFeatureExtractor) -> None:
        html = '<html><form action="http://evil.com" method="post"><input type="password"></form></html>'
        features = text_extractor.extract(html)
        assert features["text_html_risks"] > 0.0

    def test_url_density(self, text_extractor: TextFeatureExtractor) -> None:
        text = " ".join([f"http://example{i}.com/path" for i in range(12)])
        features = text_extractor.extract(text)
        assert features["text_url_count"] == 1.0

    def test_deceptive_phrases(self, text_extractor: TextFeatureExtractor) -> None:
        text = "Dear customer, click here to update your account. Enable content to view."
        features = text_extractor.extract(text)
        assert features["text_deceptive_phrases"] > 0.0

    def test_score_phishing_higher_than_legit(
        self, text_extractor: TextFeatureExtractor
    ) -> None:
        phish = text_extractor.score(
            "URGENT: Enter your password and credit card number immediately. "
            "Your account will be suspended. Click here to verify."
        )
        legit = text_extractor.score("Thank you for contacting support. Have a great day.")
        assert phish > legit

    def test_all_caps_ratio(self, text_extractor: TextFeatureExtractor) -> None:
        text = "ACT NOW VERIFY YOUR ACCOUNT IMMEDIATELY OR LOSE ACCESS"
        features = text_extractor.extract(text)
        assert features["text_all_caps_ratio"] > 0.0


# ===========================================================================
# Email header feature tests
# ===========================================================================


class TestEmailFeatureExtractor:
    def test_returns_all_feature_keys(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        features = header_extractor.extract_from_headers({"From": "test@example.com"})
        expected_keys = {
            "header_from_display_mismatch",
            "header_reply_to_differs",
            "header_return_path_mismatch",
            "header_subject_urgency",
            "header_subject_credential_lure",
            "header_subject_oci_lure",
            "header_spoofed_high_value_domain",
            "header_suspicious_received",
            "header_missing_message_id",
            "header_html_only",
        }
        assert expected_keys == set(features.keys())

    def test_all_values_in_range(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {
            "From": "PayPal <noreply@paypa1-secure.xyz>",
            "Subject": "Urgent: Verify your account",
            "Reply-To": "harvest@attacker.com",
        }
        features = header_extractor.extract_from_headers(headers)
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_reply_to_mismatch(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {
            "From": "noreply@legit.com",
            "Reply-To": "harvest@attacker.com",
        }
        features = header_extractor.extract_from_headers(headers)
        assert features["header_reply_to_differs"] == 1.0

    def test_reply_to_same_domain_ok(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {
            "From": "noreply@legit.com",
            "Reply-To": "support@legit.com",
        }
        features = header_extractor.extract_from_headers(headers)
        assert features["header_reply_to_differs"] == 0.0

    def test_subject_urgency(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {"Subject": "URGENT: Your account is suspended"}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_subject_urgency"] > 0.0

    def test_subject_credential_lure(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {"Subject": "Verify your password and login credentials"}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_subject_credential_lure"] > 0.0

    def test_spoofed_microsoft(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {"From": "Microsoft Security <security@microsoft-alert.com>"}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_spoofed_high_value_domain"] == 1.0

    def test_legit_microsoft_not_flagged(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {"From": "Microsoft <noreply@microsoft.com>"}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_spoofed_high_value_domain"] == 0.0

    def test_missing_message_id(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {"From": "test@example.com", "Message-ID": ""}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_missing_message_id"] == 1.0

    def test_present_message_id(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {
            "From": "test@example.com",
            "Message-ID": "<abc123@mail.example.com>",
        }
        features = header_extractor.extract_from_headers(headers)
        assert features["header_missing_message_id"] == 0.0

    def test_return_path_mismatch(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        headers = {
            "From": "noreply@legit.com",
            "Return-Path": "<bounce@attacker.com>",
        }
        features = header_extractor.extract_from_headers(headers)
        assert features["header_return_path_mismatch"] == 1.0

    def test_score_in_range(self, header_extractor: EmailFeatureExtractor) -> None:
        headers = {
            "From": "PayPal <noreply@paypa1-secure.xyz>",
            "Subject": "Urgent: Verify your account now",
            "Reply-To": "harvest@attacker.com",
        }
        score = header_extractor.score_from_headers(headers)
        assert 0.0 <= score <= 1.0

    def test_oci_subject_lure(self, header_extractor: EmailFeatureExtractor) -> None:
        headers = {"Subject": "VPN access credentials for SCADA system update required"}
        features = header_extractor.extract_from_headers(headers)
        assert features["header_subject_oci_lure"] > 0.0

    def test_from_raw_parses_headers(
        self, header_extractor: EmailFeatureExtractor
    ) -> None:
        raw = (
            "From: Attacker <phish@evil.com>\r\n"
            "Subject: URGENT verify your account\r\n"
            "Message-ID: <xyz@evil.com>\r\n"
            "\r\n"
            "Body text here"
        )
        features = header_extractor.extract_from_raw(raw)
        assert isinstance(features, dict)
        assert len(features) > 0
