import unittest
from unittest.mock import patch

from herald.core.security import SSRFProtectionError, validate_url_safe
from herald.investigation.persistence import render_markdown
from herald.investigation.scoring import analyze_lexical, combine_scores
from herald.investigation.targets import normalize_target


class InvestigationCliTests(unittest.TestCase):
    def test_normalize_target_defaults_to_https(self):
        url, domain = normalize_target("www.paypal-login-alert.com/path")

        self.assertEqual(url, "https://paypal-login-alert.com/path")
        self.assertEqual(domain, "paypal-login-alert.com")

    def test_suspicious_lexical_reasons_are_explained(self):
        lexical = analyze_lexical("paypal-login-alert.com")

        self.assertGreaterEqual(lexical["score"], 0.35)
        self.assertIn("login", lexical["suspicious_keywords"])
        self.assertTrue(any(factor["name"] == "Suspicious keywords" for factor in lexical["risk_factors"]))

    def test_score_combination_returns_risk_factors(self):
        lexical = analyze_lexical("paypal-login-alert.com")
        verdict, score, factors = combine_scores(
            lexical,
            {"errors": ["A: missing"], "a_records": []},
            {"has_tls": False},
            {"skipped": True},
        )

        self.assertEqual(verdict, "Suspected")
        self.assertGreaterEqual(score, 0.35)
        self.assertTrue(any(factor["name"] == "DNS unresolved" for factor in factors))

    def test_markdown_report_contains_operational_sections(self):
        markdown = render_markdown({
            "trace_id": "trc-test",
            "url": "https://example.com",
            "domain": "example.com",
            "verdict": "Likely Clean",
            "phishing_score": 0.1,
            "evidence_dir": "evidence/trc-test_example.com",
            "completed_at": "2026-05-25T00:00:00+00:00",
            "summary": {"headline": "example.com is Likely Clean.", "why_flagged": ["No strong phishing indicators were observed."]},
            "risk_factors": [],
            "dns": {},
            "tls": {},
            "visual": {},
            "stages": [],
            "errors": [],
        })

        self.assertIn("## Executive Summary", markdown)
        self.assertIn("## Why Flagged", markdown)
        self.assertIn("## Risk Factors", markdown)
        self.assertIn("## Evidence", markdown)

    def test_ssrf_blocks_private_ip_by_default(self):
        with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.7", 0))]):
            with self.assertRaises(SSRFProtectionError) as raised:
                validate_url_safe("https://internal.example")

        self.assertEqual(raised.exception.resolved_ip, "10.0.0.7")
        self.assertIn("private", raised.exception.reason)

    def test_ssrf_allows_private_ip_with_explicit_override(self):
        private_overrides = []
        with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.7", 0))]):
            hostname = validate_url_safe(
                "https://internal.example",
                allow_private=True,
                trace_id="trc-test",
                private_overrides=private_overrides,
            )

        self.assertEqual(hostname, "internal.example")
        self.assertEqual(private_overrides[0]["resolved_ip"], "10.0.0.7")
        self.assertEqual(private_overrides[0]["trace_id"], "trc-test")

    def test_ssrf_metadata_endpoint_remains_blocked_with_override(self):
        with self.assertRaises(SSRFProtectionError) as raised:
            validate_url_safe("http://169.254.169.254", allow_private=True)

        self.assertIn("metadata", raised.exception.reason.lower())


if __name__ == "__main__":
    unittest.main()
