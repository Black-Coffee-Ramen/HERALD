"""Core phishing detection orchestrator.

:class:`PhishingDetector` is the primary entry point for HERALD.  It:

1. Extracts features from the supplied input (URL, email headers, or message
   body) via the specialised feature extractors.
2. Combines the per-domain risk scores using configurable weights.
3. Queries the :class:`~herald.model.PhishingModel` for an ML probability.
4. Merges the heuristic and ML signals into a final :class:`~herald.result.DetectionResult`.

Critical-infrastructure deployments can tune the threshold and feature
weights via a custom :class:`~herald.config.Config` instance.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from herald.config import Config
from herald.features.email_features import EmailFeatureExtractor
from herald.features.text_features import TextFeatureExtractor
from herald.features.url_features import URLFeatureExtractor
from herald.logger import get_logger
from herald.model import HEADER_FEATURES, TEXT_FEATURES, URL_FEATURES, PhishingModel
from herald.result import DetectionResult, RiskLevel

logger = get_logger("detector")


class PhishingDetector:
    """AI-based phishing detector for critical infrastructure environments.

    Args:
        config: Runtime configuration.  Defaults to :class:`~herald.config.Config`
            with values read from environment variables via
            :meth:`~herald.config.Config.from_env`.
        model_path: Path to a saved :class:`~herald.model.PhishingModel`.
            When *None* the built-in default model (trained on synthetic
            data) is used.

    Example::

        detector = PhishingDetector()

        # Analyse a URL
        result = detector.analyze_url("http://paypa1-secure.xyz/login")

        # Analyse email headers
        result = detector.analyze_headers({
            "From": "PayPal Security <noreply@paypa1-secure.xyz>",
            "Subject": "Urgent: Verify your account",
        })

        # Analyse message body
        result = detector.analyze_text(email_body)

        # Full email analysis
        result = detector.analyze_email(
            headers={"From": ..., "Subject": ...},
            body=email_body,
            urls=["http://paypa1-secure.xyz/login"],
        )
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self._config = config or Config.from_env()
        self._model = PhishingModel(model_path=model_path)
        self._url_extractor = URLFeatureExtractor(self._config)
        self._text_extractor = TextFeatureExtractor()
        self._header_extractor = EmailFeatureExtractor()

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def analyze_url(self, url: str) -> DetectionResult:
        """Analyse a single URL for phishing indicators.

        Args:
            url: The URL string to analyse.

        Returns:
            A :class:`~herald.result.DetectionResult` with URL-specific
            feature scores and an overall risk assessment.
        """
        logger.debug("Analysing URL: %s", url)
        url_features = self._url_extractor.extract(url)
        url_score = self._url_extractor.score(url)

        # Build the full feature vector (text/header features default to 0)
        all_features = {f: 0.0 for f in URL_FEATURES + TEXT_FEATURES + HEADER_FEATURES}
        all_features.update(url_features)

        ml_label, ml_prob = self._model.predict(all_features)

        # Combine heuristic URL score with ML probability
        combined = self._combine_scores(
            url_score=url_score,
            text_score=0.0,
            header_score=0.0,
            ml_prob=ml_prob,
        )

        indicators = self._url_indicators(url_features)
        return self._build_result(
            combined_score=combined,
            ml_prob=ml_prob,
            indicators=indicators,
            url_score=url_score,
            text_score=0.0,
            header_score=0.0,
            feature_scores=url_features,
        )

    def analyze_text(self, text: str) -> DetectionResult:
        """Analyse message body text for phishing indicators.

        Args:
            text: Plain-text or HTML email body.

        Returns:
            A :class:`~herald.result.DetectionResult` with text-specific
            feature scores and an overall risk assessment.
        """
        logger.debug("Analysing text body (%d chars).", len(text))
        text_features = self._text_extractor.extract(text)
        text_score = self._text_extractor.score(text)

        all_features = {f: 0.0 for f in URL_FEATURES + TEXT_FEATURES + HEADER_FEATURES}
        all_features.update(text_features)

        ml_label, ml_prob = self._model.predict(all_features)

        combined = self._combine_scores(
            url_score=0.0,
            text_score=text_score,
            header_score=0.0,
            ml_prob=ml_prob,
        )

        indicators = self._text_indicators(text_features)
        return self._build_result(
            combined_score=combined,
            ml_prob=ml_prob,
            indicators=indicators,
            url_score=0.0,
            text_score=text_score,
            header_score=0.0,
            feature_scores=text_features,
        )

    def analyze_headers(self, headers: Dict[str, str]) -> DetectionResult:
        """Analyse email headers for spoofing and impersonation indicators.

        Args:
            headers: Mapping of header name → value (case-insensitive).

        Returns:
            A :class:`~herald.result.DetectionResult` with header-specific
            feature scores and an overall risk assessment.
        """
        logger.debug("Analysing %d headers.", len(headers))
        header_features = self._header_extractor.extract_from_headers(headers)
        header_score = self._header_extractor.score_from_headers(headers)

        all_features = {f: 0.0 for f in URL_FEATURES + TEXT_FEATURES + HEADER_FEATURES}
        all_features.update(header_features)

        ml_label, ml_prob = self._model.predict(all_features)

        combined = self._combine_scores(
            url_score=0.0,
            text_score=0.0,
            header_score=header_score,
            ml_prob=ml_prob,
        )

        indicators = self._header_indicators(header_features)
        return self._build_result(
            combined_score=combined,
            ml_prob=ml_prob,
            indicators=indicators,
            url_score=0.0,
            text_score=0.0,
            header_score=header_score,
            feature_scores=header_features,
        )

    def analyze_email(
        self,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        urls: Optional[List[str]] = None,
        raw_message: Optional[str] = None,
    ) -> DetectionResult:
        """Full email analysis combining headers, body, and embedded URLs.

        Provide either *raw_message* (a complete RFC 2822 string) **or** a
        combination of *headers*, *body*, and *urls*.

        Args:
            headers: Email header dictionary.
            body: Plain-text or HTML message body.
            urls: List of URLs extracted from or linked in the message.
            raw_message: Complete raw RFC 2822 email string.  When supplied,
                *headers* and *body* are derived from this string.

        Returns:
            A :class:`~herald.result.DetectionResult` combining all
            available signal sources.
        """
        if raw_message:
            import email as email_lib
            import email.policy as email_policy

            msg = email_lib.message_from_string(
                raw_message, policy=email_policy.default
            )
            headers = {k: str(v) for k, v in msg.items()}
            payload = msg.get_payload(decode=False)
            if isinstance(payload, str):
                body = payload
            elif msg.is_multipart():
                parts = []
                for part in msg.walk():
                    ct = part.get_content_type()
                    if ct in ("text/plain", "text/html"):
                        try:
                            part_payload = part.get_payload(decode=True)
                            if part_payload:
                                parts.append(part_payload.decode("utf-8", errors="replace"))
                        except Exception:
                            pass
                body = "\n".join(parts)

        # Compute per-domain scores
        url_score = 0.0
        url_features: Dict[str, float] = {f: 0.0 for f in URL_FEATURES}
        if urls:
            url_scores_list = [self._url_extractor.score(u) for u in urls]
            url_score = max(url_scores_list) if url_scores_list else 0.0
            # Use features from the highest-scoring URL
            if urls:
                worst_url = urls[
                    url_scores_list.index(max(url_scores_list))
                ]
                url_features = self._url_extractor.extract(worst_url)

        text_score = 0.0
        text_features: Dict[str, float] = {f: 0.0 for f in TEXT_FEATURES}
        if body:
            text_score = self._text_extractor.score(body)
            text_features = self._text_extractor.extract(body)

        header_score = 0.0
        header_features: Dict[str, float] = {f: 0.0 for f in HEADER_FEATURES}
        if headers:
            header_score = self._header_extractor.score_from_headers(headers)
            header_features = self._header_extractor.extract_from_headers(headers)

        # Combined feature vector for the ML model
        all_features: Dict[str, float] = {}
        all_features.update(url_features)
        all_features.update(text_features)
        all_features.update(header_features)

        ml_label, ml_prob = self._model.predict(all_features)

        combined = self._combine_scores(
            url_score=url_score,
            text_score=text_score,
            header_score=header_score,
            ml_prob=ml_prob,
        )

        # Collect all indicators
        indicators: List[str] = []
        indicators.extend(self._url_indicators(url_features))
        indicators.extend(self._text_indicators(text_features))
        indicators.extend(self._header_indicators(header_features))

        # Deduplicate while preserving order
        seen: set = set()
        unique_indicators: List[str] = []
        for ind in indicators:
            if ind not in seen:
                seen.add(ind)
                unique_indicators.append(ind)

        all_feature_scores = {**url_features, **text_features, **header_features}
        return self._build_result(
            combined_score=combined,
            ml_prob=ml_prob,
            indicators=unique_indicators,
            url_score=url_score,
            text_score=text_score,
            header_score=header_score,
            feature_scores=all_feature_scores,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine_scores(
        self,
        url_score: float,
        text_score: float,
        header_score: float,
        ml_prob: float,
    ) -> float:
        """Merge heuristic component scores with the ML probability.

        Heuristic scores are weighted only over *populated* domains so that
        analysing a URL in isolation (without headers or body) returns the
        full URL risk score rather than a diluted average.  The final
        combined score is the **maximum** of the weighted heuristic and the
        ML probability, so that a strong signal from *either* source raises
        the alert level.  This conservative (max-fusion) approach is
        appropriate for critical-infrastructure deployments where the cost
        of a missed phishing attempt is high.
        """
        cfg = self._config
        total_weight = 0.0
        weighted_sum = 0.0

        if url_score > 0:
            total_weight += cfg.url_weight
            weighted_sum += url_score * cfg.url_weight
        if text_score > 0:
            total_weight += cfg.text_weight
            weighted_sum += text_score * cfg.text_weight
        if header_score > 0:
            total_weight += cfg.header_weight
            weighted_sum += header_score * cfg.header_weight

        heuristic = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Max-fusion: take the most pessimistic risk assessment
        combined = max(heuristic, ml_prob)
        return min(combined, 1.0)

    def _build_result(
        self,
        combined_score: float,
        ml_prob: float,
        indicators: List[str],
        url_score: float,
        text_score: float,
        header_score: float,
        feature_scores: Dict[str, float],
    ) -> DetectionResult:
        is_phishing = combined_score >= self._config.phishing_threshold
        risk_level = RiskLevel.from_score(combined_score)
        # Confidence is based on distance from decision boundary (0.5)
        confidence = min(abs(ml_prob - 0.5) * 2.0, 1.0)
        return DetectionResult(
            is_phishing=is_phishing,
            risk_score=combined_score,
            risk_level=risk_level,
            confidence=confidence,
            indicators=indicators,
            url_score=url_score,
            text_score=text_score,
            header_score=header_score,
            feature_scores=feature_scores,
        )

    # ------------------------------------------------------------------
    # Indicator text generation
    # ------------------------------------------------------------------

    @staticmethod
    def _url_indicators(features: Dict[str, float]) -> List[str]:
        indicators = []
        if features.get("url_uses_ip", 0) > 0.5:
            indicators.append("URL uses an IP address instead of a domain name")
        if features.get("url_brand_impersonation", 0) > 0.5:
            indicators.append(
                "URL contains a well-known brand name on an unrelated domain"
            )
        if features.get("url_suspicious_tld", 0) > 0.5:
            indicators.append("URL uses a high-risk top-level domain")
        if features.get("url_is_shortener", 0) > 0.5:
            indicators.append("URL uses a URL-shortening service to hide the destination")
        if features.get("url_has_at_sign", 0) > 0.5:
            indicators.append("URL contains an '@' sign (possible credential lure)")
        if features.get("url_double_slash_redirect", 0) > 0.5:
            indicators.append("URL contains a double-slash redirect pattern")
        if features.get("url_is_http", 0) > 0.5:
            indicators.append("URL uses unencrypted HTTP (no TLS)")
        if features.get("url_subdomain_count", 0) > 0.6:
            indicators.append("URL has an unusual number of subdomains")
        if features.get("url_is_long", 0) > 0.6:
            indicators.append("URL is abnormally long")
        if features.get("url_suspicious_keywords", 0) > 0.3:
            indicators.append("URL contains credential-harvesting keywords")
        return indicators

    @staticmethod
    def _text_indicators(features: Dict[str, float]) -> List[str]:
        indicators = []
        if features.get("text_credential_request", 0) > 0.3:
            indicators.append("Message contains requests for credentials or personal data")
        if features.get("text_urgency", 0) > 0.3:
            indicators.append("Message uses urgency language to pressure the recipient")
        if features.get("text_threat_language", 0) > 0.3:
            indicators.append("Message contains threatening language or account-closure warnings")
        if features.get("text_oci_keywords", 0) > 0.3:
            indicators.append(
                "Message references critical-infrastructure or OT/ICS systems"
            )
        if features.get("text_deceptive_phrases", 0) > 0.3:
            indicators.append("Message contains deceptive call-to-action phrases")
        if features.get("text_html_risks", 0) > 0.3:
            indicators.append("HTML body contains risky constructs (forms, iframes, scripts)")
        if features.get("text_url_count", 0) > 0.4:
            indicators.append("Message contains an unusually high number of embedded URLs")
        return indicators

    @staticmethod
    def _header_indicators(features: Dict[str, float]) -> List[str]:
        indicators = []
        if features.get("header_from_display_mismatch", 0) > 0.5:
            indicators.append(
                "Sender display name does not match the actual sending address"
            )
        if features.get("header_spoofed_high_value_domain", 0) > 0.5:
            indicators.append(
                "Sender address impersonates a high-value brand or government domain"
            )
        if features.get("header_reply_to_differs", 0) > 0.5:
            indicators.append("Reply-To address belongs to a different domain than the sender")
        if features.get("header_return_path_mismatch", 0) > 0.5:
            indicators.append("Return-Path domain does not match the From domain")
        if features.get("header_subject_urgency", 0) > 0.3:
            indicators.append("Subject line contains urgency indicators")
        if features.get("header_subject_credential_lure", 0) > 0.3:
            indicators.append("Subject line contains credential-harvesting lure words")
        if features.get("header_subject_oci_lure", 0) > 0.3:
            indicators.append(
                "Subject line references critical-infrastructure or OT/ICS systems"
            )
        if features.get("header_missing_message_id", 0) > 0.5:
            indicators.append("Email is missing a Message-ID header")
        return indicators
