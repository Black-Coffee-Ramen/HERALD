"""Tests for PhishingModel."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from herald.model import ALL_FEATURES, PhishingModel


class TestPhishingModel:
    @pytest.fixture(autouse=True)
    def model(self) -> PhishingModel:
        self._model = PhishingModel()
        return self._model

    def test_predict_returns_label_and_probability(self) -> None:
        features = {f: 0.0 for f in ALL_FEATURES}
        label, prob = self._model.predict(features)
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_predict_phishing_sample(self) -> None:
        """High-signal phishing features should produce label=1."""
        features = {f: 0.0 for f in ALL_FEATURES}
        # Set all phishing-indicative features to maximum
        for feat in [
            "url_uses_ip",
            "url_suspicious_tld",
            "url_brand_impersonation",
            "url_has_at_sign",
            "url_is_shortener",
            "text_urgency",
            "text_credential_request",
            "text_threat_language",
            "header_from_display_mismatch",
            "header_spoofed_high_value_domain",
            "header_reply_to_differs",
        ]:
            features[feat] = 1.0
        label, prob = self._model.predict(features)
        assert label == 1
        assert prob > 0.5

    def test_predict_legitimate_sample(self) -> None:
        """All-zero features (fully benign) should produce label=0."""
        features = {f: 0.0 for f in ALL_FEATURES}
        label, prob = self._model.predict(features)
        assert label == 0
        assert prob < 0.5

    def test_missing_features_default_to_zero(self) -> None:
        """Passing a partial feature dict should not raise."""
        label, prob = self._model.predict({"url_uses_ip": 1.0})
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_feature_names_property(self) -> None:
        names = self._model.feature_names
        assert names == ALL_FEATURES
        assert len(names) > 0

    def test_refit_on_custom_data(self) -> None:
        """fit() should not raise and should produce a usable model."""
        rng = np.random.default_rng(0)
        n = 50
        feature_dicts = [
            {f: float(rng.uniform(0, 1)) for f in ALL_FEATURES} for _ in range(n)
        ]
        labels = [int(rng.integers(0, 2)) for _ in range(n)]
        self._model.fit(feature_dicts, labels)
        label, prob = self._model.predict({f: 0.5 for f in ALL_FEATURES})
        assert label in (0, 1)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            self._model.save(path)
            assert os.path.exists(path)

            loaded = PhishingModel(model_path=path)
            features = {f: 0.5 for f in ALL_FEATURES}
            label1, prob1 = self._model.predict(features)
            label2, prob2 = loaded.predict(features)
            assert label1 == label2
            assert abs(prob1 - prob2) < 1e-6

    def test_load_nonexistent_falls_back_to_default(self) -> None:
        """Providing a non-existent model path should use the default model."""
        model = PhishingModel(model_path="/non/existent/path.joblib")
        label, prob = model.predict({f: 0.0 for f in ALL_FEATURES})
        assert label in (0, 1)
