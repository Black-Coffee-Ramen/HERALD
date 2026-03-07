# HERALD

**HERALD** is an AI-based phishing detection system designed for **critical infrastructure** environments.  It combines rule-based heuristics with a machine-learning classifier to analyse URLs, email headers, and message bodies for phishing indicators — with special attention to operational-technology (OT) and industrial-control-system (ICS) lures that target energy, utilities, government, and other critical-infrastructure organisations.

---

## Features

| Capability | Description |
|---|---|
| **URL analysis** | IP hosts, suspicious TLDs, URL shorteners, brand impersonation, credential-harvesting keywords, redirect patterns, and more |
| **Email header analysis** | From/Reply-To domain mismatches, spoofed high-value brands, missing Message-ID, urgency subjects, OT/ICS lures |
| **Body text analysis** | Urgency language, credential requests, threat phrases, OCI-specific keywords (SCADA, VPN, PLC, …), HTML risks |
| **ML classifier** | Gradient-boosted model trained on synthetic data; re-trainable on deployment-specific data |
| **Risk levels** | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` with a continuous score in [0, 1] |
| **Configurable threshold** | Tune the decision boundary via config or `HERALD_THRESHOLD` env var |
| **JSON output** | Machine-readable results for SIEM integration |
| **CLI** | `herald url`, `herald text`, `herald headers`, `herald email` sub-commands |

---

## Installation

```bash
pip install .
```

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

### Requirements

* Python ≥ 3.9
* `scikit-learn ≥ 1.3`
* `numpy ≥ 1.24`
* `joblib ≥ 1.3`

---

## Quick start

### Python API

```python
from herald import PhishingDetector

detector = PhishingDetector()

# Analyse a URL
result = detector.analyze_url("http://paypa1-secure.xyz/login?verify=account")
print(result.summary())
# Verdict   : PHISHING
# Risk Level: MEDIUM
# Risk Score: 0.2537
# Confidence: 0.9975
# Indicators:
#   • URL uses a high-risk top-level domain
#   • URL uses unencrypted HTTP (no TLS)
#   • URL contains credential-harvesting keywords

# Analyse email headers
result = detector.analyze_headers({
    "From": "PayPal Security <noreply@paypal-alert.xyz>",
    "Subject": "Urgent: Verify your account",
    "Reply-To": "harvest@attacker.com",
})

# Full email analysis from a raw .eml file
with open("suspicious.eml") as f:
    result = detector.analyze_email(raw_message=f.read())
print(result.to_dict())  # JSON-serialisable dict for SIEM
```

### Command-line interface

```bash
# Analyse a URL
herald url "http://paypa1-secure.xyz/login"

# Analyse email headers inline
herald headers \
  --from-addr "PayPal <noreply@paypal-verify.xyz>" \
  --subject "URGENT: Account suspended" \
  --reply-to "steal@attacker.com"

# Full email analysis from a .eml file
herald email suspicious.eml

# Analyse a text file
herald text --file body.txt

# JSON output for SIEM integration
herald --format json url "http://192.168.1.1/admin/login"

# Lower the detection threshold (more sensitive)
herald --threshold 0.3 email suspicious.eml
```

**Exit codes**: `0` = legitimate, `1` = phishing detected, `2` = error.

---

## Configuration

Configuration can be supplied via a `Config` object or environment variables:

| Environment variable | Default | Description |
|---|---|---|
| `HERALD_THRESHOLD` | `0.5` | Phishing score threshold (0–1) |
| `HERALD_LOG_LEVEL` | `INFO` | Logging verbosity |
| `HERALD_URL_WEIGHT` | `0.4` | Weight of URL heuristics in combined score |
| `HERALD_TEXT_WEIGHT` | `0.35` | Weight of text heuristics |
| `HERALD_HEADER_WEIGHT` | `0.25` | Weight of header heuristics |

```python
from herald.config import Config
from herald import PhishingDetector

# Stricter configuration for critical infrastructure
config = Config(phishing_threshold=0.35)
detector = PhishingDetector(config=config)
```

---

## Training a custom model

```python
from herald.model import PhishingModel

model = PhishingModel()

# Supply labelled feature dictionaries
feature_dicts = [...]  # list of {feature_name: score} dicts
labels = [1, 0, 1, ...]  # 1 = phishing, 0 = legitimate

model.fit(feature_dicts, labels)
model.save("custom_model.joblib")

# Use with the detector
from herald import PhishingDetector
detector = PhishingDetector(model_path="custom_model.joblib")
```

---

## Running tests

```bash
pytest
# or with coverage
pytest --cov=herald --cov-report=term-missing
```

---

## Architecture

```
herald/
├── __init__.py          # Public API: PhishingDetector, DetectionResult
├── config.py            # Configuration dataclass
├── result.py            # DetectionResult and RiskLevel
├── logger.py            # Logging helpers
├── model.py             # Gradient-boosted ML model
├── detector.py          # Orchestrator (heuristics + ML)
└── features/
    ├── url_features.py   # URL-based feature extraction
    ├── text_features.py  # Text/body feature extraction
    └── email_features.py # Email header feature extraction
```

The **risk score** is computed as `max(heuristic_score, ml_probability)`, where the heuristic score is a weighted combination over only the *populated* analysis domains.  This max-fusion approach ensures that a strong signal from *either* the rule-based system or the ML model will raise an alert — appropriate for environments where missed detections are more costly than false positives.

---

## Security considerations

* HERALD is designed as a **detection aid**, not a sole gatekeeper.  All alerts should be reviewed by a security analyst.
* The default model is trained on **synthetic data**.  For production deployments, re-train with organisation-specific labelled data for best results.
* Feature weights and the detection threshold should be tuned for the specific deployment environment using historical data and acceptable false-positive rates.
