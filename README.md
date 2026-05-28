<div align="center">

<img src="public/logo-positive.png" width="220" alt="HERALD Logo">

# HERALD — Phishing Domain Intelligence Platform

> Autonomous phishing domain detection for critical sector entities. No third-party APIs. Fully on-premises.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Precision: 0.981](https://img.shields.io/badge/Precision-0.981-brightgreen.svg)](#performance)

HERALD is an open-source phishing investigation platform that monitors the internet for lookalike domains targeting banks, government portals, and financial institutions. It catches threats within minutes of domain registration by combining Certificate Transparency log monitoring, multi-stage ML detection, live network enrichment, and Playwright-powered visual analysis — all without relying on VirusTotal, Shodan, or any paid threat intelligence feed.

The most reliable and battle-tested path today is the **CLI investigation workflow**. The API, Redis worker queue, and Next.js dashboard are operational but under active stabilization.

---

## Table of Contents

- [Why HERALD](#why-herald)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage — CLI Reference](#usage--cli-reference)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [ML Model Evolution](#ml-model-evolution)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Known Limitations](#known-limitations)
- [Contact](#contact)

---

## Why HERALD

Commercial threat intelligence platforms cost tens of thousands of dollars annually and create data sovereignty concerns through third-party API dependencies. Small banks, fintech companies, and government agencies in developing markets need the same level of protection.

HERALD is:

- **Self-hosted** — your domain watchlist and scan data never leave your infrastructure
- **API-free** — no VirusTotal, Shodan, or commercial feeds required
- **Real-time** — catches phishing domains within minutes of CT log registration
- **Explainable** — every verdict comes with a human-readable risk factor breakdown
- **Resilient** — individual stage failures (DNS, TLS, OCR) degrade gracefully without aborting an investigation

---

## How It Works

### Discovery

HERALD discovers suspicious domains through three channels:

- **Certificate Transparency logs** via Certstream WebSocket (`wss://certstream.calidog.io`) with `crt.sh` fallback polling
- **Newly registered domain feeds** polled hourly
- **Social media scraping** of public Telegram channels for shared phishing links

### Detection Pipeline

Domains enter a two-stage pipeline:

1. **Stage 1 — Lexical screening**: Fast heuristic analysis of domain name structure — length, digit ratio, hyphen count, brand keyword position, suspicious TLDs, punycode, entropy, and edit distance from monitored brands.

2. **Stage 2 — Network enrichment**: Borderline scores trigger live WHOIS age lookup, DNS record analysis, TLS certificate inspection, and optional headless browser screenshot with OCR phrase matching.

3. **Score fusion**: Lexical score, domain age, TLS anomalies, and OCR findings combine into an additive verdict capped at 1.0.

4. **Visual fallback**: Borderline detections trigger a Playwright screenshot and perceptual analysis against known brand templates — catching phishing pages with no URL similarity to their target brand.

---

## Performance

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| Indian CSE-Filtered Test Set | **0.981** | **0.841** | **0.906** |
| Sanity check — phishing (n=6) | 1.000 | 1.000 | 1.000 |
| Sanity check — legitimate (n=6) | 1.000 | 1.000 | 1.000 |

> External validation run on March 10, 2026 on PhishTank data filtered for the Indian financial and government sector.

---

## Features

### Detection Capabilities

| Capability | Description |
|---|---|
| Typosquatting | Edit distance, keyboard adjacency, character substitution |
| IDN / Homoglyph | Unicode confusable character detection (Cyrillic, Greek) |
| Fuzzy brand matching | Levenshtein distance catches `5bi`, `hdfc1`, `uldai` variants |
| Path-based phishing | Brand keywords buried in URL paths on generic domains |
| TLD risk scoring | Explicit penalty for high-risk gTLDs (`.xyz`, `.top`, `.buzz`, `.tk`) |
| Tunnelling detection | Flags Ngrok, Vercel, Cloudflare Tunnel subdomains |
| Visual similarity | OCR + perceptual hashing against known CSE page templates |
| Suspected monitoring | Re-monitors parked domains up to 90 days; escalates on activation |

### Per-Domain Reports

Every detected domain generates a full evidence package:

- Domain creation date, registrar, and registrant details
- IP address, ASN, and hosting country
- MX and DNS records
- SSL/TLS certificate issuer, SAN, and expiry metadata
- Full-page screenshot
- PDF evidence export
- Maliciousness confidence score with explainable risk factors

---

## Tech Stack

**Backend**

| Layer | Technology |
|---|---|
| CLI / entrypoint | Python 3.12, argparse via `setup.py` console script |
| Investigation pipeline | Custom `InvestigationPipeline` in `herald/investigation/` |
| API | FastAPI + Uvicorn + SlowAPI (rate limiting) |
| ML ensemble | scikit-learn Random Forest + XGBoost, joblib serialization |
| Browser automation | Playwright (headless Chromium) |
| OCR | Tesseract via pytesseract |
| Feature extraction | dnspython, python-whois, tldextract, BeautifulSoup |
| Queue / workers | Redis + `RedisReliableQueue` (leases, DLQ, retries) |
| Database | SQLAlchemy — SQLite default, PostgreSQL optional |
| Telemetry | Redis pub/sub → WebSocket bridge |
| Reports | reportlab (PDF), structlog (structured logging) |

**Frontend**

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router), React 19 |
| Styling | Tailwind CSS 4 |
| Charts | Recharts |
| Icons | lucide-react |
| Real-time | WebSocket client connected to `/ws/telemetry` |

---

## Architecture Overview

HERALD has three operational layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI-first investigation path  (primary, reliable today)        │
│                                                                  │
│  herald CLI → InvestigationPipeline                             │
│    → SSRF validation → Lexical → DNS/WHOIS → TLS →             │
│      Playwright/OCR → Score fusion → Evidence persistence       │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  API / Redis worker path  (partially active, stabilizing)       │
│                                                                  │
│  FastAPI → Redis queues → Domain worker (PhishingPredictorV3)  │
│    → SQLAlchemy DB                                              │
│    → Visual worker (Playwright subprocess)                      │
│    → Redis pub/sub telemetry → WebSocket /ws/telemetry          │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Next.js operations console  (mock-first, real hooks present)   │
│                                                                  │
│  Dashboard → TelemetryClient (MOCK default)                     │
│    → REAL/HYBRID: WebSocket to FastAPI backend                  │
└─────────────────────────────────────────────────────────────────┘
```

**API/Worker data flow:**

```
Client → FastAPI → Redis reliable queues → Domain worker
                                               ↓
                                         DomainScan DB
                                               ↓
                                 Borderline → Visual worker
                                               ↓
                                         Redis pub/sub
                                               ↓
                                 FastAPI → WebSocket /ws/telemetry
```

---


## Repository Structure

```text
herald/                                  # Core backend package powering phishing intelligence workflows
├── cli.py                               # Unified CLI entrypoint for investigations, reporting, screenshots, and analysis
│
├── investigation/                       # End-to-end investigation orchestration pipeline
│   ├── pipeline.py                      # InvestigationPipeline coordinating the full analysis lifecycle
│   ├── scoring.py                       # Heuristic scoring engine, confidence fusion, and verdict generation
│   ├── intelligence.py                  # DNS, WHOIS, TLS, and infrastructure intelligence collectors
│   ├── targets.py                       # URL normalization, parsing, validation, and safe-domain helpers
│   └── persistence.py                   # Evidence persistence layer for JSON, Markdown, and JSONL artifacts
│
├── core/                                # Shared security, browser, authentication, and utility primitives
│   ├── security.py                      # SSRF mitigation, IP validation, and private-range blocking
│   ├── playwright_analyzer.py           # Headless Chromium automation, OCR extraction, and screenshot capture
│   ├── auth.py                          # JWT authentication, bcrypt password hashing, and access control
│   └── homoglyph_generator.py           # Unicode homoglyph and confusable-domain generation utilities
│
├── features/                            # Feature engineering and extraction modules
│   ├── lexical_features.py              # Lexical phishing indicators and brand impersonation detection
│   ├── content_features.py              # HTTP content inspection and page-level behavioral analysis
│   └── dns_features.py                  # DNS resolution, record parsing, and infrastructure enrichment
│
├── api/                                 # FastAPI backend services and API layer
│   └── main.py                          # REST API routes, WebSocket bridge, queue submission, and orchestration
│
├── db/                                  # Database abstraction and persistence models
│   └── models.py                        # SQLAlchemy models for scans, whitelists, and historical tracking
│
├── monitoring/                          # Distributed queue processing and operational infrastructure
│   ├── redis_queue.py                   # Reliable Redis queue with retries, leasing, and dead-letter handling
│   ├── queue_worker.py                  # Domain analysis worker consuming queued scan jobs
│   ├── visual_worker.py                 # Isolated OCR/browser subprocess worker for visual inspection
│   ├── metrics.py                       # Prometheus-style runtime metrics and instrumentation
│   ├── resilience.py                    # Redis circuit breaker and fault-tolerance utilities
│   └── scheduler.py                     # Automated re-scan scheduling for suspicious domains
│
├── ingestion/                           # Real-time domain intelligence and threat ingestion services
│   ├── certstream_monitor.py            # Certificate Transparency log stream monitoring
│   ├── new_domains_monitor.py           # Newly registered domain discovery and polling pipeline
│   ├── social_monitor.py                # Telegram public-channel phishing intelligence scraper
│   └── tunnel_monitor.py                # Detection of tunneling-service generated subdomains
│
├── telemetry/                           # Redis pub/sub telemetry transport and event envelopes
│
├── predict_with_fallback.py             # ML inference pipeline with resilient fallback prediction handling
│
└── utils/                               # Shared utilities for exports, logging, and reporting
    ├── logging/                         # Structured logging helpers and runtime diagnostics
    ├── exporters/                       # JSON, CSV, and structured evidence export utilities
    └── reporting/                       # HTML/PDF report generation and formatting helpers

frontend/                                # Next.js operational dashboard and analyst console
├── app/                                 # App Router pages, layouts, and API routes
├── components/                          # Dashboard widgets, traces, DLQ views, and investigation panels
├── hooks/                               # Custom React hooks including telemetry subscriptions
├── services/                            # WebSocket clients, API adapters, and mock data generators
└── types/                               # Shared TypeScript interfaces and telemetry schemas

models/                                  # Machine learning model artifacts and serialized assets
├── ensemble_v7.joblib                   # Production ensemble model (Random Forest + XGBoost)
├── domain_transformer.pt                # Experimental transformer-based character model
└── char_vocab.json                      # Character vocabulary mapping for transformer inference

research/                                # Experimental ML pipelines, datasets, notebooks, and training scripts

legacy/                                  # Archived legacy implementations and deprecated tooling

tests/                                   # Pytest suite covering scoring, CLI flows, APIs, and security logic

docker/                                  # Containerization assets and deployment orchestration files

evidence/                                # Runtime-generated investigation evidence and forensic artifacts

requirements-runtime.txt                 # Minimal runtime dependencies for production deployments
requirements-dev.txt                     # Development, linting, formatting, and testing dependencies
requirements-research.txt                # Research and experimentation dependencies
requirements-lock.txt                    # Fully pinned dependency lock file

setup.py                                 # Python package metadata and installation configuration

config.yaml                              # Centralized runtime and infrastructure configuration

docker-compose.yml                       # Multi-service local orchestration setup

.env.example                             # Environment variable template for local setup and deployment
```
---

## Quick Start

The fastest path to a working investigation — no server or database required:

```bash
git clone https://github.com/Black-Coffee-Ramen/HERALD
cd HERALD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python -m playwright install chromium
herald investigate paypal-login-alert.com
```

---

## Installation

### Prerequisites

- Python 3.12+
- Node.js 18+ (frontend only)
- Tesseract OCR (optional — enables OCR text extraction)

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements-dev.txt
pip install -e .
python -m playwright install chromium
```

### Optional: Tesseract OCR

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

Without Tesseract, screenshots still capture but OCR text extraction is skipped. The investigation continues with a degraded visual stage.

### Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
# Available at http://localhost:3000
```

The frontend defaults to mock/synthetic telemetry. Set `NEXT_PUBLIC_TELEMETRY_MODE=REAL` and run the API backend to connect live data.

### Docker (Experimental — Fix Required)

Before building, update `docker/Dockerfile` to reference the correct requirements file:

```dockerfile
# Replace:
COPY requirements.txt .
RUN pip install -r requirements.txt

# With:
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt
```

Then:

```bash
docker compose up --build
```

---

## Usage — CLI Reference

The `herald` console script is installed by `setup.py` as `herald = herald.cli:main`.

### `herald investigate`

Runs the full investigation pipeline: SSRF validation → lexical analysis → DNS/WHOIS → TLS → screenshot/OCR → score fusion → evidence persistence.

```bash
herald investigate <target> [--json] [--no-visual] [--allow-private]
```

```bash
# Standard investigation with Rich terminal output
herald investigate paypal-login-alert.com

# JSON output for scripting and automation
herald investigate https://example.com/login --json

# Skip Playwright and OCR (faster, no browser required)
herald investigate suspicious.example --no-visual

# Permit private/internal IP resolution (metadata endpoints remain blocked)
herald investigate internal.test --allow-private
```

Output includes trace ID, verdict, phishing score, evidence path, risk factor explanations, DNS/TLS intelligence, and pipeline stage lifecycle.

**Verdict thresholds:**

| Verdict | Score |
|---|---|
| `Phishing` | ≥ 0.70 |
| `Suspected` | ≥ 0.35 |
| `Likely Clean` | < 0.35 |

### `herald analyze`

Runs the investigation pipeline without Playwright screenshot or OCR. Faster and suitable for bulk analysis.

```bash
herald analyze <domain> [--json] [--allow-private]
```

### `herald screenshot`

Runs the investigation with visual analysis and prints only the visual evidence summary.

```bash
herald screenshot <target> [--json] [--allow-private]
```

Screenshot saved to: `evidence/<trace_id>_<domain>/screenshots/homepage.png`

### `herald report`

Loads a previously persisted investigation by trace ID.

```bash
herald report <trace_id> [--json]
```

Trace IDs follow the format `trc-<10 hex chars>`. Lookup scans `evidence/<trace_id>*/investigation.json`.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Completed successfully |
| `1` | Report not found or no command given |
| `2` | SSRF protection blocked the target |

### Evidence Layout

```
evidence/
  investigations.jsonl                          ← index of all runs
  trc-1a2b3c4d5e_paypal-login-alert.com/
    investigation.json                          ← complete structured result
    report.md                                   ← human-readable Markdown report
    screenshots/
      homepage.png                              ← full-page screenshot
```

Top-level JSON fields: `trace_id`, `input`, `url`, `domain`, `started_at`, `completed_at`, `elapsed_ms`, `verdict`, `phishing_score`, `evidence_dir`, `lexical`, `dns`, `tls`, `visual`, `summary`, `risk_factors`, `stages`, `errors`.

---

## API Reference

The FastAPI application runs at `http://localhost:8000`. Interactive Swagger docs are available at `/docs`.

> The API is functional but less battle-tested than the CLI. Queue submission endpoints have a known globals issue — see [Known Limitations](#known-limitations).

### Authentication

```bash
# Register a local user
POST /api/auth/register
{"username": "analyst", "password": "secret"}

# Obtain a bearer token (OAuth2 password form)
POST /api/auth/token
# Form fields: username, password
# Returns: {"access_token": "...", "token_type": "bearer"}
```

### Public Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/ready` | Database, Redis, and telemetry readiness |
| `GET` | `/metrics` | In-process Prometheus-style metrics |
| `GET` | `/api/metrics-summary` | Queue depths, worker state, circuit breaker status |
| `WS` | `/ws/telemetry` | Redis pub/sub → WebSocket bridge |

### Queue Submission (Authenticated)

```bash
# Enqueue a domain for background analysis
POST /api/scan
Authorization: Bearer <token>
{"domain": "sbi-login-secure.xyz", "target_cse": "Unknown"}

# Enqueue a URL — normalizes to domain, returns job and trace IDs
POST /api/investigate
Authorization: Bearer <token>
{"url": "https://sbi-login-secure.xyz/login"}
```

### Data Retrieval (Authenticated)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/suspected` | List `DomainScan` rows with `Suspected` verdict |
| `GET` | `/api/detections` | 50 most recent `DomainScan` rows |
| `GET` | `/api/export/{domain}/json` | Full JSON export for a domain |
| `GET` | `/api/export/{domain}/pdf` | PDF evidence report for a domain |

### Analyst Tools (Authenticated)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/feedback` | Submit analyst verdict override |
| `GET` | `/api/whitelist` | List whitelisted domains |
| `POST` | `/api/whitelist` | Add a domain to the whitelist |
| `DELETE` | `/api/whitelist/{domain}` | Remove a domain from the whitelist |
| `GET` | `/api/admin/failed-jobs` | View dead-letter queue entries |
| `POST` | `/api/admin/failed-jobs/retry` | Drain DLQ back to the ready queue |

---

## Environment Variables

Copy `.env.example` to `.env` and configure before running.

### Database and Cache

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///domain_history.db` | SQLAlchemy database URL |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

### API Authentication

| Variable | Description |
|---|---|
| `JWT_SECRET_KEY` | Secret key for JWT signing — **must be changed in production** |
| `JWT_ALGORITHM` | Algorithm for JWT (e.g. `HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime in minutes |

### Queue Tuning

| Variable | Description |
|---|---|
| `DOMAIN_QUEUE_MAX_READY` | Queue-pressure threshold before API backpressure |
| `VISUAL_QUEUE_MAX_READY` | Domain worker threshold for enqueuing visual jobs |
| `VISUAL_ANALYSIS_TIMEOUT_SECONDS` | Visual worker child-process timeout |
| `VISUAL_CIRCUIT_FAILURE_THRESHOLD` | Failures before circuit opens |
| `VISUAL_CIRCUIT_RESET_SECONDS` | Seconds before circuit half-opens |

### Browser

| Variable | Description |
|---|---|
| `PLAYWRIGHT_PAGE_LOAD_TIMEOUT` | Page navigation timeout in milliseconds |
| `EVIDENCE_DIR` | Default output directory for visual analysis |

### Frontend

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_TELEMETRY_MODE` | `MOCK` | `MOCK`, `REAL`, or `HYBRID` |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/telemetry` | WebSocket backend URL |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | REST backend URL |

---

## Deployment

### Recommended: CLI-only (no infrastructure dependencies)

```bash
pip install -r requirements-runtime.txt
pip install -e .
python -m playwright install chromium
herald investigate example.com
```

Evidence writes to `evidence/` locally. No Redis or database required.

### API + Worker Stack

Requires Redis. SQLite is the default; set `DATABASE_URL` for PostgreSQL.

```bash
# API server
uvicorn herald.api.main:app --host 0.0.0.0 --port 8000

# Domain analysis worker
python -m herald.monitoring.queue_worker

# Visual analysis worker (isolated subprocess for browser/OCR timeouts)
python -m herald.monitoring.visual_worker
```

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU | 8 cores | 16+ cores |
| RAM | 8 GB | 32 GB |
| Storage | 50 GB | 200 GB |

For large-scale monitoring of 50+ CSEs with real-time CT log processing, 48+ cores and 256 GB RAM support parallel scanning of thousands of domains per hour.

---

## ML Model Evolution

HERALD has two independent detection paths:

**CLI path** (`herald/investigation/scoring.py`): Rule-based heuristic scoring — fast, fully explainable, no model file required.

**Worker path** (`herald/predict_with_fallback.py`): `PhishingPredictorV3` loads `models/ensemble_v7.joblib` — a Random Forest (40%) + XGBoost (60%) ensemble with content-feature adjustment for borderline scores.

| Version | Precision | Recall | Key Improvements |
|---|---|---|---|
| v3 | 0.877 | 0.546 | Baseline — lexical features only |
| v4 | 0.455 | 0.957 | Biased — missing legitimate class |
| v5 | 0.941 | 0.814 | Added legitimate class via Tranco list |
| v6 | 0.950 | 0.824 | WHOIS + SSL/DNS + Indian domain corpus |
| **v7** | **0.981** | **0.841** | Two-stage inference + content features |
| v8 (exp.) | 0.969 | 0.847 | Char-Transformer ensemble (experimental) |

**Research finding:** Domain-name-only detection hits a hard F1 ceiling around 0.91 regardless of architecture. Crossing it requires live page content analysis, as implemented in v7 Stage 2.

The `models/` directory contains artifacts from v2 through v9. The production worker defaults to `ensemble_v7.joblib`; all others are historical or experimental.

---

## Configuration

```yaml
# config.yaml
monitoring:
  suspected_duration_days: 90     # Re-monitor parked domains for this long
  check_interval_hours: 24        # How often to re-scan suspected domains

classification:
  phishing_threshold: 0.571       # Tuned for precision/recall balance
  suspected_threshold: 0.35       # Below this = likely legitimate

crawler:
  max_threads: 50
  screenshot_timeout: 30

whitelist:
  domains:
    - accounts.mgovcloud.in       # Known-legitimate domains to suppress false positives
```

### Adding a CSE Watchlist

Edit `herald/features/lexical_features.py`:

```python
CSE_KEYWORDS = [
    "sbi", "hdfc", "icici", "pnb", "uidai", "irctc",
    # Add your brands here
    "yourbank", "yourbrand",
]
```

Then retrain the model:

```bash
python research/scripts/retrain_v3.py --training_data research/datasets/
```

### Adding Telegram Channels to Monitor

```yaml
# config.yaml
social:
  telegram_channels:
    - your_channel_name    # public channel username — no @ prefix
  scrape_interval_minutes: 30
  max_posts_per_scrape: 50
```

---

## Contributing

Contributions are welcome. Please open an issue before starting any large change to discuss scope and approach.

Areas where help is most valuable:

- **CSE keyword lists** for countries and sectors beyond India
- **New data source integrations** — additional CT log providers, passive DNS feeds
- **Fix Docker deployment** — update `docker/Dockerfile` to reference `requirements-runtime.txt`
- **Fix API queue globals** — replace unqualified `domain_queue` with explicit `get_domain_queue()` calls in `herald/api/main.py`
- **Integration tests** — `herald investigate --json` with mocked DNS/TLS/Playwright
- **Frontend wiring** — connect DLQ page, trace page, and health/readiness routes to real backend endpoints
- **Model documentation** — model cards for `ensemble_v7.joblib` covering feature list, thresholds, training data lineage, and validation metrics

### Development Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python -m playwright install chromium

# Run the focused CLI test suite
python -m pytest tests/test_investigation_cli.py -q

# Verify compile-time correctness
python -m compileall herald -q
```

---

## Known Limitations

The following issues are tracked and not yet resolved:

| Issue | Location | Impact |
|---|---|---|
| `requirements.txt` missing | `docker/Dockerfile` | Docker builds fail without manual fix |
| Unqualified queue globals | `herald/api/main.py` | `/api/scan` and `/api/investigate` likely raise `NameError` |
| Stale Streamlit path | `docker/docker-compose.yml` | References `dashboard/dashboard.py` (moved to `legacy/`) |
| Frontend defaults to mock | `NEXT_PUBLIC_TELEMETRY_MODE` | Dashboard shows synthetic data unless set to `REAL` |
| Split detection engines | `scoring.py` vs `predict_with_fallback.py` | CLI and worker verdicts use different logic and thresholds |
| Browser SSRF gaps | `playwright_analyzer.py` | Subresource loads and post-navigation redirects are not re-validated |
| Open user registration | `/api/auth/register` | Must be restricted for any externally accessible deployment |
| Hard-coded JWT default | `JWT_SECRET_KEY` | Must be overridden in production via environment variable |
| Permissive CORS | `herald/api/main.py` | Allows all origins — restrict before production deployment |

---

## External Network Dependencies

HERALD makes outbound calls to the following public infrastructure only:

- `python-whois` — WHOIS lookups via public WHOIS servers
- `playwright` — Headless Chromium browsing of target domains
- `certstream` — WebSocket to `wss://certstream.calidog.io` for Certificate Transparency
- `crt.sh` — Fallback HTTP polling for CT data
- Public DNS resolution via Python `socket` / `aiodns`
- `requests` + `BeautifulSoup` — Telegram public channel scraping (`t.me/s/channel`)

No commercial threat intelligence APIs. No VirusTotal, Shodan, or external detection services.

---

## Roadmap

- [ ] React dashboard replacing Streamlit for production deployments *(in progress)*
- [ ] STIX/TAXII export for sharing indicators with other platforms
- [ ] Webhook alerts — Slack, email, PagerDuty
- [ ] Multi-tenant support for monitoring multiple organizations
- [ ] BERT-based domain name similarity model

---

## Contact

<div align="center">

Built by Athiyo — IIIT Delhi
Mail: [athiyo22118@iiitd.ac.in]

---

*0.981 precision on live PhishTank data · Zero third-party APIs · Fully on-premises*