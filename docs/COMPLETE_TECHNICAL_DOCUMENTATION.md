# Herald — Complete Technical Documentation

## 1. Executive Summary

HERALD, short for **Heuristic & Ensemble Risk Assessment for Lookalike Domains**, is an AI-powered phishing-domain intelligence and investigation platform. It is built for organizations that need to discover, score, investigate, and preserve evidence for suspicious domains without sending sensitive telemetry to commercial threat-intelligence vendors. In the current repository, Herald is not a generic SIEM replacement. Its strongest implemented capability is **phishing and lookalike-domain detection**, especially for domains that impersonate Indian banks, government portals, telecom providers, public utilities, and other critical service entities.

Herald exists because phishing investigations usually fail in one of two ways. First, high-volume feeds such as Certificate Transparency streams, newly registered domain lists, and manually reported URLs produce far more candidates than an analyst can inspect. Second, many machine-learning classifiers return only a score, leaving the analyst with no evidence package: no screenshot, no OCR text, no DNS context, no TLS observations, and no explanation of why a domain was flagged. Herald addresses both problems by pairing fast ML triage with a slower evidence-first investigation path.

The current codebase has four active surfaces:

| Surface | Primary files | What it does |
|---|---|---|
| CLI investigation | `herald/cli.py`, `herald/investigation/pipeline.py` | Runs an end-to-end investigation for one URL or domain and writes evidence artifacts. |
| FastAPI service | `herald/api/main.py` | Accepts authenticated scan/investigation requests, exposes detections, exports reports, and bridges telemetry to WebSocket clients. |
| Worker pipeline | `herald/monitoring/queue_worker.py`, `herald/monitoring/visual_worker.py` | Pulls Redis jobs, runs ML/domain analysis, optionally performs Playwright/OCR visual analysis, and updates the database. |
| Next.js console | `frontend/app`, `frontend/components`, `frontend/services` | Shows operational telemetry, queue status, trace timelines, domain detail pages, and demo scenarios. |

The core thesis is simple: Herald reduces **MTTD** (**Mean Time To Detect**) by catching suspicious domains soon after registration or submission, then reduces analyst burden by producing explainable artifacts instead of an opaque binary label. A high-confidence lexical or ensemble verdict can be emitted quickly. Borderline cases can be escalated into live content analysis and visual inspection. The analyst receives a risk score, a verdict, risk factors, screenshots when available, OCR findings, DNS/WHOIS/TLS intelligence, and exported JSON/PDF reports.

The main users are:

| User | What they need from Herald |
|---|---|
| SOC analyst | A queue of prioritized suspicious domains, evidence, screenshots, OCR text, and enough explanation to decide whether to escalate or close. |
| Threat hunter | Discovery signals from CT logs, NRD feeds, suspicious TLDs, lookalike spelling, brand keywords, and visual indicators. |
| CISO | A self-hosted platform with explainable detections, local data custody, measurable precision/recall, auditability, and clear operational controls. |
| Data scientist | Training datasets, feature extraction scripts, model lineage, evaluation reports, false-positive/false-negative analysis, and reproducible retraining paths. |
| DevOps engineer | Docker Compose, health endpoints, readiness checks, Redis queue metrics, worker liveness, Prometheus metrics, and deployment/runbook guidance. |

If you read nothing else, read this: **Herald is a self-hosted phishing-domain investigation platform that turns suspicious domains into defensible evidence.** It uses lexical, network, content, ML, and visual/OCR signals to score domains; Redis-backed workers to process them reliably; SQLAlchemy to persist detection state; and a Next.js dashboard to visualize operational telemetry. Its production-shaped path is still stabilizing, but the CLI investigation workflow is already the clearest end-to-end evidence generator.

Example analyst flow:

```bash
herald investigate https://hdfc-netbanking-verify.top --visual
```

The expected result is not merely `"phishing": true`. The pipeline normalizes the target, validates it for SSRF safety, extracts lexical signals, collects DNS/WHOIS/TLS intelligence where possible, captures a screenshot and OCR text when visual analysis is enabled, fuses the scores, and writes JSON/Markdown evidence under `evidence/`.

> **Key Takeaways**
> - Herald is implemented as a phishing/lookalike-domain intelligence platform, not a broad all-telemetry SOC product.
> - Its value is fast triage plus analyst-ready evidence.
> - The CLI path is the most direct investigation route; the API, Redis workers, and dashboard form the platform mode.

## 2. System Architecture

Herald is organized around a pipeline that gets progressively more expensive as confidence becomes less obvious. Cheap lexical and ML features run first. Live network, content, screenshot, and OCR analysis are reserved for suspicious or borderline domains. This structure protects throughput while still giving analysts rich evidence when it matters.

```text
                         +-----------------------------+
                         | Analysts / Feeds / API      |
                         | CLI, CT monitor, NRD feed   |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         | Target normalization        |
                         | URL -> domain, lowercase    |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         | SSRF safety validation      |
                         | private IP and unsafe URL   |
                         | rejection unless allowed    |
                         +--------------+--------------+
                                        |
                +-----------------------+-----------------------+
                |                                               |
                v                                               v
   +-----------------------------+                 +-----------------------------+
   | CLI InvestigationPipeline   |                 | FastAPI / Redis platform    |
   | synchronous evidence path   |                 | async worker path           |
   +--------------+--------------+                 +--------------+--------------+
                  |                                               |
                  v                                               v
   +-----------------------------+                 +-----------------------------+
   | Lexical scoring             |                 | domain_analysis_queue       |
   | DNS / WHOIS / TLS           |                 | RedisReliableQueue          |
   | Playwright / OCR            |                 +--------------+--------------+
   +--------------+--------------+                                |
                  |                                               v
                  |                                +-----------------------------+
                  |                                | Domain worker               |
                  |                                | PhishingPredictorV3         |
                  |                                +--------------+--------------+
                  |                                               |
                  |                                  borderline?  |
                  |                                               v
                  |                                +-----------------------------+
                  |                                | visual_analysis_queue       |
                  |                                +--------------+--------------+
                  |                                               |
                  |                                               v
                  |                                +-----------------------------+
                  |                                | Visual worker               |
                  |                                | Playwright + OCR subprocess |
                  |                                +--------------+--------------+
                  |                                               |
                  v                                               v
   +-----------------------------+                 +-----------------------------+
   | evidence/ artifacts         |                 | SQLAlchemy database         |
   | JSON, Markdown, screenshots |                 | DomainScan, User, etc.     |
   +--------------+--------------+                 +--------------+--------------+
                  |                                               |
                  +-----------------------+-----------------------+
                                          v
                         +-----------------------------+
                         | Reports and dashboard       |
                         | JSON/PDF export, Next.js UI |
                         +-----------------------------+
```

### Layers

The **ingestion layer** accepts candidate domains and URLs. In the repository this includes API submission (`POST /api/scan`, `POST /api/investigate`), legacy/monitoring feed modules under `herald/ingestion/`, and CLI invocation. The ingestion responsibility is not to decide whether a domain is malicious; it captures the target and source context, normalizes the input, and hands work to the next stage.

The **safety layer** lives mainly in `herald/core/security.py`. It protects the platform from becoming a blind SSRF browser or scanner. Before a worker fetches or screenshots a user-submitted URL, the platform validates schemes, hostnames, and private-address behavior. The CLI allows an explicit `allow_private` override because local testing may require private targets.

The **feature layer** lives under `herald/features/`. The most important implemented extractor is `lexical_features.py`, which turns a domain into numeric signals: length, digit ratio, hyphen ratio, CSE brand keyword position, malicious TLD indicator, Levenshtein distance to known brands, entropy, punycode flag, suspicious keyword flags, and character trigram statistics. Other feature modules (`dns_features.py`, `whois_features.py`, `ssl_features.py`, `content_features.py`) separate live intelligence signals by domain.

The **ML layer** is centered on `herald/predict_with_fallback.py`. `PhishingPredictorV3` loads a joblib ensemble, aligns extracted features with the model's expected feature order, combines Random Forest and XGBoost probabilities, applies a threshold, and triggers content or visual fallback logic for borderline confidence. Model artifacts live in `models/`; research scripts and processed datasets live under `research/`.

The **detection layer** lives under `herald/detection/` and provides a unified interface for all scoring paths. It centralizes `DetectionEngine`, which lazily loads underlying models (`MLScorer` wraps `PhishingPredictorV3`, `HeuristicScorer` wraps lexical heuristics). The CLI, Redis background workers, and Telegram/social ingestion scripts all share this same scoring interface. This ensures all domains are evaluated against a consistent, strongly-typed `Verdict` contract regardless of how they enter the platform.

The **investigation layer** lives under `herald/investigation/`. `InvestigationPipeline` is a synchronous pipeline that produces a complete `InvestigationResult`, records stage timing and degraded failures, and writes evidence artifacts. Its scoring path uses `investigation/scoring.py`, DNS/TLS collection from `investigation/intelligence.py`, normalized targets from `investigation/targets.py`, and persistence helpers from `investigation/persistence.py`.

The **queue and monitoring layer** lives under `herald/monitoring/`. `RedisReliableQueue` provides ready, processing, delayed, and dead-letter queues. `queue_worker.py` processes domain jobs and persists first-stage verdicts. `visual_worker.py` handles browser/OCR jobs in child processes with timeouts and circuit breaker protection. `metrics.py` renders Prometheus-style counters, gauges, histograms, and timers.

The **telemetry layer** lives under `herald/telemetry/`. It wraps operational events in a schema-stable `EventEnvelope`, publishes them to Redis pub/sub, and allows FastAPI's `/ws/telemetry` endpoint to broadcast events to frontend clients.

The **frontend layer** is a Next.js 16, React 19, TypeScript application. It uses typed telemetry models, mock scenarios, charts, system-health panels, domain-detail pages, trace views, and dead-letter queue pages. The frontend can operate in mock mode and has real/hybrid hooks for backend health and metrics.

### State

| State | Location | Contents |
|---|---|---|
| Submitted jobs | Redis ready queues | Domain jobs and visual-analysis jobs with `job_id`, `trace_id`, `attempts`, and source metadata. |
| Leased jobs | Redis processing lists | Jobs currently held by a worker with `lease_expires_at`. |
| Delayed retries | Redis sorted sets | Failed jobs waiting for exponential backoff retry. |
| Dead-letter jobs | Redis DLQ lists | Jobs that exhausted retries or hit immediate DLQ conditions such as schema mismatch. |
| Detection records | SQLAlchemy database | `DomainScan` rows with domain, target CSE, label, confidence, lifecycle, screenshot path, OCR text, DNS JSON. |
| Users and access | SQLAlchemy database | `User` rows with hashed password, role, and activity state. |
| Evidence artifacts | `evidence/` | JSON reports, Markdown reports, screenshots, and visual artifacts. |
| Model artifacts | `models/` | Joblib ensembles, vocabularies, augmented training workbooks. |
| Research datasets | `research/datasets` | Raw PS-02 workbooks, external feeds, processed CSVs, feature matrices. |

### Communication

The CLI path is direct function invocation:

```text
herald.cli -> InvestigationPipeline.investigate() -> evidence files
```

The platform path is queue based:

```text
FastAPI -> RedisReliableQueue -> domain worker -> database -> optional visual queue -> visual worker -> database
```

Telemetry is pub/sub based:

```text
worker -> TelemetryEmitter -> Redis channel herald.telemetry -> FastAPI WebSocket -> frontend
```

The API persists state through SQLAlchemy and exposes health, readiness, metrics, recent detections, suspected domains, whitelist controls, failed-job controls, and export endpoints.

### Technology Choices

| Technology | Why it fits Herald |
|---|---|
| Python | Mature security tooling, ML libraries, DNS/TLS/WHOIS libraries, FastAPI, Playwright integration, joblib model loading. |
| FastAPI | Typed request models, dependency injection, OAuth2 helpers, OpenAPI docs, async WebSocket support. |
| SQLAlchemy | Portable SQLite/PostgreSQL database layer with explicit domain models. |
| Redis | Lightweight queueing, pub/sub telemetry, worker liveness keys, circuit-breaker counters, and retry state. |
| Next.js / React / TypeScript | Strong dashboard experience, typed telemetry contracts, app router pages, charts, and mock/real data modes. |
| Playwright | Browser automation for screenshots, DOM inspection, OCR preparation, and phishing-page visual evidence. |
| scikit-learn / XGBoost / joblib | Practical tabular-domain feature modeling with serializable ensembles. |
| Docker Compose | Local orchestration for API, Redis, Postgres, workers, and frontend dependencies. |

Concrete example: A submitted `hdfc-netbanking-verify.top` job first enters `domain_analysis_queue`. The domain worker validates the URL, extracts features, loads `models/ensemble_v7.joblib`, computes an ensemble confidence, marks it `Phishing` if above threshold, or marks it `Suspected` and enqueues visual analysis if the score is borderline. The visual worker runs Playwright in a subprocess, captures OCR findings, boosts confidence if credential prompts are present, and updates `domain_scans`.

> **Key Takeaways**
> - Herald uses direct CLI investigation and queued platform investigation.
> - Cheap scoring runs first; expensive browser/OCR work runs only when justified.
> - State is split across Redis queues, SQLAlchemy records, filesystem evidence, and model artifacts.

## 3. Module-by-Module Breakdown

### `herald/api`

**Purpose:** The API module exposes Herald as an HTTP service. It accepts scan requests, manages authentication, serves operational status, exposes detection history, handles whitelist and DLQ operations, exports reports, and streams telemetry to dashboard clients.

**Responsibilities:**

- Create the FastAPI application.
- Initialize and validate database tables on startup.
- Connect to Redis and instantiate reliable queues.
- Enforce rate limits through SlowAPI.
- Authenticate users with OAuth2 bearer tokens and JWT.
- Push scan and investigation jobs into Redis.
- Return detection and suspected-domain records.
- Manage whitelist entries.
- Export JSON and PDF investigation reports.
- Expose `/metrics`, `/api/health`, `/api/ready`, and `/api/metrics-summary`.
- Bridge Redis pub/sub telemetry to `/ws/telemetry`.

**Key concepts:** A **scan** is a domain-only submission. An **investigation** is a URL submission that is normalized into a domain and sent to the same first-stage queue. A **trace ID** ties API, queue, worker, and telemetry events together. A **lifecycle state** tracks whether a domain is queued, processing, degraded, failed, or verdict-ready.

**How it works:**

1. FastAPI starts and calls `init_db()`.
2. The startup handler validates expected `domain_scans` columns.
3. Redis is probed. If available, API state receives `domain_queue`, `visual_queue`, and `visual_circuit`.
4. Users authenticate through `/api/auth/token`.
5. Authenticated clients call `/api/scan` or `/api/investigate`.
6. The API checks queue pressure and enqueues a job with domain, source, target CSE, trace ID, and lifecycle state.
7. Workers process jobs asynchronously.
8. Clients poll detections/export endpoints or subscribe to `/ws/telemetry`.

**Code walkthrough:**

```python
async def startup_event() -> None
```

Parameters: none. It uses environment variables such as `REDIS_HOST`, database configuration from `herald.db.models`, and app state. Return: none. Example input is application startup; output is initialized database tables and optional Redis queue handles. Edge case: if Redis is unavailable, the API still starts but queue-dependent endpoints return `500`.

```python
def trigger_scan(request: Request, scan_req: ScanRequest, current_user: User, domain_queue) -> dict
```

Parameters: `request` supplies headers and client context; `scan_req.domain` is the candidate domain; `scan_req.target_cse` is the protected brand/service; `current_user` is the JWT-authenticated user; `domain_queue` is injected from app state. Return: a JSON object with status and job ID. Example:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "target_cse": "HDFC"
}
```

Expected response:

```json
{
  "status": "ok",
  "job_id": "9b9f9cf4-3b6a-4fb0-bc7d-0dc3e704b54a",
  "message": "Domain hdfc-netbanking-verify.top queued for analysis"
}
```

Edge case: if `DOMAIN_QUEUE_MAX_READY` is exceeded, the endpoint returns `429` to avoid unbounded backlog.

```python
def investigate_url(request: Request, inv_req: InvestigateRequest, current_user: User, domain_queue) -> dict
```

Parameters: `inv_req.url` can be a full URL or bare domain. Return: job metadata including normalized domain and trace ID. Example input `https://secure-sbi-login.xyz/path` becomes domain `secure-sbi-login.xyz`. Edge case: the normalization is simple string splitting; unusual URLs should be validated again by workers through SSRF checks.

**Inputs and outputs:**

| Endpoint | Input | Output |
|---|---|---|
| `POST /api/auth/register` | username, password, role | user record, if registration enabled |
| `POST /api/auth/token` | OAuth2 form username/password | JWT bearer token |
| `POST /api/scan` | domain, target CSE | Redis job ID |
| `POST /api/investigate` | URL | Redis job ID, trace ID, normalized domain |
| `GET /api/detections` | bearer token | recent `DomainScan` records |
| `GET /api/export/{domain}/json` | domain | persisted scan report |
| `GET /metrics` | none | Prometheus text |

**Integration points:** `api` talks to `db.models`, `core.auth`, `monitoring.redis_queue`, `monitoring.resilience`, `monitoring.metrics`, and `utils.export`. It receives telemetry from Redis through `redis.asyncio` and forwards it to frontend WebSocket clients.

**Common failure modes:** Redis offline causes scan endpoints to fail. Database schema mismatch causes startup exit. Missing JWT secret hardening can weaken auth. Queue pressure returns `429`. WebSocket clients may disconnect silently; broadcast catches exceptions and drops failed sends.

### `herald/core`

**Purpose:** The core module contains security, authentication, browser analysis, and domain-specific low-level logic used by several higher layers.

**Responsibilities:**

- Hash and verify user passwords.
- Create JWT access tokens.
- Validate URLs for SSRF safety.
- Run Playwright-based page analysis.
- Detect/generate homoglyph-like domain variants.

**Key concepts:** **SSRF** (**Server-Side Request Forgery**) is a risk where an attacker makes the platform fetch internal resources. Because Herald accepts URLs and may launch a browser, URL validation is a first-class security control. **Homoglyphs** are visually similar characters from different scripts, such as Cyrillic characters that resemble Latin letters.

**How it works:** Authentication helpers are used by FastAPI dependencies. `validate_url_safe()` is called by the CLI pipeline and queue worker before fetching or analyzing targets. `PlaywrightVisualAnalyzer` is loaded lazily because browser automation is slow and dependency-heavy.

**Code walkthrough:**

```python
def validate_url_safe(url: str, *, allow_private: bool = False, trace_id: str | None = None, private_overrides: list | None = None) -> None
```

Parameters: `url` is the target; `allow_private` permits local/private IPs for testing; `trace_id` supports logging; `private_overrides` captures exceptions when private access is allowed. Return: none; unsafe URLs raise an exception. Example:

```python
validate_url_safe("https://hdfc-netbanking-verify.top")
```

Edge case: a public hostname can resolve to a private IP after DNS lookup, so validation must consider resolution, not only string format.

```python
async def PlaywrightVisualAnalyzer.run_analysis(domain_or_url: str) -> dict
```

Parameters: a URL or domain. Return: dictionary containing `success`, `screenshot_path`, `ocr_text`, and `ocr_findings`. Example output:

```json
{
  "success": true,
  "screenshot_path": "evidence/trc-abc/example/screenshot.png",
  "ocr_text": "Login Verify Account Password",
  "ocr_findings": {
    "is_suspicious": true,
    "ocr_risk_score": 0.82
  }
}
```

Edge case: browser launch can hang or time out, which is why the worker runs visual analysis in a child process.

**Integration points:** `core.security` is called by `investigation.pipeline` and `monitoring.queue_worker`. `core.auth` is called by `api.main`. `core.playwright_analyzer` is called by the CLI pipeline, predictor fallback, and visual worker.

**Common failure modes:** Missing Playwright browsers, blocked DNS, non-HTTP schemes, internal IP submissions, screenshot timeouts, OCR dependency failures, and JWT secret misconfiguration.

### `herald/db`

**Purpose:** The database module defines persistent entities and database connection behavior.

**Responsibilities:**

- Define SQLAlchemy declarative models.
- Configure SQLite by default and PostgreSQL when `DATABASE_URL` is provided.
- Create engine/session factories.
- Initialize tables.

**Key concepts:** `DomainScan` is the main detection record. It does not store the full evidence graph; it stores enough state for dashboards, exports, and operational views. Detailed evidence lives in files under `evidence/`.

**How it works:** `DATABASE_URL` defaults to `sqlite:///domain_history.db`. PostgreSQL-specific pool settings are added when the URL starts with `postgresql`. SQLite receives a busy timeout for local concurrency tolerance. `SessionLocal` is used by API and workers.

**Code walkthrough:**

```python
def init_db() -> None
```

Parameters: none. Return: none. It calls `Base.metadata.create_all(bind=engine)`. Example:

```bash
python setup_db.py
```

Edge case: `create_all` does not perform full migrations for changed columns, so the API startup explicitly checks for required columns and can ask you to run setup/migration logic.

Model example:

```python
class DomainScan(Base):
    __tablename__ = "domain_scans"
```

Important fields:

| Field | Type | Meaning |
|---|---|---|
| `domain` | string | Candidate domain. Indexed. |
| `target_cse` | string | Protected entity or brand, such as HDFC or SBI. |
| `source` | string | `api_manual`, `api_investigate`, feed name, or worker source. |
| `label` | string | `Clean`, `Suspected`, `Phishing`, `Rejected`, etc. |
| `confidence` | float | Model or fused confidence. |
| `lifecycle_state` | string | Queue/investigation state. |
| `screenshot_path` | string | Path to visual evidence. |
| `ocr_text` | string | Extracted page text. |
| `dns_records` | string | JSON stored as text. |

**Example:** When `visual_worker.py` confirms suspicious OCR, it updates `DomainScan.label` to `Phishing`, raises confidence, stores screenshot/OCR fields, sets lifecycle to `VERDICT_READY`, and commits.

**Integration points:** API reads and writes users, whitelist entries, scans, and exports. Queue workers upsert scans. Visual workers update evidence fields. Metrics summary does not directly query the database beyond readiness checks.

**Common failure modes:** Schema mismatch, SQLite write contention, missing PostgreSQL pool configuration, stale records from repeated domain scans, and text-encoded JSON being harder to query than JSONB in PostgreSQL.

### `herald/features`

**Purpose:** The features module converts raw domain, network, content, SSL, WHOIS, and DNS observations into structured features usable by heuristic scoring or ML models.

**Responsibilities:**

- Extract lexical domain features.
- Identify protected CSE brand keywords.
- Flag suspicious TLDs and punycode.
- Compute entropy and character composition.
- Compute fuzzy brand distance with Levenshtein.
- Extract content signals such as forms and obfuscated JavaScript.
- Represent DNS, WHOIS, and SSL signals separately for model and investigation use.

**Key concepts:** A **feature vector** is an ordered numeric representation of a domain. ML models cannot directly reason over strings such as `hdfc-netbanking-verify.top`; the string becomes numeric columns such as `num_hyphens`, `is_malicious_gtld`, `has_verify`, and `min_brand_levenshtein`.

**How it works:** `extract_url_features(df, domain_col="domain")` accepts a pandas DataFrame, parses each URL/domain, and returns the original DataFrame plus feature columns. The predictor then selects `self.feature_names` from the output to align with the trained model.

**Code walkthrough:**

```python
def extract_url_features(df: pandas.DataFrame, domain_col: str = "domain") -> pandas.DataFrame
```

Parameters: `df` is an input DataFrame; `domain_col` names the column containing URLs or domains. Return: DataFrame with appended numeric features. Example input:

```python
import pandas as pd
from herald.features.lexical_features import extract_url_features

df = pd.DataFrame([{"domain": "hdfc-netbanking-verify.top"}])
features = extract_url_features(df)
print(features[["domain_length", "num_hyphens", "is_malicious_gtld", "has_verify"]])
```

Expected shape:

```text
domain_length  num_hyphens  is_malicious_gtld  has_verify
26.0           2.0          1.0                1.0
```

Edge case: malformed URLs or invalid IPv6 bracket syntax are caught in parsing and handled through fallback splitting.

```python
def calculate_entropy(domain: str) -> float
```

Parameters: raw domain string. Return: Shannon entropy. Example: `calculate_entropy("aaaa")` returns lower entropy than `calculate_entropy("a8xqz")`. Edge case: empty string returns `0`.

**Inputs and outputs:** Input is usually a URL/domain string in a DataFrame. Output is numeric columns. Example feature categories include:

| Category | Examples |
|---|---|
| Length and ratios | `domain_length`, `digit_ratio`, `hyphen_ratio`, `special_char_ratio` |
| Brand position | `brand_keyword_position`, `has_cse_keyword_in_subdomain`, `brand_to_reg_length_ratio` |
| Suspicious structure | `is_malicious_gtld`, `is_punycode`, `subdomain_depth`, `has_ip` |
| Text intent | `has_login`, `has_secure`, `has_verify`, `has_banking`, `has_auth` |
| Fuzzy matching | `min_brand_levenshtein` |
| Character statistics | `entropy`, `suspicious_trigram_count`, `unique_trigram_ratio` |

**Integration points:** `predict_with_fallback.py` imports `extract_url_features`. `investigation.scoring` uses lexical concepts for heuristic scoring. Research scripts import feature extractors to build feature matrices.

**Common failure modes:** Training and inference feature-order mismatch, missing dependency `python-Levenshtein`, overfitting to visible brand keywords, and treating public-suffix parsing too simplistically for multi-label suffixes such as `co.in`.

### `herald/ingestion`

**Purpose:** The ingestion module contains monitors that discover candidate domains from external or live sources.

**Responsibilities:**

- Monitor Certificate Transparency streams.
- Track newly registered/newly observed domains.
- Monitor social or tunnel-related sources.
- Normalize discovered candidates into jobs for downstream analysis.

**Key concepts:** **Certificate Transparency** logs reveal newly issued TLS certificates. Phishing domains often request certificates soon after registration, so CT monitoring can surface threats before victim reports arrive. **Tunneling domains** such as public tunnel services can host temporary phishing pages and should be treated differently from ordinary registered domains.

**How it works:** The repository contains ingestion modules such as `certstream_monitor.py`, `new_domains_monitor.py`, `social_monitor.py`, and `tunnel_monitor.py`. These are feed-specific adapters. Their operational contract should be: discover a candidate, attach source metadata, optionally attach target CSE hints, and enqueue or hand off to the domain analysis queue.

> ⚠️ Inferred from structure — verify against actual source.
> The exact runtime wiring for every ingestion monitor may be incomplete or legacy. The README identifies CT and NRD sources as part of the design, while the API/worker queue path is the more explicit current platform integration.

**Code walkthrough:**

```python
def monitor_certstream(redis_queue: RedisReliableQueue, target_keywords: list[str]) -> None
```

Parameters: a queue for discovered jobs and keywords such as `sbi`, `hdfc`, `uidai`. Return: none; it continuously emits jobs. Example job:

```json
{
  "domain": "secure-hdfc-kyc.top",
  "source": "certstream",
  "target_cse": "HDFC"
}
```

Edge case: CT logs can produce wildcard certificates, duplicate domains, and benign domains containing brand-like substrings.

**Inputs and outputs:** Inputs are external streams and feed files. Outputs are normalized domain job dictionaries.

**Integration points:** Ingestion should talk to `monitoring.redis_queue`, `features.lexical_features.CSE_KEYWORDS`, and potentially `core.security` if it performs any fetches.

**Common failure modes:** Feed disconnection, high duplicate rates, overmatching short keywords such as `bob`, API/feed rate limiting, and queue pressure during bursts.

### `herald/investigation`

**Purpose:** The investigation module is the evidence-first pipeline. It takes a URL/domain and produces a complete `InvestigationResult` with stage-level timings, risk factors, summaries, artifacts, and degraded-stage errors.

**Responsibilities:**

- Normalize input targets into `(url, domain)`.
- Create evidence directories.
- Run SSRF validation.
- Run lexical analysis.
- Collect DNS/WHOIS intelligence.
- Collect TLS intelligence.
- Optionally run screenshot/OCR analysis.
- Combine scores into a verdict.
- Save JSON/Markdown evidence.

**Key concepts:** A **stage** is one named unit of investigation such as "Lexical analysis" or "TLS intelligence". A **degraded stage** records an error without aborting the whole investigation. A **risk factor** is an explainable reason contributing to the final phishing score.

**How it works:** `InvestigationPipeline.investigate()` runs sequentially. SSRF validation is mandatory and aborts on failure. DNS, TLS, and visual stages are degraded, so a WHOIS outage or screenshot timeout does not destroy the investigation. The final verdict is produced by `combine_scores()`, and reports are written through `save_investigation()`.

**Code walkthrough:**

```python
class InvestigationPipeline:
    def __init__(self, evidence_root: str = "evidence") -> None
```

Parameters: `evidence_root` controls where artifacts are written. Return: an initialized pipeline. Example:

```python
pipeline = InvestigationPipeline(evidence_root="evidence")
```

Edge case: if the evidence path is not writable, the pipeline will fail before analysis.

```python
def investigate(self, target: str, *, include_visual: bool = True, allow_private: bool = False) -> InvestigationResult
```

Parameters: `target` is URL/domain; `include_visual` controls Playwright/OCR; `allow_private` permits private network targets. Return: `InvestigationResult`. Example:

```python
result = pipeline.investigate("https://hdfc-netbanking-verify.top", include_visual=True)
print(result.verdict, result.phishing_score)
```

Example output:

```text
phishing 0.91
```

Edge case: visual analysis can fail while the final investigation still completes with a degraded visual stage.

```python
@contextmanager
def _degraded_stage(self, stages: list[StageResult], errors: list[str], name: str) -> Iterator[dict[str, Any]]
```

Parameters: stage collection, error collection, and stage name. Return: a context dictionary that receives `details`. Example use:

```python
with self._degraded_stage(stages, errors, "TLS intelligence") as stage:
    tls = collect_tls_intelligence(domain)
    stage["details"] = {"has_tls": tls.get("has_tls")}
```

Edge case: any exception is captured as a degraded stage and appended to `errors`.

**Inputs and outputs:** Input is a target string. Output is `InvestigationResult.to_dict()`:

```json
{
  "trace_id": "trc-abc123def0",
  "domain": "hdfc-netbanking-verify.top",
  "verdict": "phishing",
  "phishing_score": 0.91,
  "risk_factors": [
    {"name": "Suspicious TLD", "weight": 0.15},
    {"name": "Brand keyword with auth terms", "weight": 0.25}
  ],
  "stages": [
    {"name": "SSRF validation", "status": "ok", "duration_ms": 12}
  ],
  "errors": []
}
```

**Integration points:** Uses `core.security`, `core.playwright_analyzer`, `investigation.targets`, `investigation.intelligence`, `investigation.scoring`, and `investigation.persistence`.

**Common failure modes:** Unsafe URL rejection, DNS/WHOIS timeout, TLS handshake failure, Playwright browser failure, OCR extraction failure, and filesystem write issues.

### `herald/monitoring`

**Purpose:** The monitoring module contains queue infrastructure, worker loops, metrics, resilience controls, and worker startup orchestration.

**Responsibilities:**

- Provide reliable Redis queue semantics.
- Run the domain worker.
- Run the visual worker.
- Track metrics.
- Track browser pressure and worker liveness.
- Implement circuit breaker behavior for visual analysis.
- Send failed jobs to DLQ and retry eligible jobs with backoff.

**Key concepts:** A **reliable queue** moves a job from ready to processing before execution and only removes it after `ack`. A **lease** makes crashed worker jobs reclaimable. A **DLQ** (**dead-letter queue**) stores jobs that cannot be processed safely. A **circuit breaker** prevents repeated visual-analysis failures from consuming all resources.

**How it works:** `RedisReliableQueue.dequeue()` promotes due delayed jobs, reclaims expired processing leases, and then uses `BRPOPLPUSH` to atomically move work from ready to processing. Workers call `ack()` after success. On failure, workers call `retry_or_dlq()` or `send_to_dlq()`.

**Code walkthrough:**

```python
def enqueue(self, payload: dict[str, Any], *, delay_seconds: int = 0) -> str
```

Parameters: job payload and optional delay. Return: job ID. Example:

```python
job_id = domain_queue.enqueue({"domain": "secure-sbi-login.xyz", "source": "api_manual"})
```

Edge case: delayed jobs are stored in a sorted set and must be promoted by future dequeue calls.

```python
def process_domain(job_data: dict) -> None
```

Parameters: a decoded Redis job with `domain`, `source`, `target_cse`, and trace metadata. Return: none; it persists a `DomainScan`. Example job:

```json
{
  "domain": "secure-sbi-login.xyz",
  "source": "api_manual",
  "target_cse": "SBI",
  "trace_id": "trc-123"
}
```

Edge case: if `validate_url_safe()` rejects the target, the domain is saved as `Rejected` with lifecycle `FAILED`.

```python
def apply_visual_result(job_data: dict) -> None
```

Parameters: visual job with domain, target CSE, and initial confidence. Return: none; it updates the scan row. Edge case: if visual analysis times out, the worker marks the record `DEGRADED` instead of hanging indefinitely.

**Inputs and outputs:** Inputs are Redis job JSON strings. Outputs are database updates, telemetry events, queue acknowledgements, retries, or DLQ entries.

**Integration points:** Workers use `db.models`, `predict_with_fallback`, `core.security`, `telemetry.emitter`, `telemetry.stream`, and `monitoring.metrics`.

**Common failure modes:** Redis outage, stale processing leases, schema mismatch, model file missing, browser timeout, circuit breaker open, duplicate submissions, and SQLite contention.

### `herald/telemetry`

**Purpose:** The telemetry module emits operational events and trace spans in a schema that the frontend can understand.

**Responsibilities:**

- Define telemetry envelope schemas.
- Publish worker events to Redis.
- Represent event priority, severity, degraded state, worker type, and payload.
- Feed `/ws/telemetry` through Redis pub/sub.

**Key concepts:** An **event envelope** is a stable wrapper around all telemetry payloads. The frontend TypeScript `EventEnvelope<T>` mirrors the backend Pydantic `EventEnvelope`, allowing type-safe dashboard development.

**How it works:** Workers initialize `TelemetryEmitter` with a `TelemetryStream`. The emitter publishes events such as `JOB_ACCEPTED`, `BROWSER_TELEMETRY_UPDATED`, `DEGRADED_STATE_ACTIVATED`, verdicts, and trace spans. FastAPI listens to `herald.telemetry` and broadcasts raw JSON to active WebSocket clients.

**Code walkthrough:**

```python
class EventEnvelope(BaseModel):
    event_id: str
    event_type: str
    trace_id: str | None
    timestamp: str
    worker_type: str
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    telemetry_priority: Literal["HIGH", "MEDIUM", "LOW"]
    degraded_state: bool
    source_service: str
    payload: Any
    version: str
```

Parameters are Pydantic fields. Return: serialized JSON. Example:

```json
{
  "event_id": "evt-70e8a3b1f2c4",
  "event_type": "VERDICT_EMITTED",
  "trace_id": "trc-123",
  "timestamp": "2026-06-02T09:00:00",
  "worker_type": "domain_worker",
  "severity": "INFO",
  "telemetry_priority": "HIGH",
  "degraded_state": false,
  "source_service": "herald",
  "payload": {
    "domain": "secure-sbi-login.xyz",
    "verdict": "PHISHING",
    "confidence": 0.91
  },
  "version": "1.0"
}
```

Edge case: frontend and backend schema drift will break real-time rendering. Keep Python and TypeScript envelope fields aligned.

**Integration points:** `monitoring.queue_worker`, `monitoring.visual_worker`, `api.main`, and frontend services/components.

**Common failure modes:** Redis pub/sub offline, WebSocket disconnects, event payload shape mismatch, and noisy low-priority telemetry overwhelming UI state.

### `herald/utils`

**Purpose:** The utils module contains cross-cutting helpers used by investigation, export, logging, CSE mapping, HTML fetching, PDF generation, and legitimate-service detection.

**Responsibilities:**

- Set up structured logging.
- Generate PDF reports.
- Export JSON/PDF artifacts.
- Fetch HTML safely for content analysis.
- Map protected CSE keywords/entities.
- Detect legitimate service providers that may otherwise look suspicious.

**Key concepts:** **CSE** appears in this project as a protected critical/service entity label: banks, government services, telecom brands, and similar targets. **Legitimate service detection** prevents false positives when a domain is hosted on known providers or contains brand-adjacent strings for benign reasons.

**How it works:** API export endpoints call `generate_pdf_report(scan)`. Worker and API modules call `setup_logging()` to configure structured logs. Content and investigation stages can use HTML fetching and service detection helpers.

**Code walkthrough:**

```python
def generate_pdf_report(scan: DomainScan) -> io.BytesIO
```

Parameters: a persisted `DomainScan`. Return: in-memory PDF buffer. Example API output:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/export/hdfc-netbanking-verify.top/pdf \
  --output herald_report.pdf
```

Edge case: missing screenshot paths should produce a report without crashing.

```python
def setup_logging() -> None
```

Parameters: none. Return: none. It configures logs for structured operational debugging. Edge case: double initialization can lead to duplicate handlers if not guarded.

**Integration points:** API exports, queue workers, visual workers, investigation report writing, and research/document generation scripts.

**Common failure modes:** PDF generation failing on missing optional fields, HTML fetcher timeouts, overly broad legitimate-service whitelisting, and inconsistent log context when trace IDs are not bound.

> **Key Takeaways**
> - `api`, `monitoring`, and `investigation` are the main runtime modules.
> - `features` and `predict_with_fallback.py` form the ML inference path.
> - The repository keeps production code, research code, frontend code, and evidence/reporting code deliberately separated.

## 4. ML Pipeline — Complete Technical Deep Dive

### 4a. The Problem Being Solved

Herald's ML pipeline detects and prioritizes suspicious phishing/lookalike domains. The current model is a tabular ensemble operating on lexical, DNS, SSL, WHOIS, and content-derived features. Its output is a probability-like confidence score that is turned into labels such as `Clean`, `Suspected`, or `Phishing`.

The main threat is not malware payload classification or endpoint anomaly detection. The implemented model is aimed at domains that impersonate known entities: for example, `hdfc-netbanking-verify.top`, `secure-sbi-login.xyz`, or `uidai-update-kyc.info`. These domains may use:

- Brand keywords in the registered domain or subdomain.
- Authentication words such as `login`, `verify`, `secure`, `account`, `kyc`, or `update`.
- Suspicious TLDs such as `.top`, `.xyz`, `.click`, or `.tk`.
- Fresh certificates.
- Missing MX/SPF records.
- Login forms, password fields, external form actions, or obfuscated scripts.
- Visual/OCR cues that resemble banking or government login pages.

**PS-02** likely refers to a problem-statement identifier from a cybersecurity challenge or competition dataset. The repository contains files such as `PS02_Training_set.xlsx`, `PS-02_Shortlisting_set`, and `PS02_Mock_Data_24_09_2025.xlsx`. The README and file names indicate an Indian public-sector/CSE phishing-detection setting. In that context, PS-02 can be read as "Problem Statement 02", with shortlisting, mock, and training workbooks used during staged evaluation.

> ⚠️ Inferred from structure — verify against actual source.
> The repository does not define the competition authority or official PS-02 statement in code. The dataset names and CSE-focused keyword list strongly imply a challenge dataset for phishing/lookalike domain detection involving Indian critical service entities.

**Evidences** in the training set likely refer to labelled incident evidence: URLs/domains, target brands, page screenshots, DNS/TLS metadata, WHOIS dates, content snippets, or manual labels explaining why a candidate is phishing or legitimate. In model terms, these evidence records become rows in a labelled dataset. In analyst terms, they are the artifacts used to justify an alert.

### 4b. Data Pipeline: Raw to Processed to Model Input

The dataset lifecycle is visible in the repository:

```text
research/datasets/raw/
  Mock_data/
  PS-02_Shortlisting_set/
  PS02_Training_set/

research/datasets/external/
  phishtank_online.csv
  urlhaus.csv
  tranco_legitimate.csv
  majestic_million.csv
  fresh_phishing_domains.csv

research/datasets/processed/
  full_dataset_v4.csv
  full_dataset_v5.csv
  full_dataset_v7.csv
  full_dataset_v8.csv
  full_dataset_v9_raw.csv
  full_features_v4.csv
  full_features_v5.csv
  full_features_v6.csv
  full_features_v7.csv
  full_features_v8.csv
  full_features_v9.csv
```

A realistic transformation sequence is:

1. Load raw PS-02 workbook rows and external CSV feeds.
2. Normalize URL/domain strings.
3. Deduplicate by normalized registered domain and source.
4. Assign labels such as phishing/legitimate or suspected/clean.
5. Enrich rows with external feed metadata where available.
6. Extract lexical features through `herald.features.lexical_features`.
7. Optionally extract DNS, WHOIS, SSL, and content features.
8. Save the feature matrix as `full_features_vN.csv`.
9. Train model versions through `research/scripts/retrain_vN.py`.
10. Evaluate on holdout, shortlisting, mock, or live validation sets.
11. Save selected artifacts under `models/`.

Research scripts such as `build_dataset_v7.py`, `extract_features_v8.py`, `retrain_v9.py`, `compare_versions_v6.py`, `analyze_errors.py`, `fn_deep_dive.py`, and `validate_phishtank_v6.py` indicate an iterative model-development workflow: build datasets, extract features, retrain, compare versions, investigate false negatives, and validate against fresh phishing feeds.

The difference between `research/` and production `herald/features/` is important:

| Area | Role |
|---|---|
| `research/scripts/` | Batch experiments, dataset construction, evaluation, enrichment, version comparisons. |
| `research/datasets/` | Raw, external, and processed datasets used to train/evaluate. |
| `herald/features/` | Production feature extractors used at inference time. |
| `herald/predict_with_fallback.py` | Production predictor that loads trained artifacts and runs inference. |

They should share feature definitions. If research extraction and production extraction diverge, the model may train on one feature distribution and infer on another.

### 4c. Feature Engineering

Herald uses domain-specific tabular features. Concrete features include:

| Feature | Meaning | Security intuition |
|---|---|---|
| `domain_length` | Number of characters in host/domain | Very long domains often hide intent or mimic login paths. |
| `digit_ratio` | Fraction of digits | Randomized phishing domains often include digits. |
| `num_hyphens` | Count of hyphens | Phishing domains use hyphens to combine brand and action words. |
| `hyphen_ratio` | Hyphens divided by length | Normalizes hyphen count by domain length. |
| `subdomain_count` | Number of subdomain labels | Deep subdomains can place a brand before an unrelated registered domain. |
| `brand_keyword_position` | Brand in registered domain vs subdomain | Brand in subdomain may indicate deceptive hosting. |
| `has_cse_keyword_in_subdomain` | CSE keyword before registered domain | Example: `hdfc.login.attacker.top`. |
| `is_malicious_gtld` | TLD in suspicious list | Cheap/free TLDs are common in phishing. |
| `brand_to_reg_length_ratio` | Brand length vs registered domain length | High ratios can indicate pure impersonation. |
| `min_brand_levenshtein` | Fuzzy distance to known brands | Captures typos like `hdfcbamk`. |
| `has_brand_in_path` | Brand appears in URL path | May mimic official paths. |
| `is_punycode` | Starts with `xn--` | Can indicate Unicode homograph abuse. |
| `entropy` | Shannon entropy | Random-looking strings have higher entropy. |
| `has_login` | Domain contains `login` | Credential harvesting signal. |
| `has_verify` | Domain contains `verify` | KYC/account verification lure. |
| `has_banking` | Domain contains `banking` | Banking phishing signal. |
| `suspicious_trigram_count` | Count of suspicious trigrams | Captures short lexical patterns. |
| `has_ssl` | TLS available | Phishing domains increasingly use HTTPS. |
| `cert_age_days` | Certificate age | Very new certificates can be suspicious. |
| `has_mx` | Mail exchange records | Missing MX can suggest a short-lived web-only lure. |
| `domain_age_days` | WHOIS creation age | Newly registered domains are higher risk. |
| `has_password_field` | Content has password input | Direct credential-harvesting signal. |
| `form_action_external` | Form submits to different domain | Exfiltration indicator. |

Example feature vector:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "domain_length": 26,
  "num_hyphens": 2,
  "is_malicious_gtld": 1,
  "has_brand_keyword": 1,
  "has_verify": 1,
  "has_banking": 1,
  "entropy": 4.18,
  "has_ssl": 1,
  "cert_age_days": 2,
  "has_mx": 0,
  "domain_age_days": 1
}
```

The model sees only the numeric columns in the exact order saved in the joblib artifact:

```python
X = df_features[self.feature_names].fillna(-1)
```

That alignment line is critical. If `self.feature_names` contains a feature missing from inference output, inference fails. If the order changes without retraining, predictions become meaningless.

### 4d. Model Training

The production predictor says it loads an ensemble from `models/ensemble_v7.joblib` and expects an object containing:

```python
self.ensemble = joblib.load(model_path)
self.rf = self.ensemble["rf"]
self.xgb = self.ensemble["xgb"]
self.feature_names = self.ensemble["features"]
self.threshold = self.ensemble.get("threshold", 0.65)
```

This implies a Random Forest plus XGBoost ensemble:

```python
ml_conf = 0.6 * xgb_proba + 0.4 * rf_proba
```

That is a reasonable choice for phishing-domain features because the data is tabular, heterogeneous, partially nonlinear, and not huge enough to require deep learning for the primary classifier. Random forests are robust and interpretable through feature importance. Gradient-boosted trees often perform strongly on tabular structured data. The ensemble smooths individual model weaknesses.

Hypothetical training loop:

```python
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support

data = pd.read_csv("research/datasets/processed/full_features_v9.csv")
feature_names = [c for c in data.columns if c not in ["domain", "label", "source"]]

train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

X_train = train[feature_names].fillna(-1)
y_train = train["label"].map({"legitimate": 0, "phishing": 1})
X_test = test[feature_names].fillna(-1)
y_test = test["label"].map({"legitimate": 0, "phishing": 1})

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
xgb = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=400, eval_metric="logloss")

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

score = 0.4 * rf.predict_proba(X_test)[:, 1] + 0.6 * xgb.predict_proba(X_test)[:, 1]
pred = (score >= 0.65).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")

joblib.dump(
    {"rf": rf, "xgb": xgb, "features": feature_names, "threshold": 0.65},
    "models/ensemble_v9.joblib",
)
```

Security train/test splitting should be time-based when possible. Random splits can leak campaign structure: the same phishing kit, domain pattern, or feed batch can appear in both train and test, making performance look better than it will be on tomorrow's domains. A better split is:

| Split | Use |
|---|---|
| Older training period | Fit model and tune features. |
| Later validation period | Tune threshold and fallback policy. |
| Newest holdout/live feed | Estimate real-world performance. |

The PS-02 shortlisting set likely differs from the full training set by being an evaluation or challenge shortlist rather than a broad training corpus. It may have fewer rows, more adversarial examples, or hidden labels. Use it as a validation/benchmark set, not as ordinary training data, unless the competition rules permit training on it.

### 4e. Model Evaluation

The metrics that matter in security are not interchangeable.

| Metric | Meaning | Security interpretation |
|---|---|---|
| Precision | Of domains flagged phishing, how many are truly phishing? | High precision protects analyst trust and prevents alert fatigue. |
| Recall | Of truly phishing domains, how many were caught? | High recall reduces missed attacks. |
| F1 | Harmonic mean of precision and recall | Useful single score when classes are imbalanced. |
| ROC AUC | Ranking quality across thresholds | Helpful but can be optimistic on imbalanced data. |
| PR AUC | Precision/recall quality across thresholds | Often more useful for rare phishing positives. |
| False positive rate | Legitimate domains flagged as phishing | Can cause business disruption and analyst fatigue. |
| False negative rate | Phishing domains marked clean | Can lead to credential theft and incident escalation. |

False negatives are catastrophic because a missed phishing domain can continue harvesting credentials. False positives damage trust because analysts stop believing alerts, especially when benign government/banking domains are flagged. Herald's staged design addresses this tension: the fast model can call obvious cases, while borderline cases become `Suspected` and receive more evidence before being promoted to `Phishing`.

`monitoring/` tracks model quality indirectly in production through labels, confidence, worker status, DLQ, and analyst feedback endpoints. The `/api/feedback` endpoint is present but incomplete in the visible code. A mature implementation should persist `TP`, `FP`, `Escalated`, and analyst notes, then feed those records into periodic evaluation and retraining.

### 4f. Model Inference

Live inference through `PhishingPredictorV3.predict()` works like this:

1. Build a one-row DataFrame with the domain.
2. Extract lexical features.
3. Add v7-specific flags: `is_common_tld`, `has_brand_keyword`.
4. Collect network features: SSL and DNS.
5. Collect WHOIS domain age.
6. Select and order model features using the artifact's `features` list.
7. Predict Random Forest and XGBoost probabilities.
8. Combine probabilities as `0.6 * xgb + 0.4 * rf`.
9. If score is above threshold, return `Phishing`.
10. If score is borderline, run content analysis and adjust score.
11. If still uncertain and above fallback trigger, mark visual analysis required or run OCR fallback.

Example:

```python
from herald.predict_with_fallback import PhishingPredictorV3

predictor = PhishingPredictorV3()
result = predictor.predict("hdfc-netbanking-verify.top", cse_name="HDFC", include_visual=False)
```

Possible output:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "ml_confidence": 0.58,
  "ml_confidence_adjusted": 0.73,
  "status": "Phishing",
  "analysis_type": "ML + Content Analysis",
  "content_features": {
    "has_password_field": true,
    "has_login_form": true,
    "form_action_external": true
  }
}
```

The confidence score is a model-derived probability-like ranking signal, not a legal proof. It should be interpreted alongside evidence and calibration results. A score of `0.80` means the model is strongly ranking the domain as phishing-like under its training distribution; it does not guarantee an 80% real-world probability unless the model has been calibrated and monitored.

Low-confidence predictions are handled through `Clean` or `Suspected` states. Borderline predictions trigger content analysis and visual analysis rather than forcing a binary verdict too early.

### 4g. Model Artifacts

`models/` contains serialized model and supporting artifacts. Visible files include `char_vocab.json` and `augmented_training.xlsx`; the code expects joblib ensembles such as `ensemble_v7.joblib`, and the git status command surfaced an LFS-managed `ensemble_v9.joblib`. The expected model artifact shape is:

```python
{
    "rf": fitted_random_forest,
    "xgb": fitted_xgboost_classifier,
    "features": ["domain_length", "digit_ratio", "..."],
    "threshold": 0.65
}
```

`outputs/` stores inference outputs and reports. `evidence/` stores investigation-specific artifacts generated by the CLI pipeline. A mature model deployment process should version:

- Dataset version.
- Feature extractor version.
- Model algorithm and hyperparameters.
- Feature names and order.
- Threshold.
- Evaluation metrics.
- Training date.
- Commit SHA.

Example model metadata:

```json
{
  "model_version": "v9",
  "trained_at": "2026-05-31T10:30:00Z",
  "dataset": "full_features_v9.csv",
  "threshold": 0.65,
  "precision": 0.981,
  "recall": 0.934,
  "features_sha256": "..."
}
```

> **Key Takeaways**
> - Herald's ML pipeline is a tabular ensemble for phishing/lookalike-domain scoring.
> - Feature parity between research and production is the highest-risk ML contract.
> - Borderline scores are intentionally escalated into content or visual analysis.

## 5. Research Layer

The `research/` directory is the laboratory for the production system. It keeps datasets, experiment scripts, validation utilities, and model comparison workflows outside the runtime `herald/` package.

```text
research/
  datasets/
    raw/
    processed/
    external/
  ml_experiments/
  scripts/
  utils/
```

Raw datasets preserve original source material. Processed datasets store cleaned, normalized, merged, or feature-extracted versions. External datasets add third-party context such as PhishTank, URLhaus, Tranco, and Majestic. `ml_experiments/` contains retraining, ablation, and comparison experiments. `research/utils/` provides loading, enrichment, validation, and submission helper utilities.

The notebook-to-production flow should look like this:

1. Explore raw PS-02 and external data in research scripts or notebooks.
2. Encode a repeatable extraction step in `research/scripts/extract_features_vN.py`.
3. Train and evaluate in `research/scripts/retrain_vN.py`.
4. Compare against older model versions.
5. Analyze false positives and false negatives.
6. Promote stable feature logic into `herald/features/`.
7. Save model artifacts to `models/`.
8. Add tests that prove production inference can load the model and align features.

Concrete example:

```bash
python research/scripts/build_dataset_v8.py
python research/scripts/extract_features_v8.py
python research/scripts/retrain_v8.py
python research/scripts/compare_versions.py
```

Dataset management rules:

| Dataset type | Rule |
|---|---|
| Raw | Never mutate in place. Keep source filenames and provenance. |
| Processed | Version by pipeline/model generation, such as `full_features_v9.csv`. |
| External | Record download date and feed source. |
| Shortlisting | Treat as benchmark/evaluation unless permitted otherwise. |
| Mock | Use for demo and integration checks, not final model claims. |

> **Key Takeaways**
> - `research/` is for reproducible experiments, not runtime business logic.
> - Feature extraction must be promoted carefully into `herald/features/`.
> - Every model artifact should be traceable to dataset, feature, and evaluation versions.

## 6. Frontend Architecture

The frontend is a Next.js 16 application using React 19, TypeScript, Tailwind, lucide-react icons, shadcn-style UI primitives, and Recharts. Its job is to present operational telemetry and investigation state to analysts and operators.

Important directories:

```text
frontend/app/
  page.tsx
  observability/page.tsx
  dlq/page.tsx
  domain/[id]/page.tsx
  traces/[trace_id]/page.tsx
  api/health/route.ts
  api/ready/route.ts
  api/metrics-summary/route.ts

frontend/components/
  charts/
  domain-detail/
  system-health/
  threat-feed/
  traces/
  ui/

frontend/services/
  websocket.ts
  mock-generator.ts
  mock-traces.ts
  mock-scenarios/

frontend/types/index.ts
```

The TypeScript model file defines the frontend's view of `ThreatEvent`, `QueueMetrics`, `WorkerMetrics`, `BrowserWorkerTelemetry`, `CircuitBreakerStatus`, `TraceSpan`, `TraceEvent`, `DLQEntry`, and `DomainIntelligence`. These are dashboard contracts. They do not exactly equal database models; instead, they are presentation-oriented shapes.

Mock scenarios currently include:

| Scenario | What it simulates |
|---|---|
| `NORMAL` | Baseline queue, infrastructure, and threat stream. |
| `PHISHING_BURST` | Many malicious events and higher queue depth. |
| `REDIS_PRESSURE` | Increased event latency, Redis degraded status, circuit instability. |
| `RETRY_STORM` | Failed worker stages, retry backlog, DLQ growth, low throughput. |
| `DEGRADED` | PostgreSQL degraded mode and slow processing. |

To add a new scenario:

1. Add the scenario name to `ScenarioState`.
2. Add a new entry in `scenarios`.
3. Implement `mutateThreats`, `mutateQueue`, `mutateInfra`, and `mutateBreakers`.
4. Add the new state to the rotating `states` array if it should appear automatically.
5. Update UI controls if the dashboard exposes manual scenario switching.

Example:

```typescript
export type ScenarioState =
  | "NORMAL"
  | "PHISHING_BURST"
  | "REDIS_PRESSURE"
  | "RETRY_STORM"
  | "DEGRADED"
  | "VISUAL_BACKLOG";
```

The frontend talks to the backend in two ways. First, Next.js route handlers under `frontend/app/api` can proxy health/ready/metrics summary requests. Second, `frontend/services/websocket.ts` is the natural place to connect to FastAPI `/ws/telemetry` when `NEXT_PUBLIC_TELEMETRY_MODE=REAL` or hybrid mode is enabled.

The investigation UI should show:

- Domain and verdict.
- Confidence score.
- Queue lifecycle state.
- DNS records.
- TLS metadata.
- WHOIS metadata.
- OCR findings.
- Screenshot availability.
- Related domains.
- Timeline events.
- Trace spans and retry/degraded status.

> **Key Takeaways**
> - The frontend is an operations console with mock-first and real-data hooks.
> - TypeScript types mirror telemetry and investigation concepts.
> - Mock scenarios are deliberate tools for dashboard development and demos.

## 7. API Reference

Base URL:

```text
http://localhost:8000
```

Authentication uses OAuth2 password flow with bearer JWTs.

### Register User

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/api/auth/register` |
| Auth | none, but requires `ALLOW_REGISTRATION=true` |

Request:

```json
{
  "username": "analyst",
  "password": "change-me",
  "role": "analyst"
}
```

Response:

```json
{
  "id": 1,
  "username": "analyst",
  "role": "analyst",
  "is_active": true
}
```

Curl:

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"analyst","password":"change-me","role":"analyst"}'
```

### Login

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/api/auth/token` |
| Auth | none |

Request is form encoded:

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=analyst&password=change-me"
```

Response:

```json
{
  "access_token": "jwt...",
  "token_type": "bearer"
}
```

### Queue Domain Scan

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/api/scan` |
| Auth | bearer token |

Request:

```json
{
  "domain": "secure-sbi-login.xyz",
  "target_cse": "SBI"
}
```

Curl:

```bash
curl -X POST http://localhost:8000/api/scan \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"domain":"secure-sbi-login.xyz","target_cse":"SBI"}'
```

Response:

```json
{
  "status": "ok",
  "job_id": "uuid",
  "message": "Domain secure-sbi-login.xyz queued for analysis"
}
```

### Queue URL Investigation

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/api/investigate` |
| Auth | bearer token |

Request:

```json
{
  "url": "https://secure-sbi-login.xyz/verify"
}
```

Response:

```json
{
  "status": "ok",
  "job_id": "uuid",
  "trace_id": "uuid-or-request-id",
  "domain": "secure-sbi-login.xyz",
  "lifecycle_state": "QUEUED"
}
```

### Recent Detections

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/detections
```

Response:

```json
[
  {
    "domain": "secure-sbi-login.xyz",
    "label": "Phishing",
    "confidence": 0.91,
    "target_cse": "SBI",
    "source": "api_manual",
    "scan_date": "2026-06-02T09:00:00",
    "analyst_verdict": null
  }
]
```

### Suspected Domains

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/suspected
```

Returns records where `DomainScan.label == "Suspected"`.

### Feedback

| Field | Value |
|---|---|
| Method | `POST` |
| Path | `/api/feedback` |
| Auth | bearer token |

Request:

```json
{
  "domain": "secure-sbi-login.xyz",
  "verdict": "TP"
}
```

> ⚠️ Inferred from structure — verify against actual source.
> The visible endpoint looks incomplete after fetching the scan row. A complete implementation should update `analyst_verdict`, write an audit log, and commit.

### Whitelist

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/whitelist
```

Add:

```bash
curl -X POST http://localhost:8000/api/whitelist \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"domain":"example.gov.in","reason":"Official domain"}'
```

Delete:

```bash
curl -X DELETE \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/whitelist/example.gov.in
```

### Failed Jobs

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/failed-jobs
```

Retry:

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/admin/failed-jobs/retry
```

### Export

JSON:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/export/secure-sbi-login.xyz/json
```

PDF:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/export/secure-sbi-login.xyz/pdf \
  --output herald_report_secure-sbi-login.xyz.pdf
```

### Health and Metrics

| Path | Purpose |
|---|---|
| `/api/health` | Liveness check. |
| `/api/ready` | DB/Redis/readiness status. |
| `/api/metrics-summary` | Dashboard-friendly queue/worker/browser/circuit summary. |
| `/metrics` | Prometheus text metrics. |
| `/ws/telemetry` | WebSocket stream of Redis telemetry events. |

Error conventions:

| Status | Meaning |
|---|---|
| `401` | Missing/invalid bearer token. |
| `403` | Registration disabled or forbidden action. |
| `404` | Domain/whitelist record not found. |
| `429` | Rate limit or queue pressure. |
| `500` | Redis offline, export failure, or internal dependency failure. |

> **Key Takeaways**
> - Most operational endpoints require a bearer token.
> - Scan endpoints enqueue work; they do not synchronously return a final verdict.
> - Health, readiness, metrics, and WebSocket telemetry support production operations.

## 8. Database Schema

The visible schema contains four SQLAlchemy models: `DomainScan`, `User`, `AuditLog`, and `Whitelist`.

### `domain_scans`

| Field | Type | Index | Description |
|---|---|---|---|
| `id` | integer | primary key | Internal row ID. |
| `domain` | string | yes | Candidate domain. |
| `target_cse` | string | no | Protected entity or brand. |
| `source` | string | no | Source such as API, CT feed, or manual. |
| `scan_date` | datetime | no | Last scan/update time. |
| `label` | string | no | `Clean`, `Suspected`, `Phishing`, `Rejected`, etc. |
| `confidence` | float | no | Confidence score. |
| `is_live` | boolean | no | Whether target was reachable. |
| `analyst_verdict` | string | nullable | Analyst feedback such as `TP`, `FP`, `Escalated`. |
| `lifecycle_state` | string | no | Queue/investigation state. |
| `screenshot_path` | string | nullable | Path to screenshot artifact. |
| `ocr_text` | string | nullable | Extracted visual text. |
| `dns_records` | string | nullable | JSON-as-string DNS details. |

### `users`

| Field | Type | Index | Description |
|---|---|---|---|
| `id` | integer | primary key | Internal user ID. |
| `username` | string | unique/indexed | Login name. |
| `hashed_password` | string | no | Password hash. |
| `role` | string | no | `admin` or `analyst`. |
| `is_active` | boolean | no | Account status. |

### `audit_logs`

| Field | Type | Index | Description |
|---|---|---|---|
| `id` | integer | primary key | Internal event ID. |
| `timestamp` | datetime | no | When action occurred. |
| `user_id` | string | yes | User or system actor. |
| `action` | string | no | Action name. |
| `domain` | string | nullable | Domain affected. |
| `result` | string | no | Result text. |
| `ip_address` | string | nullable | Request/client IP. |

### `whitelist`

| Field | Type | Index | Description |
|---|---|---|---|
| `id` | integer | primary key | Internal whitelist ID. |
| `domain` | string | unique/indexed | Whitelisted domain. |
| `added_by` | string | no | Actor. |
| `added_on` | datetime | no | Creation time. |
| `reason` | string | nullable | Human reason. |

Likely additional entities for a future mature schema:

> ⚠️ Inferred from structure — verify against actual source.

| Entity | Why it would help |
|---|---|
| `Investigation` | Store one investigation run per trace, not only one row per domain. |
| `EvidenceArtifact` | Track screenshots, Markdown, JSON, PDF, OCR snippets, and hashes. |
| `FeatureVector` | Store model input features for explainability and retraining. |
| `ModelVersion` | Connect predictions to artifact versions and thresholds. |
| `TelemetryEvent` | Optional durable event store beyond Redis pub/sub. |

Data lifecycle:

| Data type | Suggested retention |
|---|---|
| Queue jobs | Until ack/retry/DLQ; DLQ retained until reviewed. |
| Domain scans | 90-365 days depending on policy. |
| Evidence artifacts | 30-180 days, longer for confirmed incidents. |
| Screenshots/OCR | Treat as sensitive; retain only as needed. |
| Audit logs | 1 year or compliance-driven retention. |
| Model training data | Long-term with provenance and access controls. |

Query patterns:

- Fetch recent detections by `scan_date DESC`.
- Fetch suspected domains by `label`.
- Fetch exact domain record by `domain`.
- Check whitelist by normalized domain.
- Export a domain report by exact domain.

> **Key Takeaways**
> - `DomainScan` is the central persisted detection object.
> - Current schema is intentionally small but could benefit from normalized evidence/model tables.
> - Screenshots, OCR text, and DNS records require retention and sensitivity controls.

## 9. Data Flow — End-to-End Worked Example

Scenario:

> A user encounters `https://hdfc-netbanking-verify.top/login`. The page advertises HDFC netbanking, asks for customer ID and password, and was registered recently.

This is adapted to Herald's actual domain-risk model. The pasted example about login failures is a generic SOC telemetry scenario; Herald's implemented pipeline is about phishing domains.

### 1. API submission

Request:

```bash
curl -X POST http://localhost:8000/api/investigate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "x-request-id: trc-demo-hdfc-001" \
  -d '{"url":"https://hdfc-netbanking-verify.top/login"}'
```

API job:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "original_url": "https://hdfc-netbanking-verify.top/login",
  "source": "api_investigate",
  "target_cse": "Unknown",
  "trace_id": "trc-demo-hdfc-001",
  "lifecycle_state": "QUEUED"
}
```

### 2. Redis queue

`RedisReliableQueue.enqueue()` adds fields:

```json
{
  "job_id": "96b68d2e-824f-4e47-9735-75fd59af24ff",
  "trace_id": "trc-demo-hdfc-001",
  "attempts": 0,
  "enqueued_at": 1780387200.0,
  "domain": "hdfc-netbanking-verify.top",
  "source": "api_investigate",
  "target_cse": "Unknown"
}
```

### 3. Domain worker

The domain worker dequeues and validates:

```python
validate_url_safe("https://hdfc-netbanking-verify.top/login")
```

If safe, it checks whitelist:

```sql
SELECT * FROM whitelist WHERE domain = 'hdfc-netbanking-verify.top';
```

No whitelist match is found.

### 4. Feature extraction

Lexical and network features:

```json
{
  "domain_length": 26,
  "num_hyphens": 2,
  "is_malicious_gtld": 1,
  "brand_keyword_position": 1,
  "has_brand_keyword": 1,
  "has_verify": 1,
  "has_banking": 1,
  "min_brand_levenshtein": 99.0,
  "has_ssl": 1,
  "is_lets_encrypt": 1,
  "cert_age_days": 1,
  "has_mx": 0,
  "domain_age_days": 1
}
```

### 5. Model score

The predictor computes:

```python
rf_proba = 0.67
xgb_proba = 0.74
ml_conf = 0.6 * 0.74 + 0.4 * 0.67
```

Result:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "ml_confidence": 0.712,
  "status": "Phishing",
  "analysis_type": "ML-v7-Ensemble"
}
```

If the score were `0.52`, content analysis could add a boost:

```json
{
  "has_password_field": true,
  "has_login_form": true,
  "has_obfuscated_js": false,
  "form_action_external": true,
  "redirected_to_different_domain": false
}
```

Boost: `0.15 + 0.15 = 0.30`, adjusted score capped at `0.82`.

### 6. Database update

`upsert_domain_scan()` stores:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "target_cse": "Unknown",
  "source": "api_investigate",
  "label": "Phishing",
  "confidence": 0.712,
  "lifecycle_state": "PROCESSING"
}
```

### 7. Optional visual worker

If the domain is borderline and visual analysis is required, visual job:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "target_cse": "HDFC",
  "initial_confidence": 0.58,
  "source": "api_investigate",
  "parent_analysis_type": "ML-v7-Ensemble + Visual Pending"
}
```

Visual output:

```json
{
  "success": true,
  "screenshot_path": "evidence/trc-demo-hdfc-001/hdfc-netbanking-verify.top/screenshot.png",
  "ocr_text": "HDFC Bank NetBanking Login Customer ID Password Verify Account",
  "ocr_findings": {
    "is_suspicious": true,
    "ocr_risk_score": 0.88
  },
  "cv_ocr_confirmed": true,
  "final_confidence": 0.98,
  "cv_ocr_status": "suspicious"
}
```

Final database row:

```json
{
  "domain": "hdfc-netbanking-verify.top",
  "label": "Phishing",
  "confidence": 0.98,
  "lifecycle_state": "VERDICT_READY",
  "screenshot_path": "evidence/.../screenshot.png",
  "ocr_text": "HDFC Bank NetBanking Login Customer ID Password Verify Account"
}
```

### 8. Analyst UI

The analyst sees:

- Verdict: `MALICIOUS` or `Phishing`.
- Confidence: `98%`.
- Stage timeline: received, lexical, content, visual, OCR, completed.
- Screenshot available.
- OCR detections present.
- DNS/TLS/WHOIS tables if enriched.
- Trace spans showing browser launch, screenshot capture, OCR, and verdict generation.

### 9. Analyst action

The analyst can:

1. Export JSON/PDF evidence.
2. Mark feedback as `TP`.
3. Escalate to takedown workflow outside Herald.
4. Add false-positive domains to whitelist.
5. Review related domains or repeat investigation.

> **Key Takeaways**
> - A Herald investigation starts as a domain/URL job and ends as a verdict plus evidence.
> - Borderline cases receive visual/OCR analysis.
> - The analyst journey is evidence-driven, not score-only.

## 10. Mock Scenarios — Demo Guide

The frontend mock scenarios let you demonstrate operational states without requiring a live Redis/Postgres/worker stack.

### `NORMAL`

Represents ordinary throughput. Threat events, queue metrics, infrastructure state, and circuit breakers pass through unchanged.

Trigger:

```typescript
setScenario("NORMAL");
```

Detection exercised: baseline event rendering and healthy dashboards.

Analyst journey: monitor recent detections and system health without intervention.

### `PHISHING_BURST`

Mutates all threat events to `MALICIOUS`, raises confidence to `95`, increases queue depth by `5000`, and increases throughput.

Trigger:

```typescript
setScenario("PHISHING_BURST");
```

Detection exercised: burst handling, visual emphasis for malicious feeds, queue growth.

Analyst journey: triage active malicious events and watch backlog.

### `REDIS_PRESSURE`

Adds latency, increases queue depth, marks Redis degraded, raises API latency, and shifts Redis circuit breaker state to `HALF_OPEN`.

Trigger:

```typescript
setScenario("REDIS_PRESSURE");
```

Detection exercised: degraded infrastructure display and queue pressure status.

Analyst journey: pause noncritical submissions, notify operator, monitor recovery.

### `RETRY_STORM`

Marks threat stages as failed, increases retry queue and DLQ sizes, drops throughput, and opens worker-related breakers.

Trigger:

```typescript
setScenario("RETRY_STORM");
```

Detection exercised: DLQ page, retry visibility, degraded worker behavior.

Analyst journey: inspect failed-job classes, retry eligible jobs, identify dependency failure source.

### `DEGRADED`

Forces slow latency, low throughput, PostgreSQL degraded state, and degraded mode.

Trigger:

```typescript
setScenario("DEGRADED");
```

Detection exercised: resilience UX and degraded platform affordances.

Analyst journey: keep reviewing existing evidence while operators stabilize persistence.

To add a custom scenario:

1. Edit `frontend/services/mock-scenarios/index.ts`.
2. Extend `ScenarioState`.
3. Add a mutator object.
4. Decide whether it should auto-rotate.
5. Add tests or Storybook/demo checks if available.

Example custom mutator:

```typescript
VISUAL_BACKLOG: {
  mutateThreats: (events) => events.map((evt) => ({ ...evt, workerStage: "VISUAL" })),
  mutateQueue: (metrics) => ({ ...metrics, queueDepth: metrics.queueDepth + 2500 }),
  mutateInfra: (infra) => ({ ...infra, degradedModeActive: true }),
  mutateBreakers: (breakers) => breakers
}
```

> **Key Takeaways**
> - Mock scenarios are frontend simulation tools.
> - They mutate typed dashboard state rather than backend records.
> - They are useful for demos, resilience design, and visual regression checks.

## 11. Testing Strategy

The `tests/` directory currently includes tests such as:

```text
tests/test_api_queue.py
tests/test_investigation_cli.py
tests/test_vibe_fixes.py
```

A complete testing strategy should cover correctness, security controls, queue behavior, ML contracts, and analyst-facing outputs.

Unit tests per module:

| Module | Test focus |
|---|---|
| `api` | Auth, rate limits, request validation, queue pressure, export errors. |
| `core.security` | SSRF rejection, private IP handling, scheme validation, DNS edge cases. |
| `db` | Schema creation, upsert behavior, PostgreSQL/SQLite configuration. |
| `features` | Deterministic feature values for known domains. |
| `predict_with_fallback` | Model artifact loading, feature alignment, threshold decisions. |
| `investigation` | Stage ordering, degraded-stage behavior, evidence persistence. |
| `monitoring.redis_queue` | Enqueue, dequeue, ack, retry, DLQ, lease reclamation. |
| `monitoring.queue_worker` | Whitelist handling, duplicate skip, SSRF failure, DB commit. |
| `monitoring.visual_worker` | Timeout handling, circuit breaker, DB update. |
| `telemetry` | Envelope schema and frontend compatibility. |
| `frontend` | Component rendering, mock scenarios, websocket parsing. |

Security telemetry mocks:

```python
def test_lexical_features_for_bank_phish():
    df = pd.DataFrame([{"domain": "hdfc-netbanking-verify.top"}])
    features = extract_url_features(df)
    assert features.loc[0, "is_malicious_gtld"] == 1
    assert features.loc[0, "has_verify"] == 1
```

Queue integration test:

```python
def test_queue_retry_moves_to_dlq(fake_redis):
    queue = RedisReliableQueue(fake_redis, DOMAIN_ANALYSIS_QUEUE, max_retries=1)
    queue.enqueue({"domain": "example.top"})
    leased, job = queue.dequeue(timeout=0)
    queue.retry_or_dlq(leased, job, RuntimeError("boom"))
    assert queue.depth()["dlq"] == 1
```

Detection quality tests should not only assert that code runs. They should assert that known phishing examples score above threshold, known legitimate CSE domains score below threshold, and hard false-positive/false-negative regressions are tracked as fixtures.

Recommended commands:

```bash
python -m pytest tests/
python -m pytest tests/test_api_queue.py
python -m pytest tests/test_investigation_cli.py
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

ML contract check:

```python
def test_model_feature_contract():
    predictor = PhishingPredictorV3(model_path="models/ensemble_v7.joblib")
    df = pd.DataFrame([{"domain": "secure-sbi-login.xyz"}])
    features = extract_url_features(df)
    missing = set(predictor.feature_names) - set(features.columns)
    assert not missing
```

> **Key Takeaways**
> - Test detection quality, not just application mechanics.
> - SSRF and queue reliability need dedicated regression tests.
> - ML feature contracts should be tested whenever features or models change.

## 12. Deployment & Operations

### Local Development

Prerequisites:

- Python 3.12+
- Node.js 18+
- Docker and Docker Compose
- Redis and PostgreSQL when running platform mode
- Playwright browser dependencies for visual analysis

Backend setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python setup_db.py
```

Infrastructure:

```bash
docker compose up -d redis postgres
```

API:

```bash
uvicorn herald.api.main:app --host 0.0.0.0 --port 8000
```

Workers:

```bash
python -m herald.monitoring.queue_worker
python -m herald.monitoring.visual_worker
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

CLI:

```bash
herald investigate https://hdfc-netbanking-verify.top --visual
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLite/PostgreSQL connection string. |
| `REDIS_HOST` | Redis hostname, default `localhost`. |
| `REDIS_PORT` | Redis port, default `6379`. |
| `ALLOW_REGISTRATION` | Enables `/api/auth/register` when `true`. |
| `DOMAIN_QUEUE_MAX_READY` | Rejects new scan jobs when domain queue is too deep. |
| `VISUAL_QUEUE_MAX_READY` | Stops visual enqueue when browser backlog is too high. |
| `VISUAL_ANALYSIS_TIMEOUT_SECONDS` | Child-process visual timeout. |
| `VISUAL_CIRCUIT_FAILURE_THRESHOLD` | Failures before opening visual circuit. |
| `VISUAL_CIRCUIT_RESET_SECONDS` | Circuit reset interval. |
| `IDEMPOTENCY_TTL_SECONDS` | Duplicate domain skip window. |
| `DB_POOL_SIZE` | PostgreSQL pool size. |
| `DB_MAX_OVERFLOW` | PostgreSQL overflow connections. |
| `DB_POOL_TIMEOUT` | PostgreSQL pool wait timeout. |
| `SQLITE_BUSY_TIMEOUT_SECONDS` | SQLite lock wait timeout. |

### Health Checks

Liveness:

```bash
curl http://localhost:8000/api/health
```

Readiness:

```bash
curl http://localhost:8000/api/ready
```

Prometheus:

```bash
curl http://localhost:8000/metrics
```

Dashboard summary:

```bash
curl http://localhost:8000/api/metrics-summary
```

### Operational Playbook

1. If `/api/ready` reports Redis disconnected, check Redis container and `REDIS_HOST`.
2. If domain queue is growing, start or scale `queue_worker`.
3. If visual queue is growing, inspect browser timeouts and visual circuit state.
4. If DLQ grows, fetch `/api/admin/failed-jobs` and group by `last_error`.
5. If schema mismatch appears, run `python setup_db.py` or apply migration.
6. If model loading fails, verify model artifact path and Git LFS checkout.
7. If screenshots fail, verify Playwright browser installation and runtime permissions.

`scripts/` and root utilities likely contain setup, migration, reporting, or operational helpers. Use `setup_db.py` for schema initialization/migration checks and Docker Compose for local services.

> **Key Takeaways**
> - Platform mode needs API, Redis, workers, database, and optionally frontend.
> - Health/readiness/metrics endpoints are already present.
> - Browser/OCR failures should degrade investigations rather than halt the entire platform.

## 13. Security Considerations

Herald handles suspicious URLs, screenshots, OCR text, analyst accounts, and potentially sensitive evidence. Securing Herald itself is as important as detecting phishing.

### Data Handling

Screenshots and OCR text may contain login forms, brand assets, victim-specific query strings, or submitted test credentials. Treat them as sensitive artifacts. Store them with restricted permissions, encrypt at rest in production, and define retention periods.

Example policy:

```text
Confirmed incident evidence: retain 180 days.
False positives: retain 30 days.
Screenshots containing PII: redact or delete after review.
Audit logs: retain 365 days.
```

### API Security

The API uses JWT bearer tokens and password hashing helpers. Production hardening should include:

- Strong `SECRET_KEY`.
- Registration disabled except during controlled bootstrap.
- Admin-only whitelist and failed-job retry controls.
- Role checks for analyst vs admin actions.
- Rate limiting on scan endpoints.
- Request IDs and audit logs.
- CORS restricted to trusted dashboard origins.
- HTTPS termination.

### SSRF and Browser Safety

URL validation is mandatory because Herald fetches and screenshots untrusted targets. Controls should include:

- Reject private and loopback IPs unless explicitly allowed.
- Re-resolve hostnames near fetch time.
- Restrict schemes to HTTP/HTTPS.
- Isolate browser processes.
- Enforce timeouts.
- Prevent local file access.
- Avoid sending internal cookies or credentials.

### Model Security

Model risks include poisoning, adversarial domain strings, and distribution drift. Defenses:

- Track dataset provenance.
- Separate training, validation, shortlisting, and live holdouts.
- Require review before adding external feed labels.
- Monitor false positives and false negatives.
- Version model artifacts and feature definitions.
- Treat joblib files as trusted-code artifacts; do not load untrusted joblib files.

### Access Control

Analysts may not need admin controls. Suggested roles:

| Role | Permissions |
|---|---|
| Analyst | View detections, export evidence, submit feedback. |
| Hunter | Submit scans, view related domains, create investigations. |
| Admin | Manage users, whitelist, DLQ retry, configuration. |
| Auditor | Read-only access to evidence and audit logs. |

### Abuse Cases

| Abuse | Mitigation |
|---|---|
| Attacker submits internal URL | SSRF validation and private IP rejection. |
| Attacker floods scan queue | Rate limits and queue pressure rejection. |
| Malicious page crashes browser | Child process isolation, timeouts, circuit breaker. |
| Poisoned model artifact | Trusted artifact storage and hash verification. |
| Analyst exports sensitive evidence | Role checks and audit logging. |

> **Key Takeaways**
> - Untrusted URL handling is Herald's highest runtime security risk.
> - Evidence artifacts are sensitive and need retention/access controls.
> - Joblib model artifacts must be treated as trusted executable assets.

## 14. Glossary

| Term | Definition |
|---|---|
| AI | Artificial Intelligence; in Herald, mostly ML-assisted phishing/domain scoring. |
| Analyst verdict | Human feedback such as true positive, false positive, or escalation. |
| API | Application Programming Interface; Herald exposes FastAPI endpoints. |
| AUC | Area Under Curve; ranking metric for classifiers. |
| Certificate Transparency | Public logs of issued TLS certificates. |
| Confidence | Model or fused score indicating phishing-likeness. |
| CSE | Critical/service entity label used for protected brands such as banks or government portals. |
| CT | Certificate Transparency. |
| DLQ | Dead-letter queue for jobs that failed processing. |
| DNS | Domain Name System. |
| Entropy | Measure of randomness in a string. |
| Evidence | Artifacts supporting a verdict: JSON, Markdown, screenshot, OCR text, DNS/TLS/WHOIS data. |
| False negative | A phishing domain incorrectly marked benign/clean. |
| False positive | A legitimate domain incorrectly marked malicious/suspicious. |
| F1 | Harmonic mean of precision and recall. |
| FP | False positive. |
| FN | False negative. |
| Homoglyph | Character that visually resembles another character, often used in domain impersonation. |
| IOC | Indicator of Compromise; observable artifact linked to malicious activity. |
| JWT | JSON Web Token used for bearer authentication. |
| Levenshtein distance | Edit distance between two strings, used for typo detection. |
| MTTD | Mean Time To Detect. |
| NRD | Newly Registered Domain. |
| OCR | Optical Character Recognition; converts screenshot text into machine-readable text. |
| PhishTank | Public phishing URL feed. |
| Playwright | Browser automation library used for screenshots and page analysis. |
| Precision | Fraction of flagged positives that are true positives. |
| PS-02 | Likely "Problem Statement 02" dataset identifier for the phishing-detection challenge data in this repo. |
| Punycode | ASCII encoding for internationalized domain names, often beginning `xn--`. |
| Recall | Fraction of actual positives that were detected. |
| Redis | In-memory data store used for queues, pub/sub, counters, and circuit state. |
| SOC | Security Operations Center. |
| SSRF | Server-Side Request Forgery, where a service is tricked into fetching internal resources. |
| TLD | Top-Level Domain, such as `.com`, `.in`, `.xyz`. |
| TLS | Transport Layer Security. |
| TTP | Tactics, Techniques, and Procedures; adversary behavior patterns. |
| URLhaus | Public malicious URL feed. |
| WHOIS | Domain registration metadata. |

Example: `secure-hdfc-verify.xyz` contains the CSE keyword `hdfc`, suspicious action word `verify`, and suspicious TLD `.xyz`; these are lexical IOCs that contribute to the phishing score.

> **Key Takeaways**
> - Herald terminology mixes ML, phishing investigation, queues, and operations.
> - CSE and PS-02 are project/dataset-specific terms that should be defined in onboarding docs.
> - Precision, recall, FP, and FN must be interpreted through analyst workload and missed-attack risk.

## 15. Contribution Guide

### Add a New Ingestion Source

1. Create or update a module under `herald/ingestion/`.
2. Normalize every discovered target into a domain.
3. Attach `source`, `target_cse` when known, and `trace_id`.
4. Enqueue through `RedisReliableQueue`.
5. Add duplicate control if the source is noisy.
6. Add tests with fake feed events.

Example job:

```python
domain_queue.enqueue({
    "domain": "uidai-update-kyc.info",
    "source": "new_feed_name",
    "target_cse": "UIDAI",
})
```

### Add a New Detection Rule

1. Decide whether it is lexical, network, content, visual, or post-model scoring.
2. Add feature extraction in the relevant feature module.
3. Add tests for positive and negative examples.
4. If the ML model uses the feature, retrain and save a new artifact.
5. If the heuristic scorer uses it, update `investigation/scoring.py`.
6. Update documentation and explain the analyst-visible risk factor.

Example lexical rule:

```python
features["has_kyc"] = domains.str.contains("kyc", case=False, regex=False).fillna(False).astype(int)
```

### Retrain and Deploy a Model

1. Build or update the processed dataset.
2. Extract features with the current production-compatible extractor.
3. Train the model version.
4. Evaluate on time-based holdout and live feed validation.
5. Compare with previous model.
6. Analyze false positives and false negatives.
7. Save artifact with feature list and threshold.
8. Update `PhishingPredictorV3` default path or configuration.
9. Run inference contract tests.
10. Deploy with rollback plan.

Example:

```bash
python research/scripts/build_dataset_v8.py
python research/scripts/extract_features_v8.py
python research/scripts/retrain_v8.py
python research/scripts/compare_versions.py
python -m pytest tests/
```

### Code Style and Conventions

- Keep runtime code under `herald/`.
- Keep experiments under `research/`.
- Keep frontend contracts typed in `frontend/types`.
- Prefer explicit domain names over generic utility sprawl.
- Preserve trace IDs across API, queue, worker, telemetry, and evidence.
- Add tests for security and operational edge cases.
- Treat browser automation as unreliable and isolate it.
- Do not silently change model feature order.
- Do not load untrusted joblib artifacts.

### Practical Contribution Example

Goal: add a new protected brand keyword `upi`.

1. Update `CSE_KEYWORDS` in `herald/features/lexical_features.py`.
2. Add test examples:

```python
def test_upi_keyword_detection():
    df = pd.DataFrame([{"domain": "upi-verify-login.top"}])
    features = extract_url_features(df)
    assert features.loc[0, "brand_keyword_position"] == 1
```

3. Retrain if the model depends on keyword-derived distributions.
4. Validate on legitimate UPI-related domains to avoid false positives.
5. Document the keyword and examples.

> **Key Takeaways**
> - Contributions should preserve the boundary between research and runtime code.
> - New features require tests and, when model-facing, retraining.
> - Security controls and evidence quality are part of the product, not optional add-ons.
