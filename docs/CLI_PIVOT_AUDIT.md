# HERALD CLI Pivot Audit

## Current Architecture

HERALD currently contains several overlapping product directions:

- Python detection backend: lexical features, WHOIS/DNS/TLS enrichment, ML predictors, screenshot/OCR analyzers.
- FastAPI platform layer: auth, scan submission, exports, readiness, metrics, WebSocket telemetry.
- Redis worker layer: reliable queues, domain worker, visual worker, retry/DLQ behavior, circuit breaker metrics.
- Dashboard layers: Streamlit dashboard plus a newer Next.js telemetry dashboard.
- Research/training layer: scripts and `ml/` experiments for model retraining and validation.

The strongest operational path is the backend investigation logic. The weakest path for a CLI-first tool is the distributed API/Redis/dashboard coupling, which is useful for a platform demo but unnecessary for direct investigations.

## Reusable Components

- `herald.features.lexical_features`: strong lexical feature extractor and suspicious keyword features.
- `herald.core.security`: SSRF guard for safer URL investigation.
- `herald.core.playwright_analyzer`: Playwright screenshot and pytesseract OCR path, now usable directly by CLI evidence runs.
- `herald.features.content_features`: useful later for HTML form/content enrichment.
- `herald.predict_with_fallback`: reusable model ideas and network feature logic, but too coupled to optional visual fallback for the initial CLI.
- `herald.db.models`: possible future SQLite persistence, but JSON/Markdown evidence is simpler for phase 1.
- `herald.utils.logging_config`: useful structured logging style, but the CLI should keep terminal output human-first.

## Complexity To Avoid For CLI Phase

- FastAPI auth, rate limiting, WebSockets, and dashboard-specific metrics.
- Redis reliable queues, DLQ retry loops, worker heartbeats, circuit breaker state.
- Next.js mock telemetry and observability dashboard.
- Certstream/social/new-domain monitors as always-on ingestion services.
- Selenium/EasyOCR visual paths where Playwright/pytesseract can cover the CLI demo path more simply.

## CLI-First Architecture

New focused modules live under `herald/investigation/`:

- `targets.py`: URL/domain normalization and safe evidence path fragments.
- `scoring.py`: lexical analysis and simple score fusion.
- `intelligence.py`: direct DNS, WHOIS, and TLS enrichment.
- `pipeline.py`: synchronous investigation workflow with graceful degradation.
- `persistence.py`: JSON, Markdown, and JSONL evidence persistence.
- `models.py`: typed result/stage dataclasses.
- `herald/cli.py`: argparse + Rich terminal interface.

The CLI executes investigation stages directly:

1. Normalize and validate target.
2. Run lexical analysis.
3. Collect DNS/WHOIS intelligence.
4. Inspect TLS certificate.
5. Optionally capture screenshot and OCR.
6. Combine signals into a simple operational verdict.
7. Persist evidence under `evidence/<trace_id>_<domain>/`.

## Migration Strategy

Phase 1 is implemented as a narrow additive path. The API, Redis workers, dashboard, and ML scripts remain untouched except for the CLI entrypoint and Playwright analyzer reuse.

Recommended next phases:

1. Move stable model scoring from `predict_with_fallback.py` behind a lightweight optional `ModelScorer`.
2. Add a local SQLite report index only if JSONL lookup becomes limiting.
3. Add Markdown export options and terminal report filters.
4. De-emphasize or archive dashboard/worker docs so the CLI is the primary README path.
5. Add small tests around normalization, lexical scoring, report persistence, and degraded visual capture.

