# Contributing to HERALD

Welcome! We are thrilled that you are interested in contributing to HERALD. This document outlines the process for contributing, how to set up your development environment, and where you can make the most impact.

## Getting Started

To get a local development environment running, follow these steps:

### Prerequisites
- **Python 3.12+**
- **Node.js 18+** (for the Next.js frontend)
- **Docker & Docker Compose** (for running Redis and local databases)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HERALD.git
   cd HERALD
   ```

2. **Start the Infrastructure:**
   ```bash
   docker compose up -d redis postgres
   ```

3. **Backend Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

4. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   # Create a .env.local file to connect to the backend
   echo "NEXT_PUBLIC_TELEMETRY_MODE=REAL" > .env.local
   echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" >> .env.local
   npm run dev
   ```

5. **Run Tests:**
   We use `pytest` for our test suite. Ensure all tests pass before submitting a pull request.
   ```bash
   pytest tests/
   ```

## Where You Can Contribute

HERALD is actively evolving, and there are many areas where you can help. Below are some excellent starting points:

### 1. Technical Debt & Refactoring
- **Unifying the Detection Engine:** Currently, the CLI uses a heuristic scorer (`herald/investigation/scoring.py`), while the Redis workers rely on an ML model (`PhishingPredictorV3`). Unifying these so the CLI can optionally utilize the ML model (or at least share core logic) would be a huge improvement.
- **Improving Test Coverage:** We recently added end-to-end integration tests for the API-to-Queue flow (`tests/test_api_queue.py`). However, unit test coverage for the Redis workers (`herald/monitoring/queue_worker.py` and `visual_worker.py`) is still sparse.
- **Frontend Real-time Telemetry:** The Next.js dashboard now proxies data to the backend when `NEXT_PUBLIC_TELEMETRY_MODE=REAL`. However, the WebSockets integration for real-time streaming updates (`/ws/telemetry`) is currently stubbed out or underutilized on the frontend. Hooking this up to live React context would be a great feature!

### 2. New Features
- **Enhanced OCR Pipelines:** Improving the robustness of screenshot capture and OCR text extraction for visual analysis.
- **Advanced Threat Feeds:** Integrating external threat intelligence feeds (e.g., VirusTotal, URLhaus) into the investigation pipeline.
- **Reporting & Exporting:** Expanding the PDF and JSON export functionalities to include more comprehensive evidence graphs.

## Contribution Guidelines

1. **Diagnose Before Patching:** If you are fixing a bug, please ensure you have isolated the root cause (and ideally written a failing test case) before applying the fix.
2. **Commit Messages:** Keep commit messages concise and descriptive. Use prefixes like `fix:`, `feat:`, `docs:`, or `test:` to clarify the purpose.
3. **Pull Requests:** Open a PR against the `main` branch. Provide a clear description of the problem and your proposed solution. Reference any relevant GitHub issues.
4. **Code Quality:** Ensure your code is well-documented and typed. We use `mypy` and standard `flake8` / `black` formatting for Python, and ESLint/Prettier for TypeScript.

Thank you for helping make HERALD a more robust and powerful phishing investigation platform!
