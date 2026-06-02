## Installation

### Prerequisites

* Python 3.12+
* Node.js 18+ (Frontend only)
* Tesseract OCR (required for OCR text extraction)
* PostgreSQL development libraries (`libpq-dev`)
* Playwright browser dependencies

---

### Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install Python dependencies:

```bash
pip install -r requirements-runtime.txt
```

Install HERALD:

```bash
pip install -e .
```

Install Playwright browsers:

```bash
playwright install
```

Verify installation:

```bash
herald --help
```

Expected commands:

```text
investigate
analyze
screenshot
report
```

---

### System Dependencies

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr libpq-dev
```

#### macOS

```bash
brew install tesseract
```

Tesseract enables OCR extraction from captured screenshots. Without it, screenshot capture still works but OCR extraction is skipped.

---

### Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Frontend available at:

```text
http://localhost:3000
```

The frontend defaults to mock/synthetic telemetry. Set:

```bash
NEXT_PUBLIC_TELEMETRY_MODE=REAL
```

and run the API backend to connect live investigation data.

---

### Docker

Build and start the platform:

```bash
docker compose up --build
```

---

### Troubleshooting

#### `Command 'herald' not found`

Make sure HERALD itself is installed:

```bash
pip install -e .
```

#### `ModuleNotFoundError: No module named 'rich'`

Install Rich:

```bash
pip install rich
```

If this occurs, add `rich` to `requirements-runtime.txt` and reinstall dependencies.

#### Playwright Browser Errors

Reinstall browser binaries:

```bash
playwright install
```

#### Verify Installation

```bash
which herald
pip show herald
herald --help
```

The `herald` executable should resolve inside your virtual environment.
