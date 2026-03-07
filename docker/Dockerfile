FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    unzip \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (Modern Method)
RUN curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver using Chrome for Testing (modern versions 115+)
RUN apt-get update && apt-get install -y jq \
    && CHROME_VERSION=$(google-chrome --version | cut -d ' ' -f 3 | cut -d '.' -f 1-3) \
    && CHROMEDRIVER_URL=$(curl -s "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json" | jq -r ".channels.Stable.downloads.chromedriver[] | select(.platform == \"linux64\") | .url") \
    && wget -O /tmp/chromedriver.zip "$CHROMEDRIVER_URL" \
    && unzip /tmp/chromedriver.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver* /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies - Optimization: Use CPU-only versions for ML libraries
# This significantly reduces image size (from GBs to MBs)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Install Phase 5 / Extension dependencies
# Ensuring opencv-python-headless is used to avoid GUI dependencies
RUN pip install --no-cache-dir \
    certstream \
    Levenshtein \
    redis \
    APScheduler \
    sqlalchemy \
    psycopg2-binary \
    fastapi \
    uvicorn \
    aiodns \
    python-dotenv \
    fuzzywuzzy \
    easyocr \
    opencv-python-headless

# Copy application code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports (for documentation)
EXPOSE 8000 8501
