import time
import requests
import logging
import sys

# Configure basic logging to both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("health_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HealthMonitor")

API_URL = "http://127.0.0.1:8000/api/health"

def send_email_alert(subject, body):
    """
    Placeholder for SMTP email alert logic.
    You will need to configure SMTP_USER, SMTP_PASS, etc.
    """
    logger.error(f"*** ALERT NOTIFICATION: {subject} ***")
    # import smtplib
    # from email.message import EmailMessage
    # msg = EmailMessage()
    # msg.set_content(body)
    # ...

def check_health():
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code != 200:
            logger.error(f"Health check failed with status code: {response.status_code}")
            send_email_alert(f"API Health Check Failed: {response.status_code}", response.text)
            return

        data = response.json()
        if data.get("status") != "healthy":
            logger.warning(f"System degraded: {data}")
            send_email_alert("System Degraded", str(data))
        else:
            logger.info("System healthy.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Health check request failed: {e}")
        send_email_alert("API Offline", str(e))

if __name__ == "__main__":
    logger.info("Running health check poll...")
    check_health()
