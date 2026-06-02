import os
import time
import asyncio
import re
import structlog
from typing import Dict, Any, Optional

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    import platform
    if platform.system() == 'Windows':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Default timeouts
PAGE_LOAD_TIMEOUT = int(os.getenv("PLAYWRIGHT_PAGE_LOAD_TIMEOUT", "20000")) # 20 seconds
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "evidence")


def _safe_error(exc: Exception) -> str:
    return str(exc).encode("ascii", "replace").decode("ascii")


SUSPICIOUS_OCR_PATTERNS = [
    ("verify your account", r"\bverif(?:y|ication)\s+(?:your\s+)?account\b", 25),
    ("verify your identity", r"\bverif(?:y|ication)\s+(?:your\s+)?identit(?:y|ies)\b", 25),
    ("confirm your identity", r"\bconfirm\s+(?:your\s+)?identit(?:y|ies)\b", 25),
    ("sign in to your account", r"\bsign\s*in(?:\s+to)?(?:\s+your)?\s+account\b", 20),
    ("login required", r"\b(?:log\s*in|login|sign\s*in)\s+(?:required|now|here|to\s+continue)\b", 20),
    ("enter your password", r"\b(?:enter|type|provide|input)\s+(?:your\s+)?password\b", 25),
    ("password", r"\bpassword\b", 15),
    ("one time password", r"\b(?:one\s*time\s*password|otp)\b", 25),
    ("account suspended", r"\baccount\s+(?:has\s+been\s+)?(?:suspended|blocked|locked|restricted|disabled)\b", 30),
    ("unauthorized access", r"\bunauthori[sz]ed\s+access\b", 25),
    ("security alert", r"\bsecurity\s+(?:alert|warning|notice|verification|check)\b", 20),
    ("update kyc", r"\b(?:update|complete|verify)\s+(?:your\s+)?kyc\b", 30),
    ("update billing", r"\b(?:update|confirm|verify)\s+(?:your\s+)?billing\b", 25),
    ("card number", r"\b(?:(?:enter|type|provide|input)\s+(?:your\s+)?(?:debit\s+|credit\s+)?card|(?:debit\s+|credit\s+)?card\s+number)\b", 20),
    ("cvv", r"\b(?:cvv|cvc|security\s+code)\b", 25),
    ("expiry date", r"\bexpir(?:y|ation)\s+date\b", 15),
    ("net banking", r"\bnet\s*banking\b", 15),
    ("limited time", r"\b(?:limited\s+time|act\s+now|immediate\s+action)\b", 15),
    ("suspected phishing", r"\bsuspected\s+phishing\b", 35),
    ("site dangerous", r"\bfound\s+this\s+site\s+dangerous\b", 35),
    ("isp blocked", r"\bblocked(?:!|\.)?\s+(?:potentially\s+risky\s+website|due\s+to\s+security)\b", 30),
]


def _normalize_ocr_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("|", "l")
    normalized = normalized.replace("0tp", "otp")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class PlaywrightVisualAnalyzer:
    def __init__(self, evidence_dir: str = EVIDENCE_DIR):
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        
        # Configure pytesseract path for Windows if needed
        # Adjust path if tesseract is installed in a different location
        if PYTESSERACT_AVAILABLE and os.name == 'nt':
            tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            else:
                logger.warning("tesseract_not_found_in_default_path")

    def _safe_domain_path(self, domain: str) -> str:
        return "".join(char if char.isalnum() or char in ".-" else "_" for char in domain)[:255]

    async def capture_screenshot(self, target: str) -> Optional[str]:
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("playwright_unavailable")
            return None

        screenshot_dir = os.path.join(self.evidence_dir, "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, "homepage.png")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Isolate context completely
                context = await browser.new_context(
                    viewport={'width': 1366, 'height': 768},
                    ignore_https_errors=True,
                    java_script_enabled=True
                )
                page = await context.new_page()

                urls_to_try = [target] if target.startswith("http") else [f"https://{target}", f"http://{target}"]

                for url in urls_to_try:
                    try:
                        logger.info("playwright_navigation_started", url=url)
                        await page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until="domcontentloaded")
                        await page.wait_for_timeout(1000)
                        await page.screenshot(path=screenshot_path, full_page=True)
                        await browser.close()
                        logger.info("playwright_screenshot_captured", target=target, path=screenshot_path)
                        return screenshot_path
                    except PlaywrightTimeoutError:
                        logger.warning("playwright_navigation_timeout", url=url)
                    except Exception as exc:
                        logger.warning("playwright_navigation_failed", url=url, error=_safe_error(exc))

                await browser.close()
                return None
                
        except PlaywrightTimeoutError:
            return None
        except Exception as exc:
            logger.warning("playwright_capture_failed", target=target, error=_safe_error(exc))
            return None

    def extract_text(self, image_path: str) -> str:
        if not PYTESSERACT_AVAILABLE or not os.path.exists(image_path):
            return ""
            
        try:
            # Optional: Preprocess image with OpenCV to improve OCR accuracy
            # img = cv2.imread(image_path)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # text = pytesseract.image_to_string(gray)
            
            text = pytesseract.image_to_string(Image.open(image_path))
            return text.strip()
        except Exception as exc:
            logger.warning("ocr_extraction_failed", image_path=image_path, error=_safe_error(exc))
            return ""

    def check_suspicious_phrases(self, text: str) -> Dict[str, Any]:
        normalized_text = _normalize_ocr_text(text or "")
        found: list[str] = []
        score = 0

        for label, pattern, weight in SUSPICIOUS_OCR_PATTERNS:
            if re.search(pattern, normalized_text):
                found.append(label)
                score += weight

        # A lone low-signal term such as "password" is useful evidence, but
        # require a stronger aggregate before OCR alone marks the page suspicious.
        is_suspicious = score >= 20
        
        return {
            "is_suspicious": is_suspicious,
            "phrases_found": found,
            "ocr_risk_score": min(100, score)
        }

    async def run_analysis(self, domain: str) -> Dict[str, Any]:
        screenshot_path = await self.capture_screenshot(domain)
        
        if not screenshot_path:
            return {
                "success": False,
                "screenshot_path": None,
                "ocr_text": None,
                "ocr_findings": {},
                "error": "Screenshot failed"
            }
            
        ocr_text = self.extract_text(screenshot_path)
        findings = self.check_suspicious_phrases(ocr_text)
        
        return {
            "success": True,
            "screenshot_path": screenshot_path,
            "ocr_text": ocr_text,
            "ocr_findings": findings
        }
