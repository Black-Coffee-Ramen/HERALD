import ast
import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

import cv2
import structlog
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = structlog.get_logger(__name__)

DEFAULT_BROWSER_TIMEOUT_SECONDS = int(os.getenv("BROWSER_TIMEOUT_SECONDS", "20"))
ALLOW_PRIVATE_NETWORKS = os.getenv("BROWSER_ALLOW_PRIVATE_NETWORKS", "false").lower() == "true"
ALLOW_UNSAFE_SANDBOX = os.getenv("BROWSER_ALLOW_UNSAFE_NO_SANDBOX", "false").lower() == "true"


class CVOCRAnalyzer:
    def __init__(self):
        self.reader = self._setup_ocr_reader()
        self.chromedriver_paths = [
            os.getenv("CHROMEDRIVER_PATH", ""),
            "/usr/local/bin/chromedriver",
            "/usr/bin/chromedriver",
            "chromedriver",
            "C:/Users/athiy/chromedriver-win64/chromedriver.exe",
            "chromedriver.exe",
        ]

    def _setup_ocr_reader(self):
        if not EASYOCR_AVAILABLE:
            logger.warning("easyocr_unavailable")
            return None

        try:
            reader = easyocr.Reader(["en"])
            logger.info("easyocr_initialized")
            return reader
        except Exception as exc:
            logger.warning("easyocr_initialization_failed", error=str(exc))
            return None

    def setup_chromedriver(self):
        options = self._build_chrome_options()

        for driver_path in self.chromedriver_paths:
            if not driver_path:
                continue

            try:
                service = ChromeService(executable_path=driver_path)
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(DEFAULT_BROWSER_TIMEOUT_SECONDS)
                driver.set_script_timeout(DEFAULT_BROWSER_TIMEOUT_SECONDS)
                logger.info("chromedriver_ready", driver_path=driver_path)
                return driver
            except Exception as exc:
                logger.warning("chromedriver_path_failed", driver_path=driver_path, error=str(exc))

        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(DEFAULT_BROWSER_TIMEOUT_SECONDS)
            driver.set_script_timeout(DEFAULT_BROWSER_TIMEOUT_SECONDS)
            logger.info("chromedriver_ready", driver_path="system")
            return driver
        except Exception as exc:
            logger.error("chromedriver_unavailable", error=str(exc))
            return None

    def _build_chrome_options(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1365,768")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--password-store=basic")
        options.add_argument("--use-mock-keychain")
        options.add_argument("--user-data-dir=/tmp/herald-chrome")

        if ALLOW_UNSAFE_SANDBOX:
            options.add_argument("--no-sandbox")
            logger.warning("browser_no_sandbox_enabled")

        return options

    def is_domain_reachable(self, domain):
        try:
            clean_domain = self._extract_hostname(domain)
            for result in socket.getaddrinfo(clean_domain, None):
                ip_address = ipaddress.ip_address(result[4][0])
                if not ALLOW_PRIVATE_NETWORKS and not ip_address.is_global:
                    logger.warning("browser_private_network_blocked", domain=domain, ip=str(ip_address))
                    return False
            return True
        except (socket.gaierror, ValueError) as exc:
            logger.info("browser_domain_unreachable", domain=domain, error=str(exc))
            return False

    def capture_screenshot(self, domain, driver):
        if not self.is_domain_reachable(domain):
            return None

        for protocol in ["https", "http"]:
            url = f"{protocol}://{self._extract_hostname(domain)}"
            try:
                logger.info("browser_navigation_started", url=url)
                driver.get(url)
                time.sleep(1)

                safe_domain = self._safe_domain_path(domain)
                path = f"evidence/{safe_domain}/screenshot.png"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                driver.save_screenshot(path)
                logger.info("browser_screenshot_captured", domain=domain, path=path)
                return path
            except (TimeoutException, WebDriverException) as exc:
                logger.warning("browser_navigation_failed", url=url, error=str(exc))

        return None

    def perceptual_hash(self, img_path, hash_size=8):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = cv2.resize(img, (hash_size + 1, hash_size))
            diff = img[1:, :] > img[:-1, :]
            return sum([2**i if val else 0 for i, val in enumerate(diff.flatten())])
        except Exception as exc:
            logger.warning("perceptual_hash_failed", img_path=img_path, error=str(exc))
            return None

    def hamming_distance(self, hash1, hash2):
        return bin(hash1 ^ hash2).count("1") if hash1 and hash2 else 100

    def extract_text_ocr(self, img_path):
        if not EASYOCR_AVAILABLE or not self.reader:
            return []
        try:
            results = self.reader.readtext(img_path)
            return [result[1].lower() for result in results]
        except Exception as exc:
            logger.warning("ocr_extract_failed", img_path=img_path, error=str(exc))
            return []

    def find_best_template(self, cse_name):
        templates_dir = "data/templates"
        if not os.path.exists(templates_dir):
            return None

        template_files = [f for f in os.listdir(templates_dir) if f.endswith(".png")]
        if not template_files:
            return None

        cse_normalized = cse_name.lower().replace("(", "").replace(")", "").replace("&", "and").replace(" ", "_")
        best_match = None
        best_score = -1

        for template_file in template_files:
            template_path = os.path.join(templates_dir, template_file)
            template_lower = template_file.lower().replace(".png", "")
            score = 0
            cse_keywords = ["irctc", "sbi", "state_bank", "icici", "hdfc", "pnb", "bob", "bank_of_baroda", "airtel", "iocl", "nic", "rgcci"]

            for keyword in cse_keywords:
                if keyword in cse_normalized and keyword in template_lower:
                    score += 3

            if cse_normalized in template_lower:
                score += 4

            if score > best_score:
                best_score = score
                best_match = template_path

        return best_match

    def analyze_domain(self, domain, cse_name, initial_confidence):
        logger.info("visual_domain_analysis_started", domain=domain, target_cse=cse_name)

        driver = self.setup_chromedriver()
        if not driver:
            return self._result(False, "ChromeDriver not available", initial_confidence, "No analysis")

        try:
            screenshot_path = self.capture_screenshot(domain, driver)
            if not screenshot_path:
                return self._result(False, "Unable to capture screenshot", initial_confidence, "No screenshot")

            extracted_text = self.extract_text_ocr(screenshot_path)
            text_match = any(pattern in " ".join(extracted_text).lower() for pattern in ["login", "password", "username", "signin"])

            template_path = self.find_best_template(cse_name)
            visual_match = False
            visual_distance = 100

            if template_path and os.path.exists(template_path):
                hash1 = self.perceptual_hash(screenshot_path)
                hash2 = self.perceptual_hash(template_path)
                visual_distance = self.hamming_distance(hash1, hash2)
                visual_match = visual_distance <= 20

            phishing_score = 0
            if text_match:
                phishing_score += 3
            if visual_match:
                phishing_score += 4

            is_phishing = phishing_score >= 4
            indicators = {"text_match": text_match, "visual_match": visual_match, "score": phishing_score}
            logger.info("visual_domain_analysis_finished", domain=domain, confirmed=is_phishing, score=phishing_score, visual_distance=visual_distance)

            return {
                "cv_ocr_confirmed": is_phishing,
                "cv_ocr_status": "Confirmed" if is_phishing else "Not Confirmed",
                "final_confidence": 1.0 if is_phishing else initial_confidence * 0.7,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phishing_indicators": str(indicators),
                "visual_similarity": f"distance_{visual_distance}",
            }
        finally:
            driver.quit()

    @staticmethod
    def parse_indicators(raw_indicators: str) -> dict:
        try:
            parsed = ast.literal_eval(raw_indicators)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    @staticmethod
    def _extract_hostname(domain: str) -> str:
        parsed = urlparse(domain if "://" in domain else f"http://{domain}")
        return parsed.hostname or domain.split("/")[0]

    @staticmethod
    def _safe_domain_path(domain: str) -> str:
        return "".join(char if char.isalnum() or char in ".-" else "_" for char in domain)[:255]

    @staticmethod
    def _result(confirmed: bool, status: str, confidence: float, visual_similarity: str) -> dict:
        return {
            "cv_ocr_confirmed": confirmed,
            "cv_ocr_status": status,
            "final_confidence": confidence,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phishing_indicators": "{}",
            "visual_similarity": visual_similarity,
        }
