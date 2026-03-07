import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# ✅ Simplified and corrected dictionary of official CSE sites
CSE_SITES = {
    "NIC": "https://nic.gov.in",
    "RGCCI": "https://dc.crsorgi.gov.in",
    "IRCTC": "https://www.irctc.co.in/nget/train-search",
    "SBI": "https://onlinesbi.sbi",
    "ICICI": "https://www.icicibank.com",
    "HDFC": "https://www.hdfcbank.com",
    "PNB": "https://pnb.bank.in",
    "BoB": "https://www.bankofbaroda.in",
    "Airtel": "https://www.airtel.in",
    "IOCL": "https://iocl.com"
}


def capture_screenshots(output_dir="reference"):
    """
    Captures screenshots of official CSE websites for reference.
    Saves each screenshot as <CSE>.png inside the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configure Chrome WebDriver (headless mode)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1366,768")

    driver = webdriver.Chrome(options=chrome_options)

    for name, url in CSE_SITES.items():
        try:
            print(f"📸 Capturing {url} ({name}) ...")
            driver.get(url)
            time.sleep(10)  # wait for full page load

            safe_name = f"{name}.png"
            filepath = os.path.join(output_dir, safe_name)
            driver.save_screenshot(filepath)

            print(f"✅ Saved: {filepath}")

        except WebDriverException as e:
            print(f"❌ Failed to capture {url} ({name}): {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error for {url} ({name}): {e}")

    driver.quit()
    print("\n🎉 All screenshots captured successfully!")


if __name__ == "__main__":
    capture_screenshots()
