# src/utils/pdf_generator.py
from playwright.sync_api import sync_playwright
import os
import time

def take_screenshot_as_pdf(url: str, output_path: str, timeout: int = 10000):
    """
    Visit URL and save full-page screenshot as PDF.
    Args:
        url: Full URL (e.g., "http://example.com")
        output_path: Path to save PDF (e.g., "evidences_temp/example.com.pdf")
        timeout: Max load time in ms
    """
    # Ensure URL has scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until='networkidle', timeout=timeout)
            # Wait a bit for dynamic content
            time.sleep(2)
            # Save as PDF
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            page.pdf(path=output_path, format='A4')
            print(f"✅ Saved PDF: {output_path}")
        except Exception as e:
            print(f"❌ Failed to capture {url}: {str(e)}")
            # Create empty PDF or skip
        finally:
            browser.close()

# Example usage
if __name__ == "__main__":
    take_screenshot_as_pdf("airtelpoint.top", "evidences_temp/airtelpoint.top.pdf")