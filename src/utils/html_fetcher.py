# src/utils/html_fetcher.py
import requests
from urllib.parse import urlparse
import time
import logging

class HTMLFetcher:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def fetch_html(self, domain):
        """Fetch HTML content from domain"""
        try:
            # Try HTTPS first, then HTTP
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    logging.info(f"Fetching HTML from: {url}")
                    response = self.session.get(url, timeout=self.timeout, verify=False)
                    if response.status_code == 200:
                        return response.text
                except requests.exceptions.SSLError:
                    continue
                except requests.exceptions.ConnectionError:
                    continue
                except requests.exceptions.Timeout:
                    continue
                except Exception as e:
                    logging.warning(f"Failed to fetch {protocol}://{domain}: {e}")
                    continue
            return None
        except Exception as e:
            logging.error(f"Error fetching HTML for {domain}: {e}")
            return None
    
    def fetch_with_retry(self, domain, retries=2):
        """Fetch HTML with retry logic"""
        for attempt in range(retries):
            html = self.fetch_html(domain)
            if html:
                return html
            time.sleep(1)  # Wait before retry
        return None