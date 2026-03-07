import time
import re
import random
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import yaml
import threading

from herald.db.models import SessionLocal, DomainScan
from herald.predict_with_fallback import PhishingPredictorV3

# Basic logger
logger = logging.getLogger("SocialMonitor")
logger.setLevel(logging.INFO)

class SocialMonitor:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.social_cfg = self.config.get('social', {})
        self.channels = self.social_cfg.get('telegram_channels', [])
        self.interval = self.social_cfg.get('scrape_interval_minutes', 30)
        self.max_posts = self.social_cfg.get('max_posts_per_scrape', 50)
        
        whitelist_cfg = self.config.get('whitelist', {}).get('domains', [])
        self.whitelist = set(whitelist_cfg + [
            't.me', 'telegram.org', 'google.com', 'youtu.be', 'youtube.com',
            'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com'
        ])
        
        self.predictor = PhishingPredictorV3()
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
        ]

    def _get_headers(self):
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _extract_domain(self, url):
        try:
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return domain.lower()
        except:
            return None

    def scrape_channel(self, channel):
        url = f"https://t.me/s/{channel}"
        logger.info(f"Scraping channel: {channel}")
        
        retries = 0
        max_retries = 3
        backoff = 5
        
        while retries <= max_retries:
            try:
                resp = requests.get(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 429:
                    logger.warning(f"Rate limited on {channel}. Backing off {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    retries += 1
                    continue
                
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Telegram web preview messages
                messages = soup.find_all('div', class_='tgme_widget_message_text', limit=self.max_posts)
                
                found_urls = set()
                for msg in messages:
                    # Extract URLs from href attributes in message
                    for link in msg.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('http'):
                            found_urls.add(href)
                    
                    # Extract URLs from raw text
                    text = msg.get_text()
                    urls_in_text = re.findall(r'https?://[^\s"\'<>]+', text)
                    for u in urls_in_text:
                        found_urls.add(u)
                        
                return list(found_urls)
                
            except Exception as e:
                logger.error(f"Error scraping {channel}: {e}")
                return []
                
        return []

    def process_urls(self, urls, channel):
        session = SessionLocal()
        try:
            for url in urls:
                domain = self._extract_domain(url)
                if not domain or domain in self.whitelist:
                    continue
                    
                # Deduplicate
                existing = session.query(DomainScan).filter(DomainScan.domain == domain).first()
                if existing:
                    continue
                    
                logger.info(f"Scanning new domain from {channel}: {domain}")
                try:
                    res = self.predictor.predict(domain)
                    
                    # Store in DB
                    new_scan = DomainScan(
                        domain=domain,
                        label=res.get('status', 'Unknown'),
                        confidence=res.get('ml_confidence', 0.0),
                        is_live=True
                        # We could add source="telegram", channel=channel if DB schema allowed it
                    )
                    session.add(new_scan)
                    session.commit()
                    
                    if res['status'] == 'Phishing':
                        logger.warning(f"PHISHING DETECTED via Telegram ({channel}): {domain}")
                        
                except Exception as e:
                    logger.error(f"Error predicting {domain}: {e}")
                    session.rollback()
        finally:
            session.close()

    def run_cycle(self):
        logger.info(f"Starting social monitor cycle across {len(self.channels)} channels.")
        for channel in self.channels:
            urls = self.scrape_channel(channel)
            if urls:
                logger.info(f"Got {len(urls)} URLs from {channel}. Processing...")
                self.process_urls(urls, channel)
            
            # Rate limiting
            time.sleep(random.uniform(2.0, 3.5))

def start_social_monitor():
    monitor = SocialMonitor()
    if not monitor.channels:
        logger.warning("No Telegram channels configured. Social Monitor exiting.")
        return
        
    while True:
        try:
            monitor.run_cycle()
        except Exception as e:
            logger.error(f"Social monitor cycle crashed: {e}")
            
        sleep_secs = monitor.interval * 60
        logger.info(f"Social monitor sleeping for {sleep_secs} seconds.")
        time.sleep(sleep_secs)

if __name__ == "__main__":
    start_social_monitor()
