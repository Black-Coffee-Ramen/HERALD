import requests
import zipfile
import io
import os
import redis
import logging
from datetime import datetime
import time
import sys
from Levenshtein import distance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from herald.utils.cse_mapper import map_phishing_domain_to_cse

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Connect to Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logging.info("Connected to Redis queue.")
except redis.ConnectionError:
    logging.warning("Could not connect to Redis. Ensure Redis is running for production.")
    redis_client = None

def download_and_extract_daily_domains(url="https://whoisds.com/whois-database/newly-registered-domains"):
    """
    Mock/example function to download a daily domain feed.
    In reality, WhoisDS uses Base64 encoded URLs on their site that change daily.
    For this implementation, we will mock the content retrieval if the direct download fails,
    or process an existing dataset.
    """
    logging.info("Simulating daily domain feed fetch...")
    
    # In a full implementation without a paid API, we would scrape the base64 link from the HTML:
    # response = requests.get(url)
    # soup = BeautifulSoup(response.text, 'html.parser')
    # ... parse the specific download button for today ...
    
    # For demonstration/offline capability, return a mock list of domains representing 
    # a feed for the day.
    return [
        "sbi-online-kyc-update.com",
        "hdfcbank-rewards-points.in",
        "legitwebsite.org",
        "airtel-5g-upgrade.xyz",
        "irctc-ticket-booking.net"
    ]

def is_suspicious(domain):
    cse_name, official = map_phishing_domain_to_cse(domain, threshold=80)
    
    if cse_name != "Unknown CSE":
        return True, cse_name
    
    keywords = ['irctc', 'sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'iocl', 'nic', 'crsorgi']
    
    # Simple direct keyword extraction + Levenshtein checking
    domain_stripped = domain.lower().split('.')[0]
    for kw in keywords:
        # Check substring match
        if kw in domain_stripped:
            return True, kw
        # Check Levenshtein distance
        dist = distance(kw, domain_stripped)
        threshold_dist = 2 if len(kw) <= 5 else 3
        if dist <= threshold_dist and len(domain_stripped) < len(kw) + 3:
            return True, kw
            
    return False, None

def poll_daily_domains():
    logging.info(f"Starting daily domain poll at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    domains = download_and_extract_daily_domains()
    
    for domain in domains:
        suspicious, target_cse = is_suspicious(domain)
        if suspicious:
            logging.warning(f"ðŸš¨ NEW DOMAIN DETECTED: {domain} -> Target: {target_cse}")
            
            if redis_client:
                import json
                job_data = json.dumps({"domain": domain, "source": "daily_feed", "target_cse": target_cse})
                redis_client.rpush("domain_analysis_queue", job_data)

def start_polling(interval_hours=24):
    while True:
        poll_daily_domains()
        logging.info(f"Sleeping for {interval_hours} hours...")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    start_polling(interval_hours=24)
