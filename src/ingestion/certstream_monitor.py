import certstream
import json
import redis
import logging
import threading
from Levenshtein import distance
import sys
import os
import requests
import time
import redis
import logging
import threading
from Levenshtein import distance
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.cse_mapper import map_phishing_domain_to_cse

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Connect to Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logging.info("Connected to Redis queue.")
except redis.ConnectionError:
    logging.warning("Could not connect to Redis. Ensure Redis is running for production.")
    redis_client = None

# We can cache the CSE list if we need to load it
# For now, we will rely on the direct mapping logic from cse_mapper
def is_suspicious(domain):
    """
    Checks if the domain is suspicious using fuzzy matching.
    """
    # map_phishing_domain_to_cse uses exact substring matching + token_set_ratio > 80
    cse_name, official = map_phishing_domain_to_cse(domain, threshold=80)
    
    if cse_name != "Unknown CSE":
        # It's highly suspicious based on fuzzy token match or direct keyword match
        return True, cse_name
    
    # Alternatively, you can explicitly use Levenshtein distance against key domains here
    keywords = ['irctc', 'sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'iocl', 'nic', 'crsorgi']
    for kw in keywords:
        dist = distance(kw, domain.split('.')[0])
        # Allow up to 2 typos for short keywords, 3 for longer ones
        threshold_dist = 2 if len(kw) <= 5 else 3
        if dist <= threshold_dist and len(domain.split('.')[0]) < len(kw) + 3:
            return True, kw
            
    return False, None

def print_callback(message, context):
    if message['message_type'] == "heartbeat":
        return

    if message['message_type'] == "certificate_update":
        all_domains = message['data']['leaf_cert']['all_domains']
        
        for domain in all_domains:
            # Skip wildcards for matching simplicity
            clean_domain = domain.replace('*.', '')
            
            is_susp, target_cse = is_suspicious(clean_domain)
            if is_susp:
                logging.warning(f"🚨 SUSPICIOUS CERT DETECTED: {clean_domain} -> Target: {target_cse}")
                
                # Push to Redis Queue for background analysis
                if redis_client:
                    job_data = json.dumps({"domain": clean_domain, "source": "certstream", "target_cse": target_cse})
                    redis_client.rpush("domain_analysis_queue", job_data)

def poll_crt_sh():
    logging.info("Polling crt.sh as a fallback...")
    keywords = ['irctc', 'sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'iocl', 'nic', 'crsorgi']
    
    for kw in keywords:
        try:
            url = f"https://crt.sh/?q=%{kw}%&output=json"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for entry in data[:50]:  # Limit to recent 50 to avoid overloading queue
                    domain = entry.get('name_value', '')
                    for d in domain.split('\n'):
                        clean_domain = d.replace('*.', '')
                        is_susp, target_cse = is_suspicious(clean_domain)
                        if is_susp:
                            logging.warning(f"🚨 SUSPICIOUS CERT (Fallback): {clean_domain} -> Target: {target_cse}")
                            if redis_client:
                                job_data = json.dumps({"domain": clean_domain, "source": "crt.sh", "target_cse": target_cse})
                                redis_client.rpush("domain_analysis_queue", job_data)
        except Exception as e:
            logging.error(f"crt.sh polling error for {kw}: {e}")
            
def start_certstream():
    logging.info("Starting Certstream monitoring with crt.sh fallback...")
    while True:
        try:
            certstream.listen_for_events(print_callback, url='wss://certstream.calidog.io/')
        except Exception as e:
            logging.error(f"Certstream connection dropped: {e}")
        
        # Fallback
        poll_crt_sh()
        logging.info("Waiting 60 seconds before retrying WebSocket...")
        time.sleep(60)

if __name__ == "__main__":
    start_certstream()
