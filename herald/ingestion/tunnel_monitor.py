import certstream
import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from herald.core.cv_ocr_analyzer import CVOCRAnalyzer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s - %(message)s')

TUNNEL_DOMAINS = [
    ".ngrok.io", ".ngrok-free.app", ".vercel.app", ".cloudflare.com",
    ".trycloudflare.com", ".loca.lt", ".serveo.net", ".onrender.com", ".workers.dev"
]

analyzer = CVOCRAnalyzer()
executor = ThreadPoolExecutor(max_workers=3)  # Keep low to avoid massive concurrent browser spawning

TARGET_CSES = [
    "State Bank of India (SBI)", "HDFC Bank", "ICICI Bank", 
    "Indian Railway Catering and Tourism Corporation (IRCTC)",
    "National Informatics Centre (NIC)", "Punjab National Bank (PNB)",
    "Bank of Baroda (BoB)", "Airtel", "Indian Oil Corporation Limited (IOCL)"
]

def analyze_tunnel_domain(domain):
    logging.info(f"Checking tunnel domain: {domain}")
    # We check the most critical assets to see if the tunnel is spoofing them
    for cse in TARGET_CSES:
        result = analyzer.analyze_domain(domain, cse, initial_confidence=0.5)
        if result.get('cv_ocr_confirmed'):
            logging.critical(f"ðŸš¨ TUNNEL PHISHING DETECTED: {domain} masquerading as {cse}")
            return
    logging.info(f"âœ… Tunnel domain {domain} is clean.")

def print_callback(message, context):
    if message['message_type'] == "certificate_update":
        all_domains = message['data']['leaf_cert']['all_domains']
        
        for domain in all_domains:
            clean_domain = domain.replace('*.', '')
            if any(clean_domain.endswith(t_domain) for t_domain in TUNNEL_DOMAINS):
                logging.warning(f"Tunnel domain recorded: {clean_domain}")
                executor.submit(analyze_tunnel_domain, clean_domain)

def start_tunnel_monitor():
    logging.info("Starting Tunnel Domain Monitor...")
    while True:
        try:
            certstream.listen_for_events(print_callback, url='wss://certstream.calidog.io/')
        except Exception as e:
            logging.error(f"Certstream connection dropped (Tunnel Monitor): {e}")
            time.sleep(60)

if __name__ == "__main__":
    start_tunnel_monitor()
