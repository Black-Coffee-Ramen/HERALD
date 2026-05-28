import certstream
import logging
import sys
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s - %(message)s')

TUNNEL_DOMAINS = [
    ".ngrok.io", ".ngrok-free.app", ".vercel.app", ".cloudflare.com",
    ".trycloudflare.com", ".loca.lt", ".serveo.net", ".onrender.com", ".workers.dev"
]

executor = ThreadPoolExecutor(max_workers=3)  # Keep low to avoid massive concurrent browser spawning

TARGET_CSES = [
    "State Bank of India (SBI)", "HDFC Bank", "ICICI Bank", 
    "Indian Railway Catering and Tourism Corporation (IRCTC)",
    "National Informatics Centre (NIC)", "Punjab National Bank (PNB)",
    "Bank of Baroda (BoB)", "Airtel", "Indian Oil Corporation Limited (IOCL)"
]

def analyze_tunnel_domain(domain):
    logging.info(f"Checking tunnel domain: {domain}")
    from herald.core.playwright_analyzer import PlaywrightVisualAnalyzer
    analyzer = PlaywrightVisualAnalyzer()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyzer.run_analysis(domain))
        loop.close()
    except Exception as e:
        logging.error(f"Error analyzing tunnel domain {domain}: {e}")
        return
        
    if not result.get("success"):
        logging.warning(f"Failed to capture screenshot/OCR for tunnel domain {domain}: {result.get('error')}")
        return
        
    ocr_text = result.get("ocr_text", "").lower()
    
    for cse in TARGET_CSES:
        keywords = []
        if "sbi" in cse.lower() or "state bank" in cse.lower():
            keywords = ["sbi", "state bank"]
        elif "hdfc" in cse.lower():
            keywords = ["hdfc"]
        elif "icici" in cse.lower():
            keywords = ["icici"]
        elif "irctc" in cse.lower() or "railway" in cse.lower():
            keywords = ["irctc", "railway"]
        elif "nic" in cse.lower() or "informatics" in cse.lower():
            keywords = ["nic", "national informatics"]
        elif "pnb" in cse.lower() or "punjab national" in cse.lower():
            keywords = ["pnb", "punjab national"]
        elif "bob" in cse.lower() or "baroda" in cse.lower():
            keywords = ["bob", "bank of baroda"]
        elif "airtel" in cse.lower():
            keywords = ["airtel"]
        elif "iocl" in cse.lower() or "indian oil" in cse.lower():
            keywords = ["iocl", "indian oil"]
            
        if any(kw in ocr_text for kw in keywords):
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
