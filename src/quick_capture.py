## this is temporary delete later

# quick_capture.py
from playwright.sync_api import sync_playwright
import os

domains = [
    "breathwellchestclinic.site",
    "campaigncanonicalcatalog.xyz",
    "ancientheroeschronicle.pro",
    "estadiounicotickets.online",
    "unicorninnovationsedge.info",
    "koreamongolianlasikclinic.com",
    "ocean-telecommunications.com",
    "catalogcanonicalautomation.xyz",
    "turbo10telecomunicacoes.fun",
    "bobbygonzalezhospitality.com",
    "nichequalifyforcapital.com",
    "twobraveunicornspodcast.com"
]

os.makedirs("final_evidence", exist_ok=True)
for domain in domains:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://{domain}", timeout=40000)
            page.pdf(path=f"final_evidence/{domain}.pdf")
            print(f"✅ Captured {domain}")
            browser.close()
    except Exception as e:
        print(f"❌ Failed {domain}: {e}")