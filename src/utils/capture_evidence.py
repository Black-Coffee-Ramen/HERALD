# src/utils/capture_evidence.py
from playwright.sync_api import sync_playwright
import os
import time
import pandas as pd
import shutil

def capture_phishing_screenshots():
    # ✅ USE TOP 500 FILE (NOT full list!)
    predictions_csv = "outputs/high_conf_phishing_top500.csv"
    
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"Run predict.py first! Missing: {predictions_csv}")
    
    df = pd.read_csv(predictions_csv)
    phishing_domains = df['domain'].tolist()
    
    print(f"📸 Capturing TOP {len(phishing_domains)} HIGH-CONFIDENCE phishing domains...")

    # 🗑️ CLEAR OLD EVIDENCE FOLDER
    if os.path.exists("evidences_temp"):
        shutil.rmtree("evidences_temp")
    os.makedirs("evidences_temp", exist_ok=True)
    
    captured = 0
    for i, domain in enumerate(phishing_domains, 1):
        print(f"[{i}/{len(phishing_domains)}] {domain}")
        screenshot_path = f"evidences_temp/{domain}.png"
        url = f"https://{domain}"  # You can switch to "https://" if needed
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle', timeout=40000)
                time.sleep(1)  # let page render fully
                page.screenshot(path=screenshot_path, full_page=True)
                print(f"✅ Captured: {domain}")
                captured += 1
                browser.close()
        except Exception as e:
            print(f"❌ Skip {domain}: {str(e)[:50]}")
        
        # Progress update every 50
        if i % 50 == 0:
            print(f"📊 Progress: {i}/{len(phishing_domains)} | Captured: {captured}")
    
    print(f"\n🎉 FINISHED! Total captured: {captured} / {len(phishing_domains)}")

if __name__ == "__main__":
    capture_phishing_screenshots()
