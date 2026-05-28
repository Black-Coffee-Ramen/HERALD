import os
import re
import pandas as pd

def add_indian_legitimate():
    # 1. Definitively legitimate Indian domains
    indian_legitimate = [
        # Banking
        "sbi.co.in", "hdfcbank.com", "icicibank.com",
        "axisbank.com", "pnbindia.in", "bankofbaroda.in",
        "unionbankofindia.co.in", "canarabank.com",
        "kotakbank.com", "indusind.com", "yesbank.in",
        "federalbank.co.in", "idfcfirstbank.com",
        # Government
        "india.gov.in", "mygov.in", "nic.in",
        "incometax.gov.in", "gst.gov.in", "uidai.gov.in",
        "irctc.co.in", "indianrailways.gov.in",
        "passportindia.gov.in", "digilocker.gov.in",
        "epfindia.gov.in", "esic.in",
        # Telecom
        "airtel.in", "jio.com", "bsnl.co.in", "vi.in",
        # Fintech
        "npci.org.in", "phonepe.com", "paytm.com",
        "razorpay.com", "groww.in", "zerodha.com",
        "cred.club", "bharatpe.com", "mobikwik.com",
        # Insurance
        "licindia.in", "hdfclife.com", "iciciprulife.com",
        "starhealth.in", "bajajfinserv.in",
        # Oil/Energy
        "iocl.com", "bpcl.in", "hpcl.com",
        # News
        "timesofindia.com", "ndtv.com",
        "hindustantimes.com", "thehindu.com",
        "moneycontrol.com", "economictimes.com",
    ]

    # 2. Extract from data/templates/
    template_dir = "data/templates"
    extracted_domains = []
    
    # Regex to find last segment before .png that contains a dot
    # Matches patterns like HDFC_Group_hdfcbank.com.png
    regex = r"_([a-z0-9.-]+\.[a-z]{2,})\.png$"
    
    if os.path.exists(template_dir):
        files = [f for f in os.listdir(template_dir) if f.endswith(".png")]
        for f in files:
            match = re.search(regex, f, re.IGNORECASE)
            if match:
                domain = match.group(1).strip("_")
                if domain:
                    extracted_domains.append(domain.lower())
    
    print(f"Extracted {len(extracted_domains)} domains from templates: {set(extracted_domains)}")
    
    # Combine and deduplicate
    all_new_domains = list(set(indian_legitimate + extracted_domains))
    
    # 3. Deduplicate against existing data/external/tranco_legitimate.csv
    tranco_path = "data/external/tranco_legitimate.csv"
    if os.path.exists(tranco_path):
        existing_df = pd.read_csv(tranco_path)
        existing_domains = set(existing_df['domain'].str.lower().tolist())
    else:
        existing_df = pd.DataFrame(columns=['domain', 'label'])
        existing_domains = set()
    
    to_add = [d for d in all_new_domains if d not in existing_domains]
    
    # 4. Append and save
    if to_add:
        new_data = pd.DataFrame({
            'domain': to_add,
            'label': 'Legitimate'
        })
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        updated_df.to_csv(tranco_path, index=False)
        print(f"Added {len(to_add)} new Indian legitimate domains")
    else:
        print("No new domains to add.")

if __name__ == "__main__":
    add_indian_legitimate()
