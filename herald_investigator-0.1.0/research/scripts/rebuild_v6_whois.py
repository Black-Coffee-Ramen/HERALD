import os
import re
import pandas as pd
import datetime
import time
from tqdm import tqdm

# Import get_domain_age_days logic from the existing script to maintain consistency
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from enrich_with_whois import get_domain_age_days

def rebuild_v6_whois():
    print("=" * 60)
    print("Rebuilding HERALD v6 WHOIS Data")
    print("=" * 60)

    # Paths
    v5_path = 'data/processed/full_dataset_v5.csv'
    v6_ref_path = 'data/processed/full_features_v6.csv'
    v5_whois_path = 'data/processed/full_dataset_v5_whois.csv'

    if not os.path.exists(v5_path):
        print(f"ERROR: {v5_path} not found.")
        return

    # 1. Load v5 dataset
    df_v5 = pd.read_csv(v5_path)
    print(f"Loaded {len(df_v5)} domains from {v5_path}")

    # 2. Identify the 55 Indian domains (same logic as add_indian_legitimate.py)
    indian_legitimate = [
        "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "pnbindia.in", 
        "bankofbaroda.in", "unionbankofindia.co.in", "canarabank.com", "kotakbank.com", 
        "indusind.com", "yesbank.in", "federalbank.co.in", "idfcfirstbank.com",
        "india.gov.in", "mygov.in", "nic.in", "incometax.gov.in", "gst.gov.in", 
        "uidai.gov.in", "irctc.co.in", "indianrailways.gov.in", "passportindia.gov.in", 
        "digilocker.gov.in", "epfindia.gov.in", "esic.in", "airtel.in", "jio.com", 
        "bsnl.co.in", "vi.in", "npci.org.in", "phonepe.com", "paytm.com", 
        "razorpay.com", "groww.in", "zerodha.com", "cred.club", "bharatpe.com", 
        "mobikwik.com", "licindia.in", "hdfclife.com", "iciciprulife.com", 
        "starhealth.in", "bajajfinserv.in", "iocl.com", "bpcl.in", "hpcl.com",
        "timesofindia.com", "ndtv.com", "hindustantimes.com", "thehindu.com",
        "moneycontrol.com", "economictimes.com"
    ]
    
    extracted_from_templates = [
        "airtel.in", "bankofbaroda.bank.in", "dc.crsorgi.gov.in", "hdfcbank.com",
        "hdfcergo.com", "hdfclife.com", "iocl.com", "mgovcloud.in", "email.gov.in",
        "pnb.bank.in", "onlinesbi.sbi.bank.in", "sbi.bank.in", "sbicard.com", "sbilife.co.in"
    ]
    
    indian_domains = set([d.lower() for d in indian_legitimate + extracted_from_templates])
    print(f"Targeting {len(indian_domains)} unique Indian legitimate domains for age=3650.")

    # 3. Load existing WHOIS data from v6 reference
    if os.path.exists(v6_ref_path):
        df_v6 = pd.read_csv(v6_ref_path)
        # Create a mapping of domain -> domain_age_days
        whois_map = pd.Series(df_v6.domain_age_days.values, index=df_v6.domain).to_dict()
        print(f"Loaded existing WHOIS data for {len(whois_map)} domains from {v6_ref_path}")
    else:
        whois_map = {}
        print(f"WARNING: {v6_ref_path} not found. Will query any non-Indian domains.")

    # 4. Enrich
    new_ages = []
    to_query = []
    
    for domain in df_v5['domain']:
        if domain.lower() in indian_domains:
            new_ages.append(3650)
        elif domain in whois_map:
            new_ages.append(whois_map[domain])
        else:
            new_ages.append(None) # Mark for query
            to_query.append(domain)

    df_v5['domain_age_days'] = new_ages
    
    print(f"Applied estimates to Indian domains and reused existing data.")
    print(f"New domains requiring WHOIS query: {len(to_query)}")

    # 5. Query remaining domains
    if to_query:
        print(f"Processing {len(to_query)} new unique domains...")
        for i, domain in enumerate(tqdm(to_query, desc="WHOIS Queries")):
            age = get_domain_age_days(domain)
            df_v5.loc[df_v5['domain'] == domain, 'domain_age_days'] = age
            # Rate limiting
            time.sleep(0.5)
            # Periodic save (to temporary path just in case)
            if (i + 1) % 50 == 0:
                df_v5.to_csv(v5_whois_path + ".tmp", index=False)

    # 6. Final Polish
    df_v5['domain_age_days'] = df_v5['domain_age_days'].fillna(-1).astype(int)
    
    # Save final
    df_v5.to_csv(v5_whois_path, index=False)
    print(f"\nSaved enriched dataset to {v5_whois_path}")
    
    if os.path.exists(v5_whois_path + ".tmp"):
        os.remove(v5_whois_path + ".tmp")

if __name__ == '__main__':
    rebuild_v6_whois()
