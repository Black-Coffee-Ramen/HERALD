"""
scripts/enrich_with_whois.py

Enriches the HERALD dataset with WHOIS creation date and computes domain age.
Includes rate limiting and checkpointing.
"""

import os
import sys
import pandas as pd
import datetime
import time
import re
from tqdm import tqdm
from whois.whois import NICClient

def get_domain_age_days(domain):
    """Query WHOIS using direct sockets and regex. Return -1 on failure/not-found."""
    try:
        nic = NICClient()
        # Direct lookup via socket (usually port 43)
        raw_text = nic.whois_lookup(None, domain, 0, quiet=True, timeout=5)
        
        if not raw_text or "not responding" in raw_text.lower():
            return -1
            
        # Regex patterns for various WHOIS formats
        patterns = [
            r'Creation Date:\s*([0-9-]{10}T[0-9:]{8}Z)',        # Standard 
            r'creation date:\s*([0-9-]{10}T[0-9:]{8}Z)',        # Standard lower
            r'Creation Date:\s*([0-9-]{10})',                  # Date only
            r'created:\s*([0-9-]{10})',                        # Alternative
            r'Registered on:\s*([0-9-]{10})',                  # Another one
            r'\[Created on\]\s*([0-9]{4}/[0-9]{2}/[0-9]{2})',   # JP etc
            r'Creation Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})',  # Common ISO
            r'Registration Time:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})' # CN etc
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Parse YYYY-MM-DD
                try:
                    # Handle T separator if present
                    clean_date = date_str.split('T')[0].replace('/', '-')
                    creation_date = datetime.datetime.strptime(clean_date, '%Y-%m-%d')
                    age = (datetime.datetime.now() - creation_date).days
                    return age
                except ValueError:
                    continue
                    
        return -1
    except Exception:
        return -1

def main():
    print("=" * 60)
    print("HERALD WHOIS Enrichment Tool")
    print("=" * 60)

    in_path = 'data/processed/full_features_v5.csv'
    out_path = 'data/processed/full_features_v6.csv'
    checkpoint_path = 'data/processed/whois_checkpoint.csv'

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found.")
        return

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} domains from {in_path}")

    # Initialize or load checkpoint
    if os.path.exists(checkpoint_path):
        df_checkpoint = pd.read_csv(checkpoint_path)
        # Merge to see what's already done
        df = df.merge(df_checkpoint[['domain', 'domain_age_days']], on='domain', how='left')
        print(f"Resuming from checkpoint: {df['domain_age_days'].notna().sum()} domains already processed.")
    else:
        df['domain_age_days'] = None

    # Process unique domains to save queries (though Tranco/Phish usually unique here)
    unique_domains = df[df['domain_age_days'].isna()]['domain'].unique()
    print(f"Processing {len(unique_domains)} new unique domains...")

    # Dictionary to store results for unique domains
    results = {}
    
    start_time = time.time()
    for i, domain in enumerate(tqdm(unique_domains, desc="WHOIS Queries")):
        age = get_domain_age_days(domain)
        results[domain] = age
        
        # Rate limiting: 0.5s sleep between queries
        time.sleep(0.5)
        
        # Save checkpoint every 100 domains
        if (i + 1) % 100 == 0:
            # Update the main DF and save
            for d, a in results.items():
                df.loc[df['domain'] == d, 'domain_age_days'] = a
            df[df['domain_age_days'].notna()].to_csv(checkpoint_path, index=False)
            
            elapsed = time.time() - start_time
            print(f"\nProgress: {i+1}/{len(unique_domains)} | Elapsed: {elapsed/60:.2f}m")
            # Clear dictionary to save memory once checkpointed
            results = {}

    # Final update
    for d, a in results.items():
        df.loc[df['domain'] == d, 'domain_age_days'] = a

    # Fill any remaining NaNs (shouldn't be any) with -1
    df['domain_age_days'] = df['domain_age_days'].fillna(-1).astype(int)

    df.to_csv(out_path, index=False)
    print(f"\nSaved enriched dataset to {out_path}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file.")

if __name__ == '__main__':
    main()
