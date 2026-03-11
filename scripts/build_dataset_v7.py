"""
scripts/build_dataset_v7.py

Consolidates URLs and domains from various sources:
- OpenPhish
- URLhaus
- PhishTank (if available)
- Tranco
- Majestic Million
- Existing v5 training data

Applies cleaning and deduplication.
"""

import os
import pandas as pd
from urllib.parse import urlparse
import re

def extract_domain(url):
    try:
        if not url or pd.isna(url):
            return None
        url = str(url).strip()
        if '//' not in url:
            url = '//' + url
        domain = urlparse(url).netloc.lower()
        # Remove port if any
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain
    except:
        return None

def clean_domain(domain):
    if not domain:
        return None
    domain = domain.lower().strip()
    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    # Remove trailing dots
    domain = domain.rstrip('.')
    # Remove rows where domain is IP address
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain):
        return None
    # Remove domains shorter than 4 characters
    if len(domain) < 4:
        return None
    return domain

def main():
    print("=" * 50)
    print("HERALD v7 Dataset Builder")
    print("=" * 50)
    
    data_sources = []
    
    # 1. OpenPhish (Phishing)
    if os.path.exists('data/external/openphish.txt'):
        with open('data/external/openphish.txt', 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        domains = [clean_domain(extract_domain(u)) for u in urls]
        data_sources.append(pd.DataFrame({'domain': domains, 'label': 'Phishing', 'source': 'OpenPhish'}))
        print(f"OpenPhish: Added {len(urls)} URLs.")

    # 2. URLhaus (Phishing)
    if os.path.exists('data/external/urlhaus.csv'):
        df_uh = pd.read_csv('data/external/urlhaus.csv', comment='#', names=['id', 'dateadded', 'url', 'url_status', 'last_online', 'threat', 'tags', 'url_sha256', 'reporter'])
        # Filter for online or unknown
        df_uh = df_uh[df_uh['url_status'].isin(['online', 'unknown'])]
        df_uh['domain'] = df_uh['url'].apply(extract_domain).apply(clean_domain)
        data_sources.append(df_uh[['domain']].assign(label='Phishing', source='URLhaus'))
        print(f"URLhaus: Added {len(df_uh)} URLs.")

    # 3. PhishTank (Phishing)
    if os.path.exists('data/external/phishtank_online.csv'):
        df_pt = pd.read_csv('data/external/phishtank_online.csv')
        df_pt['domain'] = df_pt['url'].apply(extract_domain).apply(clean_domain)
        data_sources.append(df_pt[['domain']].assign(label='Phishing', source='PhishTank'))
        print(f"PhishTank: Added {len(df_pt)} URLs.")

    # 4. Tranco (Legitimate)
    # Re-extract if not saved, but for now we'll just re-extract as domains
    tranco_path = 'data/external/tranco.zip'
    if os.path.exists(tranco_path):
        import zipfile
        with zipfile.ZipFile(tranco_path, 'r') as z:
            with z.open('top-1m.csv') as f:
                df_tr = pd.read_csv(f, names=['rank', 'domain'])
        df_tr = df_tr[(df_tr['rank'] >= 1000) & (df_tr['rank'] <= 101000)]
        df_tr['domain'] = df_tr['domain'].apply(clean_domain)
        data_sources.append(df_tr[['domain']].assign(label='Legitimate', source='Tranco'))
        print(f"Tranco: Added {len(df_tr)} domains.")

    # 5. Majestic Million (Legitimate)
    if os.path.exists('data/external/majestic_million.csv'):
        df_mj = pd.read_csv('data/external/majestic_million.csv')
        df_mj_filtered = df_mj.iloc[1000:51000]
        df_mj_filtered = df_mj_filtered.copy()
        df_mj_filtered['domain'] = df_mj_filtered['Domain'].apply(clean_domain)
        data_sources.append(df_mj_filtered[['domain']].assign(label='Legitimate', source='Majestic'))
        print(f"Majestic: Added {len(df_mj_filtered)} domains.")

    # 6. Existing v5 training data
    v5_path = 'data/processed/full_dataset_v5.csv'
    if os.path.exists(v5_path):
        df_v5 = pd.read_csv(v5_path)
        # Columns in v5: domain, label
        df_v5['domain'] = df_v5['domain'].apply(clean_domain)
        data_sources.append(df_v5[['domain', 'label']].assign(source='v5_training'))
        print(f"Existing v5 training data: Added {len(df_v5)} rows.")

    # Combine
    full_df = pd.concat(data_sources, ignore_index=True)
    
    # Final cleanup
    full_df = full_df.dropna(subset=['domain'])
    full_df = full_df.drop_duplicates(subset=['domain'], keep='first')
    
    # Class balancing
    phish_count = len(full_df[full_df['label'] == 'Phishing'])
    legit_count = len(full_df[full_df['label'] == 'Legitimate'])
    suspect_count = len(full_df[full_df['label'] == 'Suspected']) if 'Suspected' in full_df['label'].values else 0
    
    print("\nPre-balancing Counts:")
    print(f"Phishing: {phish_count}")
    print(f"Legitimate: {legit_count}")
    if suspect_count > 0: print(f"Suspected: {suspect_count}")
    print(f"Total: {len(full_df)}")

    # Target: Roughly balanced. Target Phishing count is much lower than Legitimate.
    # We should probably sub-sample Legitimate to match Phishing better?
    # User said: "Target: ~200k-250k total rows, roughly balanced (if phishing >> legitimate, cap phishing at 2x legitimate count)"
    # Our case is legitimate >> phishing. 
    # If we want 200k-250k total, we can't really balance it if phishing is only 20k.
    # Let's see how many phishing we have after cleaning.
    
    # PhishTank has ~100k URLs. URLhaus ~20k. OpenPhish 300.
    # Deduplicated phishing might be around 50k-80k?
    
    os.makedirs('data/processed', exist_ok=True)
    full_df.to_csv('data/processed/full_dataset_v7.csv', index=False)
    print(f"\nFinal dataset saved to data/processed/full_dataset_v7.csv with {len(full_df)} rows.")

if __name__ == '__main__':
    main()
