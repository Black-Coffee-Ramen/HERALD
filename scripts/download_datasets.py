"""
scripts/download_datasets.py

Downloads external data sources for HERALD v7:
1. OpenPhish live feed
2. URLhaus recent CSV
3. Majestic Million CSV
4. Tranco (extract from existing zip)
"""

import os
import urllib.request
import zipfile
import pandas as pd
import io

# Setup directories
os.makedirs('data/external', exist_ok=True)

def download_openphish():
    print("Downloading OpenPhish live feed...")
    path = 'data/external/openphish.txt'
    urllib.request.urlretrieve('https://openphish.com/feed.txt', path)
    with open(path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"OpenPhish: Loaded {len(urls)} URLs.")
    return urls

def download_urlhaus():
    print("Downloading URLhaus recent CSV...")
    path = 'data/external/urlhaus.csv'
    urllib.request.urlretrieve('https://urlhaus.abuse.ch/downloads/csv_recent/', path)
    # URLhaus CSV has comment lines starting with #
    df = pd.read_csv(path, comment='#', names=['id', 'dateadded', 'url', 'url_status', 'last_online', 'threat', 'tags', 'url_sha256', 'reporter'])
    
    # Filter for online or unknown
    df_filtered = df[df['url_status'].isin(['online', 'unknown'])]
    print(f"URLhaus: Loaded {len(df_filtered)} online/unknown URLs (from {len(df)} total).")
    return df_filtered['url'].tolist()

def extract_tranco():
    print("Extracting Tranco from data/external/tranco.zip...")
    zip_path = 'data/external/tranco.zip'
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found. Please ensure it's available.")
        return []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Assuming top-1m.csv is inside
        with z.open('top-1m.csv') as f:
            df = pd.read_csv(f, names=['rank', 'domain'])
    
    # Take ranks 1000 to 101000
    df_filtered = df[(df['rank'] >= 1000) & (df['rank'] <= 101000)]
    print(f"Tranco: Extracted {len(df_filtered)} domains.")
    return df_filtered['domain'].tolist()

def download_majestic():
    print("Downloading Majestic Million...")
    path = 'data/external/majestic_million.csv'
    urllib.request.urlretrieve('https://downloads.majestic.com/majestic_million.csv', path)
    df = pd.read_csv(path)
    
    # Take rows 1000-51000 (skip top 1000)
    # Majestic columns: GlobalRank, TldRank, Domain, TLD, RefSubNets, RefIPs, IDN_Domain, IDN_TLD, PrevGlobalRank, PrevTldRank, PrevRefSubNets, PrevRefIPs
    df_filtered = df.iloc[1000:51000]
    print(f"Majestic: Loaded {len(df_filtered)} domains.")
    return df_filtered['Domain'].tolist()

def main():
    print("=" * 50)
    print("HERALD v7 Data Download")
    print("=" * 50)
    
    openphish_urls = download_openphish()
    urlhaus_urls = download_urlhaus()
    tranco_domains = extract_tranco()
    majestic_domains = download_majestic()
    
    # Load existing v5
    v5_path = 'data/processed/full_dataset_v5.csv'
    if os.path.exists(v5_path):
        df_v5 = pd.read_csv(v5_path)
        print(f"Existing v5 training data: {len(df_v5)} rows.")
    else:
        print("Warning: data/processed/full_dataset_v5.csv not found.")

    print("\nDownload complete.")

if __name__ == '__main__':
    main()
