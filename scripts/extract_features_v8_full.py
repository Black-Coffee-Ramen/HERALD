"""
scripts/extract_features_v8_full.py

Extracts features for all 147,264 domains in HERALD v7.
- Updated Lexical Features (includes n-grams)
- Tranco Rank Features (is_in_tranco, tranco_rank)
Batch processing (20k) to manage memory efficiently.
"""

import os
import sys
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
import warnings

# Add parent dir to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features, CSE_KEYWORDS

# Suppress warnings
warnings.filterwarnings('ignore')

def load_tranco_lookup():
    print("Loading Tranco rank lookup from data/external/tranco.zip...")
    zip_path = 'data/external/tranco.zip'
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found.")
        return {}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('top-1m.csv') as f:
            df = pd.read_csv(f, names=['rank', 'domain'])
    
    # Create lookup dict: domain -> rank
    lookup = dict(zip(df['domain'].str.lower(), df['rank']))
    print(f"Tranco lookup loaded with {len(lookup)} domains.")
    return lookup

def main():
    print("=" * 60)
    print("HERALD v8 Full Dataset Feature Extraction (147k Domains)")
    print("=" * 60)
    
    input_path = 'data/processed/full_dataset_v7.csv'
    output_path = 'data/processed/full_features_v8_full.csv'
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} domains from {input_path}.")
    
    tranco_lookup = load_tranco_lookup()
    
    batch_size = 20000
    all_features = []
    
    from tldextract import extract as tld_extract
    
    for i in tqdm(range(0, len(df), batch_size), desc="Extracting Batches"):
        batch_df = df.iloc[i:i+batch_size].copy()
        
        # 1. Base Lexical Features (includes n-grams)
        batch_feat = extract_url_features(batch_df, domain_col='domain')
        
        # 2. Add Tranco Rank Features
        def get_tranco_info(domain):
            domain_clean = str(domain).lower().strip()
            if domain_clean.startswith('www.'):
                domain_clean = domain_clean[4:]
            rank = tranco_lookup.get(domain_clean, -1)
            return rank, (1 if rank != -1 else 0)

        tranco_data = [get_tranco_info(d) for d in batch_df['domain']]
        batch_feat['tranco_rank'] = [x[0] for x in tranco_data]
        batch_feat['is_in_tranco'] = [x[1] for x in tranco_data]
        
        # 3. v7 specific fast features
        def get_v7_info(domain):
            ext = tld_extract(domain)
            is_common = 1 if ext.suffix in ['com', 'org', 'net', 'in', 'gov', 'edu'] else 0
            has_brand = 1 if any(kw in domain.lower() for kw in CSE_KEYWORDS) else 0
            return ext.suffix, is_common, has_brand

        v7_data = [get_v7_info(d) for d in batch_df['domain']]
        batch_feat['tld'] = [x[0] for x in v7_data]
        batch_feat['is_common_tld'] = [x[1] for x in v7_data]
        batch_feat['has_brand_keyword'] = [x[2] for x in v7_data]
        
        # Labels and sources
        batch_feat['label'] = batch_df['label'].values
        batch_feat['source'] = batch_df['source'].values
        
        all_features.append(batch_feat)
    
    print("\nSaving final feature matrix...")
    full_feat_df = pd.concat(all_features, ignore_index=True)
    full_feat_df.to_csv(output_path, index=False)
    print(f"Success! Final matrix: {full_feat_df.shape}")
    print(f"Saved to {output_path}.")

if __name__ == '__main__':
    main()
