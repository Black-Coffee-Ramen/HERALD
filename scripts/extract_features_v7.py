"""
scripts/extract_features_v7.py

Extracts lexical features for HERALD v7 at scale.
Uses ONLY lexical features for speed.
Batch processing (10k) to manage memory.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Add parent dir to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features, CSE_KEYWORDS

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 50)
    print("HERALD v7 Feature Extraction (Lexical Only)")
    print("=" * 50)
    
    input_path = 'data/processed/full_dataset_v7.csv'
    output_path = 'data/processed/full_features_v7.csv'
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} domains from {input_path}.")
    
    batch_size = 10000
    all_features = []
    
    from tldextract import extract as tld_extract
    
    for i in tqdm(range(0, len(df), batch_size), desc="Extracting Batches"):
        batch_df = df.iloc[i:i+batch_size].copy()
        
        # 1. Base Lexical Features
        # extract_url_features expects 'domain' column or 'url'. Our df has 'domain'.
        batch_feat = extract_url_features(batch_df, domain_col='domain')
        
        # 2. Additional fast features
        def get_tld_info(domain):
            ext = tld_extract(domain)
            tld = ext.suffix
            is_common = 1 if tld in ['com', 'org', 'net', 'in', 'gov', 'edu'] else 0
            has_brand = 1 if any(kw in domain.lower() for kw in CSE_KEYWORDS) else 0
            return tld, is_common, has_brand

        # Applying extra features
        tld_data = [get_tld_info(d) for d in batch_df['domain']]
        batch_feat['tld'] = [x[0] for x in tld_data]
        batch_feat['is_common_tld'] = [x[1] for x in tld_data]
        batch_feat['has_brand_keyword'] = [x[2] for x in tld_data]
        
        # Keep track of original label and source for training
        batch_feat['label'] = batch_df['label'].values
        batch_feat['source'] = batch_df['source'].values
        
        all_features.append(batch_feat)
    
    print("\nMerging features...")
    full_feat_df = pd.concat(all_features, ignore_index=True)
    
    print(f"Final feature matrix: {full_feat_df.shape}")
    full_feat_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}.")

if __name__ == '__main__':
    main()
