"""
scripts/build_full_dataset_v5.py

Loads full_dataset_v4.csv (Phishing/Suspected) and tranco_legitimate.csv (Legitimate).
Combines them into full_dataset_v5.csv and extracts features.
"""

import os
import sys
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features

def main():
    print("=" * 60)
    print("Building HERALD v5 Dataset (Tranco-based)")
    print("=" * 60)

    # 1. Load v4 dataset (mostly Phishing/Suspected)
    v4_path = 'data/processed/full_dataset_v4.csv'
    if not os.path.exists(v4_path):
        print(f"ERROR: {v4_path} not found.")
        return
    
    df_v4 = pd.read_csv(v4_path)
    print(f"Loaded {len(df_v4)} rows from {v4_path}")

    # 2. Load Tranco legitimate domains
    tranco_path = 'data/external/tranco_legitimate.csv'
    if not os.path.exists(tranco_path):
        print(f"ERROR: {tranco_path} not found.")
        return
    
    df_tranco = pd.read_csv(tranco_path)
    df_tranco['source'] = 'tranco_external'
    print(f"Loaded {len(df_tranco)} rows from {tranco_path}")

    # 3. Combine
    df_v5 = pd.concat([df_v4, df_tranco], ignore_index=True)
    
    # Clean up: ensure no duplicates, though Tranco is new
    df_v5 = df_v5.drop_duplicates(subset=['domain'])
    
    # Final class distribution
    print("\nFinal Class Distribution:")
    print(df_v5['label'].value_counts())

    # 4. Save dataset
    out_path = 'data/processed/full_dataset_v5.csv'
    df_v5.to_csv(out_path, index=False)
    print(f"\nSaved v5 dataset to {out_path} ({len(df_v5)} rows)")

    # 5. Feature Extraction
    print("\nRunning feature extraction on full v5 dataset...")
    # Using the optimized batch version in extract_url_features
    df_features = extract_url_features(df_v5, domain_col='domain')
    
    # Drop object cols that aren't the label/domain/source
    for col in df_features.columns:
        if df_features[col].dtype == 'object' and col not in ('domain', 'label', 'source'):
            df_features[col] = df_features[col].astype('category').cat.codes

    out_feat_path = 'data/processed/full_features_v5.csv'
    df_features.to_csv(out_feat_path, index=False)
    print(f"Saved v5 features to {out_feat_path}")

if __name__ == '__main__':
    main()
