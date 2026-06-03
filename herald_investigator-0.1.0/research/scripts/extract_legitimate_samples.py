"""
scripts/extract_legitimate_samples.py

1. Load both Shortlisting files (Part 1 + Part 2), combine = ~1M domains
2. Run v4 model on ALL of them (batch, no WHOIS/DNS — lexical features only for speed)
3. Filter: keep only domains where model confidence < 0.15 
   (model is very confident they are NOT phishing)
4. From those, randomly sample 3,000 domains
5. Label them all as "Legitimate"
6. Add to full_dataset_v4.csv alongside existing Phishing/Suspected rows
7. Save as data/processed/full_dataset_v5.csv
8. Print: how many passed the <0.15 filter, how many sampled
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features

BATCH_SIZE = 1000

def main():
    print("=" * 60)
    print("Extracting Legitimate Samples for v5")
    print("=" * 60)

    # 1. Load both Shortlisting files
    part1_path = 'data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_1.xlsx'
    part2_path = 'data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_2.xlsx'
    
    print(f"Loading {part1_path}...")
    df1 = pd.read_excel(part1_path)
    print(f"Loading {part2_path}...")
    df2 = pd.read_excel(part2_path)
    
    # Combine domains
    domains = pd.concat([df1['Domain User Form'], df2['Domain User Form']], ignore_index=True).dropna().unique()
    print(f"Total unique domains: {len(domains)}")
    
    df_all = pd.DataFrame({'domain': domains})

    # 2. Load v4 model
    model_path = 'models/ensemble_v4.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: Model v4 not found at {model_path}")
        return
    
    print(f"Loading v4 model from {model_path}...")
    checkpoint = joblib.load(model_path)
    rf_v4 = checkpoint['rf']
    xgb_v4 = checkpoint['xgb']
    feature_cols = checkpoint['features']
    threshold = checkpoint.get('threshold', 0.571)

    # 3. Process in batches to filter
    print(f"Running v4 model to identify legitimate candidates (threshold < 0.15)...")
    legit_candidates = []
    total = len(df_all)
    
    # We only need lexical features
    for start in range(0, total, BATCH_SIZE):
        batch = df_all.iloc[start:start + BATCH_SIZE].copy()
        
        try:
            # Extract lexical features
            feat_batch = extract_url_features(batch, domain_col='domain')
            
            # Predict
            X = feat_batch[feature_cols].fillna(0)
            
            # Ensemble probability: (RF + XGB) / 2
            rf_probs = rf_v4.predict_proba(X)[:, 1]
            xgb_probs = xgb_v4.predict_proba(X)[:, 1]
            probs = (rf_probs + xgb_probs) / 2
            
            # Filter confidence < 0.15
            batch_legit = batch[probs < 0.15].copy()
            legit_candidates.append(batch_legit)
            
        except Exception as e:
            print(f"  Error in batch {start}: {e}")
            
        if (start + BATCH_SIZE) % 10000 == 0 or (start + BATCH_SIZE) >= total:
            print(f"  Processed {min(start + BATCH_SIZE, total)}/{total} domains...")

    if not legit_candidates:
        print("ERROR: No legitimate candidates found.")
        return
    
    df_legit_pool = pd.concat(legit_candidates, ignore_index=True)
    n_passed = len(df_legit_pool)
    print(f"\nDomains passing < 0.15 filter: {n_passed}")

    # 4. Randomly sample 3,000 domains
    sample_size = min(3000, n_passed)
    print(f"Sampling {sample_size} domains...")
    df_sampled = df_legit_pool.sample(n=sample_size, random_state=42).copy()
    df_sampled['label'] = 'Legitimate'
    df_sampled['source'] = 'shortlisting_set'

    # 6. Add to full_dataset_v4.csv
    v4_data_path = 'data/processed/full_dataset_v4.csv'
    print(f"Loading existing v4 dataset from {v4_data_path}...")
    df_v4 = pd.read_csv(v4_data_path)
    
    # Combine
    df_v5 = pd.concat([df_v4, df_sampled], ignore_index=True)
    
    # 7. Save as data/processed/full_dataset_v5.csv
    out_path = 'data/processed/full_dataset_v5.csv'
    df_v5.to_csv(out_path, index=False)
    print(f"\nSaved v5 dataset to {out_path}")
    print(f"Final class distribution:")
    print(df_v5['label'].value_counts())

    print("\nSummary:")
    print(f"  Passed <0.15 filter: {n_passed}")
    print(f"  Sampled:             {sample_size}")

if __name__ == '__main__':
    main()
