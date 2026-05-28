"""
scripts/compare_v5_v6_fn.py

Compares false negatives of v5 and v6.
Checks if WHOIS age helped catch the previous 69 misses.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

EXCLUDE_COLS = [
    'label', 'domain', 'source', 'true_label', 'true_label_clean', 'domain_clean',
    'S. No', 'S.No', '_source_group', 'evidence_filename', 'evidence_path', 
    'evidence_exists', 'cse_name', 'cse_domain', 'is_cse_target', 'tld'
]

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    y = (df['label'] == 'Phishing').astype(int)
    return df, y

def get_preds(model_path, df):
    checkpoint = joblib.load(model_path)
    rf = checkpoint['rf']
    xgb = checkpoint['xgb']
    t = checkpoint.get('threshold', 0.5)
    cols = checkpoint['features']
    
    X = df[cols].fillna(0).astype(float)
    probs = (rf.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 2
    preds = (probs >= t).astype(int)
    return preds, probs

def main():
    print("=" * 60)
    print("HERALD v5 vs v6 False Negative Comparison")
    print("=" * 60)

    # Load data
    df_v6, y_v6 = load_data('data/processed/full_features_v6.csv')
    
    # Reproduce test split
    _, X_test, _, y_test = train_test_split(
        df_v6, y_v6, test_size=0.15, random_state=42, stratify=y_v6
    )
    
    # Get preds for v5 (using v6 data but v5 model will ignore domain_age_days)
    v5_preds, v5_probs = get_preds('models/ensemble_v5.joblib', X_test)
    v6_preds, v6_probs = get_preds('models/ensemble_v6.joblib', X_test)
    
    X_test = X_test.copy()
    X_test['v5_pred'] = v5_preds
    X_test['v6_pred'] = v6_preds
    X_test['v6_prob'] = v6_probs
    
    # False Negatives: Actual=1, Pred=0
    fn_v5 = X_test[(y_test == 1) & (X_test['v5_pred'] == 0)]
    fn_v6 = X_test[(y_test == 1) & (X_test['v6_pred'] == 0)]
    
    print(f"v5 False Negatives: {len(fn_v5)}")
    print(f"v6 False Negatives: {len(fn_v6)}")
    
    # Caught by v6 that v5 missed
    caught = fn_v5[fn_v5['v6_pred'] == 1]
    print(f"\nCaught by v6 (previously missed by v5): {len(caught)}")
    
    if len(caught) > 0:
        print("\nTop 10 caught domains (with their ages):")
        print(caught[['domain', 'domain_age_days', 'v6_prob']].head(10).to_string(index=False))

    # Still missed
    still_missed = fn_v6[fn_v6['domain'].isin(fn_v5['domain'])]
    print(f"\nStill missed by both: {len(still_missed)}")

if __name__ == '__main__':
    main()
