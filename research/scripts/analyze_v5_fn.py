"""
scripts/analyze_v5_fn.py

Extracts the 69 false negatives from the v5 test set (Tranco-based).
Prints feature values and confidence scores.
Saves to outputs/v5_false_negatives.csv.
Analyzes patterns.
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

def main():
    print("=" * 60)
    print("HERALD v5 False Negative Analysis")
    print("=" * 60)

    # 1. Load v5 features
    feat_path = 'data/processed/full_features_v5.csv'
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return
    df = pd.read_csv(feat_path, low_memory=False)
    
    # 2. Reproduce the test split (same random_state=42 and stratify as in retrain_v5_tranco.py)
    y = (df['label'] == 'Phishing').astype(int)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0)
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Get the original domains and labels for the test set
    test_indices = X_test.index
    df_test = df.loc[test_indices].copy()
    
    # 3. Load v5 model
    model_path = 'models/ensemble_v5.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return
    checkpoint = joblib.load(model_path)
    rf = checkpoint['rf']
    xgb = checkpoint['xgb']
    t = checkpoint.get('threshold', 0.45) # We used 0.45 in the best v5
    cols = checkpoint['features']
    
    # 4. Predict
    X_sub = X_test[cols].fillna(0).astype(float)
    probs = (rf.predict_proba(X_sub)[:, 1] + xgb.predict_proba(X_sub)[:, 1]) / 2
    preds = (probs >= t).astype(int)
    
    df_test['confidence'] = probs
    df_test['predicted'] = preds
    
    # 5. Extract False Negatives (Actual=Phishing, Pred=Not-Phishing)
    # y=1 is Phishing
    fn_mask = (y_test == 1) & (preds == 0)
    df_fn = df_test[fn_mask].copy()
    
    print(f"\nFound {len(df_fn)} False Negatives (Target: 69)")
    
    if len(df_fn) > 0:
        # Save to CSV
        os.makedirs('outputs', exist_ok=True)
        out_path = 'outputs/v5_false_negatives.csv'
        df_fn.to_csv(out_path, index=False)
        print(f"False negatives saved to {out_path}")
        
        # Pattern Analysis
        print("\nPattern Analysis:")
        
        # TLDs
        def get_tld(d):
            return str(d).split('.')[-1]
        
        df_fn['tld_extracted'] = df_fn['domain'].apply(get_tld)
        print("\nTop TLDs in FN:")
        print(df_fn['tld_extracted'].value_counts().head(10))
        
        # Averages
        print(f"\nAverage Domain Length: {df_fn['domain_length'].mean():.2f}")
        print(f"Average Digit Ratio:   {df_fn['digit_ratio'].mean():.2f}")
        print(f"Average Hyphen Count:  {df_fn['num_hyphens'].mean():.2f}")
        
        # Check for brand keywords
        from herald.features.lexical_features import CSE_KEYWORDS
        def count_brands(d):
            return sum(1 for kw in CSE_KEYWORDS if kw in str(d).lower())
        
        df_fn['brand_count'] = df_fn['domain'].apply(count_brands)
        print(f"\nDomains with Brand Keywords: {(df_fn['brand_count'] > 0).sum()} / {len(df_fn)}")
        
        # Sort by confidence to see "cleanest" ones
        print("\nTop 10 missed domains (sorted by lowest confidence):")
        print(df_fn[['domain', 'confidence', 'label']].sort_values('confidence').head(10).to_string(index=False))

    else:
        print("No False Negatives found. Check threshold or dataset.")

if __name__ == '__main__':
    main()
