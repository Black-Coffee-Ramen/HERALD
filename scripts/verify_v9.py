"""
scripts/verify_v9.py

Verifies HERALD v9 performance and runs sanity checks.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from herald.features.lexical_features import extract_url_features

def main():
    print("=" * 60)
    print("HERALD v9: Final Verification")
    print("=" * 60)

    # 1. Load Model
    model_path = 'models/ensemble_v9.joblib'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return
    
    ensemble = joblib.load(model_path)
    xgb = ensemble['xgb']
    rf = ensemble['rf']
    threshold = ensemble['threshold']
    features = ensemble['features']
    whitelist = ensemble.get('whitelist', [])

    print(f"Loaded {ensemble['version']} model with threshold {threshold}")
    print(f"Whitelist size: {len(whitelist)}")

    # 2. Performance Comparison on Test Set
    # (The retrain script already gave us test metrics, but let's re-verify)
    data_path = 'data/processed/full_features_v9.csv'
    df = pd.read_csv(data_path)
    df = df[df['label'].isin(['Phishing', 'Legitimate'])]
    
    X = df[features]
    y = (df['label'] == 'Phishing').astype(int)

    # We use the same random seed 42 as in retrain_v9 to get the same test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Running evaluation on {len(X_test)} samples...")
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    probs = (xgb_probs + rf_probs) / 2
    
    # Try multiple thresholds
    print("\nThreshold Sensitivity:")
    for t in [0.45, 0.50, 0.55]:
        preds = (probs >= t).astype(int)
        p = precision_score(y_test, preds)
        r = recall_score(y_test, preds)
        print(f"T={t:.2f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {2*p*r/(p+r):.4f}")

    # 3. Sanity Check on Known Domains
    print("\nSanity Check (12 Key Domains):")
    domains = [
        'google.com', 'microsoft.com', 'apple.com', 'facebook.com', 'amazon.com', 'netflix.com', # Legit
        'login-microsoft-verify.com', 'secure-apple-id.top', 'paypal-update-account.xyz', # Phish
        'icicibank.netbanking-verify.in', 'hdfcbank.secure-login.online', 'sbi-card-verification.top' # Phish
    ]
    
    from tldextract import extract as tld_extract
    from herald.features.lexical_features import CSE_KEYWORDS

    results = []
    for d in domains:
        # Check Whitelist
        clean_d = d.lower().strip().replace('www.', '')
        if clean_d in whitelist:
            results.append({'domain': d, 'prob': 0.0, 'pred': 'Clean (Whitelist)'})
            continue
            
        # Extract base lexical features
        temp_df = pd.DataFrame({'domain': [d]})
        feat_df = extract_url_features(temp_df, domain_col='domain')
        
        # Add v7 specific features manually as they are in the model's 'features' list
        ext = tld_extract(d)
        feat_df['is_common_tld'] = 1 if ext.suffix in ['com', 'org', 'net', 'in', 'gov', 'edu'] else 0
        feat_df['has_brand_keyword'] = 1 if any(kw in d.lower() for kw in CSE_KEYWORDS) else 0
        feat_df['tld'] = ext.suffix
        
        # Note: tranco_rank and is_in_tranco are EXCLUDED from training in v9,
        # but the model 'features' list should only contain the 44 numeric ones.
        # Let's verify what 'features' contains.
        X_input = feat_df[features]
        
        xp = xgb.predict_proba(X_input)[:, 1][0]
        rp = rf.predict_proba(X_input)[:, 1][0]
        prob = (xp + rp) / 2
        pred = 'Phish' if prob >= threshold else 'Clean'
        results.append({'domain': d, 'prob': prob, 'pred': pred})

    for r in results:
        print(f"{r['domain']:35} -> {r['pred']:15} (Conf: {r['prob']:.4f})")

if __name__ == '__main__':
    main()
