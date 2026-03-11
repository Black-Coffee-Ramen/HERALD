"""
scripts/verify_v7.py

Performance verification for HERALD v7:
1. Sanity check on 12 known domains.
2. Comparison with v6 on the same test set.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from herald.features.lexical_features import extract_url_features, CSE_KEYWORDS
from tldextract import extract as tld_extract
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

PHISHING_SANITY = [
    'sbi-secure-login.xyz',
    'hdfc-netbanking-verify.top',
    'uidai-update-aadhar.buzz',
    'irctc-refund-claim.xyz',
    'icici-paylink.xyz',
    'sbi-login-secure.xyz'
]

LEGITIMATE_SANITY = [
    'sbi.co.in',
    'hdfcbank.com',
    'irctc.co.in',
    'uidai.gov.in',
    'airtel.in',
    'icicibank.com'
]

def get_lexical_v7_features(domains):
    df_in = pd.DataFrame({'domain': domains})
    feat = extract_url_features(df_in, domain_col='domain')
    
    # Extra features for v7
    def get_tld_info(domain):
        ext = tld_extract(domain)
        tld = ext.suffix
        is_common = 1 if tld in ['com', 'org', 'net', 'in', 'gov', 'edu'] else 0
        has_brand = 1 if any(kw in domain.lower() for kw in CSE_KEYWORDS) else 0
        return tld, is_common, has_brand

    tld_data = [get_tld_info(d) for d in domains]
    # 'tld' is actually string in some cases, but v7 model excludes it. 
    # Let's ensure columns match v7['features']
    feat['is_common_tld'] = [x[1] for x in tld_data]
    feat['has_brand_keyword'] = [x[2] for x in tld_data]
    return feat

def predict_v7(model_data, features):
    rf = model_data['rf']
    xgb = model_data['xgb']
    feats = model_data['features']
    thr = model_data.get('threshold', 0.45)
    
    X = features[feats].fillna(-1)
    
    prob_xgb = xgb.predict_proba(X)[:, 1]
    prob_rf = rf.predict_proba(X)[:, 1]
    prob = 0.6 * prob_xgb + 0.4 * prob_rf
    return prob

def main():
    print("=" * 60)
    print("HERALD v7 Verification & Benchmarking")
    print("=" * 60)
    
    # Load Models
    v6_path = 'models/ensemble_v6.joblib'
    v7_path = 'models/ensemble_v7.joblib'
    
    if not os.path.exists(v6_path) or not os.path.exists(v7_path):
        print("ERROR: Models missing.")
        return

    v6 = joblib.load(v6_path)
    v7 = joblib.load(v7_path)

    # 1. Sanity Check (v7 Only)
    print("\nSANITY CHECK (v7)")
    print(f"{'Domain':<30} | {'Score':<10} | {'Status':<10}")
    print("-" * 55)
    
    all_sanity = PHISHING_SANITY + LEGITIMATE_SANITY
    feat_sanity = get_lexical_v7_features(all_sanity)
    scores = predict_v7(v7, feat_sanity)
    
    pass_count = 0
    for d, s in zip(all_sanity, scores):
        is_phish = d in PHISHING_SANITY
        passed = False
        if is_phish and s >= v7['threshold']: passed = True
        if not is_phish and s < 0.30: passed = True
        
        status = "PASS" if passed else "FAIL"
        if passed: pass_count += 1
        print(f"{d:<30} | {s:<10.4f} | {status}")
    
    print(f"\nSanity Result: {pass_count}/12 passed.")

    # 2. Performance Comparison on v7 Test Set
    print("\nPERFORMANCE COMPARISON (v6 vs v7)")
    # We need to re-extract v6 features (network+whois) or dummy them.
    # The user asked to compare on v7 test set which is lexical only.
    # v6 expects network features. We'll fill them with -1 for v6.
    
    v7_feat_path = 'data/processed/full_features_v7.csv'
    df_v7_feat = pd.read_csv(v7_feat_path)
    
    # Split same as retrain_v7 (70/15/15)
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(df_v7_feat, test_size=0.15, random_state=42, stratify=df_v7_feat['label'])
    
    print(f"Comparison Count: {len(df_test)}")
    
    X_v7 = df_test[v7['features']].fillna(-1)
    y_test = (df_test['label'] == 'Phishing').astype(int)
    
    # v7 scores
    prob_v7 = predict_v7(v7, X_v7)
    pred_v7 = (prob_v7 >= v7['threshold']).astype(int)
    
    # v6 scores (Dummy network features)
    X_v6 = df_test.reindex(columns=v6['features'], fill_value=-1)
    # v6 also has domain_age_days. fill with -1 or median (30)
    X_v6['domain_age_days'] = 30 
    
    prob_xgb_v6 = v6['xgb'].predict_proba(X_v6)[:, 1]
    prob_rf_v6 = v6['rf'].predict_proba(X_v6)[:, 1]
    prob_v6 = 0.6 * prob_xgb_v6 + 0.4 * prob_rf_v6
    pred_v6 = (prob_v6 >= v6.get('threshold', 0.60)).astype(int)
    
    from sklearn.metrics import precision_recall_fscore_support
    p6, r6, f6, _ = precision_recall_fscore_support(y_test, pred_v6, average='binary')
    p7, r7, f7, _ = precision_recall_fscore_support(y_test, pred_v7, average='binary')
    
    # FN/FP counts
    fp6 = ((pred_v6 == 1) & (y_test == 0)).sum()
    fn6 = ((pred_v6 == 0) & (y_test == 1)).sum()
    fp7 = ((pred_v7 == 1) & (y_test == 0)).sum()
    fn7 = ((pred_v7 == 0) & (y_test == 1)).sum()

    results = [
        {'Version': 'v6', 'Train Size': 6434, 'Features': 48, 'Precision': p6, 'Recall': r6, 'F1': f6, 'FP': fp6, 'FN': fn6},
        {'Version': 'v7', 'Train Size': 147264, 'Features': len(v7['features']), 'Precision': p7, 'Recall': r7, 'F1': f7, 'FP': fp7, 'FN': fn7}
    ]
    
    df_res = pd.DataFrame(results)
    print("\n" + df_res.to_string(index=False))
    
    os.makedirs('outputs', exist_ok=True)
    df_res.to_csv('outputs/version_comparison_v7.csv', index=False)
    print(f"\nComparison saved to outputs/version_comparison_v7.csv")

if __name__ == '__main__':
    main()
