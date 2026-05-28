"""
scripts/compare_versions_v6.py

Benchmarks v3, v4, v5, and v6 on the SAME held-out test set from v6.
Produces a comparison table of Precision, Recall, F1, FP, FN.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

np.random.seed(42)

def evaluate_model(model_data, X_test, y_test, label):
    """Evaluate an ensemble model dict."""
    rf  = model_data['rf']
    xgb = model_data['xgb']
    feats = model_data['features']
    thr   = model_data.get('threshold', 0.5)
    
    # Weights for ensemble (v6 uses 0.6/0.4, older versions use 0.5/0.5?)
    # Let's check model version or just use equal weights for others
    version = model_data.get('version', 'v4')
    
    # Align features - fill missing with -1
    X = X_test.reindex(columns=feats, fill_value=-1)
    
    if version == 'v6':
        probs = 0.6 * xgb.predict_proba(X)[:, 1] + 0.4 * rf.predict_proba(X)[:, 1]
    else:
        probs = (rf.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 2
        
    preds = (probs >= thr).astype(int)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Version': label,
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall': round(recall_score(y_test, preds, zero_division=0), 4),
        'F1': round(f1_score(y_test, preds, zero_division=0), 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Threshold': thr,
        'Key Feature': "Network + Indian Legitimate" if version == 'v6' else ("Tranco Legitimate" if version == 'v5' else "WHOIS Age" if version == 'v4' else "Lexical Only")
    }

def main():
    print("=" * 70)
    print("HERALD Version Comparison: v3 vs v4 vs v5 vs v6")
    print("=" * 70)

    feat_path = 'data/processed/full_features_v6.csv'
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_csv(feat_path)
    y = (df['label'] == 'Phishing').astype(int)
    
    EXCLUDE_COLS = [
        'label', 'domain', 'source', 'true_label', 'true_label_clean', 
        'cse_name', 'cse_domain', 'evidence_filename', 'evidence_path'
    ]
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(-1)

    # Use the same test split as retrain_v6.py
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"Test Set Size: {len(X_test)} (Phishing: {y_test.sum()})")

    results = []
    versions = [
        ('v3', 'models/ensemble_v3.joblib'),
        ('v4', 'models/ensemble_v4.joblib'),
        ('v5', 'models/ensemble_v5.joblib'),
        ('v6', 'models/ensemble_v6.joblib')
    ]

    for label, path in versions:
        if os.path.exists(path):
            print(f"Evaluating {label}...")
            model_data = joblib.load(path)
            results.append(evaluate_model(model_data, X_test, y_test, label))
        else:
            print(f"  {path} not found - skipping {label}")

    if not results:
        print("No models evaluated.")
        return

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print(df_results[['Version', 'Precision', 'Recall', 'F1', 'FP', 'FN', 'Key Feature']].to_string(index=False))
    print("=" * 85)

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/version_comparison_v6.csv'
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

if __name__ == '__main__':
    main()
