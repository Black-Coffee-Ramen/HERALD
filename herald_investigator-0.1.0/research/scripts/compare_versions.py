"""
scripts/compare_versions.py

Benchmarks v3 vs v4 on the SAME held-out test set derived from the v4 full dataset.
Produces a side-by-side table of Precision, Recall, F1, FP, FN.

Requirements:
  - data/processed/full_features_v4.csv     (produced by extract_features_v4.py)
  - models/ensemble_v3.joblib
  - models/ensemble_v4.joblib               (produced by retrain_v4.py)

Output:
  - outputs/version_comparison_v4.csv
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

THRESHOLD = 0.571


def evaluate_model(model_data, X_test, y_test, label):
    """Evaluate an ensemble model dict with 'rf', 'xgb', 'features' keys."""
    rf  = model_data['rf']
    xgb = model_data['xgb']
    feats = model_data['features']

    # Align features — fill any missing cols with 0
    X = X_test.reindex(columns=feats, fill_value=0)

    probs = (rf.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 2
    thr   = model_data.get('threshold', THRESHOLD)
    preds = (probs >= thr).astype(int)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Version':   label,
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, preds, zero_division=0), 4),
        'F1':        round(f1_score(y_test, preds, zero_division=0), 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Threshold': thr,
    }


def main():
    print("=" * 65)
    print("Version Comparison: v3 vs v4")
    print("=" * 65)

    feat_path = 'data/processed/full_features_v4.csv'
    print(f"\nLoading features from {feat_path}...")
    df = pd.read_csv(feat_path, low_memory=False)
    print(f"  Shape: {df.shape}")

    y = (df['label'] == 'Phishing').astype(int)

    # Reproduce the EXACT same test split used in retrain_v4.py
    # (same random_state ensures identical test set)
    EXCLUDE_COLS = [
        'label', 'domain', 'cse_name', 'cse_domain', 'evidence_filename',
        'source', 'evidence_path', 'evidence_exists', 'is_cse_target',
        'domain_clean', 'tld', 'true_label', 'true_label_clean',
        'S. No', 'S.No', '_source_group'
    ]
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"  Test set size: {len(X_test)} (same split as retraining)")
    print(f"  Test phishing: {y_test.sum()} / {len(y_test)}")

    results = []

    # ── v3 ──────────────────────────────────────────────────────────────────
    print("\nEvaluating v3...")
    try:
        v3 = joblib.load('models/ensemble_v3.joblib')
        results.append(evaluate_model(v3, X_test, y_test, 'v3-Operational'))
        print(f"  v3 done: P={results[-1]['Precision']}  R={results[-1]['Recall']}  F1={results[-1]['F1']}")
    except FileNotFoundError:
        print("  models/ensemble_v3.joblib not found — skipping v3")
    except Exception as e:
        print(f"  Error evaluating v3: {e}")

    # ── v4 ──────────────────────────────────────────────────────────────────
    print("\nEvaluating v4...")
    try:
        v4 = joblib.load('models/ensemble_v4.joblib')
        results.append(evaluate_model(v4, X_test, y_test, 'v4-Full-Dataset'))
        print(f"  v4 done: P={results[-1]['Precision']}  R={results[-1]['Recall']}  F1={results[-1]['F1']}")
    except FileNotFoundError:
        print("  models/ensemble_v4.joblib not found — run scripts/retrain_v4.py first")
    except Exception as e:
        print(f"  Error evaluating v4: {e}")

    if not results:
        print("\nNo models could be evaluated.")
        return

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL VERSION COMPARISON")
    print("=" * 65)
    print(df_results[['Version', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN']].to_string(index=False))

    # Target check
    for _, row in df_results.iterrows():
        p_ok = row['Precision'] >= 0.90
        r_ok = row['Recall']    >= 0.92
        status = "TARGETS MET" if (p_ok and r_ok) else "targets not fully met"
        print(f"  {row['Version']}: {status}")

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/version_comparison_v4.csv'
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
