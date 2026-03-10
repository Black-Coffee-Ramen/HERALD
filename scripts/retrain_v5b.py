"""
scripts/retrain_v5b.py

Retrains the HERALD ensemble as v5b with new features.
Target: Recall >= 0.88 with Precision >= 0.93.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

np.random.seed(42)

EXCLUDE_COLS = [
    'label', 'domain', 'source', 'true_label', 'true_label_clean', 'domain_clean',
    'S. No', 'S.No', '_source_group', 'evidence_filename', 'evidence_path', 
    'evidence_exists', 'cse_name', 'cse_domain', 'is_cse_target', 'tld'
]

def load_v5_features():
    feat_path = 'data/processed/full_features_v5.csv'
    df = pd.read_csv(feat_path, low_memory=False)
    
    # Binary label: Phishing = 1, else 0
    y = (df['label'] == 'Phishing').astype(int)
    
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    
    return X, y, feature_cols, df

def train_ensemble(X_train, y_train, X_val, y_val):
    scale_pos = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    
    xgb = XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.03,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    rf = RandomForestClassifier(
        n_estimators=700, class_weight='balanced',
        max_depth=20, random_state=42, n_jobs=-1
    )
    
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    probs = (xgb.predict_proba(X_val)[:, 1] + rf.predict_proba(X_val)[:, 1]) / 2
    return xgb, rf, probs

def evaluate_thresholds(y_true, probs):
    thresholds = np.arange(0.20, 0.75, 0.05)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f})
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("HERALD Ensemble Retraining  —  v5b (Enhanced Features)")
    print("=" * 60)

    X, y, feature_cols, df_full = load_v5_features()
    print(f"Dataset: {len(X)} rows, {len(feature_cols)} features")

    # 70/15/15 Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )

    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)}")
    xgb, rf, val_probs = train_ensemble(X_train, y_train, X_val, y_val)

    print("\nThreshold Sweep Analysis (Validation Set):")
    sweep_df = evaluate_thresholds(y_val, val_probs)
    print(sweep_df.to_string(index=False))

    # Pick threshold: Recall >= 0.88, Precision >= 0.93
    eligible = sweep_df[(sweep_df['precision'] >= 0.93) & (sweep_df['recall'] >= 0.88)]
    if eligible.empty:
        # Fallback to maximizing F1
        print("\nWARNING: Targets Recall>=0.88, Precision>=0.93 not met simultaneously.")
        best_t = sweep_df.loc[sweep_df['f1'].idxmax(), 'threshold']
    else:
        # Pick the lowest threshold that meets precision to maximize recall
        best_t = eligible.loc[eligible['recall'].idxmax(), 'threshold']
    
    print(f"\nSelected Threshold: {best_t:.3f}")

    # Final Model Save
    model_path = 'models/ensemble_v5b.joblib'
    joblib.dump({
        'rf': rf, 
        'xgb': xgb,
        'features': feature_cols,
        'threshold': best_t,
        'version': 'v5b_enhanced'
    }, model_path)
    print(f"Model saved to {model_path}")

    # Test set evaluation
    test_probs = (xgb.predict_proba(X_test)[:, 1] + rf.predict_proba(X_test)[:, 1]) / 2
    test_preds = (test_probs >= best_t).astype(int)
    p = precision_score(y_test, test_preds, zero_division=0)
    r = recall_score(y_test, test_preds, zero_division=0)
    f = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- TEST SET (v5b) ---")
    print(f"Precision: {p:.4f} (target >= 0.93)")
    print(f"Recall:    {r:.4f} (target >= 0.88)")
    print(f"F1:        {f:.4f}")
    print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # Save results for comparison
    comp_path = 'outputs/v5b_results.csv'
    pd.DataFrame([{
        'Version': 'v5b (Enhanced)',
        'Precision': p,
        'Recall': r,
        'F1': f,
        'FP': fp,
        'FN': fn
    }]).to_csv(comp_path, index=False)
    print(f"\nFinal metrics saved to {comp_path}")

if __name__ == '__main__':
    main()
