"""
scripts/retrain_v5_tranco.py

Retrains the XGBoost + Random Forest ensemble on the Tranco-based v5 dataset.
Performs threshold sweep to hit Precision >= 0.94 and Recall >= 0.92.
Compares v3, v4, and v5.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
    X = df[feature_cols].fillna(0)
    
    return X, y, feature_cols, df

def train_ensemble(X_train, y_train, X_val, y_val):
    # Scale positive weight for XGBoost
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
    
    # Val probabilities
    probs = (xgb.predict_proba(X_val)[:, 1] + rf.predict_proba(X_val)[:, 1]) / 2
    return xgb, rf, probs

def evaluate_thresholds(y_true, probs):
    thresholds = np.arange(0.30, 0.70, 0.05)
    results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f})
        
    return pd.DataFrame(results)

def get_performance(model_path, X_test, y_test, name):
    if not os.path.exists(model_path):
        return None
    
    checkpoint = joblib.load(model_path)
    # Handle older version formats if needed
    if isinstance(checkpoint, dict):
        rf = checkpoint['rf']
        xgb = checkpoint['xgb']
        t = checkpoint.get('threshold', 0.5)
        # Check for feature mismatch if necessary
        cols = checkpoint['features']
        X_sub = X_test[cols].fillna(0)
    else:
        # Unexpected format
        return None
        
    probs = (rf.predict_proba(X_sub)[:, 1] + xgb.predict_proba(X_sub)[:, 1]) / 2
    preds = (probs >= t).astype(int)
    
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Version': name,
        'Precision': p,
        'Recall': r,
        'F1': f,
        'FP': fp,
        'FN': fn
    }

def main():
    print("=" * 60)
    print("HERALD Ensemble Retraining  —  v5 (Tranco)")
    print("=" * 60)

    X, y, feature_cols, df_full = load_v5_features()
    print(f"Dataset: {len(X)} rows, {len(feature_cols)} features")
    print("Class distribution:")
    print(df_full['label'].value_counts())

    # 70/15/15 Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )

    print(f"\nTraining on {len(X_train)} samples, Validating on {len(X_val)}")
    xgb_final, rf_final, val_probs = train_ensemble(X_train, y_train, X_val, y_val)

    print("\nThreshold Sweep Analysis (Validation Set):")
    sweep_df = evaluate_thresholds(y_val, val_probs)
    print(sweep_df.to_string(index=False))
    
    # Save sweep results
    out_sweep_path = 'outputs/v5_threshold_sweep.csv'
    sweep_df.to_csv(out_sweep_path, index=False)
    print(f"Threshold sweep saved to {out_sweep_path}")

    # Pick threshold: P >= 0.94 and R >= 0.92
    eligible = sweep_df[(sweep_df['precision'] >= 0.94) & (sweep_df['recall'] >= 0.92)]
    if eligible.empty:
        # Fallback: pick one that maximizes F1 or closest to targets
        print("\nWARNING: No threshold met both P>=0.94 and R>=0.92 simultaneously.")
        best_t = sweep_df.loc[sweep_df['f1'].idxmax(), 'threshold']
    else:
        # Pick the lowest threshold that meets precision to maximize recall
        best_t = eligible.loc[eligible['recall'].idxmax(), 'threshold']
    
    print(f"\nSelected Threshold: {best_t:.3f}")

    # Final Model Save
    model_path = 'models/ensemble_v5.joblib'
    joblib.dump({
        'rf': rf_final,
        'xgb': xgb_final,
        'features': feature_cols,
        'threshold': best_t,
        'version': 'v5_tranco'
    }, model_path)
    print(f"Model saved to {model_path}")

    # Version Comparison
    print("\n" + "=" * 60)
    print("Version Comparison Table")
    print("=" * 60)
    
    comparisons = []
    # v5 (Current)
    comparisons.append(get_performance(model_path, X_test, y_test, 'v5 (Tranco)'))
    # v4 (Saved at models/ensemble_v4.joblib)
    comparisons.append(get_performance('models/ensemble_v4.joblib', X_test, y_test, 'v4 (Pre-biased)'))
    # v3 (Saved at models/ensemble_v3.joblib)
    comparisons.append(get_performance('models/ensemble_v3.joblib', X_test, y_test, 'v3 (Original)'))
    
    comp_df = pd.DataFrame([c for c in comparisons if c is not None])
    print(comp_df.to_string(index=False))
    
    # Save comparison to file
    out_comp_path = 'outputs/v5_comparison_results.csv'
    comp_df.to_csv(out_comp_path, index=False)
    print(f"\nComparison table saved to {out_comp_path}")

if __name__ == '__main__':
    main()
