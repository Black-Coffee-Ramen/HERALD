"""
scripts/retrain_v6.py

Retrains the HERALD ensemble (XGBoost + Random Forest) as v6.
Includes SSL and DNS network features.

Architecture:
- XGBoost (500 estimators, depth 6)
- Random Forest (300 estimators, depth 10)
- Weighted average ensemble (0.6 XGB + 0.4 RF)
- 70/15/15 stratified split
- 5-fold CV
- Threshold optimization (Precision >= 0.95, Recall >= 0.90)
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
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, precision_recall_curve
)

np.random.seed(42)

EXCLUDE_COLS = [
    'label', 'domain', 'source', 'true_label', 'true_label_clean', 
    'cse_name', 'cse_domain', 'evidence_filename', 'evidence_path'
]

def main():
    print("=" * 60)
    print("HERALD Ensemble Retraining  —  v6")
    print("=" * 60)

    in_path = 'data/processed/full_features_v6.csv'
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found.")
        return

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    # 1. Label Preparation
    # Binary: Phishing = 1, Suspected/Legitimate = 0
    y = (df['label'] == 'Phishing').astype(int)
    print(f"Binary label: {y.sum()} Phishing / {(y == 0).sum()} Non-Phishing")

    # 2. Feature Selection
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(-1).astype(float)
    print(f"Targeting {len(feature_cols)} features.")

    # 3. Stratified Split 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15/0.85, random_state=42, stratify=y_trainval
    )
    print(f"Split sizes - Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # 4. 5-Fold CV
    print("\nRunning 5-Fold Stratified CV on training set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_cv_tr, X_cv_val = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
        y_cv_tr, y_cv_val = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, eval_metric='logloss')
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        
        xgb.fit(X_cv_tr, y_cv_tr)
        rf.fit(X_cv_tr, y_cv_tr)

        # 0.6 XGB + 0.4 RF
        probs = 0.6 * xgb.predict_proba(X_cv_val)[:, 1] + 0.4 * rf.predict_proba(X_cv_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        cv_scores.append({
            'fold': fold,
            'precision': precision_score(y_cv_val, preds, zero_division=0),
            'recall': recall_score(y_cv_val, preds, zero_division=0),
            'f1': f1_score(y_cv_val, preds, zero_division=0)
        })
        print(f"  Fold {fold}: P={cv_scores[-1]['precision']:.3f} R={cv_scores[-1]['recall']:.3f}")

    # 5. Final Training
    print("\nTraining final ensemble...")
    xgb_v6 = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, eval_metric='logloss')
    rf_v6 = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    
    xgb_v6.fit(X_train, y_train)
    rf_v6.fit(X_train, y_train)

    # 6. Threshold Sweep (on Validation set)
    val_probs = 0.6 * xgb_v6.predict_proba(X_val)[:, 1] + 0.4 * rf_v6.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    found_target = False
    
    print("\nThreshold Sweep:")
    for thr in np.arange(0.30, 0.75, 0.05):
        val_preds = (val_probs >= thr).astype(int)
        p = precision_score(y_val, val_preds, zero_division=0)
        r = recall_score(y_val, val_preds, zero_division=0)
        f = f1_score(y_val, val_preds, zero_division=0)
        print(f"  Thr: {thr:.2f} | Prec: {p:.3f} | Rec: {r:.3f} | F1: {f:.3f}")
        
        # Check targets: Prec >= 0.95 AND Recall >= 0.90
        if p >= 0.95 and r >= 0.90:
            if not found_target or f > best_f1:
                best_threshold = thr
                best_f1 = f
                found_target = True
        
        # Keep track of best F1 in case targets aren't met
        if not found_target and f > best_f1:
            best_threshold = thr
            best_f1 = f

    print(f"\nFinal Selected Threshold: {best_threshold:.2f} (Target Met: {found_target})")

    # 7. Evaluate on Test set
    test_probs = 0.6 * xgb_v6.predict_proba(X_test)[:, 1] + 0.4 * rf_v6.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    
    print("\n--- TEST SET EVALUATION ---")
    print(classification_report(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")

    # 8. Save Model
    model_path = 'models/ensemble_v6.joblib'
    joblib.dump({
        'rf': rf_v6,
        'xgb': xgb_v6,
        'features': feature_cols,
        'threshold': best_threshold,
        'version': 'v6'
    }, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    main()
