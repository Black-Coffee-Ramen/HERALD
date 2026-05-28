"""
scripts/retrain_v5.py

Retrains the XGBoost + Random Forest ensemble on the full v5 feature set.

Architecture: Same as v4
  - XGBoost + Random Forest, averaged probability ensemble
  - Decision threshold: 0.571
  - 70/15/15 stratified split (train / validation / test)
  - 5-fold CV on training portion
  - Binary label: Phishing = 1, Non-Phishing (Suspected + Legitimate) = 0

Input:  data/processed/full_dataset_v5.csv
Output: models/ensemble_v5.joblib
        outputs/retrain_v5_results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

np.random.seed(42)

THRESHOLD = 0.450

EXCLUDE_COLS = [
    'label', 'domain', 'cse_name', 'cse_domain', 'evidence_filename',
    'source', 'evidence_path', 'evidence_exists', 'is_cse_target',
    'domain_clean', 'tld', 'true_label', 'true_label_clean',
    'S. No', 'S.No', '_source_group'
]

def main():
    print("=" * 60)
    print("HERALD Ensemble Retraining  —  v5")
    print("=" * 60)

    # 1. Feature Extraction (need to run it since dataset changed)
    in_path = 'data/processed/full_dataset_v5.csv'
    feat_path = 'data/processed/full_features_v5.csv'
    
    print(f"\nLoading dataset from {in_path}...")
    df_raw = pd.read_csv(in_path)
    print(f"  Rows: {len(df_raw)}")
    
    print("\nExtracting features...")
    df = extract_url_features(df_raw, domain_col='domain')
    
    # Drop object cols that aren't the label/domain/source
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ('domain', 'label', 'source'):
            df[col] = df[col].astype('category').cat.codes
            
    df.to_csv(feat_path, index=False)
    print(f"  Features saved to {feat_path}")

    # ─── Label preparation ──────────────────────────────────────────────────
    if 'label' not in df.columns:
        raise ValueError("'label' column missing from features CSV")

    print("\nClass distribution:")
    print(df['label'].value_counts().to_string())

    # Binary: Phishing = 1, everything else = 0
    y = (df['label'] == 'Phishing').astype(int)
    print(f"\nBinary label: {y.sum()} Phishing  /  {(y == 0).sum()} Non-Phishing")

    # ─── Feature selection ───────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    print(f"\nFeature count: {len(feature_cols)}")

    # ─── Stratified Split: 70 / 15 / 15 ─────────────────────────────────────
    # First: 85% train+val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Then split train+val into 70/15 (i.e. 15/85 ~= 17.6% of trainval)
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )
    print(f"\nSplit sizes — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ─── 5-Fold CV on Training Set ───────────────────────────────────────────
    print("\nRunning 5-Fold Stratified CV on training set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_cv_tr, X_cv_val = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
        y_cv_tr, y_cv_val = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        scale_pos = max((y_cv_tr == 0).sum() / max((y_cv_tr == 1).sum(), 1), 1.0)

        xgb = XGBClassifier(
            n_estimators=700, max_depth=10, learning_rate=0.03,
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric='logloss', verbosity=0
        )
        rf = RandomForestClassifier(
            n_estimators=700, class_weight='balanced',
            max_depth=20, random_state=42, n_jobs=-1
        )
        xgb.fit(X_cv_tr, y_cv_tr)
        rf.fit(X_cv_tr, y_cv_tr)

        probs = (xgb.predict_proba(X_cv_val)[:, 1] + rf.predict_proba(X_cv_val)[:, 1]) / 2
        preds = (probs >= THRESHOLD).astype(int)

        cv_results.append({
            'fold': fold,
            'precision': precision_score(y_cv_val, preds, zero_division=0),
            'recall': recall_score(y_cv_val, preds, zero_division=0),
            'f1': f1_score(y_cv_val, preds, zero_division=0),
        })
        print(f"  Fold {fold}: P={cv_results[-1]['precision']:.3f}  R={cv_results[-1]['recall']:.3f}  F1={cv_results[-1]['f1']:.3f}")

    cv_df = pd.DataFrame(cv_results)
    print("\nCV Summary (mean ± std):")
    print(f"  Precision: {cv_df.precision.mean():.3f} ± {cv_df.precision.std():.3f}")
    print(f"  Recall:    {cv_df.recall.mean():.3f} ± {cv_df.recall.std():.3f}")
    print(f"  F1:        {cv_df.f1.mean():.3f} ± {cv_df.f1.std():.3f}")

    # ─── Final Training on Full Train Set ────────────────────────────────────
    print("\nTraining final v5 ensemble on full training set...")
    scale_pos_final = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

    xgb_v5 = XGBClassifier(
        n_estimators=700, max_depth=10, learning_rate=0.03,
        scale_pos_weight=scale_pos_final, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    rf_v5 = RandomForestClassifier(
        n_estimators=700, class_weight='balanced',
        max_depth=20, random_state=42, n_jobs=-1
    )
    xgb_v5.fit(X_train, y_train)
    rf_v5.fit(X_train, y_train)

    # ─── Validation Set ───────────────────────────────────────────────────────
    val_probs = (xgb_v5.predict_proba(X_val)[:, 1] + rf_v5.predict_proba(X_val)[:, 1]) / 2
    val_preds = (val_probs >= THRESHOLD).astype(int)
    print("\n--- VALIDATION SET ---")
    print(classification_report(y_val, val_preds, target_names=['Non-Phishing', 'Phishing']))

    # ─── Test Set ─────────────────────────────────────────────────────────────
    test_probs = (xgb_v5.predict_proba(X_test)[:, 1] + rf_v5.predict_proba(X_test)[:, 1]) / 2
    test_preds = (test_probs >= THRESHOLD).astype(int)

    p = precision_score(y_test, test_preds, zero_division=0)
    r = recall_score(y_test, test_preds, zero_division=0)
    f = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- TEST SET (v5) ---")
    print(classification_report(y_test, test_preds, target_names=['Non-Phishing', 'Phishing']))
    print(f"Confusion Matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\nPrecision: {p:.3f}  (target >= 0.94)")
    print(f"Recall:    {r:.3f}  (target >= 0.92)")
    print(f"F1:        {f:.3f}")

    if p >= 0.94 and r >= 0.92:
        print("\nTargets MET.")
    else:
        missed = []
        if p < 0.94: missed.append(f"precision {p:.3f} < 0.94")
        if r < 0.92: missed.append(f"recall {r:.3f} < 0.92")
        print(f"\nTargets NOT fully met: {', '.join(missed)}")

    # ─── Save Model ───────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ensemble_v5.joblib'
    joblib.dump({
        'rf': rf_v5,
        'xgb': xgb_v5,
        'features': feature_cols,
        'threshold': THRESHOLD,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # ─── Save CV + Test Results ───────────────────────────────────────────────
    os.makedirs('outputs', exist_ok=True)
    results_path = 'outputs/retrain_v5_results.csv'
    results = {
        'dataset_size': len(df),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'n_phishing': int(y.sum()),
        'n_non_phishing': int((y == 0).sum()),
        'cv_precision_mean': cv_df.precision.mean(),
        'cv_recall_mean': cv_df.recall.mean(),
        'cv_f1_mean': cv_df.f1.mean(),
        'test_precision': p,
        'test_recall': r,
        'test_f1': f,
        'test_TP': tp,
        'test_FP': fp,
        'test_TN': tn,
        'test_FN': fn,
    }
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
