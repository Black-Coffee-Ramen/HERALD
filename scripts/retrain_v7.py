"""
scripts/retrain_v7.py

Retrains HERALD ensemble model as v7 using a large-scale dataset.
- Features: Lexical only
- Scale: 147k domains
- Split: 70/15/15
- Ensemble: 0.6 XGB + 0.4 RF
"""

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 50)
    print("HERALD v7 Model Retraining (Large-Scale)")
    print("=" * 50)
    
    input_path = 'data/processed/full_features_v7.csv'
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} domains with features.")

    # Binary label: Phishing=1, Legitimate+Suspected=0
    df['target'] = df['label'].apply(lambda x: 1 if x == 'Phishing' else 0)
    
    # Features to exclude
    exclude_cols = ['domain', 'label', 'source', 'tld', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(-1)
    y = df['target']
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Class Distribution:\n{y.value_counts(normalize=True)}")

    # 1. 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 2. XGBoost
    print("\nTraining XGBoost (1000 estimators)...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1
    
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    start_time = time.time()
    xgb.fit(X_train, y_train)
    print(f"XGBoost finished in {time.time() - start_time:.2f}s")

    # 3. Random Forest
    print("\nTraining Random Forest (500 estimators)...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    print(f"Random Forest finished in {time.time() - start_time:.2f}s")

    # 4. Cross-Validation (5-fold on training set)
    # This might be slow on 100k, skip if taking too long? User asked for it.
    print("\nRunning 5-Fold Cross-Validation on training set (XGBoost Only for speed)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"CV F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f}")

    # 5. Ensemble & Threshold Sweep
    print("\nOptimizing Threshold on Validation Set...")
    prob_xgb = xgb.predict_proba(X_val)[:, 1]
    prob_rf = rf.predict_proba(X_val)[:, 1]
    prob_ensemble = 0.6 * prob_xgb + 0.4 * prob_rf
    
    thresholds = np.arange(0.25, 0.75, 0.05)
    best_threshold = 0.45
    best_f1 = 0
    priority_met = False
    sweep_results = []
    
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)
    
    for thr in thresholds:
        y_pred = (prob_ensemble >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        sweep_results.append({'threshold': thr, 'precision': p, 'recall': r, 'f1': f})
        print(f"{thr:<10.2f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")
        
        # Priority: Precision >= 0.95 and Recall >= 0.90
        if p >= 0.95 and r >= 0.90:
            best_threshold = thr
            best_f1 = f
            priority_met = True
        elif not priority_met and f > best_f1:
            best_f1 = f
            best_threshold = thr

    print(f"\nSelected Threshold: {best_threshold:.2f}")

    # 6. Final Evaluation on Test Set
    print("\nFinal Evaluation on Test Set...")
    prob_xgb_test = xgb.predict_proba(X_test)[:, 1]
    prob_rf_test = rf.predict_proba(X_test)[:, 1]
    prob_test = 0.6 * prob_xgb_test + 0.4 * prob_rf_test
    
    y_pred_test = (prob_test >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_test))
    
    p_final, r_final, f_final, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

    # 7. Save Model
    model_data = {
        'rf': rf,
        'xgb': xgb,
        'features': feature_cols,
        'threshold': best_threshold,
        'version': 'v7',
        'training_size': len(df),
        'feature_type': 'lexical_only',
        'recall': r_final,
        'precision': p_final,
        'f1': f_final
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/ensemble_v7.joblib')
    print(f"\nModel saved to models/ensemble_v7.joblib")

if __name__ == '__main__':
    main()
