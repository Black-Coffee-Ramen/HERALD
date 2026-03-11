"""
scripts/retrain_v8.py

Retrains HERALD ensemble model as v8:
- Features: Lexical + N-Grams + Tranco Rank
- Split: 70/15/15
- Ensemble: 0.6 XGB + 0.4 RF
- Whitelist migration from v7
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
    print("HERALD v8 Model Retraining")
    print("=" * 50)
    
    input_path = 'data/processed/full_features_v8_full.csv'
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples with features.")

    # Binary label: Phishing=1, Legitimate+Suspected=0
    df['target'] = df['label'].apply(lambda x: 1 if x == 'Phishing' else 0)
    
    # Features to exclude
    exclude_cols = ['domain', 'label', 'source', 'tld', 'target', 'tranco_rank', 'is_in_tranco']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(-1)
    y = df['target']
    
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)
    scale_pos = n_neg / n_pos
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Class Distribution: {n_neg} Legit, {n_pos} Phish")
    print(f"Calculated scale_pos_weight: {scale_pos:.2f}")

    # 1. 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 2. XGBoost
    print(f"\nTraining XGBoost (1000 estimators, scale_pos_weight={scale_pos:.2f})...")
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
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
    print("\nTraining Random Forest (500 estimators, class_weight='balanced')...")
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
    print("\nRunning 5-Fold Cross-Validation on training set (XGBoost Only)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"CV F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f}")

    # 5. Ensemble & Threshold Selection
    # Fixed threshold 0.55
    best_threshold = 0.55
    print(f"\nUsing Fixed Threshold: {best_threshold}")

    # 6. Feature Importances (Top 15)
    print("\nTop 15 Feature Importances (XGBoost):")
    importances = xgb.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print(feat_imp.head(15))

    # 7. Final Evaluation on Test Set
    print("\nFinal Evaluation on Test Set...")
    prob_xgb_test = xgb.predict_proba(X_test)[:, 1]
    prob_rf_test = rf.predict_proba(X_test)[:, 1]
    prob_test = 0.6 * prob_xgb_test + 0.4 * prob_rf_test
    
    y_pred_test = (prob_test >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_test))
    
    p_final, r_final, f_final, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

    # 8. Migrating Whitelist and Saving Model
    print("\nMigrating whitelist from v7...")
    whitelist = []
    if os.path.exists('models/ensemble_v7.joblib'):
        v7_model = joblib.load('models/ensemble_v7.joblib')
        whitelist = v7_model.get('whitelist', [])
    
    model_data = {
        'rf': rf,
        'xgb': xgb,
        'features': feature_cols,
        'threshold': best_threshold,
        'version': 'v8_balanced',
        'training_size': len(df),
        'whitelist': whitelist,
        'recall': r_final,
        'precision': p_final,
        'f1': f_final
    }
    
    joblib.dump(model_data, 'models/ensemble_v8_balanced.joblib')
    print(f"Model saved to models/ensemble_v8_balanced.joblib")

if __name__ == '__main__':
    main()
