"""
scripts/retrain_v9.py

Trains HERALD v9 using the expanded 213k dataset.
- Balanced Class Weighting (Option A)
- XGBoost + Random Forest Ensemble
- Threshold: 0.55
- Migrates whitelist from v8
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import os
import time

# Constants
THRESHOLD = 0.55
# Features to exclude from training
EXCLUDE = ['label', 'domain', 'source', 'tld', 'tranco_rank', 'is_in_tranco']

def main():
    print("=" * 60)
    print("HERALD v9: Model Retraining (213k Domains)")
    print("=" * 60)

    # 1. Load data
    data_path = 'data/processed/full_features_v9.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        return

    print(f"Loading features from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")

    # 2. Prepare features and labels
    # Use only rows with Phishing or Legitimate labels for training
    df = df[df['label'].isin(['Phishing', 'Legitimate'])]
    print(f"Filtered to {len(df)} samples (Phishing/Legitimate only).")

    X = df.drop(columns=EXCLUDE)
    # Binary labels: 1 for Phishing, 0 for Legitimate
    y = (df['label'] == 'Phishing').astype(int)

    print(f"Features: {X.shape[1]}")
    print(f"Class Distribution: {df['label'].value_counts().to_dict()}")

    # 3. Calculate class weights
    n_legit = (y == 0).sum()
    n_phish = (y == 1).sum()
    scale_pos_weight = n_legit / n_phish
    print(f"Scale Pos Weight (XGBoost): {scale_pos_weight:.4f}")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split: {len(X_train)} training, {len(X_test)} testing.")

    # 5. Train XGBoost
    print("\nTraining XGBoost (1000 estimators)...")
    start = time.time()
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time() - start:.2f}s.")

    # 6. Train Random Forest
    print("\nTraining Random Forest (500 estimators)...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"Random Forest trained in {time.time() - start:.2f}s.")

    # 7. Evaluate Ensemble
    print("\nEvaluating Ensemble (Simple Average)...")
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    ensemble_probs = (xgb_probs + rf_probs) / 2
    
    y_pred = (ensemble_probs >= THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save Model
    # Migrate whitelist from v8 (or v7 if v8 not found)
    whitelist = []
    for model_file in ['models/ensemble_v8_balanced.joblib', 'models/ensemble_v8.joblib', 'models/ensemble_v7.joblib']:
        if os.path.exists(model_file):
            print(f"Migrating whitelist from {model_file}...")
            old_model = joblib.load(model_file)
            whitelist = old_model.get('whitelist', [])
            break
    
    ensemble_v9 = {
        'xgb': xgb,
        'rf': rf,
        'threshold': THRESHOLD,
        'features': X.columns.tolist(),
        'whitelist': whitelist,
        'version': 'v9',
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble_v9, 'models/ensemble_v9.joblib')
    print("\nSuccess! Model saved to models/ensemble_v9.joblib")

if __name__ == '__main__':
    main()
