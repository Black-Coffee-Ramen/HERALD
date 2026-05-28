import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features

# Set random seed
np.random.seed(42)

def main():
    print("Starting v3 Retraining Pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    df_orig = load_training_data()
    df_synth = pd.read_csv("data/processed/synthetic_phish_v2.csv")
    
    # Combine
    df_combined = pd.concat([df_orig[['domain', 'label']], df_synth], ignore_index=True)
    print(f"Combined dataset size: {len(df_combined)}")

    # 2. Extract Features v3
    print("Extracting features v3...")
    df_features = extract_url_features(df_combined, domain_col='domain')
    
    # Identify feature columns
    exclude = ['label', 'domain', 'cse_name', 'cse_domain', 'evidence_filename', 'source', 
               'evidence_path', 'evidence_exists', 'is_cse_target', 'domain_clean', 
               'tld', 'true_label', 'true_label_clean', 'S. No', 'S.No']
    feature_cols = [col for col in df_features.columns if col not in exclude]
    
    X = df_features[feature_cols].fillna(0)
    y = (df_features['label'] == 'Phishing').astype(int)
    
    print(f"Total features: {len(feature_cols)}")
    print("New features included:", [f for f in feature_cols if f in ['is_malicious_gtld', 'min_brand_levenshtein', 'brand_to_reg_length_ratio', 'has_brand_in_path']])

    # 3. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # 4. 5-Fold Cross Validation
    print("Running 5-Fold Stratified CV on training set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train XGB
        scale_pos = (y_cv_train == 0).sum() / (y_cv_train == 1).sum()
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, 
                            scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
        xgb.fit(X_cv_train, y_cv_train)
        
        # Train RF
        rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_depth=15, random_state=42)
        rf.fit(X_cv_train, y_cv_train)
        
        # Ensemble Prediction
        rf_pb = rf.predict_proba(X_cv_val)[:, 1]
        xgb_pb = xgb.predict_proba(X_cv_val)[:, 1]
        probs = (rf_pb + xgb_pb) / 2
        
        # Threshold: 0.571
        preds = (probs >= 0.571).astype(int)
        
        cv_res.append({
            'precision': precision_score(y_cv_val, preds),
            'recall': recall_score(y_cv_val, preds),
            'f1': f1_score(y_cv_val, preds)
        })
        
    cv_df = pd.DataFrame(cv_res)
    print("\nCV Metrics (Threshold 0.571):")
    print(cv_df.mean())
    print("\nStd:")
    print(cv_df.std())

    # 5. Final Training
    print("\nTraining final v3 ensemble on full training set...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_v3 = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, 
                         scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
    rf_v3 = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_depth=15, random_state=42)
    
    xgb_v3.fit(X_train, y_train)
    rf_v3.fit(X_train, y_train)
    
    # 6. Evaluation on Test Set
    rf_pb_test = rf_v3.predict_proba(X_test)[:, 1]
    xgb_pb_test = xgb_v3.predict_proba(X_test)[:, 1]
    probs_test = (rf_pb_test + xgb_pb_test) / 2
    preds_test = (probs_test >= 0.571).astype(int)
    
    print("\n--- v3 TEST SET PERFORMANCE ---")
    print(classification_report(y_test, preds_test))
    
    p = precision_score(y_test, preds_test)
    r = recall_score(y_test, preds_test)
    
    print(f"Final Precision: {p:.3f} (Goal >= 0.93)")
    print(f"Final Recall: {r:.3f} (Goal >= 0.87)")
    
    # 7. Save Model
    joblib.dump({'rf': rf_v3, 'xgb': xgb_v3, 'features': feature_cols}, "models/ensemble_v3.joblib")
    print("\nModel saved to models/ensemble_v3.joblib")

if __name__ == "__main__":
    main()
