import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features

# Set random seed
np.random.seed(42)

def evaluate_model(X_train, X_test, y_train, y_test):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, 
                        scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
    rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_depth=15, random_state=42)
    
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    rf_pb = rf.predict_proba(X_test)[:, 1]
    xgb_pb = xgb.predict_proba(X_test)[:, 1]
    probs = (rf_pb + xgb_pb) / 2
    preds = (probs >= 0.571).astype(int)
    
    return {
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds)
    }

def main():
    print("Starting v3 Ablation Testing...")
    
    # 1. Load Data
    df_orig = load_training_data()
    df_synth = pd.read_csv("data/processed/synthetic_phish_v2.csv")
    df_combined = pd.concat([df_orig[['domain', 'label']], df_synth], ignore_index=True)
    
    # 2. Extract Features
    df_features = extract_url_features(df_combined, domain_col='domain')
    
    exclude = ['label', 'domain', 'cse_name', 'cse_domain', 'evidence_filename', 'source', 
               'evidence_path', 'evidence_exists', 'is_cse_target', 'domain_clean', 
               'tld', 'true_label', 'true_label_clean', 'S. No', 'S.No']
    all_features = [col for col in df_features.columns if col not in exclude]
    
    v3_new = ['is_malicious_gtld', 'brand_to_reg_length_ratio', 'min_brand_levenshtein', 'has_brand_in_path']
    v2_base = [f for f in all_features if f not in v3_new]
    
    y = (df_features['label'] == 'Phishing').astype(int)
    X_full = df_features[all_features].fillna(0)
    
    # Stratified Split
    X_tr_full, X_te_full, y_train, y_test = train_test_split(X_full, y, test_size=0.15, random_state=42, stratify=y)
    
    results = []
    
    # A. Baseline (v2 features only)
    print("\nEvaluating Baseline (v2 features)...")
    res_v2 = evaluate_model(X_tr_full[v2_base], X_te_full[v2_base], y_train, y_test)
    results.append({'feature': 'v2_baseline', **res_v2})
    
    # B. Test each v3 feature individually
    for f_new in v3_new:
        print(f"Evaluating Baseline + {f_new}...")
        current_features = v2_base + [f_new]
        res = evaluate_model(X_tr_full[current_features], X_te_full[current_features], y_train, y_test)
        results.append({'feature': f_new, **res})
        
    # C. All v3 features
    print("Evaluating full v3 model...")
    res_v3 = evaluate_model(X_tr_full, X_te_full, y_train, y_test)
    results.append({'feature': 'v3_all', **res_v3})
    
    # Report Results
    df_res = pd.DataFrame(results)
    print("\nAblation Results Summary (Threshold 0.571):")
    print(df_res)
    
    # Save to CSV for report
    df_res.to_csv("outputs/ablation_results_v3.csv", index=False)

if __name__ == "__main__":
    main()
