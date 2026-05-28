import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features
from herald.predict_with_fallback import PhishingPredictorV2

# Set random seed
np.random.seed(42)

def main():
    print("Evaluating v2 Model with Content Fallback (FAST MODE)...")
    
    predictor = PhishingPredictorV2()
    
    # 1. Prepare Test Set
    df_orig = load_training_data()
    synth_path = "data/processed/synthetic_phish_v2.csv"
    if os.path.exists(synth_path):
        df_synth = pd.read_csv(synth_path)
        df_combined = pd.concat([df_orig[['domain', 'label']], df_synth], ignore_index=True)
    else:
        df_combined = df_orig[['domain', 'label']].copy()
        
    df_features = extract_url_features(df_combined, domain_col='domain')
    feature_names = predictor.feature_names
    X = df_features[feature_names].fillna(0)
    y = (df_features['label'] == 'Phishing').astype(int)
    
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    test_domains = df_combined.loc[X_test.index, 'domain'].values
    
    # 2. Run ML Predictions First (to identify borderline)
    print("Running ML stage...")
    rf_proba = predictor.rf.predict_proba(X_test)[:, 1]
    xgb_proba = predictor.xgb.predict_proba(X_test)[:, 1]
    ml_confs = (rf_proba + xgb_proba) / 2
    
    preds = []
    
    print(f"Checking {len(test_domains)} domains. OCR will only trigger for borderline cases [0.35, 0.571].")
    
    for i, domain in enumerate(test_domains):
        ml_conf = ml_confs[i]
        
        if ml_conf >= predictor.threshold:
            preds.append(1) # Phishing
        elif ml_conf < predictor.fallback_trigger:
            preds.append(0) # Clean
        else:
            # Borderline - Trigger OCR
            print(f"  [{i+1}/{len(test_domains)}] Borderline ({ml_conf:.3f}): {domain} -> OCR FALLBACK")
            res = predictor.predict(domain)
            preds.append(1 if res['status'] == 'Phishing' else 0)
            
    # 3. Report Metrics
    print("\n--- FINAL PERFORMANCE (ML + FALLBACK) ---")
    y_test_arr = y_test.values
    preds_arr = np.array(preds)
    
    print(classification_report(y_test_arr, preds_arr))
    cm = confusion_matrix(y_test_arr, preds_arr)
    print("Confusion Matrix:")
    print(cm)
    
    p = precision_score(y_test_arr, preds_arr)
    r = recall_score(y_test_arr, preds_arr)
    f1 = f1_score(y_test_arr, preds_arr)
    
    print(f"\nPrecision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
    
    if p >= 0.92: print("âœ… Precision goal met (>= 0.92)")
    else: print("âŒ Precision goal NOT met")
    
    if r >= 0.88: print("âœ… Recall goal met (>= 0.88)")
    else: print("âŒ Recall goal NOT met")

if __name__ == "__main__":
    main()
