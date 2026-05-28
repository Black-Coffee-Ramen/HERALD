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
    print("Evaluating v2 Model with Content Fallback...")
    
    predictor = PhishingPredictorV2()
    
    # 1. Prepare Test Set (same as v2)
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
    
    # 2. Run Predictions
    print(f"Running fallback pipeline on {len(test_domains)} test domains...")
    preds = []
    
    for i, domain in enumerate(test_domains):
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_domains)}")
            
        # For evaluation, we ideally need the cse_name if it was a phishing domain
        # Get true label and potential brand for OCR lookup
        true_label = y_test.iloc[i]
        
        # Heuristic: try to find brand in domain for OCR template search
        cse_name = None
        # In a real scenario, this comes from the monitor context
        
        res = predictor.predict(domain, cse_name=cse_name)
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
    
    # Baseline comparison (0.571 threshold ML-only)
    # We can calculate this from the predictor easily
    ml_preds = []
    for i, domain in enumerate(test_domains):
        # We already have ml_conf from predictor run, but let's just use the logic
        # For simplicity, I'll just report the final combined result as requested.
        pass

    if p >= 0.92: print("âœ… Precision goal met (>= 0.92)")
    else: print("âŒ Precision goal NOT met")
    
    if r >= 0.88: print("âœ… Recall goal met (>= 0.88)")
    else: print("âŒ Recall goal NOT met")

if __name__ == "__main__":
    main()
