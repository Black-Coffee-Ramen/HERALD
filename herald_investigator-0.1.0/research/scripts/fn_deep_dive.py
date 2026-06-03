import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features, CSE_KEYWORDS

# Set random seed
np.random.seed(42)

def main():
    print("Starting FN Deep Dive...")
    
    # 1. Load v2 Ensemble
    model_path = "models/ensemble_v2.joblib"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    
    ensemble = joblib.load(model_path)
    rf = ensemble['rf']
    xgb = ensemble['xgb']
    feature_names = ensemble['features']
    
    # 2. Re-run v2 Split
    df_orig = load_training_data()
    # Need to include synthetic data if used in v2
    synth_path = "data/processed/synthetic_phish_v2.csv"
    if os.path.exists(synth_path):
        df_synth = pd.read_csv(synth_path)
        df_combined = pd.concat([df_orig[['domain', 'label']], df_synth], ignore_index=True)
    else:
        df_combined = df_orig[['domain', 'label']].copy()
        
    df_features = extract_url_features(df_combined, domain_col='domain')
    X = df_features[feature_names].fillna(0)
    y = (df_features['label'] == 'Phishing').astype(int)
    
    # Stratified Split (same as retrain_v2.py)
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 3. Generate Predictions on Test Set
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + xgb_proba) / 2
    
    # Use the 0.845 threshold from v2 report to identify the 24 FNs
    threshold = 0.845
    preds = (ensemble_proba >= threshold).astype(int)
    
    # Identify FNs
    fn_mask = (y_test == 1) & (preds == 0)
    X_fn = X_test[fn_mask]
    y_fn = y_test[fn_mask]
    conf_fn = ensemble_proba[fn_mask]
    rf_conf_fn = rf_proba[fn_mask]
    xgb_conf_fn = xgb_proba[fn_mask]
    
    domains_fn = df_combined.loc[X_fn.index, 'domain'].values
    
    results = []
    categories = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    
    print(f"\nAnalyzing {len(X_fn)} False Negatives:")
    
    for i, (idx, row) in enumerate(X_fn.iterrows()):
        domain = domains_fn[i]
        conf = conf_fn[i]
        rf_c = rf_conf_fn[i]
        xgb_c = xgb_conf_fn[i]
        
        # Determine closer model
        closer = "RF" if abs(rf_c - 1) < abs(xgb_c - 1) else "XGBoost"
        
        # Categorization
        cat = 'd'
        if row['registered_domain_length'] < 12:
            cat = 'a'
        elif not any(keyword in domain.lower() for keyword in CSE_KEYWORDS):
            cat = 'b'
        elif row['subdomain_count'] >= 2: # Heuristic for legit-looking subdomains
            cat = 'c'
            
        categories[cat] += 1
        
        print(f"\n[{i+1}] {domain}")
        print(f"    Confidence: {conf:.3f} | Closer: {closer} (RF: {rf_c:.3f}, XGB: {xgb_c:.3f})")
        print(f"    Category: {cat}")
        # Print key features
        feats = [f"{f}: {row[f]:.2f}" for f in ['registered_domain_length', 'digit_ratio', 'dot_ratio', 'brand_keyword_position', 'has_cse_keyword_in_subdomain']]
        print(f"    Key Features: {', '.join(feats)}")
        
        results.append({
            'domain': domain,
            'category': cat,
            'confidence': conf,
            'closer_model': closer
        })
        
    print("\nSummary Counts per Category:")
    print(f"  a) Short simple domains: {categories['a']}")
    print(f"  b) No CSE keyword in URL: {categories['b']}")
    print(f"  c) Legit-looking subdomains: {categories['c']}")
    print(f"  d) Other patterns: {categories['d']}")
    
    # Save to CSV for reference
    df_fn = pd.DataFrame(results)
    df_fn.to_csv("outputs/fn_deep_dive.csv", index=False)
    print("\nReport saved to outputs/fn_deep_dive.csv")

if __name__ == "__main__":
    main()
