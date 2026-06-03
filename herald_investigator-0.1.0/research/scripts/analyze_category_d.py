import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features, CSE_KEYWORDS
import re
import Levenshtein

# Set random seed
np.random.seed(42)

# Ensure uidai is included for analysis
CSE_KEYWORDS_EXT = CSE_KEYWORDS + ['uidai']

def fuzzy_match(token, keywords, threshold=0.8):
    for kw in keywords:
        # Check if kw is a substring (already handled by Lexical)
        # Check for obfuscation (edit distance)
        dist = Levenshtein.distance(token, kw)
        # Simple heuristic: if length is similar and distance is small
        if 0 < dist <= 2 and len(token) > 3:
            return kw, dist
    return None, None

def main():
    print("Starting Category (d) FN Detail Analysis...")
    
    # 1. Load v2 Ensemble
    ensemble = joblib.load("models/ensemble_v2.joblib")
    rf = ensemble['rf']
    xgb = ensemble['xgb']
    feature_names = ensemble['features']
    
    # 2. Re-run v2 Split
    df_orig = load_training_data()
    synth_path = "data/processed/synthetic_phish_v2.csv"
    if os.path.exists(synth_path):
        df_synth = pd.read_csv(synth_path)
        df_combined = pd.concat([df_orig[['domain', 'label']], df_synth], ignore_index=True)
    else:
        df_combined = df_orig[['domain', 'label']].copy()
        
    df_features = extract_url_features(df_combined, domain_col='domain')
    X = df_features[feature_names].fillna(0)
    y = (df_features['label'] == 'Phishing').astype(int)
    
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 3. Identify Cat D FNs
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    ensemble_proba = (rf_proba + xgb_proba) / 2
    threshold = 0.571 # Operational threshold
    preds = (ensemble_proba >= threshold).astype(int)
    
    fn_mask = (y_test == 1) & (preds == 0)
    X_fn = X_test[fn_mask]
    conf_fn = ensemble_proba[fn_mask]
    domains_fn = df_combined.loc[X_fn.index, 'domain'].values
    
    # Filter for Category D (using the same logic as fn_deep_dive.py)
    cat_d_indices = []
    for idx, row in X_fn.iterrows():
        domain = df_combined.loc[idx, 'domain'].lower()
        if row['registered_domain_length'] < 12: continue # Cat A
        if not any(kw in domain for kw in CSE_KEYWORDS_EXT): continue # Cat B
        if row['subdomain_count'] >= 2: continue # Cat C
        cat_d_indices.append(idx)
        
    X_cat_d = X_fn.loc[cat_d_indices]
    domains_cat_d = df_combined.loc[cat_d_indices, 'domain'].values
    conf_cat_d = ensemble_proba[X_test.index.get_indexer(cat_d_indices)]
    
    print(f"\nAnalyzing {len(X_cat_d)} Category (d) False Negatives:")
    
    suspicious_tlds = ['.xyz', '.top', '.click', '.tk', '.cn', '.cf', '.ga', '.gq', '.ml', '.buzz', '.sbs', '.red', '.bet', '.fun', '.icu', '.cyou']
    
    sub_patterns = {
        'unusual_tld': 0,
        'obfuscated_kw': 0,
        'phishing_in_path': 0,
        'ip_based': 0
    }
    
    for i, (idx, row) in enumerate(X_cat_d.iterrows()):
        domain = domains_cat_d[i]
        conf = conf_cat_d[i]
        
        tld = row.get('tld', 'unknown') # TLD feature index or name
        # We need to map TLD index back or just extract from string
        actual_tld = "." + domain.split('.')[-1] if '.' in domain else ""
        
        # Sub-pattern check
        is_unusual_tld = any(actual_tld == st for st in suspicious_tlds)
        
        # Obfuscation check
        tokens = re.split(r'[^a-zA-Z0-9]', domain)
        obf_match = None
        for token in tokens:
            kw, dist = fuzzy_match(token, CSE_KEYWORDS_EXT)
            if kw:
                obf_match = (kw, dist)
                break
        
        # Path check
        has_path_kw = "/" in domain and any(kw in domain.split('/', 1)[1] for kw in CSE_KEYWORDS_EXT)
        
        # IP check
        is_ip = row['has_ip'] == 1
        
        if is_unusual_tld: sub_patterns['unusual_tld'] += 1
        if obf_match: sub_patterns['obfuscated_kw'] += 1
        if has_path_kw: sub_patterns['phishing_in_path'] += 1
        if is_ip: sub_patterns['ip_based'] += 1
        
        print(f"\n[{i+1}] Domain: {domain}")
        print(f"    TLD: {actual_tld} (Unusual: {is_unusual_tld})")
        print(f"    Conf: {conf:.3f} | Features: reg_len={row['registered_domain_length']}, digit_ratio={row['digit_ratio']:.2f}")
        print(f"    Pattern: Obfuscated: {obf_match}, PathKW: {has_path_kw}, IP: {is_ip}")
        
        # Why it looked legit?
        # Check features: maybe domain_length is high, or repeated_digits is 0
        legit_signs = []
        if row['entropy'] < 3.5: legit_signs.append("Low entropy")
        if row['special_char_ratio'] < 0.05: legit_signs.append("Few special chars")
        if row['is_common_tld'] == 1: legit_signs.append("Common TLD")
        print(f"    Possible Legit Signs: {', '.join(legit_signs)}")

    print("\n--- Sub-pattern Statistics ---")
    for k, v in sub_patterns.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
