import os
import requests
import pandas as pd
import numpy as np
import yaml
from herald.predict_with_fallback import PhishingPredictorV3
from herald.features.lexical_features import CSE_KEYWORDS
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from urllib.parse import urlparse

# Set random seed
np.random.seed(42)

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhishDetect/1.0'}
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    print("Starting v3 External Validation (PhishTank 300 Samples)...")
    
    os.makedirs("data/external", exist_ok=True)
    pt_file = "data/external/phishtank_online.csv"
    pt_url = "http://data.phishtank.com/data/online-valid.csv"
    download_file(pt_url, pt_file)
    
    # 1. Load and Filter PhishTank
    print("Loading PhishTank...")
    df_pt = pd.read_csv(pt_file)
    
    # Filtering Criteria:
    # - Target brand contains any of our CSE keywords OR
    # - URL contains Indian financial/govt domain patterns
    
    indian_patterns = ['.gov.in', '.nic.in', '.ac.in', '.res.in', 'banking', 'payment', 'kyc', 'aadhar', 'pan-card', 'voterid']
    
    def is_indian_phish(row):
        url = str(row['url']).lower()
        # Some phishtank exports have 'target' or 'verified' columns, but the direct CSV usually has 'url'
        # Let's check for keywords in URL
        if any(kw in url for kw in CSE_KEYWORDS):
            return True
        if any(pat in url for pat in indian_patterns):
            return True
        return False

    df_pt['is_target'] = df_pt.apply(is_indian_phish, axis=1)
    df_pt_filtered = df_pt[df_pt['is_target']].copy()
    
    print(f"Found {len(df_pt_filtered)} matching Indian/Target phishing URLs.")
    
    # Remove training overlap
    from herald.utils.data_loader import load_training_data
    df_train = load_training_data()
    train_set = set(df_train['domain'].str.lower().values)
    
    df_pt_filtered['clean_domain'] = df_pt_filtered['url'].apply(lambda x: urlparse(x).netloc if isinstance(x, str) else "")
    df_pt_filtered = df_pt_filtered[~df_pt_filtered['clean_domain'].str.lower().isin(train_set)]
    print(f"Remaining after training set overlap removal: {len(df_pt_filtered)}")
    
    # Sample Phishing
    n_sample = min(len(df_pt_filtered), 50)
    df_phish = df_pt_filtered.sample(n=n_sample, random_state=42).copy()
    df_phish['label'] = 1
    df_phish = df_phish.rename(columns={'url': 'input_url'})
    
    # 2. Add Legitimate baseline (CSE domains) for Precision check
    cse_domains = df_train['cse_domain'].dropna().unique()
    n_legit = min(len(cse_domains), 50)
    df_legit = pd.DataFrame({
        'clean_domain': np.random.choice(cse_domains, n_legit, replace=False),
        'label': 0
    })
    
    # Combine
    df_test = pd.concat([
        df_phish[['clean_domain', 'label']],
        df_legit
    ], ignore_index=True)
    
    print(f"Test set created: {len(df_test)} samples ({n_sample} Phishing, {n_legit} Legitimate)")
    
    # 3. Running Predictions
    predictor = PhishingPredictorV3()
    
    results = []
    print("Running v3 predictions with fallback...")
    
    for i, row in df_test.iterrows():
        domain = row['clean_domain']
        print(f"[{i+1}/{len(df_test)}] Processing: {domain}...", end=" ", flush=True)
            
        res = predictor.predict(domain)
        status = res['status']
        print(f"Done -> {status}")
        results.append({
            'domain': domain,
            'true_label': row['label'],
            'pred_label': 1 if status == 'Phishing' else 0,
            'ml_conf': res['ml_confidence'],
            'analysis': res['analysis_type']
        })
        
    # 4. Reporting
    df_eval = pd.DataFrame(results)
    y_true = df_eval['true_label'].values
    y_pred = df_eval['pred_label'].values
    
    print("\n--- v3 EXTERNAL VALIDATION (PhishTank) ---")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))
    
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    print(f"\nExternal Precision: {p:.3f}")
    print(f"External Recall: {r:.3f}")
    
    # Analysis of FP if any
    if p < 0.85:
        print("\nWARNING: Precision dropped below 0.85! Analyzing FPs...")
        fps = df_eval[(df_eval['true_label'] == 0) & (df_eval['pred_label'] == 1)]
        for _, row in fps.iterrows():
            print(f"FP: {row['domain']} | ML Conf: {row['ml_conf']:.3f}")
            # Diagnostic: print feature values
            # (Requires re-extracting)
    
if __name__ == "__main__":
    main()
