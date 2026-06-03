import os
import requests
import pandas as pd
import numpy as np
from herald.predict_with_fallback import PhishingPredictorV2
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

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
    print("Starting External Validation (PhishTank + CSE Clean)...")
    
    os.makedirs("data/external", exist_ok=True)
    
    # 1. Download PhishTank
    pt_url = "http://data.phishtank.com/data/online-valid.csv"
    pt_file = "data/external/phishtank_online.csv"
    download_file(pt_url, pt_file)
    
    # 2. Get Legitimate Domains from CSE list
    from herald.utils.data_loader import load_training_data
    df_orig = load_training_data()
    # Get unique CSE domains
    cse_domains = df_orig['cse_domain'].dropna().unique()
    print(f"Extracted {len(cse_domains)} unique legitimate CSE domains.")
    
    # 3. Load and Filter PhishTank
    print("Filtering PhishTank for Indian sector...")
    df_pt = pd.read_csv(pt_file)
    
    indian_keywords = [
        'sbi', 'hdfc', 'icici', 'pnb', 'bob', 'airtel', 'iocl', 'nic', 
        'gov.in', 'irctc', 'uidai', 'sbicard', 'icicibank', 'hdfcbank', 'onlinesbi'
    ]
    
    # Filter PT for keywords in URL
    mask = df_pt['url'].str.lower().apply(lambda x: any(kw in x for kw in indian_keywords))
    df_pt_filtered = df_pt[mask].copy()
    print(f"Found {len(df_pt_filtered)} matching Indian phishing URLs in PhishTank.")
    
    # Unseen external: filter out any that exist in training
    train_domains = set(df_orig['domain'].str.lower().values)
    df_pt_filtered['clean_domain'] = df_pt_filtered['url'].apply(lambda x: urlparse(x).netloc if isinstance(x, str) else "")
    df_pt_filtered = df_pt_filtered[~df_pt_filtered['clean_domain'].str.lower().isin(train_domains)]
    print(f"Remaining after removing training overlap: {len(df_pt_filtered)}")

    n_phish = min(len(df_pt_filtered), 100)
    df_pt_sampled = df_pt_filtered.sample(n=n_phish, random_state=42).copy()
    df_pt_sampled['label'] = 1
    df_pt_sampled = df_pt_sampled.rename(columns={'url': 'domain'})
    
    # Sample Legitimate
    n_legit = min(len(cse_domains), 100)
    legit_domains = pd.DataFrame({'domain': np.random.choice(cse_domains, n_legit, replace=False), 'label': 0})
    
    # Combine
    df_test = pd.concat([
        df_pt_sampled[['domain', 'label']],
        legit_domains
    ], ignore_index=True)
    
    print(f"Test set created: {len(df_test)} samples ({n_phish} Phishing, {n_legit} Legitimate)")
    
    # 4. Evaluate with v2 Predictor
    predictor = PhishingPredictorV2()
    
    preds = []
    print("Running predictions...")
    
    for i, row in df_test.iterrows():
        domain = row['domain']
        if isinstance(domain, str) and domain.startswith(('http://', 'https://')):
            clean_domain = urlparse(domain).netloc
        else:
            clean_domain = domain
            
        print(f"[{i+1}/{len(df_test)}] Processing: {clean_domain}...", end=" ", flush=True)
        res = predictor.predict(clean_domain)
        preds.append(1 if res['status'] == 'Phishing' else 0)
        print(f"Status: {res['status']} (ML: {res['ml_confidence']:.3f})")
        
    # 5. Report Results
    print("\n--- EXTERNAL VALIDATION PERFORMANCE ---")
    y_true = df_test['label'].values
    y_pred = np.array(preds)
    
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    print(f"\nExternal Precision: {p:.3f} (Target >= 0.92)")
    print(f"External Recall: {r:.3f} (Target >= 0.88)")

from urllib.parse import urlparse

if __name__ == "__main__":
    main()
