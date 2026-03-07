import pandas as pd
import os
import random

def generate_synthetic_phish():
    print("Generating synthetic short phishing domains...")
    
    cse_brands = ["sbi", "hdfc", "uidai", "irctc", "icici", "pnb", "bob", "airtel", "iocl", "nic"]
    suffixes = [
        "-login", "-secure", "-verify", "-update", "-help", "-support",
        "-banking", "-online", "login", "secure", "verify", "update"
    ]
    tlds = [".net", ".org", ".in", ".co", ".info", ".xyz", ".top"]
    
    synthetic_domains = []
    
    # Method 1: brand + suffix + tld
    for _ in range(100):
        brand = random.choice(cse_brands)
        suffix = random.choice(suffixes)
        tld = random.choice(tlds)
        domain = f"{brand}{suffix}{tld}"
        synthetic_domains.append({'domain': domain, 'label': 'Phishing'})
        
    # Method 2: subdomain brand
    for _ in range(100):
        brand = random.choice(cse_brands)
        sub = random.choice(["portal", "service", "secure", "check"])
        tld = random.choice(tlds)
        # sbi.portal-verify.in
        domain = f"{brand}.{sub}-{random.choice(suffixes)}{tld}"
        synthetic_domains.append({'domain': domain, 'label': 'Phishing'})
        
    df_synthetic = pd.DataFrame(synthetic_domains).drop_duplicates()
    
    os.makedirs("data/processed", exist_ok=True)
    df_synthetic.to_csv("data/processed/synthetic_phish_v2.csv", index=False)
    print(f"Generated {len(df_synthetic)} synthetic examples -> data/processed/synthetic_phish_v2.csv")

if __name__ == "__main__":
    generate_synthetic_phish()
