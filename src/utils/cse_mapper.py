# src/utils/cse_mapper.py
import pandas as pd
import os
import re
from fuzzywuzzy import fuzz  # Requires pip install fuzzywuzzy python-Levenshtein

def load_cse_reference():
    """
    Load official CSE domains dataset.
    Returns a dict: {domain: organisation_name}
    """
    ref_path = "data/processed/PS-02  Phishing Detection CSE_Domains_Dataset_for_Stage_1.xlsx"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference dataset not found: {ref_path}")
    
    df = pd.read_excel(ref_path)
    # Forward-fill missing organisation names
    df['Organisation Name'] = df['Organisation Name'].ffill()
    df = df.dropna(subset=['Whitelisted Domains'])
    
    # Normalize domains
    df['Whitelisted Domains'] = df['Whitelisted Domains'].astype(str).str.strip().str.lower()
    cse_domain_to_name = dict(zip(df['Whitelisted Domains'], df['Organisation Name']))
    
    print(f"✅ Loaded {len(cse_domain_to_name)} CSE references")
    return cse_domain_to_name, df

def normalize_domain(domain):
    """
    Lowercase, remove 'www.', strip whitespace.
    """
    domain = domain.lower().strip()
    domain = re.sub(r'^www\.', '', domain)
    return domain

def map_phishing_domain_to_cse(phishing_domain, cse_domain_to_name=None, threshold=80):
    """
    Map a phishing domain to its CSE using fuzzy matching for typosquatting detection.
    Returns (CSE Name, official domain)
    """
    domain = normalize_domain(phishing_domain)
    
    # Load reference if not provided
    if cse_domain_to_name is None:
        cse_domain_to_name, _ = load_cse_reference()
    
    # Check for direct keyword matches (hard-coded for speed on common CSEs)
    keywords = {
        'crsorgi': ("Registrar General and Census Commissioner of India (RGCCI)", "dc.crsorgi.gov.in"),
        'irctc': ("Indian Railway Catering and Tourism Corporation (IRCTC)", "irctc.co.in"),
        'nic': ("National Informatics Centre (NIC)", "nic.gov.in"),
        'sbi': ("State Bank of India (SBI)", "onlinesbi.sbi"),
        'icici': ("ICICI Bank", "icicibank.com"),
        'hdfc': ("HDFC Bank", "hdfcbank.com"),
        'pnb': ("Punjab National Bank (PNB)", "pnbindia.in"),
        'bob': ("Bank of Baroda (BoB)", "bankofbaroda.in"),
        'airtel': ("Airtel", "airtel.in"),
        'iocl': ("Indian Oil Corporation Limited (IOCL)", "iocl.com")
    }
    
    domain_lower = domain.lower()
    for keyword, (name, official) in keywords.items():
        if keyword in domain_lower:
            return name, official
    
    # Fuzzy matching against reference domains (for typosquatting)
    best_match = None
    best_score = 0
    for official_domain, cse_name in cse_domain_to_name.items():
        score = fuzz.token_set_ratio(domain, official_domain)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = (cse_name, official_domain)
    
    if best_match:
        return best_match
    
    return "Unknown CSE", "unknown"

# Example usage
if __name__ == "__main__":
    cse_domains, ref_df = load_cse_reference()
    test_domains = [
        "onlinesbi.sbi", "dc.crsorgi.gov.in", "hdfclife.com", "unknownsite.com",
        "canonicalmetricvoice-search.xyz"  # Should map to Unknown
    ]
    for d in test_domains:
        cse_name, official = map_phishing_domain_to_cse(d, cse_domains)
        print(f"{d} → {cse_name} ({official})")