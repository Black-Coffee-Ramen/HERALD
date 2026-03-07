# src/features/whois_features.py
import pandas as pd
from datetime import datetime
import whois  # Requires 'pip install python-whois'

# Registrar risk scores (0.1 = trusted, 0.9 = high-risk)
# Keys normalized to uppercase for consistent matching
REGISTRAR_RISK_SCORE = {
    'NAMECHEAP': 0.9,
    'NAMECHEAP INC': 0.9,
    'GO DADDY, LLC': 0.7,
    'GODADDY.COM, LLC': 0.7,
    'TUCOWS DOMAINS, INC.': 0.6,
    'PORKBUN LLC': 0.8,
    'DYNADOT LLC': 0.75,
    'GMO INTERNET, INC.': 0.85,
    'HOSTINGER OPERATIONS, UAB': 0.9,
    'NAMESILO, LLC': 0.7,
    'SPACESHIP, INC.': 0.65,
    'IONOS SE': 0.3,
    'WIX.COM LTD.': 0.4,
    'SQUARESPACE DOMAINS LLC': 0.35,
    'NETWORK SOLUTIONS, LLC': 0.4,
    'AMAZON REGISTRAR, INC.': 0.2,
    'GOOGLE DOMAINS': 0.1,
    'DEFAULT': 0.5
}

def extract_whois_features(df, domain_col='domain'):
    """
    Extracts WHOIS features by querying live WHOIS data for each domain.
    Handles errors and provides defaults for failed queries.
    """
    df = df.copy()
    features = []

    for _, row in df.iterrows():
        domain = row[domain_col]
        try:
            w = whois.whois(domain)
            
            # Creation date
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            # Registrar
            registrar = str(w.registrar).upper() if w.registrar else None
            
            # Privacy protection check
            privacy_terms = ['redacted', 'privacy', 'protected', 'whoisguard', 'domain privacy']
            has_privacy = 0
            for field in ['name', 'org', 'registrant_name', 'registrant_org', 'email']:
                val = getattr(w, field, None)
                if val and any(term in str(val).lower() for term in privacy_terms):
                    has_privacy = 1
                    break
            
            # Domain age
            now = datetime.utcnow()
            if creation_date:
                if not isinstance(creation_date, datetime):
                    creation_date = pd.to_datetime(creation_date)
                age_days = (now - creation_date).days
            else:
                age_days = 9999  # Unknown = treat as old
            
            is_new = 1 if age_days < 30 else 0
            
            # Risk score
            risk_score = REGISTRAR_RISK_SCORE.get(registrar, REGISTRAR_RISK_SCORE['DEFAULT']) if registrar else 0.5
            
            has_whois = 1
        except Exception:
            # Defaults on failure (e.g., domain not registered, timeout, rate limit)
            age_days = 9999
            is_new = 0
            risk_score = 0.5
            has_privacy = 1  # Assume privacy if query fails (common for suspicious domains)
            has_whois = 0
        
        features.append({
            'domain_age_days': age_days,
            'is_new_domain': is_new,
            'registrar_risk_score': risk_score,
            'has_privacy_protection': has_privacy,
            'has_whois': has_whois
        })
    
    df_features = pd.DataFrame(features, index=df.index)
    return df_features