# src/features/ssl_features.py
import pandas as pd

def extract_ssl_features(df):
    df = df.copy()
    
    # SSL presence (from your enriched data)
    df['has_ssl'] = df['Remarks (If any)'].str.contains('SSL: Yes', na=False).astype(int)
    
    # For now, issuer trust and validity require live checks (skip for Stage 1)
    df['ssl_issuer_trusted'] = 1  # Assume trusted if SSL exists
    df['ssl_valid'] = df['has_ssl']  # Assume valid if SSL exists
    
    return df[['has_ssl', 'ssl_issuer_trusted', 'ssl_valid']]