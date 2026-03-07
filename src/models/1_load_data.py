# src/models/1_load_data.py
import pandas as pd
import os
from .config import DATA_PATH, MODEL_DIR

def load_training_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    
    df = pd.read_excel(DATA_PATH)
    print("Raw columns:", df.columns.tolist())
    
    # Map to standard names (use EXACT column names from your Excel)
    df = df.rename(columns={
        'Identified Phishing/Suspected Domain Name': 'domain',
        'Critical Sector Entity Name': 'cse_name',
        'Phishing/Suspected Domains (i.e. Class Label)': 'label'
    })
    
    # Keep only needed columns
    df = df[['domain', 'cse_name', 'label']].dropna(subset=['domain'])
    print(f"✅ Loaded {len(df)} samples | Labels: {df['label'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    df = load_training_data()
    df.to_pickle(os.path.join(MODEL_DIR, "raw_data.pkl"))