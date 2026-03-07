# src/models/augment_data.py
import pandas as pd
import numpy as np
import os
from .config import MODEL_DIR  # Assumes your config

def augment_training_data(num_synthetic=200):
    # Load original
    df_original = pd.read_pickle(os.path.join(MODEL_DIR, "raw_data.pkl"))
    
    # Generate synthetic Phishing (typosquatting CSE names)
    cse_entities = ['sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'crsorgi', 'irctc', 'rbi', 'iocl']
    tlds = ['.in', '.com', '.xyz', '.top', '.shop', '.net', '.co.in']
    typos = ['bannk', 'phish', 'fake', 'scam', 'typo', 'merch', 'reward', 'login', 'card', 'bank']
    
    synthetic_domains = []
    synthetic_cse = []
    for _ in range(num_synthetic):
        entity = np.random.choice(cse_entities)
        typo = np.random.choice(typos)
        tld = np.random.choice(tlds)
        domain = f"{typo}{entity}{np.random.choice(['-', ''])}phish{tld}"
        synthetic_domains.append(domain)
        synthetic_cse.append(f"{entity.capitalize()} Bank/Entity")  # Match original style
    
    df_synthetic = pd.DataFrame({
        'domain': synthetic_domains,
        'cse_name': synthetic_cse,
        'label': ['Phishing'] * num_synthetic
    })
    
    # Combine (shuffle)
    df_augmented = pd.concat([df_original, df_synthetic], ignore_index=True).sample(frac=1).reset_index(drop=True)
    
    # Save
    augmented_path = os.path.join(MODEL_DIR, "augmented_training.xlsx")
    df_augmented.to_excel(augmented_path, index=False)
    print(f"✅ Augmented data saved: {len(df_augmented)} rows (original {len(df_original)} + {num_synthetic} synthetic)")
    print(f"New label dist: {df_augmented['label'].value_counts().to_dict()}")
    
    return df_augmented

if __name__ == "__main__":
    augment_training_data(200)