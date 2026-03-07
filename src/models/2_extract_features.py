# src/models/2_extract_features.py
import pandas as pd
import os
from .config import MODEL_DIR
from src.features.lexical_features import extract_url_features  # Updated function

def extract_features():
    df = pd.read_pickle(os.path.join(MODEL_DIR, "raw_data.pkl"))
    df_features = extract_url_features(df)
    df_features.to_pickle(os.path.join(MODEL_DIR, "features.pkl"))
    print(f"✅ Advanced features extracted! Shape: {df_features.shape}")

if __name__ == "__main__":
    extract_features()