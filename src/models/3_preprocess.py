# src/models/3_preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from .config import MODEL_DIR

def preprocess_data():
    df = pd.read_pickle(os.path.join(MODEL_DIR, "features.pkl"))
    
    # Select numeric features (exclude metadata)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if 'label' not in col.lower()]  # Safer exclude
    
    X = df[feature_cols]
    y = df['label']  # Keep as strings for now
    
    # Fill missing values
    X = X.fillna(0)
    
    # Feature selection (fit on full for simplicity; in prod, fit on train)
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # NO BALANCING HERE - move to train_model
    
    # Save artifacts (X_scaled is full preprocessed)
    pd.to_pickle(selected_features, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    pd.to_pickle(selector, os.path.join(MODEL_DIR, "feature_selector.pkl"))  # Note: Refit in train for no leakage
    pd.to_pickle(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    pd.to_pickle(X_scaled, os.path.join(MODEL_DIR, "X_processed.pkl"))
    pd.to_pickle(y, os.path.join(MODEL_DIR, "y_processed.pkl"))  # Strings
    
    print(f"✅ Preprocessing complete! Features: {len(selected_features)} | Samples: {len(X_scaled)}")

if __name__ == "__main__":
    preprocess_data()