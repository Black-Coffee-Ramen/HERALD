# src/models/4_train_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .config import MODEL_DIR

def train_model():
    print("🔍 Loading processed data...")
    X = pd.read_pickle(os.path.join(MODEL_DIR, "X_processed.pkl"))
    y = pd.read_pickle(os.path.join(MODEL_DIR, "y_processed.pkl"))  # Strings

    print(f"📊 Shape after loading: X={X.shape}, y={y.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"📊 Feature columns: {joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))}")  # Load saved cols

    # Ensure X numeric
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        X = pd.DataFrame(X)  # If array, but should be array from preprocess

    print("🔍 Preparing for model training...")
    # Split BEFORE any preprocessing/balancing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"📊 Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"📊 Test set shape: X={X_test.shape}, y={y_test.shape}")

    # Refit selector/scaler on TRAIN only (avoid leakage)
    selector = VarianceThreshold(threshold=0.01)
    scaler = StandardScaler()
    X_train_selected = selector.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected = selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_selected)

    # Balance TRAIN only
    print("🔄 Balancing classes with SMOTEENN...")
    smoteenn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smoteenn.fit_resample(X_train_scaled, y_train)
    print(f"   After balancing: {X_train_bal.shape[0]} samples")

    # Train
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        missing=np.nan,
        verbosity=0  # Cleaner logs
    )
    print("🚀 Training the model...")
    model.fit(X_train_bal, y_train_bal)

    print("📈 Evaluating the model...")
    y_pred = model.predict(X_test_scaled)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("💾 Saving the model...")
    joblib.dump(model, os.path.join(MODEL_DIR, "phishing_detector_v3.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder_v3.pkl"))
    print("✅ Model and label encoder saved successfully!")

if __name__ == "__main__":
    train_model()