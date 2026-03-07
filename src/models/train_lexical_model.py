# src/models/train_lexical_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.combine import SMOTEENN
import xgboost as xgb

def train_lexical_model():
    # Load training data
    df = pd.read_pickle("models/raw_data.pkl")
    
    # Extract ONLY lexical features
    from src.features.lexical_features import extract_url_features
    df_features = extract_url_features(df, domain_col='domain')
    
    # Select lexical features only
    feature_cols = [col for col in df_features.columns if col not in ['label', 'domain', 'cse_name']]
    X = df_features[feature_cols]
    y = df_features['label']
    
    # Convert string labels to numerical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Preprocessing
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Split BEFORE balancing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Balance TRAIN only
    smoteenn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smoteenn.fit_resample(X_train, y_train)

    # Train
    model = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    # Eval on test
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report
    print("\n📊 Lexical Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model AND the FULL feature set
    joblib.dump(model, "models/lexical_model.pkl")
    joblib.dump(selector, "models/lexical_selector.pkl")
    joblib.dump(scaler, "models/lexical_scaler.pkl")
    joblib.dump(selected_features, "models/lexical_features.pkl")
    joblib.dump(feature_cols, "models/lexical_full_features.pkl")  # NEW: Save full feature set
    joblib.dump(label_encoder, "models/lexical_label_encoder.pkl")
    print("✅ Lexical model saved!")
    print(f"📊 Original features: {len(feature_cols)}")
    print(f"📊 Selected features: {len(selected_features)}")

if __name__ == "__main__":
    train_lexical_model()