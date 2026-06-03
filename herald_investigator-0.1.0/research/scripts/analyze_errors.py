import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from herald.utils.data_loader import load_training_data
from herald.features.lexical_features import extract_url_features

# Set random seed
np.random.seed(42)

def main():
    print("Starting Error Analysis...")
    
    # 1. Load Data
    print("Loading training dataset...")
    df = load_training_data()
    
    # 2. Extract Features
    print("Extracting features...")
    df_features = extract_url_features(df, domain_col='domain')
    
    # Prepare X and y
    feature_cols = [col for col in df_features.columns if col not in ['S. No', 'S.No', 'label', 'domain', 'cse_name', 'cse_domain', 'evidence_filename', 'source', 'evidence_path', 'evidence_exists', 'is_cse_target', 'domain_clean', 'tld', 'true_label', 'true_label_clean']]
    X = df_features[feature_cols].fillna(0)
    
    # Convert labels: Phishing=1, others=0
    y = (df_features['label'] == 'Phishing').astype(int)
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Dataset size: {len(X)}")

    # 3. Train Models
    print("Training RF and XGBoost models...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    xgb = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42, eval_metric='logloss')
    
    rf.fit(X, y)
    xgb.fit(X, y)
    
    # 4. Predictions & Ensemble
    print("Generating predictions...")
    rf_proba = rf.predict_proba(X)[:, 1]
    xgb_proba = xgb.predict_proba(X)[:, 1]
    ensemble_proba = (rf_proba + xgb_proba) / 2
    predictions = (ensemble_proba > 0.5).astype(int)
    
    # 5. Bucket Predictions
    df_analysis = df[['domain', 'label']].copy()
    df_analysis['true_label'] = y
    df_analysis['predicted_label'] = predictions
    df_analysis['confidence_score'] = ensemble_proba
    df_analysis['rf_proba'] = rf_proba
    df_analysis['xgb_proba'] = xgb_proba
    
    # Identify buckets
    tp_mask = (df_analysis['true_label'] == 1) & (df_analysis['predicted_label'] == 1)
    tn_mask = (df_analysis['true_label'] == 0) & (df_analysis['predicted_label'] == 0)
    fp_mask = (df_analysis['true_label'] == 0) & (df_analysis['predicted_label'] == 1)
    fn_mask = (df_analysis['true_label'] == 1) & (df_analysis['predicted_label'] == 0)
    
    # 6. Error Analysis Details
    def get_top_features(row, X, importances, feature_names):
        # Multiply row values by importances to see which features contributed most
        # Simplified: top 3 most important features globally for this row
        indices = np.argsort(importances)[::-1][:3]
        return ", ".join([feature_names[i] for i in indices])

    feature_names = X.columns.tolist()
    global_importance = (rf.feature_importances_ + xgb.feature_importances_) / 2
    
    df_analysis['top_3_features_by_importance'] = df_analysis.apply(
        lambda row: get_top_features(row, X, global_importance, feature_names), axis=1
    )
    
    def identify_wrong_model(row):
        if row['true_label'] == 0 and row['predicted_label'] == 1: # FP
            if row['rf_proba'] > 0.5 and row['xgb_proba'] <= 0.5: return "RF"
            if row['xgb_proba'] > 0.5 and row['rf_proba'] <= 0.5: return "XGBoost"
            return "Both"
        if row['true_label'] == 1 and row['predicted_label'] == 0: # FN
            if row['rf_proba'] <= 0.5 and row['xgb_proba'] > 0.5: return "RF"
            if row['xgb_proba'] <= 0.5 and row['rf_proba'] > 0.5: return "XGBoost"
            return "Both"
        return "None"

    df_analysis['model_that_was_wrong'] = df_analysis.apply(identify_wrong_model, axis=1)
    
    # Export CSV
    os.makedirs("outputs", exist_ok=True)
    df_analysis.to_csv("outputs/error_analysis.csv", index=False)
    print("error_analysis.csv generated.")
    
    # 7. Print Top Examples
    print("\nTOP 10 FALSE POSITIVES:")
    fps = df_analysis[fp_mask].nlargest(10, 'confidence_score')
    for _, row in fps.iterrows():
        print(f"   {row['domain']} | Conf: {row['confidence_score']:.3f} | Wrong Model: {row['model_that_was_wrong']}")
        # Print top feature values for this domain
        # Get numeric index
        dom_idx = row.name
        top_feats = row['top_3_features_by_importance'].split(", ")
        vals = [f"{f}: {X.loc[dom_idx, f]}" for f in top_feats]
        print(f"     Features: {', '.join(vals)}")

    print("\nTOP 10 FALSE NEGATIVES:")
    fns = df_analysis[fn_mask].nsmallest(10, 'confidence_score')
    for _, row in fns.iterrows():
        print(f"   {row['domain']} | Conf: {row['confidence_score']:.3f} | Wrong Model: {row['model_that_was_wrong']}")
        dom_idx = row.name
        top_feats = row['top_3_features_by_importance'].split(", ")
        vals = [f"{f}: {X.loc[dom_idx, f]}" for f in top_feats]
        print(f"     Features: {', '.join(vals)}")

    # 8. Feature Importance Report
    print("\nFEATURE IMPORTANCE REPORT:")
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    xgb_imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    all_imp = pd.DataFrame({'RF': rf_imp, 'XGBoost': xgb_imp})
    all_imp['Difference'] = (all_imp['RF'] - all_imp['XGBoost']).abs()
    all_imp['Unstable'] = all_imp['Difference'] > 0.1
    
    print(all_imp.sort_values(by='Difference', ascending=False).head(20))
    
    unstable = all_imp[all_imp['Unstable']]
    if not unstable.empty:
        print("\nUNSTABLE FEATURES (Difference > 0.1):")
        print(unstable)
    else:
        print("\nNo unstable features found.")

    # 9. Confidence Distribution Plot
    print("\nGenerating confidence distribution plot...")
    plt.figure(figsize=(12, 8))
    
    buckets = {
        'Correct Phishing (TP)': df_analysis[tp_mask]['confidence_score'],
        'Correct Legitimate (TN)': df_analysis[tn_mask]['confidence_score'],
        'False Positives (FP)': df_analysis[fp_mask]['confidence_score'],
        'False Negatives (FN)': df_analysis[fn_mask]['confidence_score']
    }
    
    colors = ['green', 'blue', 'orange', 'red']
    for (name, scores), color in zip(buckets.items(), colors):
        if not scores.empty:
            plt.hist(scores, bins=20, alpha=0.5, label=f"{name} (n={len(scores)})", color=color, density=True)
            
    plt.title("Confidence Score Distribution by Classification Outcome")
    plt.xlabel("Confidence Score (Ensemble Probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/error_analysis_confidence.png")
    print("error_analysis_confidence.png saved.")

if __name__ == "__main__":
    main()
