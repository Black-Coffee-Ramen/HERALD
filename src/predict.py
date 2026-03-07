# src/predict.py (simplified) - CV/OCR functionality removed
import pandas as pd
import joblib
import numpy as np
import os
import warnings
import urllib3
from src.features.lexical_features import extract_url_features
from src.core.content_classifier import ContentClassifier


# Disable SSL warnings
warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_cse_reference():
    """
    Load CSE reference data
    Returns domain_to_name mapping and validation patterns
    """
    # Simplified CSE reference - you can expand this as needed
    cse_domain_to_name = {
        'irctc.co.in': 'Indian Railway Catering and Tourism Corporation (IRCTC)',
        'sbi.co.in': 'State Bank of India (SBI)',
        'icicibank.com': 'ICICI Bank',
        'hdfcbank.com': 'HDFC Bank',
        'pnb.co.in': 'Punjab National Bank (PNB)',
        'bankofbaroda.in': 'Bank of Baroda (BOB)',
        'airtel.in': 'Airtel',
        'iocl.com': 'Indian Oil Corporation Limited (IOCL)',
        'nic.in': 'National Informatics Centre (NIC)',
        'crsorgi.gov.in': 'Registrar General and Census Commissioner of India (RGCCI)'
    }
    
    validation_patterns = {
        'irctc': ['irctc', 'railway'],
        'sbi': ['sbi', 'statebank'],
        'icici': ['icici'],
        'hdfc': ['hdfc'],
        'pnb': ['pnb'],
        'bob': ['bankofbaroda', 'bob'],
        'airtel': ['airtel'],
        'iocl': ['iocl', 'indianoil'],
        'nic': ['nic', 'gov.in'],
        'census': ['crsorgi', 'censusindia']
    }
    
    return cse_domain_to_name, validation_patterns

def should_analyze_domain(domain, cse_domains):
    """
    SMART FILTER: Only analyze domains that actually target our CSEs
    """
    domain_lower = domain.lower()
    
    # CSE keywords - only these matter for the competition
    cse_keywords = [
        'irctc', 'sbi', 'statebank', 'icici', 'hdfc', 'pnb', 
        'bankofbaroda', 'bob', 'airtel', 'iocl', 'indianoil',
        'nic', 'gov', 'crsorgi', 'censusindia'
    ]
    
    # Only analyze if domain contains CSE keywords
    return any(keyword in domain_lower for keyword in cse_keywords)

def map_phishing_domain_to_cse(domain, cse_domain_to_name, validation_patterns):
    """
    SIMPLIFIED and more accurate CSE mapping
    """
    domain_lower = domain.lower()
    
    # Direct CSE mapping - be more specific
    cse_mapping = {
        'Indian Railway Catering and Tourism Corporation (IRCTC)': ['irctc'],
        'State Bank of India (SBI)': ['sbi', 'statebank', 'sbicard'],
        'ICICI Bank': ['icici'],
        'HDFC Bank': ['hdfc'],
        'Punjab National Bank (PNB)': ['pnb'],
        'Bank of Baroda (BOB)': ['bankofbaroda', 'bob'],
        'Airtel': ['airtel'],
        'Indian Oil Corporation Limited (IOCL)': ['iocl', 'indianoil'],
        'National Informatics Centre (NIC)': ['nic.', 'gov.in', 'india.gov'],
        'Registrar General and Census Commissioner of India (RGCCI)': ['crsorgi', 'censusindia']
    }
    
    # Exact matches first
    for cse_name, keywords in cse_mapping.items():
        for keyword in keywords:
            if keyword in domain_lower:
                # Validate it's not a legitimate domain
                if not is_likely_legitimate_domain(domain_lower, cse_name):
                    cse_domain = get_cse_domain(cse_name, cse_domain_to_name)
                    return cse_name, cse_domain
    
    return 'Unknown', 'unknown'

def is_likely_legitimate_domain(domain, cse_name):
    """
    Quick check for obvious legitimate domains
    """
    legitimate_indicators = [
        'ifsc', 'code', 'find', 'search', 'locate', 'directory',
        'calculator', 'emi', 'rate', 'info', 'portal'
    ]
    
    # If domain has service indicators, it's probably legitimate
    return any(indicator in domain for indicator in legitimate_indicators)

def get_cse_domain(cse_name, cse_domain_to_name):
    """Get official domain for CSE"""
    for domain, name in cse_domain_to_name.items():
        if name == cse_name:
            return domain
    return 'unknown'

def filter_false_positives(df_predictions):
    """
    Remove obvious false positives before final output
    """
    false_positive_indicators = [
        'ifsc', 'code', 'find', 'search', 'locate', 'directory',
        'calculator', 'emi', 'rate', 'info', 'portal', 'service'
    ]
    
    def is_likely_false_positive(domain, target_cse):
        domain_lower = domain.lower()
        
        # Domain has service keywords
        if any(indicator in domain_lower for indicator in false_positive_indicators):
            return True
        
        # Generic target with low specificity
        if target_cse in ['Financial Institution (Generic)', 'Government Service (Generic)']:
            return True
            
        return False
    
    df_predictions['is_false_positive'] = df_predictions.apply(
        lambda row: is_likely_false_positive(row['domain'], row['target_cse']), 
        axis=1
    )
    
    df_filtered = df_predictions[df_predictions['is_false_positive'] == False]
    
    print(f"🎯 Filtered {len(df_predictions) - len(df_filtered)} false positives")
    
    return df_filtered

def predict_with_ensemble():
    print("🔍 Starting enhanced ensemble phishing domain prediction...")
    
    # Load models
    try:
        lexical_model = joblib.load("models/lexical_model.pkl")
        lexical_encoder = joblib.load("models/lexical_label_encoder.pkl")
        lexical_scaler = joblib.load("models/lexical_scaler.pkl")
        lexical_selector = joblib.load("models/lexical_selector.pkl")
        lexical_full_features = joblib.load("models/lexical_full_features.pkl")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        return None
    
    # Load CSE reference data
    cse_domain_to_name, validation_patterns = load_cse_reference()
    print("✅ Loaded CSE reference data")
    
    # Load datasets
    print("📁 Loading shortlisting datasets...")
    
    df_part1 = pd.read_excel("data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_1.xlsx")
    df_part1 = df_part1.rename(columns={df_part1.columns[0]: 'domain'})
    df_part1['source_file'] = 'Part_1'
    
    df_part2 = pd.read_excel("data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_2.xlsx")
    df_part2 = df_part2.rename(columns={df_part2.columns[0]: 'domain'})
    df_part2['source_file'] = 'Part_2'
    
    df_combined = pd.concat([df_part1, df_part2], ignore_index=True)
    
    # 🎯 CRITICAL FIX: Filter domains to only those targeting CSEs
    print("🎯 Filtering domains to CSE-targeting only...")
    initial_count = len(df_combined)
    df_combined['should_analyze'] = df_combined['domain'].apply(
        lambda x: should_analyze_domain(x, cse_domain_to_name)
    )
    df_targeted = df_combined[df_combined['should_analyze'] == True].copy()
    
    print(f"📊 Reduced from {initial_count:,} to {len(df_targeted):,} targeted domains")
    
    if len(df_targeted) == 0:
        print("❌ No CSE-targeting domains found")
        return None
    
    # Lexical prediction
    print("🔍 Extracting lexical features...")
    df_lexical = extract_url_features(df_targeted, domain_col='domain')
    
    for feature in lexical_full_features:
        if feature not in df_lexical.columns:
            df_lexical[feature] = 0
    
    X_lexical = df_lexical[lexical_full_features]
    X_lexical_selected = lexical_selector.transform(X_lexical)
    X_lexical_scaled = lexical_scaler.transform(X_lexical_selected)
    
    lex_proba = lexical_model.predict_proba(X_lexical_scaled)
    ensemble_pred = (lex_proba[:, 1] > 0.5).astype(int)
    
    labels = lexical_encoder.inverse_transform([0, 1])
    df_targeted['predicted_label'] = [labels[pred] for pred in ensemble_pred]
    df_targeted['confidence'] = np.max(lex_proba, axis=1)
    
    # 🎯 IMPROVED CSE target identification
    print("🎯 Identifying targeted CSEs...")
    cse_targets = []
    cse_domains = []
    cse_confidences = []
    
    for domain in df_targeted['domain']:
        cse_name, cse_domain = map_phishing_domain_to_cse(domain, cse_domain_to_name, validation_patterns)
        cse_targets.append(cse_name)
        cse_domains.append(cse_domain)
        
        if cse_name != 'Unknown':
            confidence = 'High'
        else:
            # Only use generic categories for clear cases
            domain_lower = domain.lower()
            if any(word in domain_lower for word in ['bank', 'credit', 'loan', 'card']):
                cse_targets[-1] = 'Financial Institution (Generic)'
                confidence = 'Medium'
            elif any(word in domain_lower for word in ['gov', 'government', 'india']):
                cse_targets[-1] = 'Government Service (Generic)'
                confidence = 'Medium'
            elif any(word in domain_lower for word in ['airtel', 'jio', 'vi']):
                cse_targets[-1] = 'Telecom Service (Generic)'
                confidence = 'Medium'
            else:
                confidence = 'Low'
        
        cse_confidences.append(confidence)
    
    df_targeted['target_cse'] = cse_targets
    df_targeted['cse_domain'] = cse_domains
    df_targeted['cse_confidence'] = cse_confidences
    
    # 🎯 REMOVE CV/OCR - it's causing more problems than solutions
    df_targeted['escalation_reason'] = "Lexical analysis only"
    
    # Apply false positive filter
    df_targeted = filter_false_positives(df_targeted)
    
    # # Apply two-stage classification
    # from src.features.content_classifier import ContentClassifier
    
    print("\n🎯 Applying two-stage classification...")
    classifier = ContentClassifier()
    df_enhanced = classifier.batch_classify(df_targeted)
    
    # Save enhanced results
    df_enhanced.to_csv("outputs/enhanced_predictions_two_stage.csv", index=False)
    
    # Summary statistics
    original_phishing = len(df_targeted[df_targeted['predicted_label'] == 'Phishing'])
    final_phishing = len(df_enhanced[df_enhanced['final_label'] == 'Phishing'])
    final_suspected = len(df_enhanced[df_enhanced['final_label'] == 'Suspected'])
    
    print(f"\n📊 TWO-STAGE CLASSIFICATION SUMMARY:")
    print(f"   - Original Phishing count: {original_phishing}")
    print(f"   - Final Phishing count: {final_phishing}")
    print(f"   - Final Suspected count: {final_suspected}")
    print(f"   - Visual confirmations: {final_phishing - original_phishing}")
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    
    df_targeted.to_csv("outputs/shortlisting_predictions.csv", index=False)
    print(f"✅ Targeted predictions saved! → outputs/shortlisting_predictions.csv")
    
    # Enhanced CSE predictions
    cse_phishing = df_enhanced[
        (df_enhanced['final_label'] == 'Phishing') & 
        (df_enhanced['target_cse'] != 'Unknown')
    ].nlargest(1000, 'confidence')
    
    cse_phishing.to_csv("outputs/enhanced_cse_predictions.csv", index=False)
    print(f"🎯 CSE-targeting phishing saved ({len(cse_phishing)} domains)")
    
    # Summary
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"   - Total domains processed: {len(df_combined):,}")
    print(f"   - CSE-targeting domains analyzed: {len(df_targeted):,}")
    print(f"   - Final Phishing: {final_phishing:,}")
    print(f"   - Final Suspected: {final_suspected:,}")
    print(f"   - CSE-targeting phishing: {len(cse_phishing):,}")
    
    # Show top examples
    print(f"\n🔍 TOP CSE-TARGETING DOMAINS:")
    sample_cse = cse_phishing.head(10)
    for _, row in sample_cse.iterrows():
        print(f"   - {row['domain']} → {row['target_cse']} (Confidence: {row['confidence']:.4f})")
    
    return df_enhanced

if __name__ == "__main__":
    df_results = predict_with_ensemble()
    print("\n✅ PREDICTION COMPLETE!")