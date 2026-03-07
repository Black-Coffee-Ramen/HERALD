# src/main_pipeline.py
import pandas as pd
import numpy as np
import joblib
from src.core.content_classifier import ContentClassifier
from src.utils.html_fetcher import HTMLFetcher
from src.features.lexical_features import extract_url_features
from src.features.whois_features import extract_whois_features
import logging

class EnhancedPhishingPipeline:
    def __init__(self):
        self.content_classifier = ContentClassifier()
        self.html_fetcher = HTMLFetcher()
        self.load_models()
    
    def load_models(self):
        """Load ML models"""
        try:
            self.lexical_model = joblib.load("models/lexical_model.pkl")
            self.lexical_scaler = joblib.load("models/lexical_scaler.pkl")
            self.lexical_selector = joblib.load("models/lexical_selector.pkl")
            self.lexical_features = joblib.load("models/lexical_full_features.pkl")
            self.models_loaded = True
        except Exception as e:
            logging.warning(f"Could not load ML models: {e}")
            self.models_loaded = False
    
    def analyze_domain(self, domain, target_cse):
        """Enhanced domain analysis with content classification"""
        logging.info(f"🔍 Analyzing: {domain} -> {target_cse}")
        
        # 1. Get lexical features
        lexical_features = self.get_lexical_features(domain)
        
        # 2. Get WHOIS features
        whois_features = self.get_whois_features(domain)
        
        # 3. Get HTML content
        html_content = self.html_fetcher.fetch_with_retry(domain)
        
        # 4. Content-based classification
        content_classification = self.content_classifier.analyze_content(
            html_content, domain, target_cse
        )
        
        # 5. Calculate lexical suspicion score
        lexical_suspicion = self.calculate_lexical_suspicion(lexical_features, domain, target_cse)
        
        # 6. Final classification
        final_label, confidence = self.final_classification(
            content_classification, lexical_suspicion, whois_features, domain, target_cse
        )
        
        return {
            'domain': domain,
            'target_cse': target_cse,
            'final_label': final_label,
            'confidence': confidence,
            'content_classification': content_classification,
            'lexical_suspicion': lexical_suspicion,
            'domain_age': whois_features.get('domain_age_days', -1),
            'html_content_length': len(html_content) if html_content else 0
        }
    
    def get_lexical_features(self, domain):
        """Extract lexical features"""
        try:
            df_temp = pd.DataFrame([{'domain': domain}])
            features_df = extract_url_features(df_temp, domain_col='domain')
            return features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
        except Exception as e:
            logging.error(f"Error extracting lexical features for {domain}: {e}")
            return {}
    
    def get_whois_features(self, domain):
        """Extract WHOIS features"""
        try:
            df_temp = pd.DataFrame([{'domain': domain}])
            features_df = extract_whois_features(df_temp, domain_col='domain')
            return features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
        except Exception as e:
            logging.error(f"Error extracting WHOIS features for {domain}: {e}")
            return {}
    
    def calculate_lexical_suspicion(self, lexical_features, domain, target_cse):
        """Calculate lexical suspicion score (0-1)"""
        if not self.models_loaded:
            # Fallback: simple rule-based scoring
            return self.rule_based_suspicion(domain, target_cse)
        
        try:
            # Prepare features for ML model
            feature_vector = []
            for feature_name in self.lexical_features:
                feature_vector.append(lexical_features.get(feature_name, 0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Transform and predict
            features_selected = self.lexical_selector.transform(feature_vector)
            features_scaled = self.lexical_scaler.transform(features_selected)
            probability = self.lexical_model.predict_proba(features_scaled)[0][1]
            
            return probability
        except Exception as e:
            logging.error(f"ML prediction failed for {domain}: {e}")
            return self.rule_based_suspicion(domain, target_cse)
    
    def rule_based_suspicion(self, domain, target_cse):
        """Rule-based fallback when ML models aren't available"""
        domain_lower = domain.lower()
        suspicion_score = 0
        
        # Typosquatting indicators
        if self.is_typosquatting(domain, target_cse):
            suspicion_score += 0.3
        
        # Suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club']
        if any(domain_lower.endswith(tld) for tld in suspicious_tlds):
            suspicion_score += 0.2
        
        # Length-based suspicion
        if len(domain) > 30:
            suspicion_score += 0.1
        
        # Hyphen count
        if domain.count('-') >= 2:
            suspicion_score += 0.1
        
        return min(suspicion_score, 1.0)
    
    def is_typosquatting(self, domain, target_cse):
        """Check for typosquatting patterns"""
        domain_clean = domain.lower().replace('www.', '').replace('-', '')
        
        # Get CSE keywords
        cse_keywords = self.content_classifier._get_brand_keywords(target_cse)
        
        for keyword in cse_keywords:
            if keyword in domain_clean and len(keyword) > 2:
                return True
        return False
    
    def final_classification(self, content_class, lexical_suspicion, whois_features, domain, target_cse):
        """Final classification logic"""
        domain_age = whois_features.get('domain_age_days', 1000)
        is_new_domain = domain_age < 30
        
        # DECISION MATRIX:
        if content_class == "Phishing":
            return "Phishing", max(lexical_suspicion, 0.8)
        
        elif content_class == "Legitimate Service":
            return "Legitimate", 0.1  # Very low suspicion
        
        elif content_class == "Suspected":
            # Suspected domains = lexical suspicion but no phishing content
            if lexical_suspicion > 0.7 and is_new_domain:
                return "Suspected", lexical_suspicion
            elif lexical_suspicion > 0.8:
                return "Suspected", lexical_suspicion
            else:
                return "Legitimate", 1 - lexical_suspicion
        
        else:
            return "Legitimate", 0.2