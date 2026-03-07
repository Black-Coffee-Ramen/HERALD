import os
import joblib
import pandas as pd
import numpy as np
import yaml
from herald.features.lexical_features import extract_url_features
from herald.core.cv_ocr_analyzer import CVOCRAnalyzer

class PhishingPredictorV3:
    def __init__(self, model_path="models/ensemble_v3.joblib", config_path="config.yaml"):
        print("Loading v3 Phishing Predictor with Fallback...")
        self.ensemble = joblib.load(model_path)
        self.rf = self.ensemble['rf']
        self.xgb = self.ensemble['xgb']
        self.feature_names = self.ensemble['features']
        
        # CSE Keywords for brand matching
        from herald.features.lexical_features import CSE_KEYWORDS
        self.cse_keywords = CSE_KEYWORDS
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.threshold = self.config['thresholds']['phishing']
        self.fallback_trigger = self.config['thresholds']['suspected']
        
        # Lazy load OCR analyzer to save resources unless needed
        self._ocr_analyzer = None

    def _extract_brand(self, domain):
        """Extract brand keyword from domain for OCR comparison"""
        domain_lower = domain.lower()
        for kw in self.cse_keywords:
            if kw in domain_lower:
                return kw
        return "Generic"

    @property
    def ocr_analyzer(self):
        if self._ocr_analyzer is None:
            self._ocr_analyzer = CVOCRAnalyzer()
        return self._ocr_analyzer

    def predict(self, domain, cse_name=None):
        """Predict if a domain is phishing with ML + Content Fallback"""
        # 1. ML Prediction
        df = pd.DataFrame([{'domain': domain}])
        df_features = extract_url_features(df, domain_col='domain')
        X = df_features[self.feature_names].fillna(0)
        
        rf_proba = self.rf.predict_proba(X)[0, 1]
        xgb_proba = self.xgb.predict_proba(X)[0, 1]
        ml_conf = (rf_proba + xgb_proba) / 2
        
        result = {
            'domain': domain,
            'ml_confidence': ml_conf,
            'status': 'Clean',
            'analysis_type': 'ML-Lexical'
        }
        
        # 2. Threshold Logic
        if ml_conf >= self.threshold:
            result['status'] = 'Phishing'
            return result
            
        # 3. Content-Based Fallback for Borderline Cases
        if ml_conf >= self.fallback_trigger:
            print(f"DEBUG: Borderline case ({ml_conf:.3f}). Triggering OCR fallback for {domain}...")
            
            if not cse_name:
                cse_name = self._extract_brand(domain)
                
            ocr_res = self.ocr_analyzer.analyze_domain(domain, cse_name, ml_conf)
            
            # Check visual similarity score/match
            # CVOCRAnalyzer.perceptual_hash + hamming_distance
            # distance <= 20 is 'visual_match' in the class implementation
            
            indicators = eval(ocr_res['phishing_indicators'])
            visual_match = indicators.get('visual_match', False)
            
            if visual_match:
                print(f"DEBUG: Visual similarity confirmed phishing!")
                result['status'] = 'Phishing'
                result['analysis_type'] = 'ML + OCR Fallback'
                result['ocr_details'] = ocr_res
            elif ocr_res['cv_ocr_status'] == 'Unable to capture screenshot':
                # Page might be empty or unreachable
                result['status'] = 'Suspected'
                result['analysis_type'] = 'ML + Reachability Check'
            else:
                result['status'] = 'Suspected'
                result['analysis_type'] = 'ML-Lexical'
                
        return result

if __name__ == "__main__":
    predictor = PhishingPredictorV2()
    # Test case
    test_domain = "onlinesbi.verify-update.com"
    print(predictor.predict(test_domain, "SBI"))
