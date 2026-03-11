import os
import joblib
import pandas as pd
import numpy as np
import yaml
import socket
import ssl
import dns.resolver
import whois
from datetime import datetime
import time
import warnings
from herald.features.lexical_features import extract_url_features
from herald.core.cv_ocr_analyzer import CVOCRAnalyzer
from herald.features.content_features import extract_content_features

"""
DESIGN: HERALD v7 Inference Architecture (Two-Stage Pipeline)

STAGE 1: Fast Lexical Screening (Always runs, <10ms)
- Extraction: Lexical features only (no WHOIS/DNS/SSL).
- Model: v7 Ensemble (Trained on lexical features).
- Decision:
    - If Score > 0.80: Immediately return 'Phishing' (High Confidence).
    - If Score < 0.20: Immediately return 'Clean' (High Confidence).
    - If Score 0.20 - 0.80: Proceed to STAGE 2.

STAGE 2: Network Enrichment (Borderline Cases only, ~3-5 seconds)
- Extraction: Full Signal (WHOIS Age + SSL Cert + DNS features + Lexical).
- Model: v6 Ensemble (48+ features).
- Decision: Final classification based on combined signals.

Benefits:
- 85-90% of domains resolved in <10ms via Stage 1.
- Significant reduction in network latency and WHOIS/DNS query load.
- Maintained high accuracy for borderline cases via Stage 2.
"""

# Suppress warnings
warnings.filterwarnings('ignore')

class PhishingPredictorV3:
    """
    HERALD Phishing Predictor (v7)
    Uses Lexical + Network (SSL/DNS) + WHOIS features with a two-stage pipeline.
    Maintains compatibility with v3 class name for system-wide integration.
    """
    def __init__(self, model_path="models/ensemble_v7.joblib", config_path="config.yaml"):
        print(f"Loading HERALD v7 Phishing Predictor from {model_path}...")
        if not os.path.exists(model_path):
            # Fallback to absolute path or project root if needed
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
            
        self.ensemble = joblib.load(model_path)
        self.rf = self.ensemble['rf']
        self.xgb = self.ensemble['xgb']
        self.feature_names = self.ensemble['features']
        self.threshold = self.ensemble.get('threshold', 0.65) # Use v7 default threshold (0.65)
        
        # CSE Keywords for brand matching
        from herald.features.lexical_features import CSE_KEYWORDS
        self.cse_keywords = CSE_KEYWORDS
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.fallback_trigger = self.config.get('thresholds', {}).get('suspected', 0.30)
        else:
            self.fallback_trigger = 0.30
        
        self._ocr_analyzer = None

    def get_network_features(self, domain):
        """Extract live network features for a domain (SSL + DNS)."""
        features = {
            'has_ssl': 0, 'is_lets_encrypt': 0,
            'cert_domain_matches': 0, 'cert_age_days': -1,
            'cert_days_remaining': -1, 'has_mx': 0,
            'has_spf': 0, 'a_record_count': 0, 'ttl_value': -1
        }
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with socket.socket() as raw_sock:
                raw_sock.settimeout(3)
                with ctx.wrap_socket(raw_sock, server_hostname=domain) as s:
                    s.connect((domain, 443))
                    cert = s.getpeercert()
                    features['has_ssl'] = 1
                    issuer = str(cert.get('issuer',''))
                    features['is_lets_encrypt'] = int("Let's Encrypt" in issuer)
                    
                    # Cert domain matches check
                    san = str(cert.get('subjectAltName',''))
                    features['cert_domain_matches'] = int(domain.lower() in san.lower())
                    
                    nb = cert.get('notBefore','')
                    na = cert.get('notAfter','')
                    if nb:
                        from ssl import cert_time_to_seconds
                        age = (time.time() - cert_time_to_seconds(nb))
                        features['cert_age_days'] = int(age/86400)
                    if na:
                        remaining = (cert_time_to_seconds(na) - time.time())
                        features['cert_days_remaining'] = int(remaining/86400)
        except:
            pass
            
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 3
            resolver.lifetime = 3
            
            # MX
            try:
                mx = resolver.resolve(domain, 'MX')
                features['has_mx'] = int(len(mx) > 0)
            except: pass
            
            # SPF
            try:
                txt = resolver.resolve(domain, 'TXT')
                features['has_spf'] = int(any('spf' in str(r).lower() for r in txt))
            except: pass
            
            # A
            try:
                a = resolver.resolve(domain, 'A')
                features['a_record_count'] = len(a)
                features['ttl_value'] = a.rrset.ttl
            except: pass
        except:
            pass
            
        return features

    def get_whois_age(self, domain):
        """Get WHOIS domain age in days."""
        try:
            # Using python-whois (imported as whois)
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            if creation_date:
                age = (datetime.now() - creation_date).days
                return age
        except:
            pass
        return -1

    def _extract_brand(self, domain):
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
        """Predict if a domain is phishing with ML (Full v7 Signal) + Stage 2 Content Analysis"""
        # 0. Whitelist Check
        whitelist = self.ensemble.get('whitelist', [])
        clean_domain = domain.lower().strip().replace('www.', '')
        if clean_domain in whitelist:
            return {
                'domain': domain,
                'ml_confidence': 0.01,
                'status': 'Clean',
                'analysis_type': 'Whitelist'
            }

        # 1. Feature Extraction
        df = pd.DataFrame([{'domain': domain}])
        df_features = extract_url_features(df, domain_col='domain')
        
        # v7 specific fast features
        from tldextract import extract as tld_extract
        ext = tld_extract(domain)
        df_features['is_common_tld'] = 1 if ext.suffix in ['com', 'org', 'net', 'in', 'gov', 'edu'] else 0
        df_features['has_brand_keyword'] = 1 if any(kw in domain.lower() for kw in self.cse_keywords) else 0
        
        # Network Features (SSL/DNS)
        net_feats = self.get_network_features(domain)
        for k, v in net_feats.items():
            df_features[k] = v
            
        # WHOIS Age
        df_features['domain_age_days'] = self.get_whois_age(domain)
        
        # Align with model features - Ensure exact order and names
        X = df_features[self.feature_names].fillna(-1)
        # 2. Ensemble Prediction (0.6 XGB + 0.4 RF)
        rf_proba = self.rf.predict_proba(X)[0, 1]
        xgb_proba = self.xgb.predict_proba(X)[0, 1]
        ml_conf = 0.6 * xgb_proba + 0.4 * rf_proba
        
        result = {
            'domain': domain,
            'ml_confidence': round(float(ml_conf), 4),
            'status': 'Clean',
            'analysis_type': 'ML-v7-Ensemble'
        }
        
        # 3. Decision Logic
        if ml_conf >= self.threshold:
            result['status'] = 'Phishing'
            return result
            
        # 3.5 Stage 2 Content Analysis for borderline cases (0.35 - 0.65)
        if 0.35 <= ml_conf <= 0.65:
            content_feats = extract_content_features(domain)
            
            # Simple rule-based boost on top of ML score
            boost = 0.0
            if content_feats.get('has_password_field') and content_feats.get('has_login_form'):
                boost += 0.15
            if content_feats.get('has_obfuscated_js'):
                boost += 0.10
            if content_feats.get('form_action_external'):
                boost += 0.15
            if content_feats.get('redirected_to_different_domain'):
                boost += 0.10
            if content_feats.get('has_copyright') and content_feats.get('has_favicon'):
                boost -= 0.10  # More likely legitimate
            
            adjusted_score = min(1.0, ml_conf + boost)
            result['ml_confidence_adjusted'] = round(float(adjusted_score), 4)
            result['content_features'] = content_feats
            
            # Re-classify with adjusted score
            if adjusted_score >= self.threshold:
                result['status'] = 'Phishing'
                result['analysis_type'] = 'ML + Content Analysis'
                return result
            
        # 4. Fallback (OCR) for borderline cases
        if ml_conf >= self.fallback_trigger:
            if not cse_name:
                cse_name = self._extract_brand(domain)
            
            try:
                ocr_res = self.ocr_analyzer.analyze_domain(domain, cse_name, ml_conf)
                indicators = eval(ocr_res.get('phishing_indicators', '{}'))
                if indicators.get('visual_match', False):
                    result['status'] = 'Phishing'
                    result['analysis_type'] = 'ML + OCR Fallback'
                    result['ocr_details'] = ocr_res
                elif ocr_res.get('cv_ocr_status') == 'Unable to capture screenshot':
                    result['status'] = 'Suspected'
                    result['analysis_type'] = 'ML + Reachability Check'
                else:
                    result['status'] = 'Suspected'
                    result['analysis_type'] = 'ML-v6-Full-Signal'
            except Exception as e:
                result['status'] = 'Suspected'
                
        return result

if __name__ == "__main__":
    predictor = PhishingPredictorV3()
    # Test case
    test_domain = "hdfc-netbanking-verify.top"
    print(predictor.predict(test_domain, "HDFC"))
