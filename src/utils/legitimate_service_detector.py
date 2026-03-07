# src/utils/legitimate_service_detector.py
import re
import pandas as pd
import requests
from urllib.parse import urlparse
import tldextract

class LegitimateServiceDetector:
    """
    Detect legitimate utility services to reduce false positives
    """
    
    # Known legitimate patterns for Indian financial/utility services
    LEGITIMATE_PATTERNS = [
        # RBI and banking utilities
        r"ifsc\.bankifsccode\.com",
        r"bankifsccode\.com",
        r"rbi\.org\.in",
        r"indianbank\.net\.in/ifsc",
        r"ifsccodebank\.in",
        r"allbankifsccode\.com",
        r"bankifsccode\.com",
        r"findifsc\.code\.in",
        
        # Government and utility services
        r"india\.gov\.in",
        r"nic\.in",
        r"digitalindia\.gov\.in",
        r"mygov\.in",
        r"incometaxindia\.gov\.in",
        r"incometaxindiaefiling\.gov\.in",
        r"epfindia\.gov\.in",
        r"uidai\.gov\.in",
        r"pgportal\.gov\.in",
        
        # Educational and research
        r"iit.*\.ac\.in",
        r"nit.*\.ac\.in",
        r"iim.*\.ac\.in",
        r"ugc\.ac\.in",
        r"aicte\.ind\.in",
        
        # Legitimate financial portals
        r"moneycontrol\.com",
        r"economictimes\.indiatimes\.com",
        r"nseindia\.com",
        r"bseindia\.com",
        
        # Payment gateways
        r"paytm\.com",
        r"phonepe\.com",
        r"googlepay\.com",
        r"amazonpay\.in",
        
        # Known safe domains with banking info
        r"bankbazaar\.com",
        r"paisabazaar\.com",
        r"policybazaar\.com"
    ]
    
    # Suspicious patterns that might be falsely flagged
    SUSPICIOUS_BUT_LEGITIMATE = [
        r".*login.*portal.*",
        r".*secure.*bank.*",
        r".*online.*payment.*",
        r".*verify.*account.*"
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.LEGITIMATE_PATTERNS]
        self.suspicious_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_BUT_LEGITIMATE]
    
    def is_legitimate_utility(self, domain):
        """
        Check if domain matches known legitimate utility patterns
        """
        domain_lower = domain.lower()
        
        # Check against legitimate patterns
        for pattern in self.compiled_patterns:
            if pattern.search(domain_lower):
                print(f"✅ Legitimate utility detected: {domain} matches {pattern.pattern}")
                return True
        
        return False
    
    def is_false_positive_candidate(self, domain, lexical_features=None):
        """
        Check if domain might be a false positive based on patterns and features
        """
        domain_lower = domain.lower()
        
        # Check legitimate patterns first
        if self.is_legitimate_utility(domain):
            return True
        
        # Check for suspicious-but-legitimate patterns with additional validation
        matches_suspicious_pattern = any(pattern.search(domain_lower) for pattern in self.suspicious_patterns)
        
        if matches_suspicious_pattern and lexical_features:
            # Additional validation for suspicious patterns
            return self.validate_suspicious_domain(domain, lexical_features)
        
        return False
    
    def validate_suspicious_domain(self, domain, lexical_features):
        """
        Additional validation for domains matching suspicious-but-legitimate patterns
        """
        # Extract domain age if available
        domain_age = lexical_features.get('domain_age_days', 0)
        
        # Long-standing domains are less likely to be phishing
        if domain_age > 365:  # More than 1 year old
            print(f"🔍 Old domain {domain} ({domain_age} days) - likely legitimate")
            return True
        
        # Check for professional TLDs
        professional_tlds = ['.com', '.org', '.net', '.in', '.co.in', '.org.in']
        domain_tld = '.' + domain.split('.')[-1]
        
        if domain_tld in professional_tlds:
            # Additional checks can be added here
            pass
        
        return False
    
    def get_legitimate_category(self, domain):
        """
        Categorize the legitimate service
        """
        domain_lower = domain.lower()
        
        categories = {
            'banking_utility': [
                r"ifsc\.", r"bankifsccode", r"ifsccode", r"bankifsc",
                r"rbi\.org\.in", r"indianbank\.net\.in/ifsc"
            ],
            'government': [
                r"\.gov\.in", r"india\.gov", r"nic\.in", r"digitalindia",
                r"mygov\.in", r"incometaxindia", r"epfindia", r"uidai"
            ],
            'educational': [
                r"\.ac\.in", r"ugc\.ac\.in", r"aicte\.ind\.in"
            ],
            'financial_portal': [
                r"moneycontrol", r"economictimes", r"nseindia", r"bseindia"
            ],
            'payment_gateway': [
                r"paytm", r"phonepe", r"googlepay", r"amazonpay"
            ]
        }
        
        for category, patterns in categories.items():
            for pattern in patterns:
                if re.search(pattern, domain_lower, re.IGNORECASE):
                    return category
        
        return 'other_legitimate'

# Singleton instance for easy import
detector = LegitimateServiceDetector()

def is_legitimate_utility(domain):
    """Convenience function for quick legitimate service detection"""
    return detector.is_legitimate_utility(domain)

def should_downgrade_prediction(domain, prediction, confidence, features=None):
    """
    Determine if prediction should be downgraded based on legitimate service patterns
    """
    if prediction in ['Phishing', 'Suspected']:
        if detector.is_false_positive_candidate(domain, features):
            print(f"🔧 Downgrading {domain} from {prediction} to Legitimate (known utility)")
            return 'Legitimate', confidence * 0.3  # Reduce confidence significantly
    return prediction, confidence