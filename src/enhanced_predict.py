import pandas as pd
import requests
import time
from src.core.content_classifier import ContentClassifier
from src.utils.legitimate_service_detector import LegitimateServiceDetector
from src.features.lexical_features import extract_url_features
from src.features.whois_features import extract_whois_features
import joblib
import os

## Temporary file 

class EnhancedPhishingDetector:
    def __init__(self):
        self.content_classifier = ContentClassifier()
        self.legitimate_detector = LegitimateServiceDetector()
        self.load_models()
    
    def load_models(self):
        """Load ML models"""
        try:
            self.lexical_model = joblib.load("models/lexical_model.pkl")
            self.lexical_scaler = joblib.load("models/lexical_scaler.pkl")
            self.lexical_selector = joblib.load("models/lexical_selector.pkl")
            self.lexical_features = joblib.load("models/lexical_full_features.pkl")
        except:
            print("⚠️  ML models not available, using rule-based approach only")
            self.lexical_model = None
    
    def fetch_html_content(self, domain, timeout=10):
        """Fetch HTML content from domain"""
        try:
            response = requests.get(f"http://{domain}", timeout=timeout, verify=False)
            return response.text
        except:
            try:
                response = requests.get(f"https://{domain}", timeout=timeout, verify=False)
                return response.text
            except:
                return None
    
    def capture_screenshot(self, domain):
        """Capture screenshot (simplified version)"""
        # Use your existing screenshot code
        return None  # Placeholder
    
    def analyze_domain(self, domain):
        """Enhanced domain analysis with content classification"""
        print(f"🔍 Analyzing: {domain}")
        
        # 1. Fetch content and screenshot
        html_content = self.fetch_html_content(domain)
        screenshot_path = self.capture_screenshot(domain)
        
        # 2. Content-based classification (PRIMARY)
        classification, content_confidence = self.content_classifier.classify_by_content(
            html_content, screenshot_path, domain
        )
        
        # 3. Check for legitimate services (FALSE POSITIVE PREVENTION)
        if html_content and self.legitimate_detector.is_legitimate_bank_service(html_content, domain):
            if classification == "Phishing":
                print(f"🔄 Overriding: {domain} is legitimate service")
                classification = "Legitimate"
                content_confidence = 0.1
        
        # 4. Lexical features (SECONDARY - for suspected domains)
        lexical_confidence = self.get_lexical_confidence(domain)
        
        # 5. Final decision
        final_result = self.final_classification(
            classification, content_confidence, lexical_confidence, domain
        )
        
        return final_result
    
    def get_lexical_confidence(self, domain):
        """Get confidence from lexical model"""
        if not self.lexical_model:
            return 0.5
            
        try:
            features_df = extract_url_features(pd.DataFrame([{'domain': domain}]))
            
            # Ensure all features are present
            for feature in self.lexical_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            X = features_df[self.lexical_features]
            X_selected = self.lexical_selector.transform(X)
            X_scaled = self.lexical_scaler.transform(X_selected)
            
            proba = self.lexical_model.predict_proba(X_scaled)[0][1]
            return proba
        except:
            return 0.5
    
    def final_classification(self, classification, content_confidence, lexical_confidence, domain):
        """Final classification decision"""
        result = {
            'domain': domain,
            'classification': classification,
            'content_confidence': content_confidence,
            'lexical_confidence': lexical_confidence,
            'final_confidence': max(content_confidence, lexical_confidence),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # If content says legitimate, trust it (prevents false positives)
        if classification == "Legitimate":
            result['final_classification'] = "Legitimate"
        # If content says phishing and high confidence, trust it
        elif classification == "Phishing" and content_confidence > 0.7:
            result['final_classification'] = "Phishing"
        # If content says suspected OR lexical says phishing but content unsure
        elif classification == "Suspected" or (lexical_confidence > 0.8 and content_confidence < 0.3):
            result['final_classification'] = "Suspected"
        else:
            result['final_classification'] = "Legitimate"
        
        return result

def main():
    """Main enhanced prediction pipeline"""
    print("🎯 ENHANCED PHISHING DETECTION WITH CONTENT ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    df_part1 = pd.read_excel("data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_1.xlsx")
    df_part1 = df_part1.rename(columns={df_part1.columns[0]: 'domain'})
    
    df_part2 = pd.read_excel("data/raw/PS-02_Shortlisting_set/Shortlisting_Data_Part_2.xlsx") 
    df_part2 = df_part2.rename(columns={df_part2.columns[0]: 'domain'})
    
    df_combined = pd.concat([df_part1, df_part2], ignore_index=True)
    print(f"📊 Total domains: {len(df_combined):,}")
    
    # Initialize detector
    detector = EnhancedPhishingDetector()
    
    # Analyze domains
    results = []
    phishing_count = 0
    suspected_count = 0
    
    for i, domain in enumerate(df_combined['domain']):
        if i % 100 == 0:
            print(f"📈 Progress: {i}/{len(df_combined)}")
        
        try:
            result = detector.analyze_domain(domain.strip())
            results.append(result)
            
            if result['final_classification'] == "Phishing":
                phishing_count += 1
                print(f"🚨 PHISHING: {domain}")
            elif result['final_classification'] == "Suspected":
                suspected_count += 1
                print(f"⚠️ SUSPECTED: {domain}")
                
        except Exception as e:
            print(f"❌ Error: {domain} - {e}")
            continue
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/enhanced_predictions.csv", index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 ENHANCED ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total analyzed: {len(results)}")
    print(f"🚨 Phishing: {phishing_count}")
    print(f"⚠️ Suspected: {suspected_count}") 
    print(f"✅ Legitimate: {len(results) - phishing_count - suspected_count}")
    print(f"💾 Results: outputs/enhanced_predictions.csv")

if __name__ == "__main__":
    main()