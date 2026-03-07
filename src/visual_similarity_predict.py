# src/visual_similarity_predict.py
import pandas as pd
import cv2
import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
import time
import socket

class VisualSimilarityAnalyzer:
    def __init__(self):
        self.templates_dir = "data/templates"
        self.setup_chromedriver()
        
    def setup_chromedriver(self):
        """Setup ChromeDriver for screenshot capture"""
        self.chromedriver_paths = [
            "/usr/local/bin/chromedriver",
            "/usr/bin/chromedriver",
            "chromedriver",
            "C:/Users/athiy/chromedriver-win64/chromedriver.exe",
            "chromedriver.exe",
        ]
    
    def get_chromedriver(self):
        """Get ChromeDriver instance"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        
        for driver_path in self.chromedriver_paths:
            try:
                service = ChromeService(executable_path=driver_path)
                driver = webdriver.Chrome(service=service, options=options)
                return driver
            except:
                continue
        return None
    
    def is_domain_reachable(self, domain):
        """Check if domain is reachable"""
        try:
            clean_domain = domain.split('/')[0]
            socket.gethostbyname(clean_domain)
            return True
        except socket.gaierror:
            return False
    
    def capture_screenshot(self, domain):
        """Capture screenshot of domain"""
        driver = self.get_chromedriver()
        if not driver:
            return None
            
        try:
            if not self.is_domain_reachable(domain):
                return None
                
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    driver.get(url)
                    time.sleep(3)
                    
                    safe_domain = domain.replace('/', '_')
                    path = f"evidence/{safe_domain}/screenshot.png"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    driver.save_screenshot(path)
                    return path
                except:
                    continue
            return None
        finally:
            driver.quit()
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for comparison"""
        if not os.path.exists(image_path):
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize to standard size for comparison
        img = cv2.resize(img, (800, 600))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def structural_similarity(self, img1, img2):
        """Calculate structural similarity between two images"""
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(img1, img2)
            return score
        except ImportError:
            # Fallback: use basic histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def template_matching(self, screenshot, template):
        """Perform template matching"""
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_val
    
    def find_best_template_match(self, screenshot_path, cse_name):
        """Find the best matching template for the screenshot"""
        if not os.path.exists(screenshot_path):
            return 0.0, "No screenshot"
            
        screenshot = self.load_and_preprocess_image(screenshot_path)
        if screenshot is None:
            return 0.0, "Invalid screenshot"
        
        best_similarity = 0.0
        best_template = None
        
        # Get all templates for this CSE
        cse_templates = self.get_cse_templates(cse_name)
        
        for template_path in cse_templates:
            template = self.load_and_preprocess_image(template_path)
            if template is None:
                continue
                
            # Calculate similarity
            similarity = self.structural_similarity(screenshot, template)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_template = os.path.basename(template_path)
        
        return best_similarity, best_template
    
    def get_cse_templates(self, cse_name):
        """Get all template files for a specific CSE"""
        if not os.path.exists(self.templates_dir):
            return []
            
        # Map CSE names to template patterns
        cse_patterns = {
            'Indian Railway Catering and Tourism Corporation (IRCTC)': ['irctc', 'railway'],
            'State Bank of India (SBI)': ['sbi', 'state_bank'],
            'ICICI Bank': ['icici'],
            'HDFC Bank': ['hdfc'],
            'Punjab National Bank (PNB)': ['pnb', 'punjab'],
            'Bank of Baroda (BOB)': ['bob', 'bank_of_baroda', 'baroda'],
            'Airtel': ['airtel'],
            'Indian Oil Corporation Limited (IOCL)': ['iocl', 'indian_oil'],
            'National Informatics Centre (NIC)': ['nic', 'national_informatics'],
            'Registrar General and Census Commissioner of India (RGCCI)': ['rgcci', 'census', 'registrar'],
            'Financial Institution (Generic)': ['bank', 'financial'],
            'Government Service (Generic)': ['gov', 'government'],
            'Telecom Service (Generic)': ['telecom', 'mobile']
        }
        
        templates = []
        template_files = [f for f in os.listdir(self.templates_dir) if f.endswith('.png')]
        
        if cse_name in cse_patterns:
            patterns = cse_patterns[cse_name]
            for template_file in template_files:
                template_lower = template_file.lower()
                if any(pattern in template_lower for pattern in patterns):
                    templates.append(os.path.join(self.templates_dir, template_file))
        
        # If no specific templates found, use all templates
        if not templates:
            templates = [os.path.join(self.templates_dir, f) for f in template_files]
            
        return templates
    
    def analyze_phishing_indicators(self, screenshot_path, domain, cse_name):
        """Analyze visual phishing indicators"""
        if not os.path.exists(screenshot_path):
            return {
                'is_phishing': False,
                'similarity_score': 0.0,
                'best_template': 'No screenshot',
                'reason': 'Cannot capture website',
                'confidence': 0.0
            }
        
        # Calculate similarity with templates
        similarity_score, best_template = self.find_best_template_match(screenshot_path, cse_name)
        
        # Phishing detection logic
        is_phishing = False
        reason = "Legitimate website"
        confidence = similarity_score
        
        # High similarity with CSE templates is strong indicator
        if similarity_score > 0.7:
            is_phishing = True
            reason = f"High visual similarity ({similarity_score:.2f}) with {cse_name} template"
            confidence = similarity_score
        elif similarity_score > 0.5:
            # Medium similarity - check additional factors
            domain_lower = domain.lower()
            suspicious_indicators = [
                'login' in domain_lower,
                'secure' in domain_lower, 
                'verify' in domain_lower,
                'account' in domain_lower,
                'update' in domain_lower,
                'portal' in domain_lower
            ]
            
            if sum(suspicious_indicators) >= 2:
                is_phishing = True
                reason = f"Medium visual similarity ({similarity_score:.2f}) + suspicious domain patterns"
                confidence = similarity_score * 0.8
            else:
                is_phishing = False
                reason = f"Medium visual similarity but no strong phishing indicators"
                confidence = similarity_score * 0.3
        else:
            is_phishing = False
            reason = f"Low visual similarity ({similarity_score:.2f}) with CSE templates"
            confidence = 0.0
        
        return {
            'is_phishing': is_phishing,
            'similarity_score': similarity_score,
            'best_template': best_template,
            'reason': reason,
            'confidence': confidence
        }

def run_visual_similarity_analysis():
    """Run visual similarity analysis on high-confidence domains"""
    print("🎯 STARTING VISUAL SIMILARITY ANALYSIS")
    print("=" * 60)
    
    # Check if predictions exist
    if not os.path.exists("outputs/enhanced_cse_predictions.csv"):
        print("❌ No enhanced predictions found. Run main prediction first.")
        print("💡 Run: python -m src.predict")
        return
    
    # Load predictions
    df_cse = pd.read_csv("outputs/enhanced_cse_predictions.csv")
    print(f"📊 Loaded {len(df_cse)} CSE-targeting domains")
    
    # Filter high-confidence domains
    high_conf_domains = df_cse[df_cse['confidence'] > 0.95].head(15)
    print(f"🔍 Analyzing {len(high_conf_domains)} high-confidence domains")
    
    # Initialize analyzer
    analyzer = VisualSimilarityAnalyzer()
    
    # Initialize result columns
    result_columns = [
        'visual_phishing', 'visual_similarity', 'best_template_match', 
        'visual_analysis_reason', 'visual_confidence', 'screenshot_path'
    ]
    
    for col in result_columns:
        if col not in df_cse.columns:
            df_cse[col] = ''
    
    analyzed_count = 0
    confirmed_phishing = 0
    
    for idx, row in high_conf_domains.iterrows():
        domain = row['domain'].strip()
        cse_name = row['target_cse']
        lexical_confidence = row['confidence']
        
        print(f"\n{'='*50}")
        print(f"🔬 [{analyzed_count + 1}/{len(high_conf_domains)}] {domain}")
        print(f"🎯 Target CSE: {cse_name}")
        print(f"📊 Lexical Confidence: {lexical_confidence:.4f}")
        print(f"{'='*50}")
        
        # Capture screenshot
        print("📸 Capturing screenshot...")
        screenshot_path = analyzer.capture_screenshot(domain)
        
        if screenshot_path:
            print(f"✅ Screenshot: {screenshot_path}")
            
            # Analyze visual similarity
            print("🔍 Analyzing visual similarity...")
            analysis_result = analyzer.analyze_phishing_indicators(screenshot_path, domain, cse_name)
            
            # Update results
            df_cse.at[idx, 'visual_phishing'] = analysis_result['is_phishing']
            df_cse.at[idx, 'visual_similarity'] = analysis_result['similarity_score']
            df_cse.at[idx, 'best_template_match'] = analysis_result['best_template']
            df_cse.at[idx, 'visual_analysis_reason'] = analysis_result['reason']
            df_cse.at[idx, 'visual_confidence'] = analysis_result['confidence']
            df_cse.at[idx, 'screenshot_path'] = screenshot_path
            
            status = "🚨 PHISHING" if analysis_result['is_phishing'] else "✅ CLEAN"
            print(f"📊 Visual Analysis: {status}")
            print(f"   Similarity: {analysis_result['similarity_score']:.3f}")
            print(f"   Reason: {analysis_result['reason']}")
            
            if analysis_result['is_phishing']:
                confirmed_phishing += 1
        else:
            print("❌ Could not capture screenshot")
            df_cse.at[idx, 'visual_analysis_reason'] = 'Cannot capture website'
        
        analyzed_count += 1
        
        # Save progress
        df_cse.to_csv("outputs/visual_similarity_analysis.csv", index=False)
        print(f"💾 Progress saved: {analyzed_count} domains analyzed")
    
    # Final save
    df_cse.to_csv("outputs/visual_similarity_analysis.csv", index=False)
    
    # Results summary
    print(f"\n{'='*60}")
    print("🎯 VISUAL SIMILARITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Domains analyzed: {analyzed_count}")
    print(f"🚨 Visually confirmed phishing: {confirmed_phishing}")
    print(f"✅ Clean domains: {analyzed_count - confirmed_phishing}")
    
    # Show confirmed phishing domains
    confirmed = df_cse[df_cse['visual_phishing'] == True]
    if len(confirmed) > 0:
        print(f"\n🔍 VISUALLY CONFIRMED PHISHING DOMAINS:")
        for _, row in confirmed.iterrows():
            similarity = row.get('visual_similarity', 0)
            print(f"   - {row['domain']} -> {row['target_cse']} (Similarity: {similarity:.3f})")
    
    # Show false positives (lexical phishing but visually clean)
    false_positives = df_cse[(df_cse['predicted_label'] == 'Phishing') & 
                            (df_cse['visual_phishing'] == False)]
    if len(false_positives) > 0:
        print(f"\n⚠️  POTENTIAL FALSE POSITIVES (Lexical phishing but visually clean):")
        for _, row in false_positives.head(5).iterrows():
            similarity = row.get('visual_similarity', 0)
            print(f"   - {row['domain']} -> {row['target_cse']} (Similarity: {similarity:.3f})")
    
    print(f"\n💾 Results saved to: outputs/visual_similarity_analysis.csv")

if __name__ == "__main__":
    run_visual_similarity_analysis()