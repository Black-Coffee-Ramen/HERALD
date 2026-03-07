# src/features/content_classifier.py
import pandas as pd
import requests
import time
import os
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import TimeoutException, WebDriverException
import easyocr
from io import BytesIO
import joblib

class ContentClassifier:
    def __init__(self, template_dir="data/templates", evidence_dir="evidence"):
        self.template_dir = template_dir
        self.evidence_dir = evidence_dir
        self.setup_dependencies()
        
    def setup_dependencies(self):
        """Setup OCR, Selenium and other dependencies"""
        # Setup EasyOCR
        try:
            self.reader = easyocr.Reader(['en'])
            self.ocr_available = True
        except Exception as e:
            print(f"⚠️  EasyOCR initialization failed: {e}")
            self.ocr_available = False
        
        # Setup Selenium
        self.selenium_available = self.setup_chromedriver(test_only=True)
        
        # Create directories
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.evidence_dir, exist_ok=True)
    
    def setup_chromedriver(self, test_only=False):
        """Setup ChromeDriver for screenshot capture"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-gpu')
            
            possible_paths = [
                "/usr/local/bin/chromedriver",
                "/usr/bin/chromedriver",
                "chromedriver",
                "C:/Users/athiy/chromedriver-win64/chromedriver.exe",
                "chromedriver.exe",
            ]
            
            for driver_path in possible_paths:
                try:
                    service = ChromeService(executable_path=driver_path)
                    driver = webdriver.Chrome(service=service, options=options)
                    if test_only:
                        driver.quit()
                        return True
                    return driver
                except:
                    continue
            
            # Final attempt with system PATH
            driver = webdriver.Chrome(options=options)
            if test_only:
                driver.quit()
                return True
            return driver
            
        except Exception as e:
            print(f"❌ ChromeDriver setup failed: {e}")
            return False if test_only else None
    
    def is_live_content(self, domain, timeout=10):
        """
        Check if domain has live content (HTTP 200 + non-trivial HTML)
        Returns: (is_live, content_length, status_code, content_type)
        """
        try:
            # Try both HTTP and HTTPS
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    response = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
                    
                    # Check if content is non-trivial
                    content = response.text
                    content_length = len(content)
                    
                    # Heuristics for non-trivial content
                    has_substantial_content = content_length > 1000
                    has_html_structure = '<html' in content.lower() or '<body' in content.lower()
                    has_meaningful_tags = any(tag in content.lower() for tag in ['<div', '<form', '<input', '<button'])
                    
                    is_non_trivial = (has_substantial_content and 
                                    (has_html_structure or has_meaningful_tags))
                    
                    print(f"🌐 {domain}: Status {response.status_code}, Length {content_length}, "
                          f"Non-trivial: {is_non_trivial}")
                    
                    return (response.status_code == 200 and is_non_trivial, 
                           content_length, response.status_code, response.headers.get('content-type', ''))
                    
                except requests.exceptions.RequestException as e:
                    continue
            
            return False, 0, 0, ''
            
        except Exception as e:
            print(f"❌ Error checking live content for {domain}: {e}")
            return False, 0, 0, ''
    
    def capture_screenshot(self, domain, driver=None, timeout=15):
        """Capture screenshot of domain using Selenium"""
        external_driver = driver is not None
        if not external_driver:
            driver = self.setup_chromedriver()
            if not driver:
                return None
        
        try:
            driver.set_page_load_timeout(timeout)
            
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    driver.get(url)
                    time.sleep(3)  # Wait for page load
                    
                    screenshot_dir = f"{self.evidence_dir}/{domain}"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    screenshot_path = f"{screenshot_dir}/screenshot.png"
                    
                    driver.save_screenshot(screenshot_path)
                    print(f"✅ Screenshot captured: {screenshot_path}")
                    return screenshot_path
                    
                except TimeoutException:
                    continue
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ Screenshot failed for {domain}: {e}")
            return None
        finally:
            if not external_driver and driver:
                driver.quit()
    
    def extract_ui_elements(self, screenshot_path):
        """
        Extract UI elements from screenshot using OCR and image processing
        Returns: dict with detected elements and confidence scores
        """
        elements = {
            'login_form': False,
            'password_field': False,
            'username_field': False,
            'brand_logo': False,
            'submit_button': False,
            'security_indicators': False,
            'extracted_text': []
        }
        
        if not os.path.exists(screenshot_path):
            return elements
        
        try:
            # OCR Text Extraction
            if self.ocr_available:
                results = self.reader.readtext(screenshot_path)
                extracted_text = [result[1].lower() for result in results]
                elements['extracted_text'] = extracted_text
                
                # Check for login-related text
                login_keywords = ['login', 'sign in', 'username', 'password', 'email', 'account']
                password_keywords = ['password', 'pwd', 'passcode']
                username_keywords = ['username', 'email', 'user id', 'login id']
                submit_keywords = ['submit', 'sign in', 'log in', 'continue', 'enter']
                
                elements['login_form'] = any(keyword in ' '.join(extracted_text) for keyword in login_keywords)
                elements['password_field'] = any(keyword in ' '.join(extracted_text) for keyword in password_keywords)
                elements['username_field'] = any(keyword in ' '.join(extracted_text) for keyword in username_keywords)
                elements['submit_button'] = any(keyword in ' '.join(extracted_text) for keyword in submit_keywords)
            
            # Image-based element detection (basic)
            img = cv2.imread(screenshot_path)
            if img is not None:
                # Detect input-like elements (rectangles)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rectangle_count = 0
                
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:  # Rectangle detection
                        rectangle_count += 1
                
                # Multiple rectangles often indicate form fields
                if rectangle_count >= 2:
                    elements['login_form'] = True
            
            # Security indicator detection
            security_text = ['secure', 'https', 'ssl', 'protected', 'verified']
            if any(indicator in ' '.join(elements['extracted_text']) for indicator in security_text):
                elements['security_indicators'] = True
            
            print(f"🔍 UI Elements detected: { {k: v for k, v in elements.items() if v} }")
            return elements
            
        except Exception as e:
            print(f"⚠️  UI element extraction error: {e}")
            return elements
    
    def calculate_visual_similarity(self, screenshot_path, cse_name):
        """
        Calculate visual similarity with CSE templates
        Returns: similarity_score (0-1), has_visual_match (bool)
        """
        if not os.path.exists(screenshot_path):
            return 0.0, False
        
        try:
            # Find matching template
            template_path = self.find_matching_template(cse_name)
            if not template_path or not os.path.exists(template_path):
                return 0.0, False
            
            # Load images
            img1 = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0, False
            
            # Resize to same dimensions
            img1 = cv2.resize(img1, (300, 200))
            img2 = cv2.resize(img2, (300, 200))
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity, _ = ssim(img1, img2, full=True)
            
            # Perceptual hash comparison
            hash1 = self.perceptual_hash(screenshot_path)
            hash2 = self.perceptual_hash(template_path)
            hamming_dist = self.hamming_distance(hash1, hash2) if hash1 and hash2 else 64
            hash_similarity = 1 - (hamming_dist / 64.0)
            
            # Combined similarity score
            combined_similarity = (similarity + hash_similarity) / 2
            
            print(f"🔍 Visual similarity: SSIM={similarity:.3f}, Hash={hash_similarity:.3f}, Combined={combined_similarity:.3f}")
            
            return combined_similarity, combined_similarity > 0.6  # Threshold
            
        except Exception as e:
            print(f"⚠️  Visual similarity calculation error: {e}")
            return 0.0, False
    
    def perceptual_hash(self, img_path, hash_size=8):
        """Generate perceptual hash for image"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = cv2.resize(img, (hash_size + 1, hash_size))
            diff = img[1:, :] > img[:-1, :]
            return sum([2 ** i if val else 0 for i, val in enumerate(diff.flatten())])
        except Exception as e:
            return None
    
    def hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        return bin(hash1 ^ hash2).count('1') if hash1 and hash2 else 64
    
    def find_matching_template(self, cse_name):
        """Find template for CSE"""
        if not os.path.exists(self.template_dir):
            return None
        
        template_files = [f for f in os.listdir(self.template_dir) if f.endswith('.png')]
        if not template_files:
            return None
        
        # Normalize CSE name for matching
        cse_normalized = cse_name.lower().replace('(', '').replace(')', '').replace('&', 'and').replace(' ', '_')
        
        best_match = None
        best_score = -1
        
        for template_file in template_files:
            template_path = os.path.join(self.template_dir, template_file)
            template_lower = template_file.lower().replace('.png', '')
            
            score = 0
            
            # Exact match scoring
            cse_keywords = ['irctc', 'sbi', 'state_bank', 'icici', 'hdfc', 'pnb', 'bob', 
                           'airtel', 'iocl', 'nic', 'rgcci', 'census']
            
            for keyword in cse_keywords:
                if keyword in cse_normalized and keyword in template_lower:
                    score += 3
            
            # Partial match scoring
            cse_words = cse_normalized.split('_')
            for word in cse_words:
                if len(word) > 3 and word in template_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = template_path
        
        return best_match
    
    def has_cse_like_ui(self, ui_elements, cse_name):
        """
        Check if UI elements match typical CSE phishing patterns
        """
        # Required elements for phishing classification
        required_elements = [
            ui_elements['login_form'],
            ui_elements['password_field'] or ui_elements['username_field']
        ]
        
        has_required = all(required_elements)
        
        # Supporting elements
        supporting_elements = [
            ui_elements['submit_button'],
            ui_elements['brand_logo'],
            len(ui_elements['extracted_text']) > 5  # Substantial text content
        ]
        
        supporting_count = sum(supporting_elements)
        
        # Bank-specific additional checks
        if any(bank in cse_name.lower() for bank in ['bank', 'sbi', 'icici', 'hdfc']):
            has_banking_terms = any(term in ' '.join(ui_elements['extracted_text']) 
                                  for term in ['account', 'bank', 'card', 'secure'])
            supporting_count += 1 if has_banking_terms else 0
        
        print(f"🔍 CSE-like UI: Required={has_required}, Supporting={supporting_count}/3")
        
        return has_required and supporting_count >= 1
    
    def two_stage_classification(self, domain, lexical_prediction, lexical_confidence, cse_name):
        """
        Two-stage classification logic
        """
        print(f"\n{'='*60}")
        print(f"🔍 Two-Stage Classification: {domain}")
        print(f"📊 Lexical: {lexical_prediction} (confidence: {lexical_confidence:.3f})")
        print(f"🎯 Target CSE: {cse_name}")
        print(f"{'='*60}")
        
        # Stage 1: Lexical prediction check
        if lexical_prediction not in ['Phishing', 'Suspected']:
            print("✅ Stage 1: Lexical prediction is not Phishing/Suspected")
            return 'Legitimate', lexical_confidence, "Lexical prediction clean"
        
        print("🚨 Stage 1: Lexical prediction indicates Phishing/Suspected")
        
        # Stage 2: Live content check
        is_live, content_length, status_code, content_type = self.is_live_content(domain)
        
        if not is_live:
            print("⚠️  Stage 2: No live content detected")
            return 'Suspected', lexical_confidence * 0.7, "No live content"
        
        print("✅ Stage 2: Live content detected, proceeding to visual analysis")
        
        # Stage 3: Visual/OCR analysis
        screenshot_path = self.capture_screenshot(domain)
        if not screenshot_path:
            print("⚠️  Stage 3: Could not capture screenshot")
            return 'Suspected', lexical_confidence * 0.8, "Screenshot failed"
        
        # Extract UI elements
        ui_elements = self.extract_ui_elements(screenshot_path)
        
        # Calculate visual similarity
        visual_similarity, has_visual_match = self.calculate_visual_similarity(screenshot_path, cse_name)
        
        # Check for CSE-like UI elements
        has_cse_ui = self.has_cse_like_ui(ui_elements, cse_name)
        
        # Final decision logic
        if has_visual_match and has_cse_ui:
            final_confidence = min(1.0, lexical_confidence + 0.2)  # Boost confidence
            print(f"🚨 Stage 3: VISUAL PHISHING CONFIRMED")
            print(f"   - Visual similarity: {visual_similarity:.3f}")
            print(f"   - CSE-like UI: {has_cse_ui}")
            print(f"   - Final confidence: {final_confidence:.3f}")
            return 'Phishing', final_confidence, "Visual confirmation"
        else:
            final_confidence = lexical_confidence * 0.9  # Slight reduction
            print(f"⚠️  Stage 3: Visual analysis inconclusive")
            print(f"   - Visual similarity: {visual_similarity:.3f}")
            print(f"   - CSE-like UI: {has_cse_ui}")
            print(f"   - Final confidence: {final_confidence:.3f}")
            return 'Suspected', final_confidence, "Visual analysis inconclusive"
    
    def batch_classify(self, df_predictions):
        """
        Apply two-stage classification to batch predictions
        """
        print("🎯 Starting two-stage batch classification...")
        
        # Initialize result columns
        df_predictions['final_label'] = df_predictions['predicted_label']
        df_predictions['final_confidence'] = df_predictions['confidence']
        df_predictions['classification_reason'] = 'Initial lexical'
        df_predictions['visual_similarity'] = 0.0
        df_predictions['has_live_content'] = False
        df_predictions['ui_elements_detected'] = ''
        
        # Filter domains for two-stage analysis
        analysis_domains = df_predictions[
            df_predictions['predicted_label'].isin(['Phishing', 'Suspected'])
        ].head(100)  # Limit for performance
        
        print(f"🔍 Applying two-stage classification to {len(analysis_domains)} domains...")
        
        for idx, row in analysis_domains.iterrows():
            domain = row['domain']
            lexical_pred = row['predicted_label']
            lexical_conf = row['confidence']
            cse_name = row.get('target_cse', 'Unknown')
            
            try:
                final_label, final_confidence, reason = self.two_stage_classification(
                    domain, lexical_pred, lexical_conf, cse_name
                )
                
                # Update DataFrame
                df_predictions.loc[idx, 'final_label'] = final_label
                df_predictions.loc[idx, 'final_confidence'] = final_confidence
                df_predictions.loc[idx, 'classification_reason'] = reason
                
            except Exception as e:
                print(f"❌ Error classifying {domain}: {e}")
                df_predictions.loc[idx, 'classification_reason'] = f"Error: {str(e)}"
        
        return df_predictions

# Usage example and testing
if __name__ == "__main__":
    # Test the content classifier
    classifier = ContentClassifier()
    
    # Test single domain
    test_domain = "example.com"
    result = classifier.two_stage_classification(
        domain=test_domain,
        lexical_prediction="Phishing",
        lexical_confidence=0.85,
        cse_name="State Bank of India (SBI)"
    )
    
    print(f"\n🎯 Final Result: {result}")
