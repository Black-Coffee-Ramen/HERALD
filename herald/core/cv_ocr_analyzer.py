# src/core/cv_ocr_analyzer.py
import pandas as pd
import cv2
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
import requests
import numpy as np

class CVOCRAnalyzer:
    def __init__(self):
        self.setup_components()
        
    def setup_components(self):
        """Setup ChromeDriver and EasyOCR"""
        # Setup EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'])
                print("EasyOCR initialized")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.reader = None
        else:
            print("EasyOCR not installed. Text extraction from images will be disabled.")
            self.reader = None
        
        # Setup ChromeDriver paths
        self.chromedriver_paths = [
            "/usr/local/bin/chromedriver",
            "/usr/bin/chromedriver",
            "chromedriver",
            "C:/Users/athiy/chromedriver-win64/chromedriver.exe",
            "chromedriver.exe",
        ]
    
    def setup_chromedriver(self):
        """Setup ChromeDriver using webdriver-manager"""
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        
        try:
            print("Auto-installing/updating ChromeDriver...")
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            driver.set_page_load_timeout(10)
            print("ChromeDriver setup successful via webdriver-manager")
            return driver
        except Exception as e:
            print(f"Webdriver-manager failed: {e}")
            
            # Fallback to manual paths
            for driver_path in self.chromedriver_paths:
                try:
                    print(f"Trying fallback ChromeDriver at: {driver_path}")
                    service = ChromeService(executable_path=driver_path)
                    driver = webdriver.Chrome(service=service, options=options)
                    print(f"ChromeDriver found at: {driver_path}")
                    return driver
                except Exception as e:
                    continue
        
        print("All ChromeDriver setup attempts failed")
        return None
    
    def is_domain_reachable(self, domain):
        """Check if domain is reachable"""
        import socket
        try:
            clean_domain = domain.split('/')[0]
            socket.gethostbyname(clean_domain)
            return True
        except socket.gaierror:
            return False
    
    def capture_screenshot(self, domain, driver):
        """Capture screenshot of domain"""
        if not self.is_domain_reachable(domain):
            return None
            
        try:
            # Try both protocols
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    print(f"Attempting: {url}")
                    driver.get(url)
                    time.sleep(3)
                    
                    # Create evidence directory
                    safe_domain = domain.replace('/', '_')
                    path = f"evidence/{safe_domain}/screenshot.png"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    driver.save_screenshot(path)
                    print(f"Screenshot captured: {path}")
                    return path
                except Exception as e:
                    continue
            return None
        except Exception as e:
            print(f"❌ Screenshot failed: {e}")
            return None
    
    def perceptual_hash(self, img_path, hash_size=8):
        """Generate perceptual hash for image comparison"""
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
        return bin(hash1 ^ hash2).count('1') if hash1 and hash2 else 100
    
    def extract_text_ocr(self, img_path):
        """Extract text from image using OCR"""
        if not EASYOCR_AVAILABLE or not self.reader:
            return []
        try:
            results = self.reader.readtext(img_path)
            return [result[1].lower() for result in results]
        except:
            return []
    
    def find_best_template(self, cse_name):
        """Find the best matching template for CSE"""
        templates_dir = "data/templates"
        if not os.path.exists(templates_dir):
            return None
            
        template_files = [f for f in os.listdir(templates_dir) if f.endswith('.png')]
        if not template_files:
            return None
        
        cse_normalized = cse_name.lower().replace('(', '').replace(')', '').replace('&', 'and').replace(' ', '_')
        
        best_match = None
        best_score = -1
        
        for template_file in template_files:
            template_path = os.path.join(templates_dir, template_file)
            template_lower = template_file.lower().replace('.png', '')
            score = 0
            
            # Scoring logic
            cse_keywords = ['irctc', 'sbi', 'state_bank', 'icici', 'hdfc', 'pnb', 'bob', 
                           'bank_of_baroda', 'airtel', 'iocl', 'nic', 'rgcci']
            
            for keyword in cse_keywords:
                if keyword in cse_normalized and keyword in template_lower:
                    score += 3
            
            if cse_normalized in template_lower:
                score += 4
                
            if score > best_score:
                best_score = score
                best_match = template_path
        
        return best_match
    
    def analyze_domain(self, domain, cse_name, initial_confidence):
        """Complete CV/OCR analysis for a domain"""
        print(f"Analyzing: {domain} -> {cse_name}")
        
        # Setup driver
        driver = self.setup_chromedriver()
        if not driver:
            return {
                'cv_ocr_confirmed': False,
                'cv_ocr_status': 'ChromeDriver not available',
                'final_confidence': initial_confidence,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'phishing_indicators': '{}',
                'visual_similarity': 'No analysis'
            }
        
        try:
            # Capture screenshot
            screenshot_path = self.capture_screenshot(domain, driver)
            if not screenshot_path:
                return {
                    'cv_ocr_confirmed': False,
                    'cv_ocr_status': 'Unable to capture screenshot',
                    'final_confidence': initial_confidence,
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'phishing_indicators': '{}',
                    'visual_similarity': 'No screenshot'
                }
            
            # OCR analysis
            extracted_text = self.extract_text_ocr(screenshot_path)
            text_match = any(pattern in ' '.join(extracted_text).lower() 
                           for pattern in ['login', 'password', 'username', 'signin'])
            
            # Visual similarity
            template_path = self.find_best_template(cse_name)
            visual_match = False
            visual_distance = 100
            
            if template_path and os.path.exists(template_path):
                hash1 = self.perceptual_hash(screenshot_path)
                hash2 = self.perceptual_hash(template_path)
                visual_distance = self.hamming_distance(hash1, hash2)
                visual_match = visual_distance <= 20
                print(f"Visual analysis: distance={visual_distance}, match={visual_match}")
            
            # Decision logic
            phishing_score = 0
            if text_match:
                phishing_score += 3
            if visual_match:
                phishing_score += 4
            
            is_phishing = phishing_score >= 4
            
            print(f"Analysis result: TextMatch={text_match}, VisualMatch={visual_match}, Score={phishing_score} -> {'PHISHING' if is_phishing else 'CLEAN'}")
            
            return {
                'cv_ocr_confirmed': is_phishing,
                'cv_ocr_status': 'Confirmed' if is_phishing else 'Not Confirmed',
                'final_confidence': 1.0 if is_phishing else initial_confidence * 0.7,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'phishing_indicators': f'{{"text_match": {text_match}, "visual_match": {visual_match}, "score": {phishing_score}}}',
                'visual_similarity': f'distance_{visual_distance}'
            }
            
        finally:
            driver.quit()