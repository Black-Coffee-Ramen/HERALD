# src/core/domain_analyzer.py
import cv2
import imagehash
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime
import json
import os
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time

class DomainAnalyzer:
    def __init__(self):
        self.cse_reference = self.load_cse_reference()
        self.visual_threshold = 15  # phash Hamming distance
        self.lexical_threshold = 0.7
        self.max_domain_age = 90  # days
        
    def load_cse_reference(self):
        """Load CSE reference images and templates"""
        return {
            'National Informatics Centre (NIC)': {
                'reference_images': ['data/cse_refs/nic_login.png', 'data/cse_refs/nic_home.png'],
                'keywords': ['nic', 'national informatics', 'gov.in', 'digital india'],
                'login_selectors': ['#username', '#password', 'input[type="password"]']
            },
            'State Bank of India (SBI)': {
                'reference_images': ['data/cse_refs/sbi_login.png'],
                'keywords': ['sbi', 'state bank', 'onlinesbi'],
                'login_selectors': ['.login', '#username', '#password']
            },
            'IRCTC': {
                'reference_images': ['data/cse_refs/irctc_login.png'],
                'keywords': ['irctc', 'indian railway'],
                'login_selectors': ['#userId', '#password']
            }
        }
    
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        from collections import Counter
        import math
        
        if not text:
            return 0
        
        counter = Counter(text)
        text_length = len(text)
        entropy = 0
        
        for count in counter.values():
            p = count / text_length
            entropy -= p * math.log2(p)
            
        return entropy
    
    def get_tld_risk(self, domain):
        """Calculate TLD risk score"""
        risky_tlds = ['.xyz', '.top', '.club', '.online', '.site', '.website']
        safe_tlds = ['.gov.in', '.nic.in', '.co.in', '.ac.in', '.org', '.edu']
        
        domain_lower = domain.lower()
        
        for tld in risky_tlds:
            if domain_lower.endswith(tld):
                return 0.8
                
        for tld in safe_tlds:
            if domain_lower.endswith(tld):
                return 0.1
                
        return 0.3  # Neutral TLDs
    
    def fuzzy_match(self, keyword, domain, threshold=0.8):
        """Simple fuzzy matching using Levenshtein distance ratio"""
        from difflib import SequenceMatcher
        
        domain_lower = domain.lower()
        keyword_lower = keyword.lower()
        
        # Check for exact substring first
        if keyword_lower in domain_lower:
            return 1.0
            
        # Check for token-based matching
        domain_tokens = set(domain_lower.replace('-', ' ').split())
        keyword_tokens = set(keyword_lower.split())
        
        if keyword_tokens.intersection(domain_tokens):
            return 0.9
            
        # Use sequence matcher for fuzzy matching
        ratio = SequenceMatcher(None, keyword_lower, domain_lower).ratio()
        return ratio if ratio >= threshold else 0.0
    
    def detect_typosquatting(self, domain):
        """Detect typosquatting patterns"""
        # This would compare against known CSE domains
        # For now, return a basic score based on domain characteristics
        if '-' in domain and len(domain) > 15:
            return 0.7
        return 0.3
    
    def extract_lexical_features(self, domain):
        """Enhanced lexical analysis with fuzzy matching"""
        features = {
            'length': len(domain),
            'hyphen_count': domain.count('-'),
            'digit_ratio': sum(c.isdigit() for c in domain) / len(domain) if domain else 0,
            'entropy': self.calculate_entropy(domain),
            'brand_inclusion_score': 0,
            'tld_risk': self.get_tld_risk(domain),
            'typosquatting_score': self.detect_typosquatting(domain)
        }
        
        # Check for brand inclusion with fuzzy matching
        for cse_name, cse_info in self.cse_reference.items():
            for keyword in cse_info['keywords']:
                match_score = self.fuzzy_match(keyword, domain, threshold=0.8)
                features['brand_inclusion_score'] = max(
                    features['brand_inclusion_score'], 
                    match_score
                )
        
        lexical_score = self.calculate_lexical_score(features)
        return lexical_score, features
    
    def calculate_lexical_score(self, features):
        """Calculate overall lexical suspicion score"""
        weights = {
            'brand_inclusion_score': 0.4,
            'tld_risk': 0.2,
            'typosquatting_score': 0.15,
            'hyphen_count': 0.1,
            'entropy': 0.1,
            'digit_ratio': 0.05
        }
        
        score = 0
        for feature, weight in weights.items():
            score += features.get(feature, 0) * weight
            
        return min(1.0, score)  # Cap at 1.0
    
    def analyze_whois(self, domain):
        """Enhanced WHOIS analysis"""
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
                
            domain_age = (datetime.now() - creation_date).days if creation_date else 0
            
            whois_features = {
                'age_days': domain_age,
                'is_new_domain': domain_age <= self.max_domain_age,
                'registrar': w.registrar,
                'has_privacy': bool(w.name and 'redacted' in w.name.lower()),
                'registrant_org': w.org,
                'country': w.country
            }
            
            return whois_features
        except Exception as e:
            print(f"WHOIS analysis failed for {domain}: {e}")
            return {'age_days': 0, 'is_new_domain': True, 'registrar': 'Unknown'}
    
    def setup_driver(self):
        """Setup Chrome driver for screenshot capture with multiple fallback options"""
        from selenium.webdriver.chrome.service import Service as ChromeService
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
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
                driver.set_page_load_timeout(30)
                return driver
            except Exception:
                continue
                
        try:
            # Final attempt with system PATH
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            print(f"Failed to setup Chrome driver: {e}")
            return None
    
    def capture_screenshot(self, domain):
        """Capture webpage screenshot"""
        driver = self.setup_driver()
        if not driver:
            return None
            
        try:
            # Try both HTTP and HTTPS
            for protocol in ['https://', 'http://']:
                try:
                    url = f"{protocol}{domain}"
                    driver.get(url)
                    time.sleep(3)  # Wait for page load
                    
                    # Create evidence directory
                    os.makedirs('evidence/temp', exist_ok=True)
                    screenshot_path = f'evidence/temp/{domain.replace(".", "_")}.png'
                    driver.save_screenshot(screenshot_path)
                    
                    driver.quit()
                    return screenshot_path
                    
                except (TimeoutException, WebDriverException):
                    continue
                    
        except Exception as e:
            print(f"Screenshot capture failed for {domain}: {e}")
            
        driver.quit()
        return None
    
    def fetch_html(self, domain):
        """Fetch HTML content of domain"""
        try:
            for protocol in ['https://', 'http://']:
                try:
                    response = requests.get(f"{protocol}{domain}", timeout=10)
                    if response.status_code == 200:
                        return response.text
                except requests.RequestException:
                    continue
        except Exception as e:
            print(f"HTML fetch failed for {domain}: {e}")
            
        return ""
    
    def save_html(self, domain, html_content):
        """Save HTML content to file"""
        try:
            safe_domain = domain.replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evidence_dir = f'evidence/{safe_domain}/{timestamp}'
            os.makedirs(evidence_dir, exist_ok=True)
            
            html_path = f'{evidence_dir}/page.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return html_path
        except Exception as e:
            print(f"Failed to save HTML for {domain}: {e}")
            return ""
    
    def analyze_visual_similarity(self, screenshot_path):
        """Analyze visual similarity to CSE references"""
        if not screenshot_path or not os.path.exists(screenshot_path):
            return {'min_distance': float('inf'), 'is_lookalike': False}
        
        try:
            from herald.core.visual_analyzer import VisualAnalyzer
            visual_analyzer = VisualAnalyzer()
            
            min_distance = float('inf')
            for cse_name in self.cse_reference.keys():
                distance = visual_analyzer.compare_with_cse_templates(screenshot_path, cse_name)
                min_distance = min(min_distance, distance.get('min_distance', float('inf')))
            
            return {
                'min_distance': min_distance,
                'is_lookalike': min_distance <= self.visual_threshold
            }
        except Exception as e:
            print(f"Visual analysis failed: {e}")
            return {'min_distance': float('inf'), 'is_lookalike': False}
    
    def analyze_page_content(self, html_content):
        """Analyze page content for phishing indicators"""
        from herald.core.content_classifier import ContentClassifier
        classifier = ContentClassifier()
        
        try:
            has_credentials = classifier.detect_credential_forms(html_content)
            form_count = len(BeautifulSoup(html_content, 'html.parser').find_all('form'))
            
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title else ""
            
            return {
                'has_credentials': has_credentials,
                'form_count': form_count,
                'title': title
            }
        except Exception as e:
            print(f"Content analysis failed: {e}")
            return {'has_credentials': False, 'form_count': 0, 'title': ''}
    
    def determine_content_state(self, visual_analysis, content_analysis, html_content):
        """Determine content state based on analysis"""
        from herald.core.content_classifier import ContentClassifier
        classifier = ContentClassifier()
        
        # First check if it's a parked page
        if classifier.detect_parked_page(html_content):
            return 'parked'
        
        # Check for lookalike with credentials
        if (visual_analysis.get('is_lookalike', False) and 
            content_analysis.get('has_credentials', False)):
            return 'lookalike'
        
        # Check for unrelated content
        if (not content_analysis.get('has_credentials', False) and 
            not visual_analysis.get('is_lookalike', False)):
            return 'unrelated'
        
        return 'unknown'
    
    def capture_and_analyze_content(self, domain):
        """Capture page content and perform visual analysis"""
        screenshot_path = self.capture_screenshot(domain)
        html_content = self.fetch_html(domain)
        
        if screenshot_path and os.path.exists(screenshot_path):
            # Visual similarity analysis
            visual_analysis = self.analyze_visual_similarity(screenshot_path)
            
            # Content analysis
            content_analysis = self.analyze_page_content(html_content)
            
            # Combine analyses
            content_state = self.determine_content_state(
                visual_analysis, content_analysis, html_content
            )
            
            return {
                'screenshot_path': screenshot_path,
                'html_path': self.save_html(domain, html_content),
                'visual_distance': visual_analysis.get('min_distance', float('inf')),
                'has_credentials': content_analysis.get('has_credentials', False),
                'content_state': content_state,
                'is_lookalike': visual_analysis.get('is_lookalike', False),
                'page_title': content_analysis.get('title', ''),
                'form_count': content_analysis.get('form_count', 0)
            }
        
        return {'content_state': 'blocked', 'has_credentials': False}