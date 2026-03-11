"""
herald/features/content_features.py

Extracts HTML content features for Stage 2 enrichment (borderline domains).
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

def default_features():
    """Returns a dictionary of default features with negative or zero values."""
    return {
        'redirected_to_different_domain': False,
        'redirect_count': 0,
        'has_password_field': False,
        'has_login_form': False,
        'form_action_external': False,
        'num_forms': 0,
        'has_favicon': False,
        'external_resource_ratio': 0.0,
        'page_text_length': 0,
        'has_copyright': False,
        'title_has_brand': False,
        'has_eval_js': False,
        'has_obfuscated_js': False,
        'iframe_count': 0,
        'hidden_element_count': 0
    }

def extract_content_features(domain, timeout=5):
    """
    Fetch and analyze the content of a domain to extract phishing-relevant features.
    """
    try:
        # 1. Fetch the page
        resp = requests.get(
            f'http://{domain}',
            timeout=timeout,
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        html = resp.text
        final_url = resp.url
        base_domain = urlparse(f'http://{domain}').netloc.lower().replace('www.', '')
        final_domain = urlparse(final_url).netloc.lower().replace('www.', '')
        
        # 2. Extract features using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        features = default_features()
        
        # REDIRECT FEATURES
        features['redirected_to_different_domain'] = (base_domain != final_domain)
        features['redirect_count'] = len(resp.history)
        
        # FORM FEATURES
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        
        for form in forms:
            pwd_field = form.find('input', {'type': 'password'})
            if pwd_field:
                features['has_password_field'] = True
                features['has_login_form'] = True
            
            action = form.get('action', '')
            if action.startswith('http'):
                action_domain = urlparse(action).netloc.lower().replace('www.', '')
                if action_domain and action_domain != base_domain:
                    features['form_action_external'] = True
        
        # CONTENT FEATURES
        features['has_favicon'] = bool(soup.find('link', rel=re.compile(r'icon', re.I)))
        
        # External resource ratio (scripts and images)
        resources = soup.find_all(['script', 'img'])
        external_count = 0
        total_resources = len(resources)
        for res in resources:
            src = res.get('src', '')
            if src.startswith('http'):
                src_domain = urlparse(src).netloc.lower().replace('www.', '')
                if src_domain and src_domain != base_domain:
                    external_count += 1
        
        if total_resources > 0:
            features['external_resource_ratio'] = external_count / total_resources
            
        features['page_text_length'] = len(soup.get_text())
        
        text_content = soup.get_text().lower()
        features['has_copyright'] = ('©' in html or 'copyright' in text_content)
        
        # Title brand check (Requires CSE keywords)
        title = soup.title.string.lower() if soup.title else ""
        from herald.features.lexical_features import CSE_KEYWORDS
        features['title_has_brand'] = any(kw.lower() in title for kw in CSE_KEYWORDS)
        
        # SUSPICIOUS PATTERNS
        features['has_eval_js'] = 'eval(' in html
        features['has_obfuscated_js'] = any(p in html for p in ['fromCharCode', 'unescape'])
        features['iframe_count'] = len(soup.find_all('iframe'))
        
        # Hidden elements (simplified check)
        hidden_styles = ['display:none', 'display: none', 'visibility:hidden', 'visibility: hidden']
        hidden_count = 0
        for tag in soup.find_all(True, style=True):
            style = tag.get('style', '').lower()
            if any(s in style for s in hidden_styles):
                hidden_count += 1
        features['hidden_element_count'] = hidden_count
        
        return features
        
    except Exception:
        return default_features()

if __name__ == "__main__":
    import sys
    test_domain = sys.argv[1] if len(sys.argv) > 1 else "google.com"
    print(f"Testing features for: {test_domain}")
    print(extract_content_features(test_domain))
