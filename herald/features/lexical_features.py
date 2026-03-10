import pandas as pd
import numpy as np
import re
import math
import idna  # for decoding Punycode safely
from urllib.parse import urlparse
import Levenshtein


CSE_KEYWORDS = [
    # Original
    'nic', 'crsorgi', 'irctc', 'sbi', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'iocl',
    # Expanded
    'onlinesbi', 'sbicard', 'icicibank', 'hdfcbank', 'pnbindia', 'bankofbaroda',
    'airtel', 'irctc', 'rail', 'railway', 'gov.in', 'crsorgi', 'census', 'uidai'
]

MALICIOUS_TLDS = [
    'xyz','top','buzz','info','click','tk','cn',
    'cf','ga','gq','ml','bet','red','sbs','fun',
    'icu','cyou','shop','live','online','site'
]


def extract_url_features(df, domain_col='domain'):
    """Extract lexical features from domains (v3 with fuzzy matching and path analysis)"""
    urls = df[domain_col].astype(str).str.lower()
    
    # Extract domain/host and path for specialized features
    def parse_url_parts(u):
        if '//' not in u:
            u = '//' + u
        try:
            parsed = urlparse(u)
            return parsed.netloc or parsed.path.split('/')[0], parsed.path
        except ValueError:
            # Fallback for invalid URLs (e.g., malformed IPv6 brackets)
            domain = u.replace('//', '').split('/')[0]
            path = '/' + '/'.join(u.replace('//', '').split('/')[1:])
            return domain, path

    parsed_parts = urls.apply(parse_url_parts)
    domains = parsed_parts.apply(lambda x: x[0])
    paths = parsed_parts.apply(lambda x: x[1])
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Generalization Ratios
    features['domain_length'] = domains.str.len()
    features['digit_ratio'] = domains.str.count(r'\d') / np.maximum(features['domain_length'], 1)
    features['dot_ratio'] = domains.str.count(r'\.') / np.maximum(features['domain_length'], 1)
    features['num_hyphens'] = domains.str.count('-')
    features['hyphen_ratio'] = features['num_hyphens'] / np.maximum(features['domain_length'], 1)
    features['num_special_chars'] = domains.str.count(r'[^a-zA-Z0-9.-]')
    features['special_char_ratio'] = features['num_special_chars'] / np.maximum(features['domain_length'], 1)
    
    # 2. Registered Domain vs Subdomain Analysis
    def analyze_domain_parts(domain):
        parts = domain.split('.')
        if len(parts) < 2:
            return 0, 0, len(domain), 0, "" # Subdomains, BrandPos, RegLen, InSub, TLD
        
        # registered domain logic: simplified
        tld = parts[-1]
        reg_domain = ".".join(parts[-2:])
        subdomains = parts[:-2]
        
        # has_cse_keyword_in_subdomain
        has_cse_in_sub = any(any(keyword in sub for keyword in CSE_KEYWORDS) for sub in subdomains)
        
        # brand_keyword_position: 0=None, 1=Registered Domain, 2=Subdomain
        brand_pos = 0
        matched_kw = None
        for kw in CSE_KEYWORDS:
            if kw in reg_domain:
                brand_pos = 1
                matched_kw = kw
                break
            
        if brand_pos == 0 and has_cse_in_sub:
            brand_pos = 2
            # Find which keyword matched in subdomains for F3
            for kw in CSE_KEYWORDS:
                if any(kw in sub for sub in subdomains):
                    matched_kw = kw
                    break
            
        return len(subdomains), brand_pos, len(reg_domain), int(has_cse_in_sub), tld, matched_kw

    parts_analysis = domains.apply(lambda x: pd.Series(analyze_domain_parts(x)))
    features['subdomain_count'] = parts_analysis[0]
    features['brand_keyword_position'] = parts_analysis[1]
    features['registered_domain_length'] = parts_analysis[2]
    features['has_cse_keyword_in_subdomain'] = parts_analysis[3]
    tlds = parts_analysis[4].astype(str)
    matched_kws = parts_analysis[5]

    # F1: is_malicious_gtld
    features['is_malicious_gtld'] = tlds.isin(MALICIOUS_TLDS).astype(int)

    # F3: brand_to_reg_length_ratio
    def calc_brand_ratio(row):
        kw = row[5]
        reg_len = row[2]
        if isinstance(kw, str) and reg_len > 0:
            return len(kw) / reg_len
        return 0.0
    features['brand_to_reg_length_ratio'] = parts_analysis.apply(calc_brand_ratio, axis=1)

    # F2: min_brand_levenshtein
    def calc_min_levenshtein(domain):
        # Split on . and -
        tokens = re.split(r'[.-]', domain)
        min_norm_dist = 99.0
        for token in tokens:
            if len(token) < 3: continue
            for kw in CSE_KEYWORDS:
                dist = Levenshtein.distance(token, kw)
                if 1 <= dist <= 3:
                    norm_dist = dist / len(kw)
                    if norm_dist < min_norm_dist:
                        min_norm_dist = norm_dist
        return min_norm_dist
    features['min_brand_levenshtein'] = domains.apply(calc_min_levenshtein)

    # F4: has_brand_in_path
    features['has_brand_in_path'] = paths.apply(lambda p: int(any(kw in p for kw in CSE_KEYWORDS)))

    # Special patterns
    features['has_ip'] = domains.str.match(r'^\d+\.\d+\.\d+\.\d+$').fillna(False).astype(int)
    repeated_digits_match = domains.str.extract(r'(\d)\1{2,}', expand=False)
    features['has_repeated_digits'] = (~repeated_digits_match.isna()).astype(int)
    
    # New Features for v5b
    features['consonant_ratio'] = domains.str.count(r'[bcdfghjklmnpqrstvwxyz]') / np.maximum(features['domain_length'], 1)
    features['vowel_ratio'] = domains.str.count(r'[aeiou]') / np.maximum(features['domain_length'], 1)
    
    def get_longest_consonant_run(d):
        runs = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', str(d).lower())
        return max([len(r) for r in runs]) if runs else 0
    features['longest_consonant_run'] = domains.apply(get_longest_consonant_run)
    
    features['is_punycode'] = domains.str.startswith('xn--').astype(int)
    features['subdomain_depth'] = domains.apply(lambda x: max(str(x).count('.') - 1, 0))
    
    # TLD features
    features['is_common_tld'] = tlds.isin(['com', 'org', 'net', 'edu', 'gov']).fillna(False).astype(int)
    features['tld_length'] = tlds.str.len().fillna(0)
    
    # Entropy
    features['entropy'] = domains.apply(calculate_entropy)
    
    # Suspicious keywords
    suspicious_keywords = ['login', 'signin', 'secure', 'account', 'verify', 'update', 'support', 
                          'banking', 'paypal', 'auth', 'security', 'confirm']
    for keyword in suspicious_keywords:
        features[f'has_{keyword}'] = domains.str.contains(keyword, case=False, regex=False).fillna(False).astype(int)
    
    features['has_https'] = urls.str.startswith('https://').fillna(False).astype(int)
    
    # Fill NaN and ensure numeric
    features = features.fillna(0)
    # Drop intermediate columns if any
    cols_to_drop = []
    # Actually, the original code converted categorical to codes. Let's keep that pattern.
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].astype('category').cat.codes
        else:
            features[col] = features[col].astype(float)
    
    return pd.concat([df, features], axis=1)

def calculate_entropy(domain):
    """Calculate Shannon entropy of domain"""
    if len(domain) == 0:
        return 0
    entropy = 0
    for char in set(domain):
        p = domain.count(char) / len(domain)
        entropy -= p * np.log2(p) if p > 0 else 0
    return entropy