import os
import pandas as pd
import numpy as np
import ssl
import socket
import dns.resolver
from datetime import datetime
from tqdm import tqdm
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress certificate verification warnings for extraction
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def get_ssl_features(domain, timeout=3):
    features = {
        'has_ssl': False,
        'cert_age_days': -1,
        'cert_days_remaining': -1,
        'is_lets_encrypt': False,
        'cert_domain_matches': False
    }
    
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # We just want to get the cert
        
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert(binary_form=True)
                features['has_ssl'] = True
                
                decoded_cert = ssock.getpeercert()
                
                if decoded_cert:
                    # Parse dates
                    # Format: 'Oct 15 00:00:00 2024 GMT'
                    not_before = datetime.strptime(decoded_cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                    not_after = datetime.strptime(decoded_cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    now = datetime.now()
                    
                    features['cert_age_days'] = (now - not_before).days
                    features['cert_days_remaining'] = (not_after - now).days
                    
                    # Check issuer
                    issuer = str(decoded_cert.get('issuer', ''))
                    if "Let's Encrypt" in issuer:
                        features['is_lets_encrypt'] = True
                    
                    # Check domain match (CN or SAN)
                    subject = dict(x[0] for x in decoded_cert.get('subject', []))
                    common_name = subject.get('commonName', '')
                    
                    # Extract SANs
                    sans = []
                    for entry in decoded_cert.get('subjectAltName', []):
                        if entry[0] == 'DNS':
                            sans.append(entry[1])
                    
                    if domain.lower() == common_name.lower() or domain.lower() in [s.lower() for s in sans]:
                        features['cert_domain_matches'] = True
                    elif common_name.startswith('*.'):
                        base_domain = common_name[2:].lower()
                        if domain.lower().endswith('.' + base_domain) or domain.lower() == base_domain:
                            features['cert_domain_matches'] = True

    except Exception:
        pass
        
    return features

def get_dns_features(domain, timeout=3):
    features = {
        'has_mx': False,
        'has_spf': False,
        'a_record_count': 0,
        'ttl_value': -1
    }
    
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout
    
    try:
        # MX Records
        try:
            mx_answers = resolver.resolve(domain, 'MX')
            if len(mx_answers) > 0:
                features['has_mx'] = True
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout, dns.resolver.NoRootSOA):
            pass
            
        # SPF Records (TXT)
        try:
            txt_answers = resolver.resolve(domain, 'TXT')
            for rdata in txt_answers:
                if 'v=spf1' in str(rdata):
                    features['has_spf'] = True
                    break
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout, dns.resolver.NoRootSOA):
            pass
            
        # A Records
        try:
            a_answers = resolver.resolve(domain, 'A')
            features['a_record_count'] = len(a_answers)
            if len(a_answers) > 0:
                features['ttl_value'] = a_answers.rrset.ttl
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers, dns.exception.Timeout, dns.resolver.NoRootSOA):
            pass
            
    except Exception:
        pass
        
    return features

def extract_for_domain(domain):
    ssl_feat = get_ssl_features(domain)
    dns_feat = get_dns_features(domain)
    res = {'domain': domain}
    res.update(ssl_feat)
    res.update(dns_feat)
    return res

def main():
    print("=" * 60)
    print("HERALD Network Feature Extraction (v6 - Fast Mode)")
    print("=" * 60)

    v5_feat_path = 'data/processed/full_features_v5.csv'
    whois_path = 'data/processed/full_dataset_v5_whois.csv'
    out_path = 'data/processed/full_features_v6.csv'
    checkpoint_path = 'data/processed/network_checkpoint.csv'

    if not os.path.exists(v5_feat_path) or not os.path.exists(whois_path):
        print("ERROR: Input files missing.")
        return

    # Load data
    df_lex = pd.read_csv(v5_feat_path)
    df_whois = pd.read_csv(whois_path)
    
    # Merge WHOIS age into features
    df = df_lex.merge(df_whois[['domain', 'domain_age_days']], on='domain', how='left')
    
    # Checkpoint logic
    if os.path.exists(checkpoint_path):
        df_cp = pd.read_csv(checkpoint_path)
        net_cols = ['has_ssl', 'cert_age_days', 'cert_days_remaining', 'is_lets_encrypt', 
                    'cert_domain_matches', 'has_mx', 'has_spf', 'a_record_count', 'ttl_value']
        # Map existing results
        cp_map = df_cp.set_index('domain')[net_cols].to_dict('index')
        for domain, f in cp_map.items():
            idx = df.index[df['domain'] == domain]
            if not idx.empty:
                for col in net_cols:
                    df.at[idx[0], col] = f[col]
        print(f"Resuming from checkpoint: {df['has_ssl'].notna().sum()} domains already processed.")
    else:
        for col in ['has_ssl', 'is_lets_encrypt', 'cert_domain_matches', 'has_mx', 'has_spf']:
            df[col] = None
        for col in ['cert_age_days', 'cert_days_remaining', 'a_record_count', 'ttl_value']:
            df[col] = None

    to_process = df[df['has_ssl'].isna()]
    domains_to_process = to_process['domain'].tolist()
    print(f"Processing {len(domains_to_process)} domains with 20 threads...")

    results_count = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(extract_for_domain, d): d for d in domains_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Network Extraction"):
            res = future.result()
            domain = res['domain']
            idx = df.index[df['domain'] == domain][0]
            for k, v in res.items():
                if k != 'domain':
                    df.at[idx, k] = v
            
            results_count += 1
            if results_count % 200 == 0:
                df[df['has_ssl'].notna()].to_csv(checkpoint_path, index=False)

    # Final cleanup and save
    bool_cols = ['has_ssl', 'is_lets_encrypt', 'cert_domain_matches', 'has_mx', 'has_spf']
    int_cols = ['cert_age_days', 'cert_days_remaining', 'a_record_count', 'ttl_value', 'domain_age_days']
    
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)
    for col in int_cols:
        df[col] = df[col].fillna(-1).astype(int)

    df.to_csv(out_path, index=False)
    print(f"\nSaved v6 features to {out_path} ({len(df)} rows)")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

if __name__ == '__main__':
    main()
