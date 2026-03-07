# src/utils/enrich_data.py
import whois
import dns.resolver
import requests
import ipinfo
import socket
import time
import pandas as pd
import os

# Get free token from https://ipinfo.io/signup (free 50K requests)
IPINFO_TOKEN = "54d8daa3f3a83b"  # ← Replace with your token

def get_whois_info(domain):
    try:
        w = whois.whois(domain)
        return {
            'registrar': str(w.registrar) if w.registrar else "",
            'creation_date': str(w.creation_date[0]) if isinstance(w.creation_date, list) else str(w.creation_date),
            'country': str(w.country) if w.country else ""
        }
    except Exception as e:
        return {'registrar': '', 'creation_date': '', 'country': ''}

def get_dns_info(domain):
    try:
        # A record (IP)
        ip = str(dns.resolver.resolve(domain, 'A')[0])
        # MX records
        mx = [str(x.exchange) for x in dns.resolver.resolve(domain, 'MX')]
        mx_str = "; ".join(mx) if mx else ""
        return {'ip': ip, 'mx_records': mx_str}
    except:
        return {'ip': '', 'mx_records': ''}

def get_ipinfo(ip):
    try:
        handler = ipinfo.getHandler(IPINFO_TOKEN)
        details = handler.getDetails(ip)
        return {
            'asn': details.all.get('asn', {}).get('asn', ''),
            'org': details.all.get('org', ''),
            'country': details.all.get('country_name', '')
        }
    except:
        return {'asn': '', 'org': '', 'country': ''}

def check_ssl(domain):
    try:
        response = requests.get(f"https://{domain}", timeout=5)
        return True
    except:
        return False

def enrich_predictions():
    # Load top predictions
    df = pd.read_csv("outputs/high_conf_phishing_top500.csv")
    
    enriched_rows = []
    
    for idx, row in df.iterrows():
        domain = row['domain']
        print(f"[{idx+1}/{len(df)}] Enriching {domain}")
        
        # WHOIS
        whois_data = get_whois_info(domain)
        
        # DNS
        dns_data = get_dns_info(domain)
        
        # IP Geolocation (only if IP found)
        ipinfo_data = {'asn': '', 'org': '', 'country': ''}
        if dns_data['ip']:
            ipinfo_data = get_ipinfo(dns_data['ip'])
            time.sleep(0.1)  # Rate limit
        
        # SSL
        has_ssl = check_ssl(domain)
        
        # Combine
        enriched_row = {
            **row.to_dict(),
            'registrar': whois_data['registrar'],
            'domain_creation_date': whois_data['creation_date'],
            'registrant_country': whois_data['country'],
            'hosting_ip': dns_data['ip'],
            'mx_records': dns_data['mx_records'],
            'asn': ipinfo_data['asn'],
            'hosting_isp': ipinfo_data['org'],
            'hosting_country': ipinfo_data['country'],
            'has_ssl': has_ssl
        }
        enriched_rows.append(enriched_row)
    
    # Save enriched data
    df_enriched = pd.DataFrame(enriched_rows)
    df_enriched.to_csv("outputs/enriched_predictions.csv", index=False)
    print("✅ Enriched data saved!")
    return df_enriched

if __name__ == "__main__":
    print("🚀 Starting domain enrichment process...")
    enrich_predictions()
