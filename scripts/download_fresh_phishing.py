import urllib.request
import pandas as pd
import os
from urllib.parse import urlparse
import io

def get_domain(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""

def main():
    print("Starting fresh phishing data download...")
    os.makedirs('data/external', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 1. OpenPhish
    print("Downloading OpenPhish feed...")
    openphish_url = 'https://openphish.com/feed.txt'
    try:
        req = urllib.request.Request(openphish_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        openphish_raw = response.read().decode('utf-8')
        with open('data/external/openphish_fresh.txt', 'w') as f:
            f.write(openphish_raw)
        openphish_domains = [get_domain(line.strip()) for line in openphish_raw.splitlines() if line.strip()]
        openphish_domains = [d for d in openphish_domains if d]
        print(f"OpenPhish: Found {len(openphish_domains)} domains.")
    except Exception as e:
        print(f"Error downloading OpenPhish: {e}")
        openphish_domains = []

    # 2. PhishTank
    print("Downloading PhishTank feed...")
    phishtank_url = 'http://data.phishtank.com/data/online-valid.csv'
    try:
        # Use a very specific User-Agent as required by PhishTank
        headers = {
            'User-Agent': 'phishtank/herald-bot-v1.0',
            'Accept': '*/*'
        }
        req = urllib.request.Request(phishtank_url, headers=headers)
        response = urllib.request.urlopen(req)
        phishtank_df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        phishtank_df.to_csv('data/external/phishtank_fresh.csv', index=False)
        phishtank_domains = [get_domain(url) for url in phishtank_df['url']]
        phishtank_domains = [d for d in phishtank_domains if d]
        print(f"PhishTank: Found {len(phishtank_domains)} domains.")
    except Exception as e:
        print(f"Error downloading PhishTank: {e}")
        phishtank_domains = []

    # 3. URLhaus (Full recent CSV)
    print("Downloading URLhaus feed...")
    urlhaus_url = 'https://urlhaus.abuse.ch/downloads/csv_recent/'
    try:
        req = urllib.request.Request(urlhaus_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        urlhaus_content = response.read().decode('utf-8')
        # URLhaus CSV doesn't have headers in the file itself usually, or it's commented out
        urlhaus_df = pd.read_csv(io.StringIO(urlhaus_content), comment='#', 
                                names=['id', 'dateadded', 'url', 'url_status', 'last_online', 'threat', 'tags', 'urlhaus_link', 'reporter'],
                                escapechar='\\', quoting=1, on_bad_lines='skip')
        urlhaus_df.to_csv('data/external/urlhaus_fresh.csv', index=False)
        
        # Filter for phishing tags or threat
        urlhaus_filtered = urlhaus_df[
            (urlhaus_df['tags'].astype(str).str.contains('phishing', case=False, na=False)) | 
            (urlhaus_df['threat'] == 'phishing') |
            (urlhaus_df['url'].astype(str).str.contains('verification|login|account|secure|update', case=False, na=False))
        ]
        urlhaus_domains = [get_domain(url) for url in urlhaus_filtered['url']]
        urlhaus_domains = [d for d in urlhaus_domains if d]
        print(f"URLhaus: Found {len(urlhaus_domains)} domains.")
    except Exception as e:
        print(f"Error downloading URLhaus: {e}")
        urlhaus_domains = []

    # 4. Mitchell Krogza - Phishing.Database (Massive source)
    print("Downloading Mitchell Krogza Phishing Database...")
    mk_url = 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt'
    try:
        req = urllib.request.Request(mk_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        mk_raw = response.read().decode('utf-8')
        mk_domains = [line.strip() for line in mk_raw.splitlines() if line.strip() and not line.startswith('#')]
        print(f"Mitchell Krogza: Found {len(mk_domains)} domains.")
    except Exception as e:
        print(f"Error downloading Mitchell Krogza: {e}")
        mk_domains = []

    # 5. Stamparm - Blackbook
    print("Downloading Stamparm Blackbook...")
    stam_url = 'https://raw.githubusercontent.com/stamparm/blackbook/master/blackbook.txt'
    try:
        req = urllib.request.Request(stam_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        stam_raw = response.read().decode('utf-8')
        stam_domains = [line.strip() for line in stam_raw.splitlines() if line.strip() and not line.startswith('#')]
        print(f"Stamparm: Found {len(stam_domains)} domains.")
    except Exception as e:
        print(f"Error downloading Stamparm: {e}")
        stam_domains = []

    # 6. Combine and Deduplicate
    all_fresh = list(set(openphish_domains + phishtank_domains + urlhaus_domains + mk_domains + stam_domains))
    print(f"Combined fresh phishing domains: {len(all_fresh)}")

    # Remove those already in v7
    v7_path = 'data/processed/full_dataset_v7.csv'
    if os.path.exists(v7_path):
        v7_df = pd.read_csv(v7_path)
        v7_domains = set(v7_df['domain'].str.lower())
        new_unique_phish = [d for d in all_fresh if d not in v7_domains]
        print(f"New unique phishing domains (not in v7): {len(new_unique_phish)}")
    else:
        print("Warning: full_dataset_v7.csv not found. Using all fresh domains.")
        v7_df = pd.DataFrame(columns=['domain', 'label', 'source'])
        new_unique_phish = all_fresh

    # Save fresh domains
    fresh_df = pd.DataFrame({'domain': new_unique_phish, 'label': 'Phishing', 'source': 'Fresh-Feed'})
    fresh_df.to_csv('data/external/fresh_phishing_domains.csv', index=False)
    print("Saved fresh phishing domains to data/external/fresh_phishing_domains.csv")

    # 5. Rebuild Dataset
    full_v9_df = pd.concat([v7_df, fresh_df], ignore_index=True)
    full_v9_df.drop_duplicates(subset=['domain'], inplace=True)
    full_v9_df.to_csv('data/processed/full_dataset_v9_raw.csv', index=False)
    
    print("\nFinal v9 Class Distribution:")
    print(full_v9_df['label'].value_counts())
    print(f"Total domains in v9: {len(full_v9_df)}")

if __name__ == "__main__":
    main()
