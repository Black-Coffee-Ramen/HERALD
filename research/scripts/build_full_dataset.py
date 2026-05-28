"""
scripts/build_full_dataset.py

Consolidates ALL available labeled domain data into a single unified CSV.

Sources:
  - data/raw/PS02_Training_set/PS02_Training_set.xlsx
  - data/raw/Mock_data/Mock_Data_*.xlsx  (15 files)
  - data/processed/synthetic_phish_v2.csv
  - data/external/phishtank_online.csv   (Indian sector filtered)

Outputs:
  - data/processed/full_dataset_v4.csv

NOTE: Shortlisting files are excluded — they contain domain names only, no labels.
"""

import os
import glob
import re
import pandas as pd
from urllib.parse import urlparse

# ─── Label normalisation ────────────────────────────────────────────────────

LABEL_MAP = {
    # Phishing variants
    'phishing': 'Phishing',
    'phishing ': 'Phishing',
    'phishing/suspected': 'Phishing',
    'phishing/ suspected': 'Phishing',
    # Suspected variants
    'suspected': 'Suspected',
    'suspected ': 'Suspected',
    'suspect': 'Suspected',
    'suspicious': 'Suspected',
    # Legitimate variants
    'legitimate': 'Legitimate',
    'legit': 'Legitimate',
    'benign': 'Legitimate',
    'safe': 'Legitimate',
}

def normalise_label(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    return LABEL_MAP.get(s, None)


# ─── Domain extraction from URL ─────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """Strip scheme/paths to get bare domain (or domain+subdomain)."""
    url = str(url).strip()
    if not url:
        return ''
    if '//' not in url:
        url = '//' + url
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path.split('/')[0]
        # Remove port if present
        host = host.split(':')[0].lower().strip()
        return host
    except Exception:
        return url.lower().strip()


# ─── Individual source loaders ──────────────────────────────────────────────

TRAINING_COL_DOMAIN = 'Identified Phishing/Suspected Domain Name'
TRAINING_COL_LABEL   = 'Phishing/Suspected Domains (i.e. Class Label)'


def load_training_set() -> pd.DataFrame:
    """PS02 official training set."""
    path = 'data/raw/PS02_Training_set/PS02_Training_set.xlsx'
    df = pd.read_excel(path)
    df = df[[TRAINING_COL_DOMAIN, TRAINING_COL_LABEL]].copy()
    df.columns = ['domain', 'label']
    df['source'] = 'training_set'
    return df


def load_mock_data() -> pd.DataFrame:
    """All 15 Mock_Data_*.xlsx files — same column schema as training set."""
    pattern = 'data/raw/Mock_data/Mock_Data_*.xlsx'
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no mock data files found at {pattern}")
        return pd.DataFrame(columns=['domain', 'label', 'source'])

    frames = []
    for fpath in files:
        try:
            df = pd.read_excel(fpath)
            fname = os.path.basename(fpath)
            # Find domain column (flexible)
            dom_col = next((c for c in df.columns if 'domain name' in c.lower() or 'domain' in c.lower()), None)
            lbl_col = next((c for c in df.columns if 'class label' in c.lower() or 'label' in c.lower()), None)
            if dom_col is None or lbl_col is None:
                print(f"  SKIP {fname}: could not find domain/label cols — {df.columns.tolist()}")
                continue
            sub = df[[dom_col, lbl_col]].copy()
            sub.columns = ['domain', 'label']
            sub['source'] = f'mock_{fname}'
            frames.append(sub)
        except Exception as e:
            print(f"  ERROR loading {fpath}: {e}")

    if not frames:
        return pd.DataFrame(columns=['domain', 'label', 'source'])
    return pd.concat(frames, ignore_index=True)


def load_synthetic() -> pd.DataFrame:
    """Synthetic phishing data generated previously."""
    df = pd.read_csv('data/processed/synthetic_phish_v2.csv')
    df = df[['domain', 'label']].copy()
    df['source'] = 'synthetic_v2'
    return df


# Indian financial/government keywords used to filter PhishTank
INDIAN_BRANDS = [
    'sbi', 'hdfcbank', 'icici', 'pnb', 'uidai', 'irctc', 'npci',
    'axisbank', 'bankofbaroda', 'indianbank', 'canarabank', 'incometax',
    'epfo', 'nsdl', 'sebi', 'nic', 'gov.in', 'india',
]

def load_phishtank() -> pd.DataFrame:
    """PhishTank CSV — filter for Indian sector, label all as Phishing."""
    df = pd.read_csv('data/external/phishtank_online.csv', low_memory=False)

    # Filter for online entries only (verified phishing still active)
    if 'online' in df.columns:
        df = df[df['online'].astype(str).str.lower() == 'yes']

    # Filter for Indian sector by target col or URL
    url_col = 'url'
    target_col = 'target' if 'target' in df.columns else None

    def is_indian(row):
        text = ''
        if target_col:
            text += str(row.get(target_col, '')).lower()
        text += str(row.get(url_col, '')).lower()
        return any(brand in text for brand in INDIAN_BRANDS)

    mask = df.apply(is_indian, axis=1)
    df_filtered = df[mask].copy()
    print(f"  PhishTank: {len(df)} total → {len(df_filtered)} Indian-sector rows kept")

    df_out = pd.DataFrame()
    df_out['domain'] = df_filtered[url_col].apply(extract_domain)
    df_out['label'] = 'Phishing'
    df_out['source'] = 'phishtank'
    return df_out


# ─── Main consolidation ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building full dataset v4")
    print("=" * 60)

    frames = []

    print("\n[1/4] Loading PS02 Training set...")
    df_train = load_training_set()
    print(f"  Rows: {len(df_train)}")
    frames.append(df_train)

    print("\n[2/4] Loading Mock data (15 files)...")
    df_mock = load_mock_data()
    print(f"  Rows: {len(df_mock)}")
    frames.append(df_mock)

    print("\n[3/4] Loading Synthetic phishing data...")
    df_synth = load_synthetic()
    print(f"  Rows: {len(df_synth)}")
    frames.append(df_synth)

    print("\n[4/4] Loading PhishTank (Indian sector filter)...")
    df_pt = load_phishtank()
    print(f"  Rows: {len(df_pt)}")
    frames.append(df_pt)

    # Combine
    print("\nCombining all sources...")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total before cleaning: {len(df)}")

    # Clean domains
    df['domain'] = df['domain'].astype(str).str.strip().str.lower()
    df['domain'] = df['domain'].apply(extract_domain)  # ensure bare domain
    df = df[df['domain'].notna() & (df['domain'] != '') & (df['domain'] != 'nan')]

    # Normalise labels
    df['label'] = df['label'].apply(normalise_label)
    before_drop = len(df)
    df = df[df['label'].notna()]
    dropped_label = before_drop - len(df)
    if dropped_label:
        print(f"  Dropped {dropped_label} rows with unrecognised labels")

    # Deduplicate on domain (keep first occurrence — training set is the most trusted source)
    before_dedup = len(df)
    df = df.drop_duplicates(subset='domain', keep='first')
    print(f"  Deduplication: {before_dedup} → {len(df)} rows")

    # Final stats
    print(f"\nFinal dataset size: {len(df)}")
    print("\nClass distribution:")
    print(df['label'].value_counts().to_string())
    print("\nSource breakdown:")
    # Summarise mock files as a single group
    df['_source_group'] = df['source'].apply(lambda s: 'mock_data' if s.startswith('mock_') else s)
    print(df['_source_group'].value_counts().to_string())
    df.drop(columns=['_source_group'], inplace=True)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/full_dataset_v4.csv'
    df[['domain', 'label', 'source']].to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
