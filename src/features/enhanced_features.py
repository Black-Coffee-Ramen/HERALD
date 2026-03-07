# src/features/enhanced_features.py
import pandas as pd
import numpy as np
import re
import math
import os
import socket
import whois
import datetime
import geoip2.database  # requires GeoLite2-City.mmdb (local)
from PIL import Image
import cv2
import imagehash
from pdf2image import convert_from_path

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def entropy(s):
    """Calculate Shannon entropy of a string."""
    if len(s) == 0:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log(p, 2) for p in prob if p > 0)


# -------------------------------------------------------------------------
# Visual Features
# -------------------------------------------------------------------------
def extract_visual_features(pdf_path):
    """Extract visual features from a domain's evidence PDF."""
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        img = images[0]

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        phash = imagehash.phash(img)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        img_array = np.array(img)
        color_diversity = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))

        return {
            "visual_complexity": edge_density,
            "color_diversity": color_diversity,
            "perceptual_hash": str(phash)
        }
    except Exception:
        return {
            "visual_complexity": 0,
            "color_diversity": 0,
            "perceptual_hash": "0" * 16
        }


# -------------------------------------------------------------------------
# Lexical Features
# -------------------------------------------------------------------------
def extract_lexical_features(df, domain_col="domain"):
    """Extract lexical features from domain strings."""
    df = df.copy()
    domains = df[domain_col].astype(str).str.lower()

    df["domain_length"] = domains.str.len()
    df["num_dots"] = domains.str.count(r"\.")
    df["num_hyphens"] = domains.str.count("-")
    df["num_underscores"] = domains.str.count("_")
    df["num_special_chars"] = domains.str.count(r"[^a-z0-9\.-]")
    df["num_digits"] = domains.str.count(r"\d")

    df["has_repeated_digits"] = domains.str.contains(r"(\d)\1{2,}").astype(int)

    def get_subdomains(domain):
        parts = domain.split(".")
        return parts[:-2] if len(parts) > 2 else []

    subdomains_list = domains.apply(get_subdomains)
    df["num_subdomains"] = subdomains_list.apply(len)
    df["avg_subdomain_length"] = subdomains_list.apply(
        lambda subs: np.mean([len(s) for s in subs]) if subs else 0
    )
    df["max_subdomain_length"] = subdomains_list.apply(
        lambda subs: max([len(s) for s in subs]) if subs else 0
    )

    df["domain_entropy"] = domains.apply(entropy)
    df["is_idn"] = domains.str.startswith("xn--").astype(int)

    suspicious_tlds = ["xyz", "top", "cyou", "sbs", "buzz", "icu", "shop", "site", "live", "ltd"]
    df["suspicious_tld"] = domains.str.split(".").str[-1].isin(suspicious_tlds).astype(int)

    return df


# -------------------------------------------------------------------------
# WHOIS + DNS Features
# -------------------------------------------------------------------------
def extract_whois_dns_features(df, domain_col="domain", geoip_db_path="data/GeoLite2-City.mmdb"):
    """
    Add WHOIS + DNS-based features:
      - domain_age_days
      - registrar_reputation
      - ip_country
      - ip_in_india
    """
    df = df.copy()
    domain_list = df[domain_col].astype(str)
    reader = None
    if os.path.exists(geoip_db_path):
        try:
            reader = geoip2.database.Reader(geoip_db_path)
        except Exception:
            reader = None

    whois_data = []
    for domain in domain_list:
        features = {
            "domain_age_days": np.nan,
            "registrar": "",
            "registrar_risk": 0,
            "ip_country": "",
            "ip_in_india": 0
        }

        # --- WHOIS ---
        try:
            w = whois.whois(domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if isinstance(creation, datetime.datetime):
                age = (datetime.datetime.now() - creation).days
                features["domain_age_days"] = age
            registrar = str(w.registrar or "").lower()
            features["registrar"] = registrar

            bad_registrars = ["namecheap", "public domain registry", "pdr ltd", "tucows"]
            if any(bad in registrar for bad in bad_registrars):
                features["registrar_risk"] = 1
        except Exception:
            pass

        # --- DNS + GEO ---
        try:
            ip = socket.gethostbyname(domain)
            if reader:
                try:
                    record = reader.city(ip)
                    country = record.country.name or ""
                    features["ip_country"] = country
                    if country.lower() == "india":
                        features["ip_in_india"] = 1
                except Exception:
                    pass
        except Exception:
            pass

        whois_data.append(features)

    whois_df = pd.DataFrame(whois_data)
    if reader:
        reader.close()

    return pd.concat([df, whois_df], axis=1)


# -------------------------------------------------------------------------
# Combined Feature Extractor
# -------------------------------------------------------------------------
def extract_all_features(df, evidence_dir="data/raw/PS02_Training_set/Evidences"):
    """
    Combine lexical, WHOIS, DNS, and visual features for full enrichment.
    """
    df = extract_lexical_features(df)
    df = extract_whois_dns_features(df)

    visual_features = []
    for _, row in df.iterrows():
        domain = row["domain"]
        evidence_path = f"{evidence_dir}/{domain}.pdf"

        if pd.notna(domain) and os.path.exists(evidence_path):
            vis = extract_visual_features(evidence_path)
        else:
            vis = {"visual_complexity": 0, "color_diversity": 0, "perceptual_hash": "0" * 16}
        visual_features.append(vis)

    vis_df = pd.DataFrame(visual_features)
    df = pd.concat([df, vis_df], axis=1)
    return df
